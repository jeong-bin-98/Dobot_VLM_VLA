# DOBOT × Pi0 / Pi0-FAST 파이프라인 가이드

---

## 1. Server Setup -- Pi0 / Pi0-FAST 설치 (A6000 서버)

Official docs: https://huggingface.co/docs/lerobot/pi0fast

### 1-1. LeRobot 설치

```bash
# Python 3.12 권장 (lerobot v0.5.0+)
conda create -y -n lerobot python=3.12
conda activate lerobot
conda install -y -c conda-forge ffmpeg

# LeRobot + Pi0 dependencies
git clone https://github.com/huggingface/lerobot.git
cd lerobot
pip install -e ".[pi]"

# LoRA 파인튜닝이 필요한 경우
pip install -e ".[pi,peft]"
```

> **Note:** lerobot v0.4.x에서는 `pip install "lerobot[pi]@git+https://github.com/huggingface/lerobot.git"` 사용.
> v0.5.0부터 pip install로 직접 설치 가능.

### 1-2. Pretrained Base Models

| Model | HuggingFace ID | 용도 |
|-------|---------------|------|
| Pi0 base | `lerobot/pi0_base` | Flow-matching 파인튜닝 |
| Pi0-FAST base | `lerobot/pi0_fast_base` | Autoregressive 파인튜닝 |
| FAST tokenizer | `physical-intelligence/fast` | 액션 토크나이저 (1M+ 시퀀스로 학습됨) |

---

## 2. Pi0 vs Pi0-FAST 비교

| | **Pi0** | **Pi0-FAST** |
|---|---------|-------------|
| 방식 | Flow-matching (50-step denoising) | Autoregressive (FAST token 예측) |
| 학습 속도 | 1x | **~5x 빠름** |
| 추론 시간 | ~1.4s (50 steps) | token-by-token (KV-cache 지원) |
| Backbone | PaliGemma (SigLIP + Gemma 2B) | 동일 |
| 언어 조건화 | O | O |
| GPU 요구 | A6000+ (48GB VRAM) | A6000+ (48GB VRAM) |
| LoRA 지원 | O | O |
| 추천 상황 | Pi0-FAST가 안 될 때 | **기본 추천** |

---

## 3. Training -- 학습

### 핵심 파라미터: `chunk_size` vs `n_action_steps`

```
chunk_size = 모델이 한 번에 예측하는 미래 액션 수 (예: 10)
n_action_steps = 실제로 실행하는 액션 수 (예: 1~2)
```

**DOBOT Magician의 경우:**
- `move_to(wait=True)` 한 번에 ~0.3~1.0초 소요 (역기구학 + 물리 이동)
- delta 액션은 위치 변화량이므로 1~2 스텝만 실행해도 충분
- 나머지는 버리고 다시 관측 -> 예측 (closed-loop)

```
n_action_steps=1  <- 가장 보수적, 매 스텝마다 재관측 (추천)
n_action_steps=2  <- 2 스텝 연속 실행 후 재관측
```

### 3-1. Pi0-FAST 학습

```bash
lerobot-train \
    --dataset.repo_id=local/dataset_v3 \
    --dataset.root=./dataset_v3 \
    --policy.type=pi0_fast \
    --policy.pretrained_path=lerobot/pi0_fast_base \
    --policy.dtype=bfloat16 \
    --policy.gradient_checkpointing=true \
    --policy.chunk_size=10 \
    --policy.n_action_steps=1 \
    --policy.max_action_tokens=256 \
    --batch_size=4 \
    --steps=100000 \
    --output_dir=outputs/pi0fast_dobot \
    --policy.device=cuda \
    --wandb.enable=false
```

### 3-2. Pi0 (flow-matching) 학습

```bash
lerobot-train \
    --dataset.repo_id=local/dataset_v3 \
    --dataset.root=./dataset_v3 \
    --policy.type=pi0 \
    --policy.pretrained_path=lerobot/pi0_base \
    --policy.dtype=bfloat16 \
    --policy.gradient_checkpointing=true \
    --policy.chunk_size=50 \
    --policy.n_action_steps=1 \
    --batch_size=4 \
    --steps=100000 \
    --output_dir=outputs/pi0_dobot \
    --policy.device=cuda \
    --wandb.enable=false
```

### 3-3. Pi0-FAST 커스텀 FAST Tokenizer 학습 (선택)

기본 `physical-intelligence/fast` tokenizer로 충분하지만,
DOBOT 데이터에 최적화된 토크나이저를 원한다면:

```bash
lerobot-train-tokenizer \
    --repo_id local/dataset_v3 \
    --action_horizon 10 \
    --encoded_dims "0:5" \
    --vocab_size 1024 \
    --scale 10.0 \
    --normalization_mode QUANTILES \
    --output_dir ./dobot_fast_tokenizer
```

학습 시 커스텀 토크나이저 지정:
```bash
--policy.action_tokenizer_name=./dobot_fast_tokenizer
```

### 3-4. LoRA 파인튜닝 (VRAM 절약)

전체 모델 학습이 VRAM 부족하면 LoRA 사용:

```bash
lerobot-train \
    --dataset.repo_id=local/dataset_v3 \
    --dataset.root=./dataset_v3 \
    --policy.type=pi0_fast \
    --policy.pretrained_path=lerobot/pi0_fast_base \
    --policy.dtype=bfloat16 \
    --policy.gradient_checkpointing=true \
    --policy.chunk_size=10 \
    --policy.n_action_steps=1 \
    --peft.type=lora \
    --peft.r=16 \
    --peft.lora_alpha=32 \
    --policy.optimizer_lr=1e-3 \
    --batch_size=4 \
    --steps=50000 \
    --output_dir=outputs/pi0fast_lora_dobot
```

> LoRA는 학습률을 일반 파인튜닝의 ~10배로 설정 (1e-4 -> 1e-3)

### 3-5. Expert-Only 파인튜닝 (VLM 동결)

VLM은 고정하고 action expert만 학습:

```bash
--policy.train_expert_only=true
```

---

## 4. Inference -- 추론

### 방식 A: HTTP Server/Client (현재 구현)

```
[DOBOT + 카메라] --base64 JPEG + state---> [A6000 서버] --delta actions---> [DOBOT]
   (local)              HTTP POST              (remote)          HTTP response
```

**장점:** 구현 완료, 분리된 구조, 디버깅 용이
**단점:** HTTP 오버헤드 (~50-100ms per request)

```bash
# 서버 (A6000)
PI0_POLICY_TYPE=pi0_fast \
PI0_MODEL_PATH=./outputs/pi0fast_dobot/checkpoints/last/pretrained_model \
python server/pi0_server.py

# 클라이언트 (DOBOT 연결 PC)
python client/pi0_dobot_client.py \
    --server http://192.168.1.100:8000 \
    --port COM4 --cam1 0 --cam2 1 \
    --task "pick up the red cup" \
    --chunk-size 1
```

### 방식 B: A6000 서버에서 직접 추론 (카메라 스트리밍)

A6000 서버에서 직접 추론하고, 카메라 영상을 서버로 스트리밍하는 방식.
HTTP 오버헤드 제거, 더 낮은 지연시간.

```
[카메라 2대]                      [A6000 서버]              [DOBOT]
  RTSP/ZMQ   --video stream--->  Pi0-FAST 추론  --serial---> move_to()
  (local)                        + DOBOT 제어               (USB forwarded)
```

**구현 방법 1: SSH 터널 + USB 포워딩**

카메라와 DOBOT 모두 서버에서 직접 접근:

```bash
# DOBOT PC에서 -- 카메라를 RTSP 서버로 노출
# (ffmpeg 사용)
ffmpeg -f v4l2 -video_size 640x480 -i /dev/video0 \
    -c:v libx264 -preset ultrafast -tune zerolatency \
    -f rtsp rtsp://0.0.0.0:8554/top

ffmpeg -f v4l2 -video_size 640x480 -i /dev/video1 \
    -c:v libx264 -preset ultrafast -tune zerolatency \
    -f rtsp rtsp://0.0.0.0:8554/wrist
```

```bash
# DOBOT 시리얼 포트를 서버에 TCP로 포워딩
# (socat 사용)
socat TCP-LISTEN:5001,reuseaddr,fork FILE:/dev/ttyUSB0,b115200,raw
```

```bash
# A6000 서버에서 -- 직접 추론 실행
python direct_inference.py \
    --model ./outputs/pi0fast_dobot/checkpoints/last/pretrained_model \
    --camera_top rtsp://192.168.1.50:8554/top \
    --camera_wrist rtsp://192.168.1.50:8554/wrist \
    --dobot_host 192.168.1.50 --dobot_port 5001 \
    --task "pick up the red cup"
```

**구현 방법 2: ZMQ 기반 스트리밍 (LeRobot G1 패턴 참조)**

LeRobot v0.5.0의 Unitree G1은 ZMQ 카메라 스트리밍을 사용:
```python
# 카메라 설정 (ZMQ 기반)
cameras = {
    "top": {
        "type": "zmq",
        "server_address": "192.168.1.50",
        "port": 5555,
        "camera_name": "top_camera",
        "width": 640, "height": 480, "fps": 30
    }
}
```

**구현 방법 3: 단순한 방식 -- DOBOT PC를 얇은 프록시로**

가장 실용적인 방식. DOBOT PC는 카메라 캡처 + DOBOT 제어만 담당하고,
추론은 전부 A6000 서버에서 수행. 현재 `client/pi0_dobot_client.py`가
이미 이 구조:

```
pi0_dobot_client.py (DOBOT PC)
  +-- 카메라 캡처 -> JPEG base64 인코딩 -> HTTP POST
  +-- 서버 응답 수신 (delta actions)
  +-- DOBOT move_to() 실행

pi0_server.py (A6000)
  +-- 이미지 디코딩
  +-- Pi0-FAST 추론
  +-- delta actions 반환
```

이 구조가 이미 "서버에서 직접 추론" 패턴입니다.
HTTP 오버헤드는 JPEG 인코딩/디코딩 포함 약 50-100ms로,
DOBOT의 move_to() 물리 이동 시간 (300-1000ms) 대비 무시 가능합니다.

> **결론:** 현재 server/client 구조를 유지하는 것이 가장 실용적.
> 카메라를 서버에 직접 연결하려면 USB 연장이 물리적으로 불가능하므로
> RTSP/ZMQ 스트리밍이 필요하고, 이는 HTTP POST보다 복잡도만 높아짐.

---

## 5. DOBOT 작업 공간 제한

```python
BOUNDS = {
    "x": (150, 310),   # mm
    "y": (-150, 150),  # mm
    "z": (-30, 150),   # mm
    "r": (-90, 90),    # degrees
}
```

Delta는 클램프하지 않음. 최종 좌표만 범위 내로 제한.

---

## 6. 그리퍼 이슈 (현장 디버깅 필요)

추론 중 그리퍼 미작동 문제. 코드 내 조정 가능한 상수:

```python
GRIPPER_DELAY_AFTER_MOVE_S = 0.1   # move 후 대기 -- 0.2~0.5로 높여볼 것
GRIPPER_ACTION_WAIT_S = 0.5        # grip 완료 대기
GRIPPER_THRESHOLD = 0.5            # action[4] > 0.5 -> grip ON
```

현장에서 우선 `GRIPPER_DELAY_AFTER_MOVE_S`를 0.3으로 올려보세요.

---

## 7. 데이터 수집 -> 학습 -> 추론 전체 플로우

```bash
# 1) 데이터 수집 (DOBOT PC)
python scripts/01_collect_data.py \
    --port COM4 --cam1 0 --cam2 1 \
    --task "pick up the red cup" \
    --save_dir ./dataset_v3

# 2) 검증 + 수정 (DOBOT PC)
python scripts/03_validate_dataset.py --dataset_dir ./dataset_v3 --fix

# 3) 데이터를 서버로 전송
scp -r ./dataset_v3 user@a6000-server:~/

# 4) 학습 (A6000 서버)
ssh user@a6000-server
cd ~/
lerobot-train \
    --dataset.repo_id=local/dataset_v3 \
    --dataset.root=./dataset_v3 \
    --policy.type=pi0_fast \
    --policy.pretrained_path=lerobot/pi0_fast_base \
    --policy.dtype=bfloat16 \
    --policy.gradient_checkpointing=true \
    --policy.chunk_size=10 \
    --policy.n_action_steps=1 \
    --batch_size=4 \
    --steps=100000 \
    --output_dir=outputs/pi0fast_dobot

# 5) 추론 서버 실행 (A6000 서버)
PI0_POLICY_TYPE=pi0_fast \
PI0_MODEL_PATH=./outputs/pi0fast_dobot/checkpoints/last/pretrained_model \
python server/pi0_server.py

# 6) 추론 실행 (DOBOT PC)
python client/pi0_dobot_client.py \
    --server http://a6000-server:8000 \
    --port COM4 --cam1 0 --cam2 1 \
    --task "pick up the red cup" \
    --chunk-size 1
```

---

## Reference Links

- Pi0 공식 문서: https://huggingface.co/docs/lerobot/pi0
- Pi0-FAST 공식 문서: https://huggingface.co/docs/lerobot/pi0fast
- PEFT/LoRA 가이드: https://huggingface.co/docs/lerobot/peft_training
- FAST Tokenizer: https://huggingface.co/physical-intelligence/fast
- OpenPI (Pi0 원본): https://github.com/Physical-Intelligence/openpi
- LeRobot GitHub: https://github.com/huggingface/lerobot
