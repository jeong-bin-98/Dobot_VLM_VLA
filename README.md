# DOBOT Magician x LeRobot x Pi0 / Pi0-FAST

**부산 로보틱스 AI 교육 프로그램 — 단일 로봇 팔 모방학습(Imitation Learning) + VLA(Vision-Language-Action) + 음성 제어 파이프라인**

DOBOT Magician에서 데이터를 수집하고, Pi0 또는 Pi0-FAST 모델로 학습한 뒤, GPU 서버에서 HTTP 추론을 실행하여 로컬 로봇을 원격 제어합니다.

---

## 전체 파이프라인

```
+----------------------------------------------------------------------+
|                        TRAINING PIPELINE                             |
|                                                                      |
|  [Mac/로컬]                                  [GPU 서버]               |
|   1) Collect Data  ──scp──► 2) Train (Pi0 또는 Pi0-FAST)             |
|   (단일 팔 순차 수집)            (LoRA 또는 Full fine-tuning)          |
|                                                                      |
+----------------------------------------------------------------------+
|                        INFERENCE PIPELINE                            |
|                                                                      |
|  [GPU 서버]                        [Mac/로컬 (DOBOT + 카메라)]         |
|   Pi0 Server v2 ◄─── POST /predict ──  Client                        |
|   (FastAPI)          {image_top,        ├─ 텍스트:                    |
|                       image_wrist,      │  --task "pick up the snack" |
|                       state,            └─ 음성:                      |
|                       language}            마이크 → STT → Claude API  |
+----------------------------------------------------------------------+
```

---

## 모델 선택: Pi0 vs Pi0-FAST

두 가지 VLA 모델을 지원합니다. 데이터셋과 목적에 따라 선택하세요.

| 항목 | **Pi0** (flow-matching) | **Pi0-FAST** (autoregressive + FAST tokenizer) |
|------|------------------------|------------------------------------------------|
| Base 모델 | `lerobot/pi0_base` | `lerobot/pi0fast-base` |
| Action 출력 | 연속값 직접 출력 (flow-matching) | FAST 토큰 시퀀스 디코딩 |
| 추론 속도 | 느림 (denoising 10 스텝) | 빠름 (5x) |
| 소량 데이터 안정성 | 높음 | 낮음 (토큰 포맷 학습 실패 위험) |
| 학습 속도 | 느림 | 빠름 |
| 언제 쓸까 | 데이터 적거나 FAST tokenizer 오버플로우 발생 시 | 데이터 충분 + 추론 속도 중요 시 |

**소량 데이터에서 Pi0-FAST가 garbage 토큰을 뱉는다면 Pi0로 전환하세요.**

---

## 학습 실행 (train.sh)

`train.sh`는 Pi0 / Pi0-FAST × LoRA / Full fine-tuning 4가지 조합을 지원합니다.

```bash
./train.sh [데이터셋경로] [GPU번호] [스텝수] [출력경로] [resume] [full] [pi0|pi0_fast]
```

### 4가지 학습 모드

```bash
# 1. pi0_fast + LoRA (기본, 추천)
./train.sh ./dataset/snack_dataset_v1 0 20000 outputs/snack_fast_lora

# 2. pi0_fast + Full fine-tuning (데이터 충분할 때)
./train.sh ./dataset/snack_dataset_v1 0 20000 outputs/snack_fast_full "" full

# 3. pi0 + LoRA
./train.sh ./dataset/snack_dataset_v1 0 20000 outputs/snack_pi0_lora "" "" pi0

# 4. pi0 + Full fine-tuning
./train.sh ./dataset/snack_dataset_v1 0 20000 outputs/snack_pi0_full "" full pi0
```

### 이어서 학습 (resume)

```bash
./train.sh ./dataset/snack_dataset_v1 0 40000 outputs/snack_fast_lora resume
```

### 여러 데이터셋 합쳐서 학습

```bash
./train.sh "./dataset/snack_v1 ./dataset/tissue_v1 ./dataset/drink_v1" 0 20000 outputs/multi_v1
```

---

## 추론 서버 (pi0_server_v2)

공식 lerobot 파이프라인(`make_pre_post_processors`)을 사용한 통합 추론 서버. Pi0/Pi0-FAST 모두 같은 방식으로 실행합니다.

### 기본 실행

```bash
# Pi0-FAST (빠름)
CUDA_VISIBLE_DEVICES=0 \
PI0_POLICY_TYPE=pi0_fast \
PI0_MODEL_PATH=./outputs/snack_fast_lora/checkpoints/last/pretrained_model \
python server/pi0_server_v2.py

# Pi0 (안정적)
CUDA_VISIBLE_DEVICES=0 \
PI0_POLICY_TYPE=pi0 \
PI0_MODEL_PATH=./outputs/snack_pi0_lora/checkpoints/last/pretrained_model \
python server/pi0_server_v2.py
```

### torch.compile 가속 (선택)

`PI0_COMPILE=1` 추가 시 `torch.compile(mode="max-autotune")`이 sample_actions에 적용되어 1.5~2배 가속.

```bash
CUDA_VISIBLE_DEVICES=0 \
PI0_COMPILE=1 \
PI0_POLICY_TYPE=pi0 \
PI0_MODEL_PATH=./outputs/snack_pi0_lora/checkpoints/last/pretrained_model \
python server/pi0_server_v2.py
```

**주의**: 서버 시작 시 warmup이 자동 실행되며 첫 컴파일에 2~5분 걸립니다. `Warmup 완료`가 뜨면 준비 완료.

### 환경변수

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `PI0_POLICY_TYPE` | `pi0` | `pi0` 또는 `pi0_fast` |
| `PI0_MODEL_PATH` | — | 체크포인트 경로 (`...checkpoints/last/pretrained_model`) |
| `PI0_COMPILE` | `0` | `1`이면 torch.compile 적용 |
| `PI0_PORT` | `8000` | HTTP 포트 |
| `CUDA_VISIBLE_DEVICES` | — | 사용할 GPU 번호 (예: `0`, `1`, `0,1`) |

---

## 클라이언트 (로봇 실행)

### 텍스트 명령 (기본)

```bash
python client/pi0_dobot_client.py \
    --server http://<서버IP>:8000 \
    --task "pick up the snack" \
    --cam1 0 --cam2 1
```

### 음성 명령 (Claude API)

`.env`에 `ANTHROPIC_API_KEY` 설정 후:

```bash
python client/pi0_voice_claude_client.py \
    --server http://<서버IP>:8000
```

음성으로 "과자 줘", "배고파", "스트레스 받아" 등을 말하면 Claude가 학습 task로 자동 변환합니다.

---

## 음성 제어 파이프라인

```
마이크 → STT (faster-whisper / Qwen2.5-Omni)
  → 한국어 텍스트 ("과자 줘" / "배고파")
  → Claude API (의도 분류)
  → COMMAND_MAP (학습 task 문자열 매칭)
  → "pick up the snack"
  → Pi0 서버 /predict → 로봇 실행
```

Pi0 VLA 모델은 학습 시 사용한 정확한 영어 task 문자열이 들어가야 정확도가 높습니다. Claude API가 사용자의 자연어(한국어)를 분류하고, `COMMAND_MAP`에서 학습 때 사용한 정확한 영어 프롬프트를 조회해 Pi0에 전달합니다.

### 음성 명령 매핑

| 한국어 키워드 | Pi0 task |
|--------------|---------|
| 과자, 칸초, 간식, 배고프다 | `pick up the snack` |
| 음료, 피크닉, 물, 목마르다 | `pick up the drink` |
| 휴지, 코 | `pick up the tissue` |
| 스트레스, 스트레스볼 | `pick up the stress ball` |

---

## 파일 구조

```
Dobot_VLM_VLA/
├── .env                              # API 키 (ANTHROPIC_API_KEY, HF_TOKEN)
├── README.md                         # 본 문서
├── requirements.txt
├── train.sh                          # Pi0 / Pi0-FAST 학습 스크립트
│
├── scripts/
│   ├── 01_collect_data.py            # 데이터 수집 (단일 팔 순차, 절대경로 저장)
│   ├── 03_validate_dataset.py        # 데이터셋 검증 + 자동 수정
│   ├── 05_inference_dobot.py         # 서버 없이 로컬 추론 (Mac)
│   ├── merge_datasets.py             # 멀티 태스크 데이터셋 합치기
│   └── test_dobot.py                 # DOBOT 동작 테스트
│
├── server/
│   ├── pi0_server.py                 # (구) 수동 tokenizer, pi0_fast 위주
│   └── pi0_server_v2.py              # (권장) 공식 lerobot 파이프라인, pi0/pi0_fast 통합
│
├── client/
│   ├── pi0_dobot_client.py           # 텍스트 명령 클라이언트 (기본)
│   ├── pi0_voice_claude_client.py    # 음성 명령 클라이언트 (Claude API)
│   ├── pi0_voice_client.py           # 음성 클라이언트 (로컬 Qwen3 LLM)
│   ├── voice_module.py               # STT 모듈 (whisper / qwen 백엔드)
│   ├── chatbot_module.py             # 챗봇 분류 모듈 (Qwen3)
│   ├── pi0_ws_client.py              # WebSocket 클라이언트 (실험)
│   └── test_voice_claude_pipeline.py # 음성 파이프라인 테스트 (로봇 없이)
│
└── docs/
    ├── team_guide.md                 # 팀원용 단계별 실행 가이드
    ├── PIPELINE.md                   # 파이프라인 설계 문서
    └── architecture_comparison.md    # 아키텍처 비교
```

---

## 설치

```bash
pip install -r requirements.txt
```

`.env` 파일 예시:

```
ANTHROPIC_API_KEY=sk-ant-xxxxx
HF_TOKEN=hf_xxxxx
```

---

## 알려진 문제 및 해결

### FAST tokenizer OverflowError

**증상**: 학습 중 `OverflowError: Python int too large to convert to C int`

**원인**: 액션 차원 중 하나가 분산이 매우 작고 극단값이 있으면, 정규화 후 DCT + scale(10)이 chr() 유니코드 한계(1,114,111)를 초과.

**해결**: `.venv/lib/python3.12/site-packages/lerobot/processor/tokenizer_processor.py`의 `_tokenize_action`에 clamp 추가 (이 프로젝트에 이미 적용됨):

```python
action_cpu = action[i : i + 1].cpu()
action_cpu = action_cpu.clamp(-6.0, 6.0)  # FAST tokenizer overflow 방지
tokens = self.action_tokenizer(action_cpu)
```

### Pi0-FAST가 garbage 토큰을 뱉는 경우

데이터가 적거나(1000 프레임 이하) 모델이 아직 `Action :` 포맷을 제대로 학습하지 못한 경우. **Pi0(flow-matching)로 전환**하면 토큰 포맷 문제 자체가 없어집니다.

```bash
./train.sh ./dataset/snack_v1 0 20000 outputs/snack_pi0 "" "" pi0
```

### torch.compile reduce-overhead 모드 에러

**증상**: `RuntimeError: Offset increment outside graph capture encountered unexpectedly`

**원인**: Pi0는 flow-matching이라 매 추론마다 랜덤 노이즈를 샘플링하는데, `mode="reduce-overhead"`의 CUDA graphs와 RNG가 충돌.

**해결**: `mode="max-autotune"` 사용 (pi0_server_v2에 이미 적용됨).

---

## 체크포인트 선택 가이드

학습 후 `outputs/*/checkpoints/` 안에 여러 체크포인트가 저장됩니다 (기본 5000 스텝마다 + `last`).

1. 먼저 **`last`** 로 추론 테스트
2. 결과가 불안정하면 **`010000`, `015000`** 등 중간 체크포인트 시도 (오버피팅 가능성)
3. 실제 로봇에서 가장 잘 되는 걸 선택

lerobot에는 자동으로 "best" 체크포인트를 고르는 기능이 없습니다. 실제 로봇 동작을 보고 판단해야 합니다.

---

## 하드웨어

| 구성 | 사양 |
|------|------|
| 로봇 | DOBOT Magician (USB, CH340/CP210x) |
| 카메라 | USB 카메라 2대 (cam1=wrist, cam2=top), 640x480 |
| 학습/추론 GPU | A6000+ (48GB VRAM 권장) |
| 로컬 클라이언트 | Mac (mac 브랜치) / Windows (window 브랜치) |

---

## 더 읽을거리

- 팀원용 단계별 가이드: [docs/team_guide.md](./docs/team_guide.md)
- 파이프라인 설계: [docs/PIPELINE.md](./docs/PIPELINE.md)

---

See [LICENSE](./LICENSE)
