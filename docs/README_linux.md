# DOBOT Magician x LeRobot x Pi0 / Pi0-FAST — 🐧 Linux 가이드

> **이 문서는 Linux 환경 전용 가이드입니다.** 다른 환경은 [→ 메인 README](../README.md)로 돌아가 Windows / macOS 가이드를 고르세요.

**부산 로보틱스 AI 교육 프로그램 — 단일 로봇 팔 모방학습(Imitation Learning) + VLA(Vision-Language-Action) + 음성 제어 파이프라인**

Linux (Ubuntu 20.04/22.04 기준) 로컬 머신에서 DOBOT Magician과 카메라를 연결하여 데이터 수집 / 추론 클라이언트를 실행하거나, **GPU 서버에서 학습과 추론 서버를 운영**하기 위한 가이드입니다.

---

## 🐧 Linux 전용 사전 준비

### 1. Python 설치

Ubuntu 22.04 기본 Python은 3.10입니다. 없다면:

```bash
sudo apt update
sudo apt install -y python3.10 python3.10-venv python3-pip
```

또는 pyenv / conda 사용:

```bash
conda create -n dobot python=3.10
conda activate dobot
```

### 2. DOBOT USB 드라이버 (CH340 / CP210x)

Ubuntu 커널에는 `ch341` / `cp210x` 모듈이 **기본 포함**되어 있습니다. 별도 드라이버 설치 없이 USB 연결만 하면 됩니다.

연결 후 확인:

```bash
dmesg | tail -20
# → ch341-uart converter now attached to ttyUSB0 같은 메시지 확인

ls /dev/ttyUSB*
# → /dev/ttyUSB0
```

### 3. `dialout` 그룹 권한 (중요)

USB 시리얼 장치(`/dev/ttyUSB*`)에 접근하려면 현재 사용자가 `dialout` 그룹에 속해야 합니다. 안 그러면 `Permission denied` 발생.

```bash
sudo usermod -a -G dialout $USER
# 로그아웃 후 재로그인 (또는 재부팅) 필요
groups   # dialout이 보이는지 확인
```

이게 싫으면 `sudo`로 실행할 수도 있지만 비권장.

### 4. 카메라 접근 권한 (`video` 그룹)

```bash
sudo usermod -a -G video $USER
# 로그아웃 후 재로그인
```

USB 카메라 인식 확인:

```bash
ls /dev/video*
# → /dev/video0 /dev/video2 ...

# v4l-utils로 상세 정보
sudo apt install -y v4l-utils
v4l2-ctl --list-devices
```

### 5. 시스템 의존성

```bash
sudo apt install -y \
    git \
    ffmpeg \
    libgl1 libglib2.0-0 \
    libsm6 libxext6 libxrender1 \
    portaudio19-dev     # 음성 클라이언트용 마이크 I/O
```

### 6. NVIDIA 드라이버 + CUDA (GPU 서버 운영 시)

```bash
nvidia-smi   # 드라이버 설치 확인

# CUDA 없으면:
sudo apt install -y nvidia-driver-535
# 재부팅 후
nvidia-smi
```

PyTorch는 `requirements.txt` 설치 시 CUDA 버전에 맞춰 자동 설치됩니다. 문제가 있으면:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 7. 저장소 클론 및 의존성 설치

```bash
git clone -b linux https://github.com/jeong-bin-98/Dobot_VLM_VLA.git
cd Dobot_VLM_VLA
pip install -r requirements.txt
```

`.env` 파일 (프로젝트 루트에 생성):

```
ANTHROPIC_API_KEY=sk-ant-xxxxx
HF_TOKEN=hf_xxxxx
```

---

## 전체 파이프라인

```
+----------------------------------------------------------------------+
|                        TRAINING PIPELINE                             |
|                                                                      |
|  [Linux 로컬]                                [Linux GPU 서버]         |
|   1) Collect Data  ──scp──► 2) Train (Pi0 또는 Pi0-FAST)             |
|   (DOBOT + USB 카메라 2대)       (LoRA 또는 Full fine-tuning)          |
|                                                                      |
+----------------------------------------------------------------------+
|                        INFERENCE PIPELINE                            |
|                                                                      |
|  [Linux GPU 서버]                   [Linux 로컬 (DOBOT + 카메라)]      |
|   Pi0 Server v2 ◄─── POST /predict ──  Client                        |
|   (FastAPI)          {image_top,        ├─ 텍스트:                    |
|                       image_wrist,      │  --task "pick up the snack" |
|                       state,            └─ 음성:                      |
|                       language}            마이크 → STT → Claude API  |
+----------------------------------------------------------------------+
```

---

## 데이터 수집 (Linux 로컬)

### 카메라 백엔드: `v4l2`

Linux는 V4L2(Video4Linux2)가 표준입니다. `--cam-backend`의 기본값(`auto`)은 Linux에서 자동으로 `CAP_V4L2`를 선택합니다. 명시하려면:

```bash
python scripts/01_collect_data.py \
    --cam1 0 --cam2 2 \
    --task "pick up the snack" \
    --save_dir ./dataset/snack_dataset_v1 \
    --cam-backend v4l2
```

> 카메라 인덱스는 `/dev/video0` = 0, `/dev/video2` = 2 등으로 매핑됩니다. `v4l2-ctl --list-devices`로 확인하세요. (Linux에서는 홀수 번호가 metadata 장치일 수 있어 짝수만 시도하는 편이 안전합니다.)

### DOBOT 포트: `/dev/ttyUSB*` 형식

`--port`를 생략하면 자동으로 CH340/CP210x 칩을 탐지합니다. 필요하면 직접 지정:

```bash
python scripts/01_collect_data.py \
    --port /dev/ttyUSB0 \
    --cam1 0 --cam2 2 \
    --task "pick up the snack" \
    --save_dir ./dataset/snack_dataset_v1
```

### 수집한 데이터 GPU 서버로 전송

```bash
scp -r ./dataset/snack_dataset_v1 user@<서버IP>:/home/user/datasets/

# 또는 rsync가 더 안정적
rsync -avz --progress ./dataset/snack_dataset_v1 user@<서버IP>:/home/user/datasets/
```

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

# 2. pi0_fast + Full fine-tuning
./train.sh ./dataset/snack_dataset_v1 0 20000 outputs/snack_fast_full "" full

# 3. pi0 + LoRA
./train.sh ./dataset/snack_dataset_v1 0 20000 outputs/snack_pi0_lora "" "" pi0

# 4. pi0 + Full fine-tuning
./train.sh ./dataset/snack_dataset_v1 0 20000 outputs/snack_pi0_full "" full pi0
```

### 이어서 학습 / 여러 데이터셋

```bash
# resume
./train.sh ./dataset/snack_dataset_v1 0 40000 outputs/snack_fast_lora resume

# multi-task
./train.sh "./dataset/snack_v1 ./dataset/tissue_v1 ./dataset/drink_v1" 0 20000 outputs/multi_v1
```

### 장시간 학습: tmux / nohup

SSH 세션이 끊겨도 학습이 계속되도록:

```bash
# tmux 사용
sudo apt install -y tmux
tmux new -s train
./train.sh ./dataset/snack_dataset_v1 0 20000 outputs/snack_fast_lora
# Ctrl+B, D 로 detach
# 나중에: tmux attach -t train

# 또는 nohup
nohup ./train.sh ./dataset/snack_dataset_v1 0 20000 outputs/snack_fast_lora > train.log 2>&1 &
tail -f train.log
```

---

## 추론 서버 (pi0_server_v2)

공식 lerobot 파이프라인(`make_pre_post_processors`)을 사용한 통합 추론 서버. Pi0/Pi0-FAST 모두 같은 방식으로 실행합니다.

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

```bash
CUDA_VISIBLE_DEVICES=0 \
PI0_COMPILE=1 \
PI0_POLICY_TYPE=pi0 \
PI0_MODEL_PATH=./outputs/snack_pi0_lora/checkpoints/last/pretrained_model \
python server/pi0_server_v2.py
```

**주의**: 서버 시작 시 warmup 자동 실행, 첫 컴파일에 2~5분. `Warmup 완료` 뜨면 준비 완료.

### 환경변수

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `PI0_POLICY_TYPE` | `pi0` | `pi0` 또는 `pi0_fast` |
| `PI0_MODEL_PATH` | — | 체크포인트 경로 |
| `PI0_COMPILE` | `0` | `1`이면 torch.compile 적용 |
| `PI0_PORT` | `8000` | HTTP 포트 |
| `CUDA_VISIBLE_DEVICES` | — | 사용할 GPU 번호 |

### 방화벽 (ufw)

서버에서 포트 개방:

```bash
sudo ufw allow 8000/tcp
sudo ufw status
```

---

## 클라이언트 (로봇 실행, Linux 로컬)

### 텍스트 명령 (기본)

```bash
python client/pi0_dobot_client.py \
    --server http://<서버IP>:8000 \
    --task "pick up the snack" \
    --cam1 0 --cam2 2
```

포트 지정이 필요하면:

```bash
python client/pi0_dobot_client.py \
    --server http://<서버IP>:8000 \
    --port /dev/ttyUSB0 \
    --task "pick up the snack" \
    --cam1 0 --cam2 2
```

### 음성 명령 (Claude API)

`.env`에 `ANTHROPIC_API_KEY` 설정 후:

```bash
python client/pi0_voice_claude_client.py --server http://<서버IP>:8000
```

마이크 장치 확인:

```bash
python -c "import sounddevice as sd; print(sd.query_devices())"
```

PulseAudio/PipeWire 기본 장치가 올바르게 잡혀 있어야 합니다.

---

## 모델 선택: Pi0 vs Pi0-FAST

| 항목 | **Pi0** (flow-matching) | **Pi0-FAST** (autoregressive + FAST tokenizer) |
|------|------------------------|------------------------------------------------|
| Base 모델 | `lerobot/pi0_base` | `lerobot/pi0fast-base` |
| Action 출력 | 연속값 직접 출력 | FAST 토큰 시퀀스 디코딩 |
| 추론 속도 | 느림 (denoising 10 스텝) | 빠름 (5x) |
| 소량 데이터 안정성 | 높음 | 낮음 |
| 언제 쓸까 | 데이터 적을 때, FAST 토큰 오버플로우 시 | 데이터 충분 + 속도 중요 시 |

**소량 데이터에서 Pi0-FAST가 garbage 토큰을 뱉는다면 Pi0로 전환하세요.**

---

## 음성 제어 파이프라인

```
마이크(Linux) → STT (faster-whisper / Qwen2.5-Omni)
  → 한국어 텍스트 ("과자 줘" / "배고파")
  → Claude API (의도 분류)
  → COMMAND_MAP (학습 task 문자열 매칭)
  → "pick up the snack"
  → Pi0 서버 /predict → DOBOT 실행
```

### 음성 명령 매핑

| 한국어 키워드 | Pi0 task |
|--------------|---------|
| 과자, 칸초, 간식, 배고프다 | `pick up the snack` |
| 음료, 피크닉, 물, 목마르다 | `pick up the drink` |
| 휴지, 코 | `pick up the tissue` |
| 스트레스, 스트레스볼 | `pick up the stress ball` |

---

## 🐧 Linux 전용 트러블슈팅

| 증상 | 해결 |
|------|------|
| `Permission denied: '/dev/ttyUSB0'` | `sudo usermod -a -G dialout $USER` 후 재로그인 |
| `/dev/ttyUSB*`가 안 보임 | `dmesg \| tail`로 커널 메시지 확인, USB 케이블/포트 교체 |
| 카메라 `Permission denied` | `sudo usermod -a -G video $USER` 후 재로그인 |
| 카메라 인덱스 불일치 | `v4l2-ctl --list-devices`로 실제 `/dev/video*` 번호 확인, `--cam1 0 --cam2 2`처럼 짝수 번호 사용 |
| `libGL.so.1: cannot open shared object` | `sudo apt install libgl1 libglib2.0-0` |
| `ALSA lib ... cannot find card` (음성) | `sudo apt install portaudio19-dev`, PulseAudio 재시작 (`systemctl --user restart pulseaudio`) |
| `CUDA out of memory` | LoRA 모드 사용 (`full` 플래그 빼기), 배치 크기 축소 |
| 학습 중 `OverflowError: Python int too large` | tokenizer_processor.py `clamp(-6.0, 6.0)` (이미 적용됨) |
| `RuntimeError: Offset increment outside graph capture` | `PI0_COMPILE=1` 사용 |

전체 트러블슈팅 목록: **[TROUBLESHOOTING.md](./TROUBLESHOOTING.md)**

---

## systemd 서비스로 추론 서버 상시 실행 (선택)

서버를 백그라운드 서비스로 관리하려면 `/etc/systemd/system/pi0-server.service` 생성:

```ini
[Unit]
Description=Pi0 Inference Server
After=network.target

[Service]
Type=simple
User=<사용자명>
WorkingDirectory=/home/<사용자명>/Dobot_VLM_VLA
Environment="CUDA_VISIBLE_DEVICES=0"
Environment="PI0_POLICY_TYPE=pi0_fast"
Environment="PI0_MODEL_PATH=/home/<사용자명>/Dobot_VLM_VLA/outputs/snack_fast_lora/checkpoints/last/pretrained_model"
ExecStart=/usr/bin/python3 server/pi0_server_v2.py
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now pi0-server
sudo systemctl status pi0-server
journalctl -u pi0-server -f
```

---

## 파일 구조

```
Dobot_VLM_VLA/
├── .env                              # API 키 (ANTHROPIC_API_KEY, HF_TOKEN)
├── README.md                         # 본 문서 (Linux 가이드)
├── requirements.txt
├── train.sh                          # Pi0 / Pi0-FAST 학습 스크립트
│
├── scripts/
│   ├── 01_collect_data.py            # 데이터 수집
│   ├── 03_validate_dataset.py        # 데이터셋 검증 + 자동 수정
│   ├── 05_inference_dobot.py         # 서버 없이 로컬 추론
│   ├── merge_datasets.py             # 멀티 태스크 데이터셋 합치기
│   └── test_dobot.py                 # DOBOT 동작 테스트
│
├── server/
│   ├── pi0_server.py                 # (구) 수동 tokenizer 기반
│   └── pi0_server_v2.py              # (권장) 통합 추론 서버
│
├── client/
│   ├── pi0_dobot_client.py           # 텍스트 명령 클라이언트
│   ├── pi0_voice_claude_client.py    # 음성 명령 클라이언트
│   ├── voice_module.py               # STT 모듈
│   └── ...
│
└── docs/
    ├── team_guide.md
    ├── TROUBLESHOOTING.md
    ├── PIPELINE.md
    └── ...
```

---

## 하드웨어

| 구성 | 사양 |
|------|------|
| 로봇 | DOBOT Magician (USB, CH340/CP210x) |
| 카메라 | USB 카메라 2대 (cam1=wrist, cam2=top), 640x480 |
| 학습/추론 GPU | A6000+ (48GB VRAM 권장) |
| 로컬/서버 OS | **Ubuntu 20.04 / 22.04** (본 브랜치) |

---

## 더 읽을거리

| 문서 | 설명 |
|------|------|
| [team_guide.md](./team_guide.md) | 비전공자용 단계별 실행 가이드 |
| [TROUBLESHOOTING.md](./TROUBLESHOOTING.md) | 오류 해결 모음 |
| [PIPELINE.md](./PIPELINE.md) | 파이프라인 아키텍처 설계 |
| [data_collection_guide.md](./data_collection_guide.md) | 데이터 수집 하드웨어 셋업 |
| [architecture_comparison.md](./architecture_comparison.md) | STT / LLM / VLA 모델 비교 |

---

See [LICENSE](../LICENSE)
