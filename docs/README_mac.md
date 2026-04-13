# DOBOT Magician x LeRobot x Pi0 / Pi0-FAST — 🍎 macOS 가이드

> **이 문서는 macOS 환경 전용 가이드입니다.** 다른 환경은 [→ 메인 README](../README.md)로 돌아가 Windows / Linux 가이드를 고르세요.
>
> 💡 **코드는 OS 자동 감지로 Mac/Windows/Linux 모두 동작합니다.** 이 가이드는 macOS 환경에 특화된 설치/설정 안내입니다.

**부산 로보틱스 AI 교육 프로그램 — 단일 로봇 팔 모방학습(Imitation Learning) + 음성 제어 파이프라인**

Mac (Intel / Apple Silicon) 로컬 머신에서 DOBOT Magician + USB 카메라 2대를 사용해 데이터 수집 및 추론 클라이언트를 실행하는 가이드입니다. 학습은 별도 Linux GPU 서버에서 수행합니다.

---

## 🍎 macOS 전용 사전 준비

### 1. Homebrew 설치 (패키지 관리)

[brew.sh](https://brew.sh) 안내대로 설치:

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

### 2. Python 설치

```bash
brew install python@3.10
# 또는
brew install pyenv
pyenv install 3.10.13
pyenv global 3.10.13
```

가상환경 권장:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
```

또는 Anaconda/Miniconda 사용:

```bash
conda create -n dobot python=3.10
conda activate dobot
```

### 3. DOBOT USB 드라이버 (CH340 / CP210x)

Mac에서는 칩셋에 따라 드라이버 설치가 필요합니다.

- **Apple Silicon (M1/M2/M3)**: macOS 11+ 에는 CH34x 드라이버가 **기본 포함**되어 별도 설치 불필요한 경우가 많습니다.
- **Intel Mac / 인식 안 되는 경우**:
  - **CH340/CH341**: [WCH macOS 드라이버](http://www.wch-ic.com/downloads/CH341SER_MAC_ZIP.html)
  - **CP210x**: [Silicon Labs macOS 드라이버](https://www.silabs.com/developers/usb-to-uart-bridge-vcp-drivers)

드라이버 설치 후 **시스템 설정 → 개인정보/보안**에서 "Allow" 클릭이 필요할 수 있습니다. 재부팅 권장.

DOBOT을 USB로 연결 후 포트 확인:

```bash
ls /dev/tty.usb* /dev/cu.usb*
# → /dev/tty.usbserial-0001   /dev/cu.usbserial-0001
```

> 자동 포트 탐지가 실패하면 `--port /dev/tty.usbserial-0001` 형태로 직접 지정하세요.

### 4. 카메라 권한 (AVFoundation)

Mac은 OpenCV가 **AVFoundation** 백엔드로 카메라에 접근합니다. 본 브랜치의 스크립트들은 이미 AVFoundation으로 고정되어 있어 별도 옵션 지정이 필요 없습니다.

첫 실행 시 **시스템 설정 → 개인정보 → 카메라**에서 터미널(또는 IDE)에 카메라 접근 권한을 허용해야 합니다. 권한 요청 팝업이 뜨지 않으면 터미널 앱을 완전히 종료 후 재실행하세요.

카메라 인덱스는 `0, 1, 2...` 순서로 `--cam1 0 --cam2 1` 처럼 지정. 바뀌는 경우가 있으니 안 되면 인덱스를 바꿔가며 시도하세요.

### 5. 시스템 의존성

```bash
brew install ffmpeg portaudio
```

`portaudio`는 음성 클라이언트의 마이크 입력에 필요합니다.

### 6. 저장소 클론 및 의존성 설치

```bash
git clone -b mac https://github.com/jeong-bin-98/Dobot_VLM_VLA.git
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
+-------------------------------------------------------------------+
|                      TRAINING PIPELINE                            |
|                                                                   |
|  [Mac 로컬]                              [Linux GPU 서버]          |
|   1) Collect → 2) Convert v3 → 3) Validate  →  4) Train (Pi0-FAST)|
|   (DOBOT + AVFoundation 카메라)              (LoRA fine-tuning)    |
+-------------------------------------------------------------------+
|                      INFERENCE PIPELINE                           |
|                                                                   |
|  [Linux GPU 서버]                     [Mac 로컬 (DOBOT + 카메라)]   |
|   Pi0-FAST Server ◄─── POST /predict ──  Client                   |
|   (FastAPI)       {image_top, image_wrist,   ├─ 텍스트:            |
|                    state, language}          └─ 음성:              |
|                                                 마이크 → STT →     |
|                                                 Claude → task      |
+-------------------------------------------------------------------+
```

---

## 음성 제어 파이프라인

```
마이크(Mac) → STT (faster-whisper / Qwen 2.5 Omni)
  → 한국어 텍스트 ("과자 줘" / "배고파")
  → Claude API (의도 분류)
  → COMMAND_MAP (학습 task 문자열 매칭)
  → "pick up the snack"
  → Pi0 서버 /predict → DOBOT 실행
```

Pi0 VLA 모델은 학습 시 사용한 정확한 영어 task 문자열이 들어가야 정확도가 높습니다. Claude API가 사용자의 자연어(한국어)를 분류하고, `COMMAND_MAP`에서 학습 때 사용한 정확한 영어 프롬프트를 조회하여 Pi0에 전달합니다.

---

## 사용법

### 1. 데이터 수집 (Mac)

```bash
python scripts/01_collect_data.py \
    --cam1 0 --cam2 1 \
    --task "pick up the snack" \
    --save_dir ./dataset/snack_dataset_v1
```

> 카메라 매핑: `--cam1` = wrist 카메라, `--cam2` = top 카메라
> 카메라 백엔드는 AVFoundation으로 자동/고정 — 별도 `--cam-backend` 지정 불필요.
> DOBOT 포트 자동 탐지 실패 시: `--port /dev/tty.usbserial-0001` 추가.

### 2. 데이터셋 포맷 변환 (v2 → v3)

```bash
python scripts/02_convert_v2_to_v3.py --input ./dataset/snack_dataset_v1
```

### 3. 검증

```bash
python scripts/03_validate_dataset.py --dataset_dir ./dataset/snack_dataset_v1 --fix
```

### 4. GPU 서버로 데이터 전송

```bash
scp -r ./dataset/snack_dataset_v1 user@<서버IP>:/home/user/datasets/
```

### 5. 학습 (GPU 서버에서 실행)

```bash
# 서버에 SSH 접속 후
./train.sh ./dataset/snack_dataset_v1 1 10000 outputs/snack_v1
```

> 인자 순서: `데이터셋경로` `GPU번호` `학습스텝수` `출력경로` `[resume]`

### 6. 추론 서버 실행 (GPU 서버)

```bash
PI0_POLICY_TYPE=pi0_fast \
PI0_MODEL_PATH=./outputs/snack_v1/checkpoints/last/pretrained_model \
python server/pi0_server.py
```

### 7. 클라이언트 실행 (Mac)

#### 텍스트 명령 (기본)

```bash
python client/pi0_dobot_client.py \
    --server http://<서버IP>:8000 \
    --task "pick up the snack"
```

#### 음성 명령 (Claude API)

`.env` 파일에 `ANTHROPIC_API_KEY` 설정 후:

```bash
python client/pi0_voice_claude_client.py \
    --server http://<서버IP>:8000
```

음성으로 "과자 줘", "배고파", "스트레스 받아" 등을 말하면 자동으로 해당 task가 실행됩니다.

### 8. 음성 파이프라인 테스트 (로봇 없이)

```bash
# 음성 입력 테스트 (마이크 → STT → Claude → VLA 데이터 형식 출력)
python client/test_voice_claude_pipeline.py

# 키보드 입력 테스트
python client/test_voice_claude_pipeline.py --keyboard

# STT 모델 변경 (base/small/medium)
python client/test_voice_claude_pipeline.py --stt-model small
```

---

## 음성 명령 매핑

| 한국어 키워드 | Pi0 task (학습 시 사용한 문자열) |
|--------------|-------------------------------|
| 과자, 칸초, 간식 | `pick up the snack` |
| 음료, 피크닉, 물 | `pick up the drink` |
| 휴지 | `pick up the tissue` |
| 스트레스, 스트레스볼 | `pick up the stress ball` |

Claude API는 직접적 명령("과자 줘")뿐 아니라 간접 표현("배고파" → 과자, "슬프다" → 휴지)도 분류하여 바로 실행합니다.

---

## 모델 상세

| 항목 | 값 |
|------|-----|
| 모델 | Pi0-FAST (autoregressive + FAST tokenizer) |
| base 모델 | `lerobot/pi0fast-base` (PaliGemma-3B + Gemma-300M) |
| fine-tuning | LoRA adapter |
| 입력 이미지 | top + wrist 카메라 (480x640 → 224x224 자동 리사이즈) |
| 입력 state | `[x, y, z, r, gripper]` 5차원 (MEAN_STD 정규화) |
| 입력 language | PaliGemma tokenizer (max_length=48) |
| 출력 action | `[Δx, Δy, Δz, Δr, grip]` 5차원 (delta) |

---

## STT 백엔드

| 백엔드 | 모델 | 환경 | 속도 |
|--------|------|------|------|
| `whisper` (기본) | faster-whisper base/small/medium | CPU (Mac에서도 OK) | 1~7초 |
| `qwen` | Qwen2.5-Omni-3B | GPU 권장 (Mac은 CPU/MPS) | 1~2초 (GPU) |

voice_module.py에서 `--backend whisper` 또는 `--backend qwen`으로 전환 가능. Mac에서는 `whisper` 백엔드 권장.

---

## 🍎 macOS 전용 트러블슈팅

| 증상 | 해결 |
|------|------|
| `/dev/tty.usb*`가 안 보임 | CH340/CP210x 드라이버 설치 → 시스템 설정 → 개인정보/보안에서 Allow → 재부팅 |
| `Permission denied` (시리얼 포트) | 다른 앱(DOBOT Studio 등)이 포트 점유 중, 모두 종료 |
| 카메라 접근 시 `Video input error` | 시스템 설정 → 개인정보 → 카메라에서 터미널 권한 ON, 터미널 재시작 |
| 카메라 인덱스가 바뀜 | 다른 카메라 앱 종료, `--cam1 1 --cam2 2` 등으로 인덱스 변경 |
| M1/M2에서 `torch` 설치 실패 | `pip install --upgrade pip setuptools`, Python 3.10/3.11 사용 |
| `portaudio` 관련 오류 (음성 입력) | `brew install portaudio` 후 `pip install --force-reinstall pyaudio` |
| `zsh: command not found: python` | `python3` 또는 가상환경 활성화 확인 |
| `ssl.SSLCertVerificationError` (모델 다운로드) | `pip install --upgrade certifi`, `/Applications/Python\ 3.10/Install\ Certificates.command` 실행 |

---

## 파일 구조

```
Dobot_VLM_VLA/
├── .env                              # API 키 (ANTHROPIC_API_KEY, HF_TOKEN)
├── README.md                         # 본 문서 (macOS 가이드)
├── requirements.txt
├── train.sh                          # 학습 실행 스크립트 (Linux GPU 서버에서 실행)
│
├── scripts/
│   ├── 01_collect_data.py            # 데이터 수집 (단일 팔 순차 방식, AVFoundation 고정)
│   ├── 02_convert_v2_to_v3.py        # v2.x → v3.0 포맷 변환
│   ├── 03_validate_dataset.py        # 데이터셋 검증 + 자동 수정
│   ├── 05_inference_dobot.py         # 서버 없이 로컬 추론 (AVFoundation 고정)
│   ├── merge_datasets.py             # 멀티 태스크 데이터셋 합치기
│   ├── test_camera_config.py         # 카메라 설정 테스트
│   └── test_dobot.py                 # DOBOT 동작 테스트
│
├── server/
│   ├── pi0_server.py                 # Pi0-FAST HTTP 추론 서버 (GPU)
│   └── pi0_ws_server.py              # WebSocket 서버
│
├── client/
│   ├── pi0_dobot_client.py           # 텍스트 명령 클라이언트 (Mac: /dev/tty.usbserial-*)
│   ├── pi0_voice_claude_client.py    # 음성 명령 클라이언트 (Claude API)
│   ├── voice_module.py               # STT 모듈 (whisper / qwen 백엔드)
│   ├── pi0_voice_client.py           # 음성 클라이언트 (Qwen3 로컬 LLM)
│   ├── chatbot_module.py             # 챗봇 분류 모듈 (Qwen3)
│   ├── pi0_ws_client.py              # WebSocket 클라이언트
│   └── test_voice_claude_pipeline.py # 음성 파이프라인 테스트
│
└── docs/
    ├── team_guide.md                 # 팀원용 실행 가이드
    └── ...
```

---

## 하드웨어

| 구성 | 사양 |
|------|------|
| 로봇 | DOBOT Magician (USB, CH340/CP210x) |
| 카메라 | USB 카메라 2대 (cam1=wrist, cam2=top), 640x480 |
| 학습/추론 GPU | A6000+ (48GB VRAM, 별도 Linux 서버) |
| 로컬 클라이언트 | **macOS** (Intel / Apple Silicon, 본 브랜치) |

---

See [LICENSE](../LICENSE)
