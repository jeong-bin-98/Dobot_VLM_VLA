# DOBOT Magician x LeRobot x Pi0 / Pi0-FAST — 🪟 Windows 가이드

> **이 문서는 Windows 환경 전용 가이드입니다.** 다른 환경은 [→ 메인 README](../README.md)로 돌아가 macOS / Linux 가이드를 고르세요.

**부산 로보틱스 AI 교육 프로그램 — 단일 로봇 팔 모방학습(Imitation Learning) + VLA(Vision-Language-Action) + 음성 제어 파이프라인**

Windows 10/11 로컬 머신에서 DOBOT Magician과 카메라를 연결하여 데이터 수집 / 추론 클라이언트를 실행하는 가이드입니다. 학습은 Linux GPU 서버에서 수행합니다.

---

## 🪟 Windows 전용 사전 준비

### 1. Python 설치

[python.org](https://www.python.org/downloads/windows/)에서 **Python 3.10 또는 3.11** 설치.
설치 시 **"Add Python to PATH"** 반드시 체크.

또는 Anaconda/Miniconda 사용 가능:

```powershell
conda create -n dobot python=3.10
conda activate dobot
```

### 2. DOBOT USB 드라이버 (CH340 / CP210x)

DOBOT Magician은 USB-직렬 변환 칩(CH340 또는 CP210x)을 사용합니다. 드라이버가 없으면 **장치 관리자에 COM 포트가 안 보입니다**.

- **CH340**: [WCH CH340 Windows 드라이버](http://www.wch-ic.com/downloads/CH341SER_EXE.html)
- **CP210x**: [Silicon Labs CP210x 드라이버](https://www.silabs.com/developers/usb-to-uart-bridge-vcp-drivers)

설치 후 DOBOT을 USB로 연결하고 **장치 관리자 → 포트(COM & LPT)**에서 `COMx` 번호 확인 (예: `COM3`, `COM4`).

### 3. Git (Git Bash 포함)

[git-scm.com](https://git-scm.com/download/win)에서 Git for Windows 설치.
설치에 포함된 **Git Bash**에서 `train.sh` 같은 `.sh` 스크립트를 실행할 수 있습니다.

### 4. Visual C++ Build Tools (선택, 일부 패키지 빌드 시)

[Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)에서 "Desktop development with C++" 설치. `dobot-python`, `opencv-python` 빌드 실패 시 필요.

### 5. 저장소 클론 및 의존성 설치

```powershell
git clone -b window https://github.com/jeong-bin-98/Dobot_VLM_VLA.git
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
|  [Windows 로컬]                              [Linux GPU 서버]         |
|   1) Collect Data  ──scp──► 2) Train (Pi0 또는 Pi0-FAST)             |
|   (DOBOT + USB 카메라 2대)       (LoRA 또는 Full fine-tuning)          |
|                                                                      |
+----------------------------------------------------------------------+
|                        INFERENCE PIPELINE                            |
|                                                                      |
|  [Linux GPU 서버]                  [Windows 로컬 (DOBOT + 카메라)]     |
|   Pi0 Server v2 ◄─── POST /predict ──  Client                        |
|   (FastAPI)          {image_top,        ├─ 텍스트:                    |
|                       image_wrist,      │  --task "pick up the snack" |
|                       state,            └─ 음성:                      |
|                       language}            마이크 → STT → Claude API  |
+----------------------------------------------------------------------+
```

---

## 데이터 수집 (Windows)

### 카메라 백엔드: `dshow` (DirectShow)

Windows에서는 OpenCV가 **DirectShow** 백엔드를 통해 USB 카메라에 접근해야 안정적입니다. `--cam-backend dshow` 옵션을 명시하거나, 기본값(`auto`)에 맡기면 자동으로 `CAP_DSHOW`가 선택됩니다.

### DOBOT 포트: `COMx` 형식

```powershell
python scripts\01_collect_data.py ^
    --port COM4 ^
    --cam1 0 --cam2 1 ^
    --task "pick up the snack" ^
    --save_dir .\dataset\snack_dataset_v1 ^
    --cam-backend dshow
```

> PowerShell에서는 줄 이어쓰기에 백틱(`` ` ``)을, cmd에서는 캐럿(`^`)을 사용하세요. 또는 Git Bash에서 `\`로 이어쓰기 사용 가능.

`--port`를 생략하면 자동으로 CH340/CP210x 칩을 탐지합니다. 잘 안되면 장치 관리자에서 확인한 COM 번호를 직접 지정하세요.

### 수집한 데이터 GPU 서버로 전송

```powershell
scp -r .\dataset\snack_dataset_v1 user@<서버IP>:/home/user/datasets/
```

OpenSSH 클라이언트는 Windows 10/11에 기본 내장되어 있습니다 (`설정 → 앱 → 선택적 기능`에서 활성화).

---

## 학습 (Linux GPU 서버에서 수행)

학습 자체는 GPU 서버에서 실행합니다. Windows에서는 SSH로 서버에 접속한 뒤 `train.sh`를 실행하세요.

```bash
# 서버 접속 후 (Git Bash 또는 PowerShell의 ssh)
ssh user@<서버IP>
./train.sh ./dataset/snack_dataset_v1 0 20000 outputs/snack_fast_lora
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

---

## 추론 서버 (Linux GPU 서버)

서버 실행은 Linux에서 합니다. Windows에서는 해당 서버 IP만 알면 됩니다.

```bash
# 서버에서 (참고용)
CUDA_VISIBLE_DEVICES=0 \
PI0_POLICY_TYPE=pi0_fast \
PI0_MODEL_PATH=./outputs/snack_fast_lora/checkpoints/last/pretrained_model \
python server/pi0_server_v2.py
```

---

## 클라이언트 (Windows에서 실행)

### 텍스트 명령 (기본)

```powershell
python client\pi0_dobot_client.py `
    --server http://<서버IP>:8000 `
    --task "pick up the snack" `
    --cam1 0 --cam2 1
```

자동 포트 탐지가 안 되면 `--port COM4` 같이 직접 지정:

```powershell
python client\pi0_dobot_client.py `
    --server http://<서버IP>:8000 `
    --port COM4 `
    --task "pick up the snack" `
    --cam1 0 --cam2 1
```

### 음성 명령 (Claude API)

`.env`에 `ANTHROPIC_API_KEY` 설정 후:

```powershell
python client\pi0_voice_claude_client.py --server http://<서버IP>:8000
```

> Windows에서 마이크 접근 허용이 필요합니다: **설정 → 개인정보 → 마이크**에서 앱 권한 ON.

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
마이크(Windows) → STT (faster-whisper / Qwen2.5-Omni)
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

## 🪟 Windows 전용 트러블슈팅

| 증상 | 해결 |
|------|------|
| 장치 관리자에 COM 포트가 안 보임 | CH340/CP210x 드라이버 재설치, USB 케이블 교체 |
| `Permission denied` 또는 `could not open port 'COM4'` | 다른 프로그램(DOBOT Studio 등)이 포트 점유 중. 전부 종료 후 재시도 |
| 카메라가 안 열리거나 검은 화면 | `--cam-backend dshow` 명시, 다른 앱(Zoom/Teams)이 카메라 점유 중인지 확인 |
| `OpenCV(4.x) can't open camera by index 0` | 인덱스 변경 (`--cam1 1 --cam2 2`), 또는 USB 허브 없이 직결 |
| `ImportError: DLL load failed` (torch/opencv) | Visual C++ Build Tools 설치, Python 버전을 3.10/3.11로 맞추기 |
| PowerShell에서 `./train.sh` 실행 안 됨 | Git Bash에서 실행 (`bash train.sh ...`) |
| 한글 경로에서 `UnicodeEncodeError` | 데이터셋을 영어 경로(`C:\dataset\...`)에 저장 |

전체 트러블슈팅 목록: **[TROUBLESHOOTING.md](./TROUBLESHOOTING.md)**

---

## 파일 구조

```
Dobot_VLM_VLA/
├── .env                              # API 키 (ANTHROPIC_API_KEY, HF_TOKEN)
├── README.md                         # 본 문서 (Windows 가이드)
├── requirements.txt
├── train.sh                          # Pi0 / Pi0-FAST 학습 스크립트 (Linux 서버에서 실행)
│
├── scripts/
│   ├── 01_collect_data.py            # 데이터 수집 (Windows에서 실행)
│   ├── 03_validate_dataset.py        # 데이터셋 검증
│   ├── 05_inference_dobot.py         # 서버 없이 로컬 추론
│   ├── merge_datasets.py             # 멀티 태스크 데이터셋 합치기
│   └── test_dobot.py                 # DOBOT 동작 테스트
│
├── server/
│   └── pi0_server_v2.py              # (Linux GPU 서버에서 실행)
│
├── client/
│   ├── pi0_dobot_client.py           # 텍스트 명령 클라이언트 (Windows)
│   ├── pi0_voice_claude_client.py    # 음성 명령 클라이언트 (Windows)
│   ├── voice_module.py               # STT 모듈
│   └── ...
│
└── docs/
    └── ...
```

---

## 하드웨어

| 구성 | 사양 |
|------|------|
| 로봇 | DOBOT Magician (USB, CH340/CP210x) |
| 카메라 | USB 카메라 2대 (cam1=wrist, cam2=top), 640x480 |
| 학습/추론 GPU | A6000+ (48GB VRAM 권장, 별도 Linux 서버) |
| 로컬 클라이언트 | **Windows 10/11** (본 브랜치) |

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
