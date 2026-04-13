# DOBOT Magician x LeRobot x Pi0 / Pi0-FAST

**부산 로보틱스 AI 교육 프로그램 — 단일 로봇 팔 모방학습(Imitation Learning) + VLA(Vision-Language-Action) + 음성 제어 파이프라인**

DOBOT Magician에서 데이터를 수집하고, Pi0 또는 Pi0-FAST 모델로 학습한 뒤, GPU 서버에서 HTTP 추론을 실행하여 로컬 로봇을 원격 제어합니다.

---

## 🖥️ 내 환경에 맞는 브랜치를 고르세요

로컬 머신(DOBOT + 카메라를 연결하는 노트북/PC)의 OS에 따라 브랜치가 다릅니다.
드라이버, 포트 형식, 카메라 백엔드, 설치 명령어가 환경마다 달라서 나눴어요.

| 내 로컬 OS | 브랜치 | 클릭 |
|-----------|--------|------|
| 🪟 **Windows** (10/11) | `window` | **[→ window 브랜치 README 보기](../../tree/window#readme)** |
| 🍎 **macOS** (Intel/Apple Silicon) | `mac` | **[→ mac 브랜치 README 보기](../../tree/mac#readme)** |
| 🐧 **Linux** (Ubuntu 등) | `linux` | **[→ linux 브랜치 README 보기](../../tree/linux#readme)** |

> GPU 학습/추론 서버는 환경에 관계 없이 보통 **Linux**입니다. 서버 운영 가이드는 `linux` 브랜치를 참고하세요.

### 브랜치 전환 방법

```bash
git clone https://github.com/jeong-bin-98/Dobot_VLM_VLA.git
cd Dobot_VLM_VLA
git checkout window   # 또는 mac / linux
```

또는 처음부터 특정 브랜치만 클론:

```bash
git clone -b window https://github.com/jeong-bin-98/Dobot_VLM_VLA.git
```

---

## 전체 파이프라인 (환경 공통)

```
+----------------------------------------------------------------------+
|                        TRAINING PIPELINE                             |
|                                                                      |
|  [로컬 (Win/Mac/Linux)]                       [GPU 서버 (Linux)]      |
|   1) Collect Data  ──scp──► 2) Train (Pi0 또는 Pi0-FAST)             |
|   (단일 팔 순차 수집)            (LoRA 또는 Full fine-tuning)          |
|                                                                      |
+----------------------------------------------------------------------+
|                        INFERENCE PIPELINE                            |
|                                                                      |
|  [GPU 서버]                    [로컬 (DOBOT + 카메라)]                 |
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

자세한 학습/추론/클라이언트 사용법은 환경별 브랜치 README를 참고하세요.

---

## 파일 구조 (모든 브랜치 공통)

```
Dobot_VLM_VLA/
├── .env                              # API 키 (ANTHROPIC_API_KEY, HF_TOKEN)
├── README.md                         # 본 문서 (main=선택기, 각 브랜치=환경별 가이드)
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
    ├── TROUBLESHOOTING.md            # 오류 해결 모음
    ├── PIPELINE.md                   # 파이프라인 설계 문서
    └── architecture_comparison.md    # 아키텍처 비교
```

---

## 하드웨어

| 구성 | 사양 |
|------|------|
| 로봇 | DOBOT Magician (USB, CH340/CP210x) |
| 카메라 | USB 카메라 2대 (cam1=wrist, cam2=top), 640x480 |
| 학습/추론 GPU | A6000+ (48GB VRAM 권장) |
| 로컬 클라이언트 | Windows (`window`) / macOS (`mac`) / Linux (`linux`) 브랜치 선택 |

---

## 더 읽을거리 (환경 공통 문서)

| 문서 | 설명 |
|------|------|
| [docs/team_guide.md](./docs/team_guide.md) | 비전공자용 단계별 실행 가이드 (데이터 수집→학습→추론) |
| [docs/TROUBLESHOOTING.md](./docs/TROUBLESHOOTING.md) | 오류 해결 모음 — 학습/추론/DOBOT/singularity/카메라/GPU |
| [docs/PIPELINE.md](./docs/PIPELINE.md) | 파이프라인 아키텍처 설계 및 모델 비교 |
| [docs/data_collection_guide.md](./docs/data_collection_guide.md) | 데이터 수집 하드웨어 셋업 및 키보드 조작법 |
| [docs/architecture_comparison.md](./docs/architecture_comparison.md) | STT / LLM / VLA 모델 선택 근거 비교 |
| [docs/execution_plan.md](./docs/execution_plan.md) | 프로젝트 로드맵 및 개발 단계 |

---

See [LICENSE](./LICENSE)
