# DOBOT Magician x LeRobot x Pi0 Pipeline

**부산 로보틱스 AI 교육 프로그램 -- 단일 로봇 팔 모방학습(Imitation Learning) + 음성 제어 파이프라인**

---

## 구조

```
+-------------------------------------------------------------------+
|                      TRAINING PIPELINE                            |
|                                                                   |
|  1) Collect Data --> 2) Convert v3 --> 3) Validate --> 4) Train   |
|  (Single Arm,       (LeRobot v3.0    (Auto-fix      (Pi0-FAST)   |
|   Sequential         Format)          Metadata)                   |
|   Teleoperation)                                                  |
+-------------------------------------------------------------------+
|                      INFERENCE PIPELINE                           |
|                                                                   |
|  [GPU Server]  Pi0-FAST Server (A6000)                            |
|       ↑  POST /predict                                            |
|       |  {image_top, image_wrist, state, language_instruction}     |
|       |                                                           |
|  [Local]  음성/텍스트 클라이언트 + DOBOT                              |
|       |                                                           |
|       ├─ 텍스트: --task "pick up the snack"                        |
|       └─ 음성:   마이크 → STT → Claude API → task 자동 변환         |
+-------------------------------------------------------------------+
```

## 음성 제어 파이프라인

```
마이크 → STT (faster-whisper / Qwen 2.5 Omni)
  → 한국어 텍스트 ("과자 줘" / "배고파")
  → Claude API (의도 분류)
  → COMMAND_MAP (학습 task 문자열 매칭)
  → "pick up the snack"
  → Pi0 서버 /predict → 로봇 실행
```

Pi0 VLA 모델은 학습 시 사용한 정확한 영어 task 문자열이 들어가야 정확도가 높다.
Claude API가 사용자의 자연어(한국어)를 분류하고, `COMMAND_MAP`에서 학습 때 사용한 정확한 영어 프롬프트를 조회하여 Pi0에 전달한다.

---

## 파일 구조

```
Dobot_VLM_VLA/
├── .env                              # API 키 (ANTHROPIC_API_KEY, HF_TOKEN)
├── README.md
├── requirements.txt
├── train.sh                          # 학습 실행 스크립트
│
├── scripts/
│   ├── 01_collect_data.py            # 데이터 수집 (단일 팔 순차 방식)
│   ├── 02_convert_v2_to_v3.py        # v2.x → v3.0 포맷 변환
│   ├── 03_validate_dataset.py        # 데이터셋 검증 + 자동 수정
│   └── merge_datasets.py             # 멀티 태스크 데이터셋 합치기
│
├── server/
│   └── pi0_server.py                 # Pi0-FAST HTTP 추론 서버 (GPU)
│
├── client/
│   ├── pi0_dobot_client.py           # 텍스트 명령 클라이언트 (기본)
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

## 사용법

### 1. 설치

```bash
pip install -r requirements.txt
```

### 2. 데이터 수집

```bash
python scripts/01_collect_data.py \
    --cam1 0 --cam2 1 \
    --task "pick up the snack" \
    --save_dir ./dataset/snack_dataset_v1
```

> 카메라 매핑: `--cam1` = wrist 카메라, `--cam2` = top 카메라

### 3. 검증

```bash
python scripts/03_validate_dataset.py --dataset_dir ./dataset/snack_dataset_v1 --fix
```

### 4. 학습 (Pi0-FAST)

```bash
./train.sh ./dataset/snack_dataset_v1 1 10000 outputs/snack_v1
```

> 인자 순서: `데이터셋경로` `GPU번호` `학습스텝수` `출력경로` `[resume]`

### 5. 추론

#### 서버 실행 (GPU)

```bash
PI0_POLICY_TYPE=pi0_fast \
PI0_MODEL_PATH=./outputs/snack_v1/checkpoints/last/pretrained_model \
python server/pi0_server.py
```

#### 텍스트 명령 (기본)

```bash
python client/pi0_dobot_client.py \
    --server http://<서버IP>:8000 \
    --task "pick up the snack"
```

#### 음성 명령 (Claude API)

`.env` 파일에 API 키 설정 후:

```bash
python client/pi0_voice_claude_client.py \
    --server http://<서버IP>:8000
```

음성으로 "과자 줘", "배고파", "스트레스 받아" 등을 말하면 자동으로 해당 task가 실행됩니다.

### 6. 음성 파이프라인 테스트 (로봇 없이)

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
| `whisper` (기본) | faster-whisper base/small/medium | CPU | 1~7초 |
| `qwen` | Qwen2.5-Omni-3B | GPU 권장 | 1~2초 (GPU) |

voice_module.py에서 `--backend whisper` 또는 `--backend qwen`으로 전환 가능.

---

## 하드웨어

| 구성 | 사양 |
|------|------|
| 로봇 | DOBOT Magician (USB, CH340/CP210x) |
| 카메라 | USB 카메라 2대 (cam1=wrist, cam2=top), 640x480 |
| 학습/추론 GPU | A6000+ (48GB VRAM) |
| 로컬 클라이언트 | Mac 또는 Windows (DOBOT + 카메라 연결) |

---

See [LICENSE](./LICENSE)
