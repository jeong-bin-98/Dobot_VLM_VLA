# DOBOT Magician × LeRobot × Pi0 Pipeline

**부산 로보틱스 AI 교육 프로그램 -- 단일 로봇 팔 임시학습(Imitation Learning) 파이프라인**

---

## 구조

```
+-----------------------------------------------------------------+
|                    TRAINING PIPELINE                            |
|                                                                 |
|  1) Collect Data ---> 2) Convert to v3 ---> 3) Validate ---> 4) Train |
|  (Single Arm,       (LeRobot v3.0       (Auto-fix      (Pi0    |
|   Sequential         Format)             Metadata)      -FAST) |
|   Teleoperation)                                                |
+-----------------------------------------------------------------+
|                    INFERENCE PIPELINE                           |
|                                                                 |
|  Pi0-FAST Server (A6000) --HTTP---> Client + DOBOT (local)      |
|                                                                 |
|  Model Output: Δ[x, y, z, r, grip]  (delta coordinates)       |
|  -> Inverse kinematics solved by DOBOT firmware (move_to API)   |
+-----------------------------------------------------------------+
```

## 설계

### 단일 팔 순차 방식

SO-100처럼 리더/팔로워가 없으므로 수동으로 팔을 위치시키며 프레임 단위로 수집.
`[S]` 관측 캡처 -> 팔 이동 -> `[E]` 델타 기록.

### 델타 액션

action은 절대 좌표가 아닌 변위값 `[Δx, Δy, Δz, Δr, grip]`으로 저장.

### 역기구학

Pi0 출력은 3D 직교좌표 delta. DOBOT 펌웨어가 `move_to(x, y, z, r)`로 역기구학 처리.

---

## 파일 구조

```
Dobot_VLM_VLA/
├── README.md
├── LICENSE
├── requirements.txt
├── scripts/
│   ├── 01_collect_data.py           # 단계별 데이터 수집 (단일 팔 순차 방식)
│   ├── 02_convert_v2_to_v3.py       # v2.x → v3.0 포맷 변환
│   ├── 03_validate_dataset.py       # 데이터셋 검증 + 자동 수정
│   ├── 04_train_pi0.sh              # Pi0-FAST 학습 스크립트
│   ├── 05_inference_dobot.py        # 로컬 추론 테스트
│   ├── task_normalizer.py           # 태스크 정규화 유틸
│   └── test_dobot.py               # DOBOT 연결 테스트
├── server/
│   ├── pi0_server.py                # Pi0-FAST HTTP 추론 서버 (GPU)
│   └── pi0_ws_server.py             # WebSocket 스트리밍 서버
├── client/
│   ├── pi0_dobot_client.py          # HTTP 클라이언트 + LLM 체이닝
│   ├── pi0_ws_client.py             # WebSocket 스트리밍 클라이언트
│   ├── pi0_voice_client.py          # 음성 명령 클라이언트
│   ├── voice_module.py              # 음성 인식 모듈
│   └── chatbot_module.py            # 챗봇 모듈
└── docs/
    ├── PIPELINE.md                  # 전체 파이프라인 가이드
    ├── architecture_comparison.md   # 아키텍처 비교
    ├── data_collection_guide.md     # 데이터 수집 가이드
    └── execution_plan.md            # 실행 계획
```

---

## 사용법

### 1. 설치

```bash
pip install -r requirements.txt
```

### 2. 데이터 수집

```bash
# Mac (포트 자동 감지, 또는 --port /dev/tty.usbserial-XXXXX)
python scripts/01_collect_data.py \
    --cam1 0 --cam2 1 \
    --task "pick up the red cup" \
    --save_dir ./dataset_v3
```

> 카메라 매핑: `--cam1` = wrist 카메라, `--cam2` = top 카메라
> 시리얼 포트: Mac은 자동 감지됨. 수동 지정 시 `--port /dev/tty.usbserial-XXXXX`

### 3. 검증

```bash
python scripts/03_validate_dataset.py --dataset_dir ./dataset_v3 --fix
```

### 4. 학습 (Pi0-FAST)

```bash
# Mac (MPS, float32, batch_size=1)
bash scripts/04_train_pi0.sh mac

# GPU 서버 (A6000+, bfloat16, batch_size=4)
bash scripts/04_train_pi0.sh gpu
```

> `n_action_steps=1`: DOBOT의 `move_to(wait=True)`가 ~0.3-1.0초 소요되므로
> 1 스텝만 실행 후 재관측하는 closed-loop 방식이 안정적.

### 5. 추론

```bash
# GPU 서버에서 추론 서버 실행:
PI0_MODEL_PATH=./outputs/pi0fast_dobot_testv2/checkpoints/000100/pretrained_model \
python server/pi0_server.py

# Mac 로컬에서 클라이언트 실행 (DOBOT 연결):
python client/pi0_dobot_client.py \
    --server http://<서버IP>:8000 \
    --task "pick up the red cup"
```

서버 LoRA 로딩 과정:
1. `lerobot/pi0fast-base` base 모델 로드
2. 체크포인트에서 LoRA 어댑터 merge
3. fine-tuned config 적용 (카메라: top + wrist, state: 5차원)

### 6. 음성 명령 모드

```bash
python client/pi0_voice_client.py \
    --server http://<서버IP>:8000
```

---

## 모델 상세

| 항목 | 값 |
|------|-----|
| 모델 | Pi0-FAST (autoregressive + FAST tokenizer) |
| base 모델 | `lerobot/pi0fast-base` (PaliGemma-3B + Gemma-300M) |
| fine-tuning | LoRA adapter |
| 입력 이미지 | top + wrist 카메라 (480x640 → 224x224 자동 리사이즈) |
| 입력 state | `[x, y, z, r, gripper]` 5차원 (MEAN_STD 정규화) |
| 출력 action | `[Δx, Δy, Δz, Δr, grip]` 5차원 (delta) |
| chunk_size | 5 |

---

## 하드웨어

| 구성 | 사양 |
|------|------|
| 로봇 | DOBOT Magician (USB, CH340/CP210x) |
| 카메라 | USB 카메라 2대 (cam1=wrist, cam2=top), 640x480 |
| 학습 GPU | A6000+ (48GB VRAM) |

---

See [LICENSE](./LICENSE)
