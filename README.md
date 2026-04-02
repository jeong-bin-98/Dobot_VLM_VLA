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

## 구조

```
dobot-lerobot-pi0/
+-- LICENSE                          # Proprietary License
+-- README.md                        # This file
+-- requirements.txt                 # Python dependencies
+-- scripts/
|   +-- 01_collect_data.py           # Step-by-step data collection
|   +-- 02_convert_v2_to_v3.py       # v2.x -> v3.0 format converter
|   +-- 03_validate_dataset.py       # Dataset validator with auto-fix
+-- server/
|   +-- pi0_server.py                # Pi0/Pi0-FAST HTTP inference server
|   +-- pi0_ws_server.py             # Pi0/Pi0-FAST WebSocket streaming server
+-- client/
|   +-- pi0_dobot_client.py          # HTTP client + LLM chaining
|   +-- pi0_ws_client.py             # WebSocket streaming client
+-- docs/
    +-- PIPELINE.md                  # Full pipeline guide (Pi0 + Pi0-FAST)
```

## 사용법

### 1. 설치

```bash
pip install -r requirements.txt
```

### 2. 데이터 수집

```bash
python scripts/01_collect_data.py \
    --port COM4 \
    --cam1 0 --cam2 1 \
    --task "pick up the red cup" \
    --save_dir ./dataset_v3
```

### 3. 검증

```bash
python scripts/03_validate_dataset.py --dataset_dir ./dataset_v3 --fix
```

### 4. 학습 (Pi0-FAST)

```bash
# Requires GPU server (A6000+)
pip install -e ".[pi]"

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
```

> `n_action_steps=1`: DOBOT의 `move_to(wait=True)`가 ~0.3-1.0초 소요되므로
> 1 스텝만 실행 후 재관측하는 closed-loop 방식이 안정적.
> See `docs/PIPELINE.md` for Pi0 (flow-matching) training commands.

### 5. 추론

```bash
# On GPU server (A6000):
PI0_POLICY_TYPE=pi0_fast \
PI0_MODEL_PATH=./outputs/pi0fast_dobot/checkpoints/last/pretrained_model \
python server/pi0_server.py

# On local machine with DOBOT:
python client/pi0_dobot_client.py \
    --server http://192.168.1.100:8000 \
    --port COM4 \
    --task "pick up the red cup"
```

---

## 알려진 이슈

### 그리퍼 미작동

추론 중 그리퍼가 안정적으로 작동하지 않음. `move_to()`와 `grip()` 간 큐 충돌 의심.
코드 내 `GRIPPER_DELAY_AFTER_MOVE_S` 값 조정으로 대응 가능하나 현장 디버깅 필요.

---

## v3.0 포맷

LeRobot >= 0.5.0 기준. `images/` 디렉토리, `tasks.jsonl`, 상대 경로 사용.
대규모 데이터셋은 `dataset.finalize()`로 multi-episode shard 통합 가능.

---

## 하드웨어

| 구성 | 사양 |
|------|------|
| 로봇 | DOBOT Magician (USB, CH340/CP210x) |
| 카메라 | USB 카메라 2대 (top + wrist), 640x480 |
| 학습 GPU | A6000+ (48GB VRAM) |

---

See [LICENSE](./LICENSE)
