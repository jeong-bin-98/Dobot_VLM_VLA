# 팀원용 실행 가이드

> DOBOT 데이터 수집 → 학습 → 추론 실행까지 복사-붙여넣기로 따라하는 가이드

---

## 목차
1. [사전 준비](#1-사전-준비)
2. [Part A: 데이터 수집 & 학습](#part-a-데이터-수집--학습)
3. [Part B: 추론 (서버 접속해서 로봇 실행)](#part-b-추론-서버-접속해서-로봇-실행)
4. [문제 해결](#문제-해결)

---

## 1. 사전 준비

### 하드웨어 연결
- DOBOT Magician USB 케이블 연결
- 카메라 2개 USB 연결 (위쪽 카메라 + 손목 카메라)
- DOBOT 전원 켜기 (초록불 확인)

---

## Part A: 데이터 수집 & 학습

### A-1. 데이터 수집

아래 명령어를 터미널에 복사-붙여넣기하세요.

```bash
conda activate lerobot
cd Dobot_VLM_VLA

python scripts/01_collect_data.py \
    --task "pick up the [물품 영어이름]" \
    --save_dir ./[물품이름]_dataset_v[버전번호]
```

> **`[물품 영어이름]`** → 수행할 물품 이름을 영어로 적기 (영진이한테 확인)
> **`[물품이름]_dataset_v[버전번호]`** → 예시: `cup_dataset_v1`, `block_dataset_v2`

#### 물품이 바뀌면?

**물품마다 프로그램을 따로 실행해야 합니다.** 실행 중에 task를 바꾸는 기능은 없습니다.

```bash
# 예시: 컵 수집
python scripts/01_collect_data.py \
    --task "pick up the cup" \
    --save_dir ./cup_dataset_v1

# 컵 끝나면 ESC로 종료 후, 블록 수집
python scripts/01_collect_data.py \
    --task "pick up the block" \
    --save_dir ./block_dataset_v1
```

> 학습할 때 여러 데이터셋을 합쳐서 훈련할 수 있으니, **물품별로 데이터셋을 분리**해서 수집하세요.

#### 키보드 조작법

| 키 | 동작 |
|---|---|
| **S** | 현재 상태 캡처 (관측 저장) |
| **E** | 팔을 움직인 후 눌러서 액션 기록 |
| **V** | 에피소드 저장 (한 세트 완료) |
| **D** | 현재 에피소드 버리기 |
| **G** | 그리퍼(흡착) ON/OFF |
| **Z / C** | 손목 회전 (좌/우 5도씩) |
| **Q** | 홈 위치로 이동 |
| **A** | 홈잉 (리밋스위치 캘리브레이션) |
| **X** | 알람 해제 (빨간불 → 초록불) |
| **R** | 마지막 에피소드 리플레이 |
| **W** | 마지막 스텝 되돌리기 (Undo) |
| **1** | 복구 모드 (이전 세션 이어서) |
| **ESC** | 종료 |

#### 수집 순서 (한 에피소드)

```
1. [S] 누르기 → 현재 카메라 영상 + 로봇 상태 캡처
2. 손으로 DOBOT 팔을 원하는 위치로 이동
3. [E] 누르기 → 이동한 만큼의 액션(델타) 기록
4. 1~3번을 여러 번 반복 (보통 5~15스텝)
5. [V] 누르기 → 에피소드 저장 완료
6. 다음 에피소드를 위해 1번부터 다시 시작
```

> **Tip:** 에피소드는 최소 **30개 이상** 수집하세요. 많을수록 학습 성능이 좋아집니다.

---

### A-2. 데이터 검증

수집이 끝나면 데이터가 정상인지 확인합니다. **데이터셋마다 각각 실행하세요.**

```bash
python scripts/03_validate_dataset.py --dataset_dir ./cup_dataset_v1 --fix
python scripts/03_validate_dataset.py --dataset_dir ./block_dataset_v1 --fix
```

에러가 나오면 `--fix` 옵션이 자동으로 고쳐줍니다.

---

### A-3. 학습

#### 데이터셋 1개로 학습

```bash
bash scripts/04_train_pi0.sh gpu ./cup_dataset_v1
```

#### 여러 데이터셋을 합쳐서 학습

물품별로 분리 수집한 데이터셋을 합쳐서 한번에 학습할 수 있습니다.

```bash
lerobot-train \
    --dataset.repo_id='["local/cup_dataset_v1", "local/block_dataset_v1"]' \
    --dataset.root='["./cup_dataset_v1", "./block_dataset_v1"]' \
    --policy.type=pi0_fast \
    --policy.pretrained_path=lerobot/pi0fast-base \
    --policy.push_to_hub=false \
    --policy.dtype=bfloat16 \
    --policy.gradient_checkpointing=true \
    --policy.chunk_size=5 \
    --policy.n_action_steps=1 \
    --batch_size=4 \
    --steps=100000 \
    --output_dir=outputs/pi0fast_dobot
```

> 학습이 끝나면 `outputs/pi0fast_dobot/checkpoints/` 안에 모델이 저장됩니다.

---

## Part B: 추론 (서버 접속해서 로봇 실행)

학습된 모델이 서버에 올라가 있으면, 클라이언트 PC에서 로봇을 실행합니다.

### B-1. 서버 실행 (서버 담당자가 미리 해둠)

> 이 부분은 보통 미리 켜져 있습니다. 서버가 안 켜져 있으면 서버 PC에서 아래를 실행하세요.

```bash
PI0_POLICY_TYPE=pi0_fast \
PI0_MODEL_PATH=./outputs/pi0fast_dobot/checkpoints/last/pretrained_model \
python server/pi0_server.py
```

서버가 잘 켜졌는지 확인:
```bash
curl http://서버IP:8000/health
```

---

### B-2. 클라이언트 실행 (여러분이 할 일)

DOBOT이 연결된 PC에서 아래 명령어를 실행하세요.

```bash
cd Dobot_VLM_VLA

python client/pi0_dobot_client.py \
    --server http://서버IP:8000 \
    --task "pick up the [물품 영어이름]" \
    --chunk-size 2
```

> **`서버IP`를 실제 서버 IP 주소로 바꾸세요** (예: `http://192.168.0.100:8000`)

#### 클라이언트 키보드 조작법

| 키 | 동작 |
|---|---|
| **SPACE** | 1회 추론 실행 (한 스텝) |
| **A** | 자동 모드 (연속 추론 반복) |
| **H** | 홈 위치로 이동 |
| **G** | 그리퍼 ON/OFF |
| **T** | 작업(task) 변경 |
| **L** | LLM 체이닝 모드 |
| **ESC** | 종료 |

#### 실행 순서

```
1. 위 명령어를 터미널에 붙여넣기
2. 카메라 미리보기 창이 뜸
3. [SPACE] 한 번 눌러서 테스트 (로봇이 한 스텝 움직임)
4. 잘 되면 [A] 눌러서 자동 모드 실행
5. 끝내려면 [ESC]
```

---

### B-3. LLM 체이닝 모드 (고급)

LLM이 복잡한 작업을 자동으로 분해해서 실행합니다.

```bash
python client/pi0_dobot_client.py \
    --server http://서버IP:8000 \
    --llm-mode \
    --llm-backend simple \
    --goal "책상 위 물건 정리"
```

---

## 문제 해결

| 증상 | 해결 방법 |
|---|---|
| `DOBOT not found` | USB 케이블 다시 꽂기, 전원 확인 |
| DOBOT 빨간불 | 프로그램에서 **X** 키 눌러서 알람 해제 |
| 카메라 안 뜸 | USB 다시 꽂기, 안 되면 `--cam1 1 --cam2 0` 등 번호 바꿔서 실행 |
| 서버 연결 안 됨 | 서버 IP와 포트(8000) 확인, `curl http://서버IP:8000/health` 테스트 |
| `ModuleNotFoundError` | `pip install -r requirements.txt` 다시 실행 |
| DOBOT이 이상하게 움직임 | **Q** 눌러서 홈 위치로, 그래도 안 되면 **A** (홈잉) |
| 데이터 수집 중 프로그램 꺼짐 | 다시 실행 후 **1** 눌러서 복구 모드 |

---

## 카메라 번호 확인법

어떤 카메라가 몇 번인지 모를 때:

```bash
python -c "
import cv2
for i in range(5):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            print(f'카메라 {i}: OK ({frame.shape[1]}x{frame.shape[0]})')
            cv2.imshow(f'Camera {i}', frame)
        cap.release()
cv2.waitKey(3000)
cv2.destroyAllWindows()
"
```

3초간 각 카메라 화면이 뜨니까 어떤 번호가 위쪽/손목인지 확인하세요.
