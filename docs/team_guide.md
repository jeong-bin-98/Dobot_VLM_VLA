# 팀원용 실행 가이드

> DOBOT 데이터 수집 → 서버에 업로드 → 학습 → 추론까지 **복사-붙여넣기만으로** 따라하는 가이드

---

## 목차
1. [사전 준비](#1-사전-준비)
2. [Part A: 데이터 수집 (Mac에서)](#part-a-데이터-수집-mac에서)
3. [Part B: 서버 접속 & 학습](#part-b-서버-접속--학습)
4. [Part C: 데이터셋 합치기 (멀티 태스크)](#part-c-데이터셋-합치기-멀티-태스크)
5. [Part D: 추론 (로봇 실행)](#part-d-추론-로봇-실행)
6. [문제 해결](#문제-해결)

---

## 1. 사전 준비

### 하드웨어 체크리스트

시작하기 전에 아래 항목을 확인하세요:

- [ ] DOBOT Magician USB 케이블이 Mac에 연결됨
- [ ] 카메라 2개가 USB로 Mac에 연결됨 (위쪽 카메라 + 손목 카메라)
- [ ] DOBOT 전원이 켜져 있음 (**초록불** 확인 - 빨간불이면 [문제 해결](#문제-해결) 참고)
- [ ] Mac에 `lerobot` conda 환경이 설치되어 있음

### 폴더 구조

```
Dobot_VLM_VLA/
├── dataset/                    ← 모든 데이터셋은 여기에 저장
│   ├── tissue_dataset_v1/
│   ├── snack_dataset_v1/
│   ├── beverage_dataset_v1/
│   └── sn_ti_be_merged_dataset/  ← 합쳐진 데이터셋
├── outputs/                    ← 학습된 모델이 저장되는 곳
├── scripts/                    ← 수집/학습/추론 스크립트
├── server/                     ← 추론 서버
├── client/                     ← 추론 클라이언트
└── train.sh                    ← 학습 실행 스크립트
```

### 용어 설명

| 용어 | 뜻 |
|------|-----|
| **에피소드** | 로봇이 물건을 집는 동작 1회분 (시작~끝) |
| **스텝** | 에피소드 안에서 팔을 한 번 움직이는 단위 |
| **데이터셋** | 에피소드 여러 개를 모은 폴더 |
| **학습 (training)** | 수집한 데이터로 AI 모델을 훈련시키는 과정 |
| **추론 (inference)** | 학습된 모델이 카메라를 보고 로봇을 움직이는 과정 |

---

## Part A: 데이터 수집 (Mac에서)

### A-1. 터미널 열고 환경 준비

1. Mac에서 **터미널** 앱을 엽니다 (Spotlight에서 "터미널" 검색)
2. 아래 명령어를 **한 줄씩** 복사-붙여넣기합니다:

```bash
conda activate lerobot
```

```bash
cd Dobot_VLM_VLA
```

> 각 줄을 붙여넣고 **Enter**를 누르세요. 앞에 `(lerobot)` 이 보이면 성공입니다.

### A-2. 데이터 수집 시작

아래 명령어에서 `[물품 영어이름]`과 `[물품이름]`을 바꿔서 실행하세요.
데이터셋은 `dataset/` 폴더 안에 저장합니다.

```bash
python scripts/01_collect_data.py \
    --task "pick up the [물품 영어이름]" \
    --save_dir ./dataset/[물품이름]_dataset_v[버전번호]
```

**예시 - 티슈를 집는 데이터를 수집할 때:**

```bash
python scripts/01_collect_data.py \
    --task "pick up the tissue" \
    --save_dir ./dataset/tissue_dataset_v1
```

**예시 - 과자를 집는 데이터를 수집할 때:**

```bash
python scripts/01_collect_data.py \
    --task "pick up the snack" \
    --save_dir ./dataset/snack_dataset_v1
```

> **물품마다 프로그램을 따로 실행해야 합니다.**
> 수집이 끝나면 ESC로 종료하고, 다음 물품으로 새로 실행하세요.

### A-3. 수집 조작법

프로그램이 실행되면 카메라 미리보기 창이 뜹니다.

#### 수집 순서 (한 에피소드 = 물건 한 번 집기)

```
1. [S] 누르기 → 현재 카메라 영상 + 로봇 위치를 기록
2. 손으로 DOBOT 팔을 원하는 위치로 이동
3. [E] 누르기 → 이동한 만큼을 기록
4. 1~3번을 반복 (보통 5~15번)
5. [V] 누르기 → 이 에피소드를 저장
6. 물건을 원래 위치에 놓고, 1번부터 다시 시작
```

#### 키보드 단축키

| 키 | 동작 | 언제 사용? |
|---|---|---|
| **S** | 현재 상태 캡처 | 팔을 움직이기 **전에** |
| **E** | 액션 기록 | 팔을 움직인 **후에** |
| **V** | 에피소드 저장 | 한 세트(물건 집기) **완료 후** |
| **D** | 현재 에피소드 버리기 | 실수했을 때 |
| **W** | 마지막 스텝 되돌리기 | 직전 동작만 실수했을 때 |
| **G** | 그리퍼(흡착) ON/OFF | 물건을 잡거나 놓을 때 |
| **Z / C** | 손목 회전 (좌/우 5도씩) | 손목 각도 조절 |
| **Q** | 홈 위치로 이동 | 팔 위치 초기화 |
| **A** | 홈잉 (캘리브레이션) | 위치가 이상할 때 |
| **X** | 알람 해제 | DOBOT 빨간불일 때 |
| **R** | 마지막 에피소드 리플레이 | 잘 됐는지 확인 |
| **1** | 복구/이어서 수집 모드 | 프로그램 재시작 후 기존 데이터 이어서 수집할 때 |
| **ESC** | 종료 | 수집 끝났을 때 |

> **에피소드는 최소 30개 이상** 수집하세요. 많을수록 성능이 좋아집니다.

### A-4. 이어서 수집하기 (Resume)

기존 데이터셋에 에피소드를 추가하고 싶을 때 `--resume` 옵션을 사용합니다.

| 실행 방식 | 동작 |
|---|---|
| `--resume` **없이** (기본) | 기존 데이터를 **자동 삭제**하고 새로 수집 |
| `--resume` **있을 때** | 기존 에피소드 이어서 추가 수집 |

```bash
python scripts/01_collect_data.py \
    --task "pick up the tissue" \
    --save_dir ./dataset/tissue_dataset_v1 \
    --resume
```

프로그램이 시작되면 기존 에피소드 번호를 이어받아 추가 수집됩니다.
비정상 종료로 임시 데이터가 남아있으면, **1** 키를 눌러 복구할 수 있습니다.

### A-5. 데이터 검증

수집이 끝나면 데이터가 정상인지 확인합니다.

```bash
python scripts/03_validate_dataset.py --dataset_dir ./dataset/tissue_dataset_v1 --fix
```

> 에러가 나오면 `--fix` 옵션이 자동으로 고쳐줍니다. 
> 데이터셋마다 각각 실행하세요.

---

## Part B: 서버 접속 & 학습

학습은 GPU가 있는 **서버**에서 합니다. Mac에서 수집한 데이터를 서버로 보내고, 서버에서 학습합니다.

### B-1. 서버에 접속하기

Mac 터미널에서 아래 명령어를 입력합니다:

```bash
ssh busan01@[서버IP]
```

**예시:**

```bash
ssh busan01@192.168.0.100
```

Enter를 누르면 아래처럼 비밀번호를 물어봅니다:

```
busan01@192.168.0.100's password: 
```

> **비밀번호를 입력하세요.** 타이핑해도 화면에 아무것도 안 보이는 게 정상입니다. 다 치고 Enter를 누르면 됩니다.

접속 성공하면 프롬프트가 바뀝니다:

```
busan01@edu02:~$
```

이제 서버 안에 들어온 것입니다.

### B-2. 서버에서 프로젝트 폴더로 이동

```bash
cd ~/snap/snapd-desktop-integration/intel_third_hands/Dobot_VLM_VLA
```

```bash
source .venv/bin/activate
```

> `(.venv)` 가 프롬프트 앞에 보이면 성공입니다.

### B-3. Mac에서 서버로 데이터 보내기

**새 터미널 탭**을 열고 (서버 접속 중인 터미널은 그대로 두세요), **Mac에서** 아래를 실행합니다:

```bash
cd Dobot_VLM_VLA
```

```bash
scp -r ./dataset/tissue_dataset_v1 busan01@[서버IP]:~/snap/snapd-desktop-integration/intel_third_hands/Dobot_VLM_VLA/dataset/
```

**예시:**

```bash
scp -r ./dataset/tissue_dataset_v1 busan01@192.168.0.100:~/snap/snapd-desktop-integration/intel_third_hands/Dobot_VLM_VLA/dataset/
```

> 비밀번호를 물어보면 서버 비밀번호를 입력하세요.
> 파일이 전송되는 동안 진행률이 표시됩니다. 끝날 때까지 기다려주세요.

### B-4. 학습 실행

**서버 터미널**로 돌아가서 아래를 실행합니다.

#### 데이터셋 1개로 학습

```bash
./train.sh [데이터셋경로] [GPU번호] [학습스텝수] [출력경로]
```

| 인자 | 기본값 | 설명 |
|------|--------|------|
| 데이터셋경로 | `./dataset_v3` | 수집한 데이터셋 폴더 |
| GPU번호 | `1` | 사용할 GPU 번호 (`nvidia-smi`로 확인) |
| 학습스텝수 | `100` | 학습 반복 횟수 |
| 출력경로 | `outputs/pi0fast_dobot_test` | 모델 저장 위치 |
| resume | (없음) | `resume` 입력 시 이전 체크포인트에서 이어서 학습 |

**예시 - 빠른 테스트 (1~2분):**

```bash
./train.sh ./dataset/tissue_dataset_v1 1 100 outputs/tissue_test
```

**예시 - 본 학습 (수 시간 소요):**

```bash
./train.sh ./dataset/tissue_dataset_v1 1 10000 outputs/tissue_v1
```

#### 이어서 학습하기 (Resume)

학습을 중간에 멈췄거나 스텝 수를 늘려서 더 학습하고 싶을 때, 마지막 체크포인트에서 이어서 학습할 수 있습니다.
**같은 출력경로**에 `resume`을 5번째 인자로 추가하세요:

```bash
./train.sh ./dataset/tissue_dataset_v1 1 20000 outputs/tissue_v1 resume
```

> 이전에 1000스텝까지 학습했다면, 1000스텝부터 이어서 20000스텝까지 학습합니다.
> **주의:** 출력경로(`outputs/tissue_v1`)가 이전 학습과 **같아야** 합니다. 다른 경로를 쓰면 체크포인트를 찾지 못합니다.

#### 여러 데이터셋을 합쳐서 학습

여러 물품의 데이터를 합쳐서 한번에 학습할 수 있습니다.
먼저 데이터셋을 합치고 ([Part C 참고](#part-c-데이터셋-합치기-멀티-태스크)), 합쳐진 데이터셋으로 학습하세요:

```bash
./train.sh ./dataset/sn_ti_be_merged_dataset 1 10000 outputs/sn_ti_be_multi_v1
```

> 각 데이터셋의 task (예: "pick up the tissue", "pick up the snack")는 **그대로 유지**됩니다.
> 추론 시 `--task`로 원하는 물품을 지정하면 해당 동작만 수행합니다.

> 학습이 시작되면 `Training: 0%|...` 진행바가 나옵니다.
> 학습이 끝나면 `outputs/*/checkpoints/` 안에 모델이 저장됩니다.

#### 학습 중 확인하기

학습이 잘 되고 있는지 보려면 **서버에서 새 터미널 탭**을 열고:

```bash
nvidia-smi
```

> GPU 사용량이 올라가 있으면 학습이 정상 진행 중입니다.

#### 학습 중단하기

학습을 중간에 멈추려면 학습 중인 터미널에서 `Ctrl + C`를 누르세요.

---

## Part C: 데이터셋 합치기 (멀티 태스크)

여러 물품(예: 티슈, 과자, 음료)의 데이터를 따로 수집한 뒤, **하나의 데이터셋으로 합쳐서** 학습할 수 있습니다.
합친 모델은 추론 시 `--task`로 원하는 물품을 지정하면 해당 동작만 수행합니다.

### C-1. 데이터셋 합치기

**서버 터미널**에서 실행합니다:

```bash
cd ~/snap/snapd-desktop-integration/intel_third_hands/Dobot_VLM_VLA
source .venv/bin/activate
```

```bash
python scripts/merge_datasets.py \
    --datasets ./dataset/tissue_dataset_v1 ./dataset/snack_dataset_v1 ./dataset/beverage_dataset_v1 \
    --output ./dataset/sn_ti_be_merged_dataset
```

> 데이터셋을 2개만 합칠 수도 있고, 4개 이상도 가능합니다:
> ```bash
> python scripts/merge_datasets.py \
>     --datasets ./dataset/tissue_dataset_v1 ./dataset/snack_dataset_v1 \
>     --output ./dataset/sn_ti_merged_dataset
> ```

실행 결과 예시:

```
  tissue_dataset_v1: 17 에피소드 추가 (task: pick up the tissue)
  snack_dataset_v1: 21 에피소드 추가 (task: pick up the snack)
  beverage_dataset_v1: 15 에피소드 추가 (task: pick up the beverage)

  합치기 완료: ./dataset/sn_ti_be_merged_dataset
  총 53 에피소드, 412 프레임
  Tasks: ['pick up the tissue', 'pick up the snack', 'pick up the beverage']
```

### C-2. 합친 데이터셋으로 학습

```bash
./train.sh ./dataset/sn_ti_be_merged_dataset 1 10000 outputs/sn_ti_be_multi_v1
```

### C-3. 합친 모델로 추론

추론 시 `--task`로 원하는 물품을 지정합니다:

```bash
# 티슈를 집고 싶을 때
python client/pi0_dobot_client.py \
    --server http://[서버IP]:8000 \
    --task "pick up the tissue"

# 과자를 집고 싶을 때
python client/pi0_dobot_client.py \
    --server http://[서버IP]:8000 \
    --task "pick up the snack"

# 음료를 집고 싶을 때
python client/pi0_dobot_client.py \
    --server http://[서버IP]:8000 \
    --task "pick up the beverage"
```

---

## Part D: 추론 (로봇 실행)

학습이 끝나면 모델로 로봇을 자동 제어할 수 있습니다.

### D-1. 서버에서 추론 서버 켜기

**서버 터미널**에서 실행합니다:

```bash
cd ~/snap/snapd-desktop-integration/intel_third_hands/Dobot_VLM_VLA
source .venv/bin/activate
```

```bash
PI0_POLICY_TYPE=pi0_fast \
PI0_MODEL_PATH=./outputs/sn_ti_be_multi_v1/checkpoints/last/pretrained_model \
python server/pi0_server.py
```

> `PI0_MODEL_PATH=` 뒤에 학습 출력 경로를 넣으세요.
> `로드 완료`가 뜨면 서버 준비 완료입니다. **이 터미널은 닫지 마세요.**

서버가 잘 켜졌는지 확인하려면 **Mac 터미널**에서:

```bash
curl http://[서버IP]:8000/health
```

`"status":"ok"` 가 보이면 성공입니다.

### D-2. Mac에서 클라이언트 실행

**DOBOT이 연결된 Mac**에서 새 터미널을 열고:

```bash
conda activate lerobot
cd Dobot_VLM_VLA
```

```bash
python client/pi0_dobot_client.py \
    --server http://[서버IP]:8000 \
    --task "pick up the tissue"
```

> `[서버IP]`를 실제 서버 IP 주소로 바꾸세요 (예: `192.168.0.100`)
> `--task`를 바꾸면 다른 물품을 집을 수 있습니다.

### D-3. 로봇 조작

프로그램이 실행되면 카메라 미리보기 창이 뜹니다.

```
1. [SPACE] 한 번 눌러서 테스트 (로봇이 한 스텝 움직임)
2. 잘 되면 [A] 눌러서 자동 모드 실행
3. 끝내려면 [ESC]
```

#### 키보드 단축키

| 키 | 동작 |
|---|---|
| **SPACE** | 1회 추론 실행 (한 스텝씩 테스트) |
| **A** | 자동 모드 ON/OFF (연속 추론) |
| **H** | 홈 위치로 이동 |
| **G** | 그리퍼 ON/OFF |
| **T** | 작업(task) 변경 |
| **L** | LLM 체이닝 모드 |
| **ESC** | 종료 |

---

## 문제 해결

### 하드웨어 문제

| 증상 | 해결 방법 |
|---|---|
| `DOBOT not found` | USB 케이블 다시 꽂기, 전원 확인 |
| DOBOT 빨간불 | 프로그램에서 **X** 키 눌러서 알람 해제 |
| DOBOT이 이상하게 움직임 | **Q** 눌러서 홈 위치로, 그래도 안 되면 **A** (홈잉) |
| 카메라 안 뜸 | USB 다시 꽂기, `--cam1 1 --cam2 0` 번호 바꿔서 실행 |

### 서버/학습 문제

| 증상 | 해결 방법 |
|---|---|
| `ssh: connect to host ... Connection refused` | 서버 IP 확인, 서버가 켜져 있는지 확인 |
| `Permission denied` (ssh) | 비밀번호 다시 확인, 대소문자 주의 |
| `lerobot-train: command not found` | `source .venv/bin/activate` 실행 안 한 것 |
| `tasks.parquet not found` | `python scripts/03_validate_dataset.py --dataset_dir ./dataset/데이터셋 --fix` |
| `FileNotFoundError: ... .jpg` | 데이터셋을 scp로 보낼 때 `-r` 옵션 빠뜨린 것. 다시 전송 |
| `CUDA out of memory` | `--batch_size` 줄이기 (train.sh 안에서 4 → 2) |
| 서버 연결 안 됨 (추론 시) | 서버 IP/포트 확인, `curl http://서버IP:8000/health` 테스트 |
| `ModuleNotFoundError` | `pip install -r requirements.txt` 다시 실행 |
| 데이터 수집 중 프로그램 꺼짐 | `--resume`으로 다시 실행, **1** 눌러서 복구 모드 |

### 카메라 번호 확인법

어떤 카메라가 몇 번인지 모를 때, Mac 터미널에서:

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

---

## 전체 흐름 요약

```
[Mac] 데이터 수집 (01_collect_data.py → dataset/ 폴더에 저장)
  ↓  --resume 으로 이어서 수집 가능
[Mac] 데이터 검증 (03_validate_dataset.py --fix)
  ↓
[Mac → 서버] scp로 dataset/ 폴더 내 데이터 전송
  ↓
[서버] ssh 접속 → source .venv/bin/activate
  ↓
[서버] (선택) 여러 데이터셋 합치기 (merge_datasets.py → dataset/ 폴더에 저장)
  ↓
[서버] ./train.sh 로 학습 (resume 으로 이어서 학습 가능)
  ↓
[서버] python server/pi0_server.py 로 추론 서버 실행
  ↓
[Mac] python client/pi0_dobot_client.py 로 로봇 실행
       --task 로 원하는 물품 지정
```
