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

#### 4가지 학습 모드 (pi0 / pi0_fast × LoRA / Full)

`train.sh`는 6, 7번째 인자로 **파인튜닝 방식**과 **모델 종류**를 고를 수 있습니다.

```bash
./train.sh [데이터셋] [GPU] [스텝] [출력경로] [resume] [full] [pi0|pi0_fast]
```

| 모드 | 학습 속도 | VRAM 사용량 | 추론 속도 | 안정성 | 언제 쓸까 |
|------|----------|------------|----------|--------|----------|
| **① pi0_fast + LoRA** (기본) | 빠름 | **~11GB** | 빠름 (5x) | 중 | **기본 추천** — 서버 부담 적고 빠르게 실험 |
| **② pi0_fast + Full** | 느림 | 매우 높음 (48GB+ 필요) | 빠름 (5x) | 중 | 데이터가 충분할 때 (현재 하드웨어에선 OOM 가능성 ❌) |
| **③ pi0 + LoRA** | 중간 | 낮음 | 느림 | **높음** | 데이터 적거나 Pi0-FAST가 garbage 토큰 뱉을 때 |
| **④ pi0 + Full** | 느림 | **~37GB** | 느림 | **가장 높음** | 데이터 충분 + 안정성 최우선 (최고 품질) |

**실행 예시:**

```bash
# ① pi0_fast + LoRA (기본, 추천)  — VRAM ~11GB
./train.sh ./dataset/snack_dataset_v3 1 20000 outputs/snack_v3_fast_lora

# ② pi0_fast + Full — 현재 서버에서는 OOM 가능성 있음 ❌
./train.sh ./dataset/snack_dataset_v3 0 20000 outputs/snack_v3_fast_full "" full

# ③ pi0 + LoRA  — 소량 데이터 안정 학습
./train.sh ./dataset/snack_dataset_v3 0 20000 outputs/snack_v3_pi0_lora "" "" pi0

# ④ pi0 + Full  — 최고 품질, VRAM ~37GB
./train.sh ./dataset/snack_dataset_v3 0 20000 outputs/snack_v3_pi0_full "" full pi0
```

> **인자 규칙:** 6번째(`full`)나 7번째(`pi0`) 인자만 쓰고 5번째(`resume`)는 건너뛰고 싶다면, 그 자리를 빈 문자열 `""`로 비워두세요. 위 예시의 `"" full`, `"" "" pi0` 패턴이 그런 경우입니다.

**모드 선택 가이드:**

- **처음 학습한다면** → ① `pi0_fast + LoRA` (가장 빠르고 VRAM 적게 씀)
- **Pi0-FAST가 이상한 토큰을 뱉는다면** → ③ `pi0 + LoRA`로 전환 (flow-matching은 토큰 포맷 문제 없음)
- **실제 로봇 성능을 극대화하고 싶다면** → ④ `pi0 + Full` (VRAM 여유 있을 때만)
- **pi0와 pi0_fast의 차이가 궁금하다면** → [README.md의 Pi0 vs Pi0-FAST 비교표](../README.md#모델-선택-pi0-vs-pi0-fast) 참고

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

#### 백그라운드 학습 (tmux)

학습은 오래 걸리므로, 터미널을 닫아도 학습이 계속되게 하려면 **tmux**를 사용합니다.

```bash
# 1. tmux 세션 생성
tmux new -s train

# 2. tmux 안에서 학습 실행
./train.sh ./dataset/sn_ti_be_merged_dataset 1 20000 outputs/sn_ti_be_multi_v1

# 3. 학습이 시작되면 tmux에서 빠져나오기
#    Ctrl+B 누른 뒤 D 누르기
```

**다음날 와서 학습 상태 확인하기:**

```bash
# 서버 접속
ssh busan01@[서버IP]

# tmux 세션에 다시 들어가기
tmux attach -t train
```

| tmux 명령 | 동작 |
|---|---|
| `tmux new -s train` | 새 세션 생성 |
| `Ctrl+B` → `D` | 세션에서 빠져나오기 (학습은 계속됨) |
| `tmux attach -t train` | 세션에 다시 들어가기 |
| `tmux ls` | 실행 중인 세션 목록 보기 |
| `tmux kill-session -t train` | 세션 종료 |

#### 학습 중 확인하기

학습이 잘 되고 있는지 보려면 **서버에서 새 터미널 탭**을 열고:

```bash
nvidia-smi
```

> GPU 사용량이 올라가 있으면 학습이 정상 진행 중입니다.

#### 학습 중단하기

학습을 중간에 멈추려면 학습 중인 터미널(또는 tmux 세션)에서 `Ctrl + C`를 누르세요.

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
두 가지 모드가 있습니다:
- **D-텍스트**: 텍스트로 task를 직접 지정 (기본)
- **D-음성**: 음성 명령으로 로봇 제어 (Claude API 사용)

### D-1. 체크포인트 선택하기

학습이 끝나면 `outputs/` 안에 여러 체크포인트가 저장됩니다:

```
outputs/snack_v3_fast_lora/checkpoints/
├── 005000/pretrained_model/   ← 5000스텝 시점
├── 010000/pretrained_model/   ← 10000스텝 시점
├── 015000/pretrained_model/   ← 15000스텝 시점
├── 020000/pretrained_model/   ← 20000스텝 시점 (마지막)
└── last/pretrained_model/     ← = 020000과 동일
```

**어떤 체크포인트를 써야 할까?**

lerobot에는 자동으로 "best" 체크포인트를 골라주는 기능이 없습니다.
실제 로봇에서 추론해보고 가장 잘 되는 걸 선택해야 합니다.

| 순서 | 방법 |
|------|------|
| 1 | 먼저 `last` (마지막 체크포인트)로 추론 테스트 |
| 2 | 결과가 안 좋으면 중간 체크포인트 (010000, 015000 등)로 시도 |
| 3 | 가장 잘 되는 체크포인트를 사용 |

**왜 마지막이 항상 최고가 아닌가?**

- 데이터가 적을수록 (1000프레임 이하) **오버피팅** 위험이 높음
- 오버피팅 = 학습 데이터에만 맞추고, 실제 환경에서는 못하는 상태
- 이 경우 중간 체크포인트 (전체 스텝의 50~75% 지점)가 더 나을 수 있음

**실전 팁:**

- 로봇이 물체 방향으로 대략 맞게 움직이면 → 그 체크포인트 OK
- 로봇이 엉뚱한 방향으로 가거나 떨림 → 오버피팅, 이전 체크포인트 시도
- 2~3개 체크포인트를 빠르게 테스트하는 게 가장 확실

### D-2. 서버에서 추론 서버 켜기

**서버 터미널**에서 실행합니다:

```bash
cd ~/snap/snapd-desktop-integration/intel_third_hands/Dobot_VLM_VLA
source .venv/bin/activate
```

#### 추론 서버 2가지 버전

| 버전 | 파일 | 특징 |
|------|------|------|
| **v1** (구) | `server/pi0_server.py` | 수동 tokenizer/normalizer, pi0_fast 전용으로 시작 |
| **v2** (권장) | `server/pi0_server_v2.py` | 공식 lerobot 파이프라인 사용, pi0/pi0_fast 통합, torch.compile 옵션 |

**추천: v2 사용** (아래 설명은 v2 기준)

#### GPU 선택 및 기본 실행

```bash
# GPU 0번 + pi0_fast
CUDA_VISIBLE_DEVICES=0 \
PI0_POLICY_TYPE=pi0_fast \
PI0_MODEL_PATH=./outputs/snack_v3_fast_lora/checkpoints/last/pretrained_model \
python server/pi0_server_v2.py
```

```bash
# GPU 0번 + pi0 (flow-matching)
CUDA_VISIBLE_DEVICES=0 \
PI0_POLICY_TYPE=pi0 \
PI0_MODEL_PATH=./outputs/snack_v3_pi0_lora/checkpoints/last/pretrained_model \
python server/pi0_server_v2.py
```

> `CUDA_VISIBLE_DEVICES=0` → 0번 GPU 사용. `nvidia-smi`로 빈 GPU를 확인하고 번호를 바꾸세요.
> `PI0_MODEL_PATH=` 뒤에 원하는 체크포인트 경로를 넣으세요 (`last`를 `010000` 등으로 바꾸면 중간 체크포인트 사용).
> `로드 완료`가 뜨면 서버 준비 완료입니다. **이 터미널은 닫지 마세요.**

#### torch.compile로 추론 가속 (선택)

추론이 느린 경우 `PI0_COMPILE=1`을 추가하면 `torch.compile(mode="max-autotune")`이 적용되어 1.5~2배 빨라집니다.

```bash
CUDA_VISIBLE_DEVICES=0 \
PI0_COMPILE=1 \
PI0_POLICY_TYPE=pi0 \
PI0_MODEL_PATH=./outputs/snack_v3_pi0_lora/checkpoints/last/pretrained_model \
python server/pi0_server_v2.py
```

**주의점:**
- 서버 시작 시 **첫 컴파일에 2~5분** 걸림 (warmup 자동 실행). `Warmup 완료`가 뜨면 끝
- 컴파일이 완료되면 이후 모든 추론이 빨라짐
- **먼저 컴파일 없이 정상 동작하는지 확인한 후에** 활성화하는 걸 추천

서버가 잘 켜졌는지 확인하려면 **Mac 터미널**에서:

```bash
curl http://[서버IP]:8000/health
```

`"status":"ok"` 가 보이면 성공입니다.

### D-3. 텍스트 명령 모드 (기본)

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

### D-5. 로봇 조작

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

### D-3. 음성 명령 모드 (Claude API)

음성으로 로봇을 제어합니다. 말한 내용을 STT가 인식하고, Claude API가 의도를 분류해서 학습된 task 명령으로 변환합니다.

#### 사전 준비

1. `.env` 파일에 API 키 설정 (프로젝트 루트):
```
ANTHROPIC_API_KEY=sk-ant-xxxxx
```

2. STT 모델 사전 다운로드 (최초 1회):
```bash
# faster-whisper (CPU용, 가벼움)
python -c "from faster_whisper import WhisperModel; WhisperModel('small')"

# Qwen 2.5 Omni (GPU용, 고정밀) — GPU 환경에서만
python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-Omni-3B')"
```

#### 실행

```bash
python client/pi0_voice_claude_client.py \
    --server http://[서버IP]:8000
```

#### 동작 흐름

```
마이크 음성 → STT(한국어 인식) → Claude API(의도 분류) → 영어 task → Pi0 → 로봇 실행
```

| 음성 입력 | Claude 분류 | Pi0에 전달되는 task |
|-----------|------------|-------------------|
| "과자 줘" | robot → 과자 | `pick up the snack` |
| "배고파" | dialog → 과자 제안 → 바로 실행 | `pick up the snack` |
| "목 말라" | dialog → 음료 제안 → 바로 실행 | `pick up the drink` |
| "너무 슬퍼" | dialog → 휴지 제안 → 바로 실행 | `pick up the tissue` |
| "스트레스 받아" | dialog → 스트레스볼 제안 → 바로 실행 | `pick up the stress ball` |
| "종료" | stop | (시스템 종료) |

> 로봇이 동작 중에는 음성 입력을 받지 않습니다. 동작이 끝나면 자동으로 다시 음성 대기 상태가 됩니다.

#### CLI 옵션

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--server` | (필수) | Pi0 서버 URL |
| `--stt-model` | `small` | STT 모델 (base/small/medium) |
| `--claude-model` | `claude-sonnet-4-20250514` | Claude 모델 |
| `--cycles` | `10` | 태스크당 최대 실행 사이클 |
| `--cam1` / `--cam2` | `1` / `2` | 카메라 ID |

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
[Mac] 실행 방식 선택:
  ├─ 텍스트: python client/pi0_dobot_client.py --task "pick up the tissue"
  └─ 음성:   python client/pi0_voice_claude_client.py --server http://서버IP:8000
```
