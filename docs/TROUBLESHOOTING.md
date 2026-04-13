# 오류 해결 모음 (Troubleshooting)

> 개발 과정에서 발생한 오류와 해결 방법을 카테고리별로 정리했습니다.
> 각 항목에는 관련 커밋 ID가 표기되어 있습니다.

---

## 목차

1. [학습(Training) 오류](#1-학습training-오류)
2. [추론(Inference) 오류](#2-추론inference-오류)
3. [DOBOT 하드웨어 오류](#3-dobot-하드웨어-오류)
4. [Singularity(특이점) 오류](#4-singularity특이점-오류)
5. [카메라 오류](#5-카메라-오류)
6. [GPU / 서버 오류](#6-gpu--서버-오류)
7. [데이터 수집 / 이관 오류](#7-데이터-수집--이관-오류)

---

## 1. 학습(Training) 오류

### 1-1. FAST tokenizer OverflowError

**증상**
```
OverflowError: Python int too large to convert to C int
```

**원인**  
액션 차원 중 하나가 분산이 매우 작고 극단값이 있으면, 정규화 후 DCT + `scale(10)`이 `chr()` 유니코드 한계(1,114,111)를 초과합니다.

**해결**  
`.venv/lib/python3.12/site-packages/lerobot/processor/tokenizer_processor.py`의 `_tokenize_action`에 clamp를 추가합니다 (이 프로젝트에 이미 적용됨):

```python
action_cpu = action[i : i + 1].cpu()
action_cpu = action_cpu.clamp(-6.0, 6.0)  # FAST tokenizer overflow 방지
tokens = self.action_tokenizer(action_cpu)
```

> 관련 커밋: `aa41209`

---

### 1-2. Pi0-FAST garbage 토큰 출력

**증상**  
모델이 의미 없는 토큰 시퀀스를 뱉어내며 로봇 동작이 이상해집니다.

**원인**  
데이터가 적거나(1,000 프레임 이하) 학습이 부족해서 `Action :` 포맷을 제대로 학습하지 못한 경우입니다.

**해결**  
Pi0(flow-matching)으로 전환합니다. 토큰 포맷 문제 자체가 없어집니다.

```bash
./train.sh ./dataset/snack_v1 0 20000 outputs/snack_pi0 "" "" pi0
```

---

### 1-3. tasks.parquet 포맷 오류

**증상**
```
tasks.parquet not found
# 또는
KeyError: 'task_index'
```

**원인**  
lerobot v0.5.1부터 `tasks.parquet`를 특정 인덱스 형식으로 요구하는데, 이전 방식으로 생성된 파일과 형식이 맞지 않습니다.

**해결**  
검증 스크립트로 자동 수정합니다:

```bash
python scripts/03_validate_dataset.py --dataset_dir ./dataset/데이터셋이름 --fix
```

> 관련 커밋: `0558e47`, `79f09a9`

---

### 1-4. 이미지 FileNotFoundError (서버 환경)

**증상**
```
FileNotFoundError: [Errno 2] No such file or directory: 'data/chunk-000/observation.images.wrist/episode_000000/frame_000000.jpg'
```

**원인**  
수집 시 parquet에 이미지 경로를 상대경로로 저장하면, 다른 머신(서버)에서 경로가 맞지 않습니다.

**해결**  
`01_collect_data.py`는 이미지 절대경로로 저장하도록 이미 수정되어 있습니다. 기존 데이터셋이 문제라면:

```bash
python scripts/03_validate_dataset.py --dataset_dir ./dataset/데이터셋이름 --fix
```

> 관련 커밋: `00e5cd5`, `aa41209`

---

### 1-5. CUDA out of memory

**증상**
```
torch.cuda.OutOfMemoryError: CUDA out of memory.
```

**해결**  
`train.sh` 내에서 `batch_size`를 줄입니다:

```bash
# train.sh 안에서
--batch_size=2   # 기본값 4 → 2로 변경
```

또는 LoRA 파인튜닝을 사용합니다:

```bash
./train.sh ./dataset/snack_v1 0 20000 outputs/snack_lora  # LoRA가 기본값
```

---

### 1-6. 멀티 데이터셋 학습 시 task 정보 소실

**증상**  
여러 데이터셋을 합쳐서 학습했는데 task별 구분이 안 됩니다.

**원인**  
lerobot 내장 merge는 task 정보를 올바르게 병합하지 않을 수 있습니다.

**해결**  
`merge_datasets.py`를 사용합니다:

```bash
python scripts/merge_datasets.py \
    --datasets ./dataset/snack_v1 ./dataset/tissue_v1 ./dataset/drink_v1 \
    --output ./dataset/merged_v1
```

또는 `train.sh`에 여러 경로를 공백으로 구분해서 전달합니다:

```bash
./train.sh "./dataset/snack_v1 ./dataset/tissue_v1" 0 20000 outputs/multi_v1
```

> 관련 커밋: `1d369d5`, `f72ab75`

---

## 2. 추론(Inference) 오류

### 2-1. torch.compile reduce-overhead RuntimeError

**증상**
```
RuntimeError: Offset increment outside graph capture encountered unexpectedly
```

**원인**  
Pi0는 flow-matching이라 매 추론마다 랜덤 노이즈를 샘플링하는데, `mode="reduce-overhead"`의 CUDA graphs와 RNG가 충돌합니다.

**해결**  
`mode="max-autotune"` 사용 (`pi0_server_v2.py`에 이미 적용됨):

```python
torch.compile(model.sample_actions, mode="max-autotune")
```

> 관련 커밋: `e70e784`

---

### 2-2. 카메라 키 불일치 (front → wrist)

**증상**  
추론 중 카메라 이미지가 서버로 전달되지만 모델이 이상하게 동작합니다.

**원인**  
수집 스크립트는 `observation.images.wrist` / `observation.images.top` 키를 사용하는데, 이전 추론 코드는 존재하지 않는 `observation.images.front` 키를 사용했습니다.

**해결**  
`pi0_dobot_client.py`의 카메라 매핑이 수정되어 있습니다:
- `cam1` = `wrist` 카메라
- `cam2` = `top` 카메라

다른 파일을 수정한다면:

```python
# 잘못된 키
obs["observation.images.front"] = frame

# 올바른 키
obs["observation.images.wrist"] = frame
obs["observation.images.top"] = frame
```

> 관련 커밋: `29671f7`, `42ec909`

---

### 2-3. State 정규화 / Action 역정규화 누락

**증상**  
로봇이 엉뚱한 위치로 움직이거나 아예 움직이지 않습니다.

**원인**  
학습 시 normalizer가 적용된 상태로 학습했는데, 추론 시 state 정규화와 action 역정규화를 빠뜨린 경우입니다.

**해결**  
`pi0_server_v2.py`의 공식 lerobot 파이프라인(`make_pre_post_processors`)을 사용하면 자동으로 처리됩니다. 이전 서버(`pi0_server.py`)를 사용한다면 `ModelNormalizer`를 수동으로 적용해야 합니다.

> 관련 커밋: `29671f7`, `42ec909`

---

### 2-4. normalizer step 번호 하드코딩 문제

**증상**
```
FileNotFoundError: .../stats/step_010000/normalizer.safetensors not found
```

**원인**  
이전 코드에서 normalizer 경로에 체크포인트 step 번호를 하드코딩했는데, 실제 체크포인트 번호와 다릅니다.

**해결**  
현재 코드는 glob으로 자동 검색합니다. 이 오류가 발생하면 `pi0_server_v2.py`를 사용하세요.

> 관련 커밋: `42ec909`

---

### 2-5. 추론 타임아웃 (torch.compile 첫 실행)

**증상**  
클라이언트가 `Timeout` 또는 추론 응답 없음으로 종료됩니다.

**원인**  
`PI0_COMPILE=1` 사용 시 첫 warmup 컴파일에 2~5분이 걸립니다.

**해결**  
클라이언트 타임아웃이 120초로 설정되어 있습니다. 서버 로그에서 `Warmup 완료`가 뜰 때까지 기다리세요:

```bash
# 서버에서 이 로그가 나오면 준비 완료
INFO: Warmup 완료
```

> 관련 커밋: `e70e784`

---

## 3. DOBOT 하드웨어 오류

### 3-1. ALARM 무한루프

**증상**  
DOBOT 빨간불 + ALARM 메시지가 계속 반복되면서 프로그램이 멈춥니다.

**원인**  
ALARM 발생 후 명령 큐가 얼어붙은 상태에서 ClearAlarm이 제대로 실행되지 않고, 복구 시도가 반복됩니다.

**해결**  
`ClearAllAlarms` (cmd 21)을 명시적으로 전송하고, 연속 3회 이상 ALARM 발생 시 복구를 중단합니다. 이미 코드에 적용됨. 운영 중 발생 시 프로그램에서 **X** 키로 수동 해제하거나 DOBOT 전원을 껐다 켜세요.

> 관련 커밋: `4c5102a`

---

### 3-2. ALARM 복구 후에도 이동 명령 무시됨

**증상**  
ALARM 해제 후 `move_to()` 호출이 무시되고 로봇이 움직이지 않습니다.

**원인**  
이전 코드에서 ClearAlarm 후 큐를 재시작하지 않았거나, 큐가 이미 돌고 있는 상태에서 알람을 해제해서 큐 상태가 꼬였습니다.

**해결**  
올바른 복구 순서:

```
1. 큐 정지 (_set_queued_cmd_stop_exec)
2. ClearAllAlarms (cmd 21)
3. 큐 클리어 (_set_queued_cmd_clear)
4. 큐 재시작 (_set_queued_cmd_start_exec)
```

> 관련 커밋: `4556d9f`, `bd3541f`

---

### 3-3. execute() 반환값 unpack 오류

**증상**
```
ValueError: not enough values to unpack (expected 2, got 3)
# 또는
ValueError: too many values to unpack (expected 2)
```

**원인**  
`execute()` 반환값이 `(cur, tgt)` 2개에서 `(cur, tgt, alarmed)` 3개로 변경됐는데, 호출 측 코드가 업데이트되지 않았습니다.

**해결**

```python
# 잘못된 코드
cur, tgt = execute(action)

# 올바른 코드
cur, tgt, alarmed = execute(action)
if alarmed:
    break  # ALARM 발생 시 루프 중단
```

> 관련 커밋: `2f8ff81`

---

### 3-4. 프로그램 시작 시 멈춤 (DOBOT 무응답)

**증상**  
DOBOT 연결 시 프로그램이 무한정 블로킹되고 Ctrl+C도 안 됩니다.

**원인**  
`pydobot.Dobot()` 생성자가 DOBOT 미응답 시 영원히 블로킹됩니다.

**해결**  
연결 및 복구 시퀀스를 daemon 스레드에서 실행하고 10초 타임아웃을 적용합니다. 이미 코드에 적용됨. 타임아웃 시 수동 리셋 안내가 출력됩니다:

```
[ALARM 복구 타임아웃] 수동 리셋 필요: DOBOT 전원 OFF/ON 후 재시작하세요.
```

> 관련 커밋: `7228af4`

---

### 3-5. 멀티스레드 serial 포트 충돌 (크래시)

**증상**  
자동 추론 중 랜덤하게 프로그램이 크래시되거나 DOBOT 통신 오류가 납니다.

**원인**  
카메라 프리뷰 스레드(`get_pose`)와 추론 스레드(`execute`, `clear_alarm`)가 동시에 serial 포트에 접근합니다.

**해결**  
`RLock`으로 serial 접근을 동기화합니다. `RLock`이므로 `execute → get_pose` 체인에서 데드락이 없습니다. 이미 `DobotController`에 적용됨.

> 관련 커밋: `eda5b02`

---

### 3-6. 그리퍼 미작동 (추론 중)

**증상**  
추론 시 로봇은 움직이는데 그리퍼가 열리거나 닫히지 않습니다.

**원인**  
그리퍼 제어 타이밍 상수가 현장 환경과 맞지 않을 수 있습니다.

**해결**  
`pi0_dobot_client.py` 상단의 상수를 조정합니다:

```python
GRIPPER_DELAY_AFTER_MOVE_S = 0.1   # 기본값 → 0.3~0.5로 높여볼 것
GRIPPER_ACTION_WAIT_S = 0.5        # grip 완료 대기
GRIPPER_THRESHOLD = 0.5            # action[4] > 0.5 → 그리퍼 ON
```

우선 `GRIPPER_DELAY_AFTER_MOVE_S`를 `0.3`으로 올려보세요.

---

## 4. Singularity(특이점) 오류

### 4-1. cos(θ2) 기반 singularity 감지 오류

**증상**  
DOBOT이 특정 위치로 이동 시 ALARM이 발생하거나 이상하게 꺾입니다. 이전 코드에서는 `r < 200mm` 영역만 차단했는데 실제로는 더 넓은 범위가 위험했습니다.

**원인**  
단순 수평 거리(r) 기반 모델이 59mm wrist 기구 오프셋을 무시해서 singularity 경계가 틀렸습니다.

**해결**  
DH 파라미터 기반 올바른 공식으로 교체:

```
r = 135 * sin(j2) + 206
  = 147(forearm) + 59(wrist_mech) = 206mm 고정 오프셋
```

현재 코드에 이미 적용됨. `tests/test_singularity.py`의 30개 테스트로 검증됩니다.

> 관련 커밋: `25fe8e0`, `7460f06`

---

### 4-2. Singularity 경유 시 ALARM (via-point 없음)

**증상**  
안전 반경 내부를 지나는 경로에서 ALARM이 발생합니다.

**원인**  
시작점~끝점 직선 경로가 singularity 영역을 통과합니다.

**해결**  
경유점(via-point) 자동 계산이 적용되어 있습니다. `REACH_SAFE_MIN`(≈229mm) 기준으로 singularity 회피 경유점을 거칩니다.

> 관련 커밋: `e0d629e`, `25fe8e0`

---

### 4-3. Z축 도달 불가능 위치 이동 시도

**증상**  
DOBOT이 목표 위치로 이동하지 못하고 ALARM이 발생합니다.

**원인**  
3D 공간에서 팔 길이(135mm)로 도달 불가능한 z값이 포함된 좌표입니다.

**해결**  
`is_reachable(x, y, z)` 함수로 이동 전 3D 도달 가능성을 검사합니다. 도달 불가 시 z를 허용 범위로 자동 클램프합니다(5% 안전 마진 적용):

```python
# DH 기하학: (r-206)² + (z-Z_pivot)² ≤ 135²
if not is_reachable(x, y, z):
    z = clamp_to_reachable(x, y, z)
```

> 관련 커밋: `e576c87`

---

## 5. 카메라 오류

### 5-1. OS별 카메라 백엔드 불일치

**증상**  
```
cv2.error: OpenCV(4.x) ... can't open camera by index
```
Mac에서 Linux용 V4L2 백엔드로 열려고 할 때 발생합니다.

**해결**  
`--cam-backend` 옵션으로 OS별 백엔드를 지정합니다:

```bash
# 자동 감지 (기본, 권장)
python scripts/01_collect_data.py --cam-backend auto ...

# Mac
python scripts/01_collect_data.py --cam-backend avfoundation ...

# Linux
python scripts/01_collect_data.py --cam-backend v4l2 ...

# Windows
python scripts/01_collect_data.py --cam-backend dshow ...
```

백엔드 실패 시 자동으로 기본값으로 재시도합니다.

> 관련 커밋: `8ef22cc`, `484214b`

---

### 5-2. 학습-추론 간 카메라 노출/화이트밸런스 불일치

**증상**  
학습 때와 다른 밝기/색감의 이미지가 들어가서 추론 성능이 저하됩니다.

**원인**  
카메라 자동 노출/WB가 켜진 상태에서 데이터를 수집하면 에피소드마다 밝기가 달라지고, 추론 환경과도 달라집니다.

**해결**  
모든 스크립트에 카메라 수동 설정이 적용됩니다:

```python
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)        # 오토노출 OFF (1=manual)
cap.set(cv2.CAP_PROP_EXPOSURE, -4)             # 수동 노출 (-4)
cap.set(cv2.CAP_PROP_AUTO_WB, 0)               # 오토WB OFF
cap.set(cv2.CAP_PROP_WB_TEMPERATURE, 5000)    # 색온도 5000K
cap.set(cv2.CAP_PROP_GAIN, 0)                  # 게인 0
```

`scripts/test_camera_config.py`로 현장에서 실시간 조정 가능합니다 (`[M]` 수동/자동 토글, `[+/-]` 노출, `[W/S]` 색온도).

> 관련 커밋: `ea66d49`, `0235400`

---

### 5-3. 카메라 ID 라벨 반전 (help text 버그)

**증상**  
`--cam1`이 top 카메라로, `--cam2`가 wrist 카메라로 연결됩니다.

**원인**  
`01_collect_data.py`의 `--cam1`, `--cam2` help text가 반대로 기재되어 있었습니다.

**해결**  
올바른 매핑:
- `--cam1` → **wrist** 카메라 (손목)
- `--cam2` → **top** 카메라 (위쪽)

help text가 수정되어 있습니다. 카메라 ID를 모를 때 확인하는 방법:

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

> 관련 커밋: `7460f06`

---

## 6. GPU / 서버 오류

### 6-1. GPU 메모리 누수 (비정상 종료 후)

**증상**  
서버를 Ctrl+C로 종료 후 재시작하면 `CUDA out of memory`가 발생합니다.

**원인**  
비정상 종료 시 GPU 메모리가 해제되지 않아 프로세스가 남아 있습니다.

**해결**  
서버가 `SIGINT`/`SIGTERM` 시 자동으로 정리합니다:

```python
# on_shutdown 함수가 자동 실행됨
del model
del processor
torch.cuda.empty_cache()
gc.collect()
```

재시작 전에 좀비 프로세스 확인:

```bash
nvidia-smi  # GPU 메모리 점유 프로세스 확인
kill -9 <PID>  # 필요 시 강제 종료
```

> 관련 커밋: `9c1f72c`

---

### 6-2. lerobot 명령어 not found

**증상**
```
lerobot-train: command not found
```

**해결**

```bash
source .venv/bin/activate
# 또는
conda activate lerobot
```

---

### 6-3. 서버 연결 불가 (추론 시)

**증상**
```
ConnectionRefusedError: [Errno 111] Connection refused
# 또는
ssh: connect to host ... Connection refused
```

**해결 순서**
1. 서버 IP 주소 확인 (`ip addr` 또는 `ifconfig`)
2. 서버에서 추론 서버가 실행 중인지 확인: `ps aux | grep pi0_server`
3. 헬스 체크: `curl http://서버IP:8000/health`
4. 방화벽 확인: `sudo ufw status`

---

## 7. 데이터 수집 / 이관 오류

### 7-1. scp 이미지 파일 누락

**증상**  
서버에서 데이터셋 경로는 있는데 이미지 `.jpg` 파일들이 없습니다.

**원인**  
`scp` 명령에서 `-r` 옵션을 빠뜨리면 폴더 안 파일이 전송되지 않습니다.

**해결**

```bash
# 잘못된 명령 (폴더 구조만 복사됨)
scp ./dataset/snack_v1 user@server:~/

# 올바른 명령 (-r 플래그 필수)
scp -r ./dataset/snack_v1 user@server:~/
```

---

### 7-2. 데이터 수집 중 프로그램 종료

**증상**  
수집 중 프로그램이 꺼지고 일부 에피소드가 손상됩니다.

**해결**  
`--resume` 플래그로 이어서 수집합니다:

```bash
python scripts/01_collect_data.py \
    --task "pick up the tissue" \
    --save_dir ./dataset/tissue_v1 \
    --resume
```

프로그램 시작 시 **1**번을 눌러 손상된 마지막 에피소드를 복구할 수 있습니다.

---

## 빠른 참조 (Quick Reference)

| 증상 | 해결 |
|------|------|
| `OverflowError: Python int too large` | tokenizer_processor.py에 clamp(-6.0, 6.0) 추가 |
| Pi0-FAST garbage 토큰 | Pi0(flow-matching)으로 전환 |
| `tasks.parquet not found` | `03_validate_dataset.py --fix` 실행 |
| `FileNotFoundError: .jpg` | `scp -r`로 재전송 |
| `CUDA out of memory` | batch_size 4→2로 감소 |
| torch.compile RuntimeError | `mode="max-autotune"` 사용 |
| 카메라 키 불일치 | cam1=wrist, cam2=top 매핑 확인 |
| ALARM 무한루프 | **X** 키 또는 전원 OFF/ON |
| 프로그램 시작 시 멈춤 | DOBOT 전원 확인, 10초 후 자동 복구 |
| 그리퍼 미작동 | `GRIPPER_DELAY_AFTER_MOVE_S` 0.1→0.3 |
| 카메라 안 열림 (Mac) | `--cam-backend avfoundation` |
| GPU 메모리 누수 | `nvidia-smi` 확인 후 좀비 프로세스 kill |
| `lerobot-train: not found` | `source .venv/bin/activate` |
