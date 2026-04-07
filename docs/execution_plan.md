# 음성 인식 기반 Dobot 제어 프로젝트 실행 계획

## 프로젝트 목표

사용자가 음성으로 명령하면 Dobot Magician이 인식하고 동작하는 시스템 구축

```
"빨간 컵 집어줘" → (음성인식) → (태스크 변환) → (로봇 제어) → Dobot 동작
```

---

## 전체 일정

| 단계 | 내용 | 예상 기간 | 상태 |
|------|------|----------|------|
| 1단계 | 환경 구축 + 데이터 수집 | 1~2일 | 진행 중 |
| 2단계 | Pi0 학습 + 검증 | 2~3일 | 진행 중 |
| 3단계 | Pi0 추론 테스트 (텍스트 입력) | 1일 | 미착수 |
| 4단계 | STT 모듈 구축 | 1일 | 미착수 |
| 5단계 | LLM 태스크 정규화 모듈 구축 | 1일 | 일부 완료 |
| 6단계 | 전체 파이프라인 통합 | 1~2일 | 미착수 |
| 7단계 | 테스트 + 디버깅 + 시연 | 2~3일 | 미착수 |

---

## 1단계: 환경 구축 + 데이터 수집 (진행 중)

### 완료된 것
- [x] conda 환경 설정 (lerobot)
- [x] Dobot 연결 + 테스트 (test_dobot.py)
- [x] 카메라 2대 설정 (top + front)
- [x] 데이터 수집 스크립트 (01_collect_data.py)
- [x] 데이터 검증 스크립트 (03_validate_dataset.py)
- [x] 테스트 데이터 2 에피소드 수집

### 해야 할 것
- [ ] 본격 데이터 수집: **최소 50 에피소드** (같은 태스크)
  - 태스크: "pick up the red cup"
  - 물체 위치를 매번 약간씩 변경
  - 에피소드당 4~8 스텝
- [ ] 데이터 품질 확인: 03_validate_dataset.py 실행

### 실행 명령어
```bash
conda activate lerobot
python scripts/01_collect_data.py --cam1 0 --cam2 1 --task "pick up the red cup"
```

### 수집 팁
- 테이프로 작업 영역 표시 (약 20x20cm)
- 조명 일정하게 유지
- 천천히, 일관되게 동작
- 실패한 에피소드는 [D]로 버리기

---

## 2단계: Pi0 학습 + 검증 (진행 중)

### 해야 할 것
- [ ] HuggingFace 로그인 + PaliGemma 접근 권한 확인
- [ ] Mac에서 테스트 학습 (1000 steps)
- [ ] 학습 로그 확인 (loss가 줄어드는지)
- [ ] (선택) GPU 서버에서 본격 학습 (10만 steps)

### Mac 테스트 학습
```bash
bash scripts/04_train_pi0.sh mac
```

### GPU 서버 학습 (선택)
```bash
# dataset_v3 폴더를 GPU 서버로 복사 후
bash scripts/04_train_pi0.sh gpu
```

### 체크포인트 위치
```
outputs/pi0_dobot_mac/checkpoints/last/pretrained_model/
```

---

## 3단계: Pi0 추론 테스트 — 텍스트 입력 (미착수)

### 목표
학습된 Pi0 모델로 **텍스트 명령 → Dobot 동작** 확인

### 해야 할 것
- [ ] 학습된 체크포인트로 추론 실행
- [ ] Dobot이 실제로 올바르게 움직이는지 확인
- [ ] 동작이 불안정하면 데이터 추가 수집 후 재학습

### 실행 명령어
```bash
python scripts/05_inference_dobot.py \
    --model_path ./outputs/pi0_dobot_mac/checkpoints/last/pretrained_model \
    --cam1 0 --cam2 1 \
    --task "pick up the red cup"
```

### 확인 사항
- SPACE 키로 시작 후 로봇이 컵 방향으로 이동하는지
- grip ON/OFF가 적절한 타이밍에 동작하는지
- 동작이 안정적인지 (떨리거나 튀지 않는지)

---

## 4단계: STT 모듈 구축 (미착수)

### 목표
마이크 음성 입력 → 텍스트 변환

### 모델
- **Qwen2.5-Omni-7B** (Q4 양자화, ~5GB)
- 한국어/영어 모두 지원

### 해야 할 것
- [ ] Qwen2.5-Omni 모델 다운로드 + 설치
- [ ] 마이크 입력 → 텍스트 변환 모듈 작성 (scripts/06_stt_module.py)
- [ ] 실시간 음성 인식 테스트
- [ ] 한국어 인식 정확도 확인

### 구현 내용
```python
# 06_stt_module.py (예시 구조)
class STTModule:
    def __init__(self, model_name="Qwen/Qwen2.5-Omni-7B"):
        # 모델 로드 (Q4 양자화)
        ...

    def listen(self) -> str:
        # 마이크 입력 대기
        # 음성 감지 → 텍스트 변환
        # 반환: "빨간 컵 집어줘"
        ...
```

### 필요 패키지
```bash
pip install sounddevice pyaudio webrtcvad
```

---

## 5단계: LLM 태스크 정규화 모듈 (일부 완료)

### 목표
자유로운 사용자 발화 → Pi0가 이해하는 표준 태스크 문자열로 변환

### 현재 상태
- task_normalizer.py 완성 (키워드 매칭 방식)
- 한국어/영어 기본 매핑 동작 확인

### 해야 할 것 (LLM 방식으로 업그레이드)
- [ ] Qwen3-8B 모델 다운로드 + 설치 (ollama 또는 llama.cpp)
- [ ] LLM 기반 태스크 정규화 모듈 작성 (scripts/07_llm_normalizer.py)
- [ ] 복잡한 지시 분해 테스트
- [ ] 키워드 매칭 → LLM 전환 (fallback으로 키워드 매칭 유지)

### 구현 내용
```python
# 07_llm_normalizer.py (예시 구조)
class LLMTaskNormalizer:
    def __init__(self, model="qwen3:8b"):
        # ollama 또는 llama.cpp로 로드
        ...

    def normalize(self, user_input: str) -> dict:
        # LLM에게 프롬프트:
        # "사용자 입력을 분석해서 로봇 태스크로 변환해줘"
        # 반환: {"task": "pick up the red cup", "confidence": 0.95}
        ...
```

### 설치 (ollama 사용 시)
```bash
# ollama 설치
brew install ollama
ollama serve  # 서버 시작
ollama pull qwen3:8b  # 모델 다운로드
```

---

## 6단계: 전체 파이프라인 통합 (미착수)

### 목표
STT → LLM → VLA → Dobot 전체를 하나의 스크립트로 연결

### 해야 할 것
- [ ] 통합 스크립트 작성 (scripts/08_voice_control.py)
- [ ] 각 모듈 순차 로딩 (메모리 관리)
- [ ] 실시간 루프 구현
- [ ] 에러 핸들링 + 안전 장치

### 통합 스크립트 흐름
```python
# 08_voice_control.py (예시 구조)

def main():
    # 1. 모델 로딩
    stt = STTModule()           # ~5GB
    llm = LLMTaskNormalizer()   # ~5GB
    vla = Pi0Policy(...)        # ~12GB
    bot = Dobot(...)

    # 2. 메인 루프
    while True:
        # 음성 대기
        print("명령을 말해주세요...")
        user_text = stt.listen()
        print(f"인식: {user_text}")

        # 태스크 변환
        result = llm.normalize(user_text)
        task = result["task"]
        print(f"태스크: {task}")

        # 사용자 확인
        confirm = input(f"'{task}' 실행할까요? (y/n): ")
        if confirm != 'y':
            continue

        # VLA 추론 + 로봇 실행
        execute_task(vla, bot, cameras, task)
```

### 메모리 관리 전략
```
방법 A: 세 모델 동시 로딩 (~22GB, 여유 있음)
방법 B: STT → 언로드 → LLM → 언로드 → VLA (느리지만 안전)
→ Mac 48GB에서는 방법 A가 가능
```

---

## 7단계: 테스트 + 디버깅 + 시연 (미착수)

### 해야 할 것
- [ ] 전체 파이프라인 end-to-end 테스트
- [ ] 다양한 음성 명령 테스트
  - "빨간 컵 집어줘"
  - "pick up the red cup"
  - "컵 좀 들어줄래?"
  - "grab the cup"
- [ ] 에러 케이스 테스트
  - 인식 불가능한 명령
  - 등록되지 않은 태스크
  - 로봇 도달 범위 밖 물체
- [ ] 응답 시간 측정 (음성 입력 → 로봇 동작 시작)
- [ ] 시연 영상 촬영

### 목표 응답 시간
```
STT:  1~2초 (음성 → 텍스트)
LLM:  0.5~1초 (태스크 변환)
VLA:  0.5~1초/step (추론)
────────────────────────────
총:   2~4초 (음성 → 로봇 첫 동작)
```

---

## 파일 구조 (최종)

```
Dobot_VLM_VLA/
├── scripts/
│   ├── 01_collect_data.py          ← 데이터 수집
│   ├── 02_convert_v2_to_v3.py      ← 데이터 변환
│   ├── 03_validate_dataset.py      ← 데이터 검증
│   ├── 04_train_pi0.sh             ← Pi0 학습
│   ├── 05_inference_dobot.py       ← Pi0 추론 (텍스트 입력)
│   ├── 06_stt_module.py            ← STT 모듈 (TODO)
│   ├── 07_llm_normalizer.py        ← LLM 태스크 정규화 (TODO)
│   ├── 08_voice_control.py         ← 전체 통합 (TODO)
│   ├── task_normalizer.py          ← 키워드 기반 정규화 (완료)
│   └── test_dobot.py               ← 로봇 테스트
├── docs/
│   ├── data_collection_guide.md    ← 데이터 수집 가이드
│   ├── architecture_comparison.md  ← 아키텍처 + 모델 비교
│   └── execution_plan.md           ← 이 문서
├── dataset_v3/                     ← 수집된 데이터
├── outputs/                        ← 학습 결과
└── requirements.txt
```

---

## 리스크 및 대응

| 리스크 | 대응 |
|--------|------|
| Pi0 학습 성능 부족 | 데이터 추가 수집 (100개+), GPU 서버 사용 |
| Mac 메모리 부족 | 모델 양자화, 순차 로딩 방식 전환 |
| STT 한국어 인식 부정확 | Whisper 대안 검토, 노이즈 제거 전처리 |
| 응답 시간 느림 | LLM을 키워드 매칭으로 대체 (이미 구현됨) |
| Dobot 동작 불안정 | 속도 제한, 안전 범위 제한, 비상 정지 키 |
