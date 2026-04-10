#!/usr/bin/env python3
"""
음성 → Claude → Pi0 파이프라인 테스트

DOBOT / 카메라 / Pi0 서버 없이, 실제 VLA 입력 데이터 형식을 print로 확인.
- 마이크 음성 → Qwen 2.5 ASR → 한국어 텍스트 (실제 STT)
- Claude API 실제 호출
- Pi0 VLA에 들어가는 데이터 형식을 그대로 출력

    python test_voice_claude_pipeline.py              # 음성 입력
    python test_voice_claude_pipeline.py --keyboard   # 키보드 입력 (STT 없이)
"""

import sys
import os
import json
import argparse
import numpy as np

# .env 로드
try:
    from dotenv import load_dotenv
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env")
    load_dotenv(env_path)
except ImportError:
    pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pi0_voice_claude_client import ClaudeRouter, COMMAND_MAP, STOP_KEYWORDS


# ============================================================
# 시뮬레이션용 가짜 데이터 (실제와 동일한 형식)
# ============================================================
FAKE_STATE_RAW = [240.0, 0.0, 80.0, 0.0, 0.0]  # [x, y, z, r, gripper] mm/deg

# 정규화 통계 (실제 학습에서 나온 값의 예시)
FAKE_STATE_MEAN = np.array([245.12, -2.34, 65.78, 1.23, 0.31], dtype=np.float32)
FAKE_STATE_STD  = np.array([28.45, 51.67, 38.92, 42.15, 0.46], dtype=np.float32)
FAKE_ACTION_MEAN = np.array([0.15, -0.08, 0.22, 0.01, 0.03], dtype=np.float32)
FAKE_ACTION_STD  = np.array([3.82, 4.15, 3.21, 2.55, 0.48], dtype=np.float32)

# 토크나이저 설정 (PaliGemma)
FAKE_TOKENIZER_MAX_LENGTH = 48


def print_box(title, width=64):
    print(f"\n  ┌{'─' * (width - 2)}┐")
    print(f"  │ {title:<{width - 4}} │")
    print(f"  └{'─' * (width - 2)}┘")


def print_separator(char="─", width=64):
    print(f"  {char * width}")


def fake_normalize_state(raw):
    return (np.array(raw, dtype=np.float32) - FAKE_STATE_MEAN) / FAKE_STATE_STD


def fake_unnormalize_action(norm):
    return norm * FAKE_ACTION_STD + FAKE_ACTION_MEAN


def print_vla_data(command, state_raw):
    """Pi0 VLA에 실제로 들어가는 데이터 형식을 상세히 출력"""

    print_box("Pi0 VLA 입력 데이터 (실제 형식)")

    # ── HTTP 요청 (클라이언트 → 서버) ──
    print(f"\n  ▶ HTTP POST /predict  (클라이언트 → 서버)")
    print(f"  ┌─ Request Body (JSON) ─────────────────────────────┐")
    print(f"  │ image_top:            str (base64 JPEG, ~50KB)    │")
    print(f"  │ image_wrist:          str (base64 JPEG, ~50KB)    │")
    print(f"  │ state:                {state_raw}  │")
    print(f"  │ language_instruction: \"{command}\"")
    print(f"  │ chunk_size:           2                           │")
    print(f"  └──────────────────────────────────────────────────-─┘")

    # ── 서버 내부 변환 1: 이미지 ──
    print(f"\n  ▶ 서버 내부 변환")
    print(f"  ┌─ 1. 이미지 디코딩 ────────────────────────────────┐")
    print(f"  │ base64 → cv2.imdecode → BGR→RGB                  │")
    print(f"  │ img_top:   numpy [480, 640, 3] uint8  (HWC, RGB)  │")
    print(f"  │ img_wrist: numpy [480, 640, 3] uint8  (HWC, RGB)  │")
    print(f"  │                                                   │")
    print(f"  │ → img_to_tensor():                                │")
    print(f"  │   permute(2,0,1) → /255.0 → unsqueeze(0)         │")
    print(f"  │   tensor [1, 3, 480, 640] float32  (0~1)          │")
    print(f"  └───────────────────────────────────────────────────┘")

    # ── 서버 내부 변환 2: state 정규화 ──
    norm_state = fake_normalize_state(state_raw)
    print(f"  ┌─ 2. State 정규화 ─────────────────────────────────┐")
    print(f"  │ raw state:  {state_raw}")
    print(f"  │ state_mean: {FAKE_STATE_MEAN.tolist()}")
    print(f"  │ state_std:  {FAKE_STATE_STD.tolist()}")
    print(f"  │                                                   │")
    print(f"  │ normalized = (raw - mean) / std                   │")
    print(f"  │ → {np.round(norm_state, 4).tolist()}")
    print(f"  │ → tensor [1, 5] float32                           │")
    print(f"  └───────────────────────────────────────────────────┘")

    # ── 서버 내부 변환 3: 언어 명령 토크나이즈 ──
    fake_token_ids = list(range(2, 2 + len(command.split())))  # 가짜 토큰 ID
    fake_token_ids += [0] * (FAKE_TOKENIZER_MAX_LENGTH - len(fake_token_ids))  # 패딩
    fake_attn_mask = [1] * len(command.split()) + [0] * (FAKE_TOKENIZER_MAX_LENGTH - len(command.split()))

    print(f"  ┌─ 3. 언어 명령 토크나이즈 (PaliGemma) ─────────────┐")
    print(f"  │ text: \"{command}\"")
    print(f"  │ max_length: {FAKE_TOKENIZER_MAX_LENGTH}, padding: max_length, truncation: True")
    print(f"  │                                                   │")
    print(f"  │ input_ids:      tensor [1, {FAKE_TOKENIZER_MAX_LENGTH}] int64")
    print(f"  │   [{', '.join(str(t) for t in fake_token_ids[:8])}, ..., 0, 0]")
    print(f"  │ attention_mask: tensor [1, {FAKE_TOKENIZER_MAX_LENGTH}] bool")
    print(f"  │   [{', '.join(str(t) for t in fake_attn_mask[:8])}, ..., 0, 0]")
    print(f"  └───────────────────────────────────────────────────┘")

    # ── observation dict (모델 입력) ──
    print(f"\n  ▶ Pi0 모델 입력 (observation dict)")
    print(f"  ┌───────────────────────────────────────────────────┐")
    print(f"  │ observation = {{                                   │")
    print(f"  │   \"observation.images.top\":    [1,3,480,640] f32  │")
    print(f"  │   \"observation.images.wrist\":  [1,3,480,640] f32  │")
    print(f"  │   \"observation.state\":         [1,5] f32          │")
    print(f"  │   \"observation.language.tokens\":       [1,{FAKE_TOKENIZER_MAX_LENGTH}] i64 │")
    print(f"  │   \"observation.language.attention_mask\":[1,{FAKE_TOKENIZER_MAX_LENGTH}] bool│")
    print(f"  │ }}                                                  │")
    print(f"  │                                                   │")
    print(f"  │ → policy.select_action(observation)               │")
    print(f"  └───────────────────────────────────────────────────┘")

    # ── 모델 출력 + 역정규화 ──
    fake_raw_actions = np.array([
        [0.62, -0.31, 0.18, -0.02, -0.65],
        [0.35, -0.12, -0.28, 0.01, 1.42],
    ], dtype=np.float32)
    real_actions = np.array([fake_unnormalize_action(a) for a in fake_raw_actions])

    print(f"\n  ▶ Pi0 모델 출력 (action)")
    print(f"  ┌───────────────────────────────────────────────────┐")
    print(f"  │ raw output (정규화된 값):                          │")
    print(f"  │   step 0: {np.round(fake_raw_actions[0], 4).tolist()}")
    print(f"  │   step 1: {np.round(fake_raw_actions[1], 4).tolist()}")
    print(f"  │                                                   │")
    print(f"  │ 역정규화: real = raw * action_std + action_mean    │")
    print(f"  │   action_mean: {FAKE_ACTION_MEAN.tolist()}")
    print(f"  │   action_std:  {FAKE_ACTION_STD.tolist()}")
    print(f"  │                                                   │")
    print(f"  │ 실제 delta (역정규화 후):                           │")
    print(f"  │   step 0: Δx={real_actions[0][0]:+.2f} Δy={real_actions[0][1]:+.2f} Δz={real_actions[0][2]:+.2f} Δr={real_actions[0][3]:+.2f} grip={real_actions[0][4]:.2f}")
    print(f"  │   step 1: Δx={real_actions[1][0]:+.2f} Δy={real_actions[1][1]:+.2f} Δz={real_actions[1][2]:+.2f} Δr={real_actions[1][3]:+.2f} grip={real_actions[1][4]:.2f}")
    print(f"  └───────────────────────────────────────────────────┘")

    # ── DOBOT 실행 ──
    print(f"\n  ▶ DOBOT 실행")
    print(f"  ┌───────────────────────────────────────────────────┐")
    for i, delta in enumerate(real_actions):
        new_pos = [round(state_raw[j] + delta[j], 2) for j in range(4)]
        grip = "ON (잡기)" if delta[4] > 0.5 else "OFF (놓기)"
        print(f"  │ Step {i}: cur{[round(v,1) for v in state_raw[:4]]} + Δ[{delta[0]:+.2f},{delta[1]:+.2f},{delta[2]:+.2f},{delta[3]:+.2f}]")
        print(f"  │       → move_to({new_pos[0]}, {new_pos[1]}, {new_pos[2]}, {new_pos[3]})  grip={grip}")
    print(f"  └───────────────────────────────────────────────────┘")


def handle_result(result, router, use_voice, stt):
    """분류 결과 처리"""

    if result["type"] == "stop":
        print(f"\n  → 종료 명령 감지. 시스템 종료.")
        return False

    elif result["type"] == "robot":
        print_vla_data(result["command"], FAKE_STATE_RAW)
        print(f"\n  ✓ 태스크 완료: \"{result['command']}\"")

    elif result["type"] == "dialog":
        print(f"\n  → 대화 응답: {result['response']}")

        if result["suggest_object"]:
            cmd_result = router.confirm_suggestion(result["suggest_object"])
            print(f"  → 바로 실행: {cmd_result['command']}")
            print_vla_data(cmd_result["command"], FAKE_STATE_RAW)

    return True


def test_pipeline(use_voice=True, stt_backend="whisper", stt_model=None, stt_device=None):
    backend_label = {"whisper": "faster-whisper", "qwen": "Qwen 2.5 Omni"}
    print(f"\n{'='*68}")
    print(f"  음성 → Claude → Pi0 VLA 파이프라인 테스트")
    print(f"  입력: {'🎤 마이크 (' + backend_label.get(stt_backend, stt_backend) + ')' if use_voice else '⌨️  키보드'}")
    print(f"  (DOBOT/카메라/Pi0 서버 없이, VLA 입력 데이터 형식 확인)")
    print(f"{'='*68}")

    # STT 초기화
    stt = None
    if use_voice:
        from voice_module import VoiceSTT
        stt = VoiceSTT(backend=stt_backend, model_name=stt_model, device=stt_device)

    # Claude 라우터
    print("\n  [LLM] Claude API 연결 중...")
    router = ClaudeRouter()

    # 커맨드 목록
    print_box("등록된 COMMAND_MAP (학습 task와 동일)")
    seen = set()
    for ko, en in COMMAND_MAP.items():
        if en not in seen:
            keywords = [k for k, v in COMMAND_MAP.items() if v == en]
            print(f"  │ {', '.join(keywords):28s} → \"{en}\"")
            seen.add(en)
    print(f"  │ 종료: {STOP_KEYWORDS}")

    print_separator()
    if use_voice:
        print(f"  말씀하세요. ('종료'로 끝)")
    else:
        print(f"  한국어로 입력하세요. ('종료'로 끝)")
    print_separator()

    try:
        while True:
            # ── 1단계: STT ──
            if use_voice:
                print(f"\n  🎤 듣고 있습니다...")
                text = stt.listen()
                if not text:
                    print("  (인식 실패, 다시 말해주세요)")
                    continue
            else:
                try:
                    text = input("\n  ⌨️  입력> ").strip()
                except (KeyboardInterrupt, EOFError):
                    break
                if not text:
                    continue

            print_box(f"파이프라인 시작: \"{text}\"")

            print(f"\n  ▶ 1단계: STT {'(' + backend_label.get(stt_backend, stt_backend) + ')' if use_voice else '(키보드 시뮬레이션)'}")
            print(f"    입력: {'마이크 음성' if use_voice else '키보드'}")
            print(f"    출력: \"{text}\"")

            # ── 2단계: Claude API ──
            print(f"\n  ▶ 2단계: Claude API 의도 분류")
            print(f"    입력: \"{text}\"")
            result = router.process(text)
            print(f"    출력: {{")
            print(f"      type:           \"{result['type']}\"")
            print(f"      command:        {json.dumps(result['command'], ensure_ascii=False)}")
            print(f"      response:       {json.dumps(result['response'], ensure_ascii=False)}")
            print(f"      suggest_object: {json.dumps(result['suggest_object'], ensure_ascii=False)}")
            print(f"    }}")

            if not handle_result(result, router, use_voice, stt):
                break

            print_separator("═")

    except KeyboardInterrupt:
        print("\n\n  Ctrl+C 종료")
    finally:
        if stt:
            stt.close()
        print("  테스트 종료")


def main():
    parser = argparse.ArgumentParser(description="음성 → Claude → Pi0 VLA 파이프라인 테스트")
    parser.add_argument("--keyboard", action="store_true",
                        help="키보드 입력 모드 (마이크 대신 타이핑)")
    parser.add_argument("--stt-backend", type=str, default="whisper",
                        choices=["whisper", "qwen"],
                        help="STT 백엔드: whisper(CPU 빠름) / qwen(GPU 권장)")
    parser.add_argument("--stt-model", type=str, default=None,
                        help="STT 모델명 (whisper: base/small/medium, qwen: Qwen/Qwen2.5-Omni-3B)")
    parser.add_argument("--stt-device", type=str, default=None,
                        help="STT 디바이스 (cuda/mps/cpu)")
    args = parser.parse_args()

    test_pipeline(
        use_voice=not args.keyboard,
        stt_backend=args.stt_backend,
        stt_model=args.stt_model,
        stt_device=args.stt_device,
    )


if __name__ == "__main__":
    main()
