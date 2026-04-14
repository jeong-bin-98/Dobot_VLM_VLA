#!/usr/bin/env python3
"""
음성 명령 → Claude API 분류 → Pi0 → DOBOT 통합 클라이언트

로컬(Mac)에서 실행:
  - STT: Qwen 2.5 ASR (마이크 → 한국어 텍스트)
  - LLM: Claude API (한국어 → 의도 분류 → COMMAND_MAP 매칭)
  - Pi0: 원격 GPU 서버의 /predict 엔드포인트로 영어 task 전송

사용법:
    export ANTHROPIC_API_KEY=sk-ant-xxxxx
    python pi0_voice_claude_client.py \\
        --server http://192.168.1.100:8000 \\
        --cam1 1 --cam2 2
"""

import sys
import os
import time
import json
import re
import argparse

# .env 로드 (프로젝트 루트의 .env 파일에서 ANTHROPIC_API_KEY 등 읽기)
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env"))
except ImportError:
    pass  # dotenv 없으면 환경변수에서 직접 읽음

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from voice_module import VoiceSTT
from pi0_dobot_client import Pi0Client, DobotController, CameraManager

try:
    import anthropic
except ImportError:
    print("pip install anthropic")
    sys.exit(1)


# ============================================================
# 커맨드 매핑 — 학습 시 사용한 task 문자열과 정확히 일치
# ============================================================
COMMAND_MAP = {
    "과자":       "pick up the snack",
    "칸초":       "pick up the snack",
    "간식":       "pick up the snack",
    "음료":       "pick up the drink",
    "피크닉":     "pick up the drink",
    "물":         "pick up the drink",
    "휴지":       "pick up the tissue",
    "스트레스":   "pick up the stress ball",
    "스트레스볼": "pick up the stress ball",
}

STOP_KEYWORDS = ["종료", "그만", "멈춰", "스톱", "stop"]

# Claude 분류 프롬프트
CLASSIFY_PROMPT = """당신은 DOBOT 로봇 팔의 음성 명령 분류기입니다.
사용자의 한국어 문장을 분석해서 JSON으로 응답하세요.

분류 기준:
- "robot": 사용자가 물건을 가져다달라는 요청 (예: "과자 줘", "음료 가져와", "연필 좀")
- "dialog": 일상 대화, 감정 표현, 질문 등 (예: "피곤해", "심심해", "오늘 날씨 어때")

사용 가능한 물체: 과자, 음료, 휴지, 스트레스볼

[STT 오인식 보정]
- "과장 줘"는 로봇 명령 맥락에서 "과자 줘"의 오인식일 수 있음
- "음뇨"→"음료", "연삐"→"연필" 등 음성인식 오류 감안하여 판단
- 단, "과장님 전화해"처럼 명확한 대화 맥락이면 dialog 처리

[상황-물체 매핑]
- 피곤해/스트레스 받아 → 스트레스볼 제안
- 배고프다/출출해 → 과자 제안
- 목마르다 → 음료 제안
- 슬프다/울고 싶다/눈물 난다 → 휴지 제안
- 코 풀어야 해/닦아야 해 → 휴지 제안

반드시 아래 JSON 형식으로만 응답 (다른 텍스트 없이):
{"type":"robot","object":"과자","response":null,"suggest_object":null}
또는
{"type":"dialog","object":null,"response":"힘드시군요.","suggest_object":"스트레스볼"}"""


class ClaudeRouter:
    """Claude API를 사용한 의도 분류 + 커맨드 매핑"""

    def __init__(self, model: str = "claude-sonnet-4-20250514"):
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            print("ANTHROPIC_API_KEY 환경변수를 설정하세요.")
            print("  export ANTHROPIC_API_KEY=sk-ant-xxxxx")
            sys.exit(1)

        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        print(f"  [LLM] Claude API ({model})")

    def process(self, korean_text: str) -> dict:
        """
        한국어 텍스트 → 분류 → 결과

        Returns:
            {
                "type": "robot" | "dialog" | "stop",
                "command": "pick up the snack" | None,
                "response": "대화 응답" | None,
                "suggest_object": "과자" | None,
                "raw_text": "원본 한국어",
            }
        """
        if not korean_text:
            return {"type": "error", "command": None, "response": "인식 실패",
                    "suggest_object": None, "raw_text": ""}

        # 종료 키워드 체크
        for kw in STOP_KEYWORDS:
            if kw in korean_text:
                return {"type": "stop", "command": None, "response": "종료합니다.",
                        "suggest_object": None, "raw_text": korean_text}

        # Claude API 호출
        llm_result = self._classify(korean_text)

        result = {
            "type": llm_result.get("type", "dialog"),
            "command": None,
            "response": llm_result.get("response"),
            "suggest_object": llm_result.get("suggest_object"),
            "raw_text": korean_text,
        }

        if result["type"] == "robot":
            obj = llm_result.get("object", "")
            result["command"] = self._get_command(obj)
            if result["command"] is None:
                # 매칭 실패 시 dialog로 전환
                result["type"] = "dialog"
                result["response"] = (
                    f"'{obj}'은(는) 등록된 물체가 아닙니다. "
                    f"사용 가능: 과자, 음료, 휴지, 스트레스볼"
                )

        return result

    def confirm_suggestion(self, suggest_object: str) -> dict:
        """dialog에서 제안된 물체를 사용자가 확인 시 커맨드 반환"""
        command = self._get_command(suggest_object)
        return {
            "type": "robot",
            "command": command,
            "response": f"{suggest_object}을(를) 가져다 드릴게요.",
            "suggest_object": None,
            "raw_text": f"(확인: {suggest_object})",
        }

    def _classify(self, text: str) -> dict:
        """Claude API로 의도 분류"""
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=256,
                temperature=0.1,
                system=CLASSIFY_PROMPT,
                messages=[{"role": "user", "content": text}],
            )
            content = response.content[0].text.strip()
            return self._parse_json(content)
        except Exception as e:
            print(f"  [LLM] Claude API 오류: {e}")
            # API 실패 시 키워드 매칭 폴백
            return self._keyword_fallback(text)

    def _parse_json(self, text: str) -> dict:
        """Claude 응답에서 JSON 추출"""
        match = re.search(r'\{[^{}]*\}', text)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        return {"type": "dialog", "object": None,
                "response": text[:100], "suggest_object": None}

    def _keyword_fallback(self, text: str) -> dict:
        """API 실패 시 키워드 매칭으로 폴백"""
        text_clean = text.lower().replace(" ", "")
        for keyword in COMMAND_MAP:
            if keyword in text_clean:
                return {"type": "robot", "object": keyword,
                        "response": None, "suggest_object": None}
        return {"type": "dialog", "object": None,
                "response": f"'{text}' — 등록된 물체가 없어요.",
                "suggest_object": None}

    @staticmethod
    def _get_command(object_name: str) -> str | None:
        """한국어 물체명 → COMMAND_MAP에서 영어 프롬프트 조회"""
        if not object_name:
            return None
        if object_name in COMMAND_MAP:
            return COMMAND_MAP[object_name]
        # 부분 매칭
        for key, cmd in COMMAND_MAP.items():
            if key in object_name or object_name in key:
                return cmd
        return None


class VoiceClaudePipeline:
    """음성 → STT → Claude → Pi0 → DOBOT 전체 파이프라인"""

    def __init__(self, args):
        print(f"\n{'='*60}")
        print(f"  음성 명령 Dobot 제어 시스템 (Claude API)")
        print(f"{'='*60}\n")

        # 1. STT (Qwen 2.5 ASR)
        print("[1/4] STT 모듈 로딩...")
        self.stt = VoiceSTT(
            backend=args.stt_backend,
            model_name=args.stt_model,
            device=args.stt_device,
        )

        # 2. LLM (Claude API)
        print("[2/4] Claude API 연결...")
        self.router = ClaudeRouter(model=args.claude_model)

        # 3. Pi0 서버
        print("[3/4] Pi0 서버 연결...")
        self.pi0 = Pi0Client(
            server_url=args.server,
            chunk_size=args.chunk_size,
        )

        # 4. DOBOT + 카메라
        print("[4/4] DOBOT + 카메라 연결...")
        self.dobot = DobotController(args.port)
        self.cameras = CameraManager(args.cam1, args.cam2)

        self.max_cycles = args.cycles

        print(f"\n{'='*60}")
        print(f"  시스템 준비 완료!")
        print(f"  Pi0 서버: {args.server}")
        print(f"  LLM: Claude API ({args.claude_model})")
        print(f"  명령을 말해주세요. '종료'로 끝낼 수 있습니다.")
        print(f"{'='*60}\n")

    def run(self):
        """메인 루프: 음성 대기 → Claude 분류 → 실행"""
        try:
            self.dobot.home()

            while True:
                # === 1단계: 음성 인식 (Qwen 2.5 ASR) ===
                text = self.stt.listen()
                if not text:
                    continue

                print(f"\n  [음성] \"{text}\"")

                # === 2단계: Claude API 분류 ===
                result = self.router.process(text)
                print(f"  [분류] type={result['type']}")

                if result["type"] == "stop":
                    print(f"  {result['response']}")
                    break

                elif result["type"] == "robot":
                    print(f"  [명령] {result['command']}")
                    self._execute_robot_task(result["command"])

                elif result["type"] == "dialog":
                    if result["response"]:
                        print(f"  [응답] {result['response']}")

                    # 물체 제안이 있으면 바로 실행
                    if result["suggest_object"]:
                        cmd_result = self.router.confirm_suggestion(result["suggest_object"])
                        print(f"  [실행] {cmd_result['response']}")
                        self._execute_robot_task(cmd_result["command"])

        except KeyboardInterrupt:
            print("\n\n  Ctrl+C 감지, 종료합니다.")
        finally:
            self.close()

    def _execute_robot_task(self, command: str):
        """Pi0 추론 → DOBOT 실행 루프"""
        if not command:
            print("  [에러] 실행할 커맨드가 없습니다.")
            return

        print(f"\n  === 태스크 실행: {command} ===")

        for cycle in range(self.max_cycles):
            f1, f2 = self.cameras.capture()
            if f1 is None or f2 is None:
                continue

            state = self.dobot.get_state()

            # Pi0 서버 추론
            actions, raw_out, dt_ms = self.pi0.predict(f1, f2, state, command)

            if actions is None:
                print(f"  추론 실패, 재시도...")
                time.sleep(1)
                continue

            # 액션 실행
            for i, delta in enumerate(actions):
                cur, tgt, alarmed = self.dobot.execute(delta)
                print(
                    f"  Cycle {cycle+1} [{i+1}/{len(actions)}] "
                    f"Δ[{delta[0]:+.1f},{delta[1]:+.1f},{delta[2]:+.1f}] "
                    f"G:{'ON' if self.dobot.grip_on else 'OFF'} "
                    f"{dt_ms:.0f}ms"
                )
                if alarmed:
                    print(f"  >> ALARM 복구됨 — 새로 관측합니다")
                    break

        print(f"  === 태스크 완료 ===\n")

    @staticmethod
    def _is_confirm(text: str) -> bool:
        """사용자 확인 발화인지 판별"""
        confirm_words = {"응", "네", "예", "좋아", "그래", "부탁", "해줘", "가져다", "줘", "yes", "ok", "okay"}
        return any(w in text for w in confirm_words)

    def close(self):
        """모든 리소스 해제"""
        self.stt.close()
        self.dobot.close()
        self.cameras.close()
        print("  시스템 종료 완료")


def main():
    parser = argparse.ArgumentParser(
        description="음성 명령 → Claude API → Pi0 → DOBOT 제어"
    )

    # Pi0 서버
    parser.add_argument("--server", type=str, required=True,
                        help="Pi0 서버 URL (예: http://192.168.1.100:8000)")

    # DOBOT
    parser.add_argument("--port", type=str, default=None, help="DOBOT 시리얼 포트")
    parser.add_argument("--cam1", type=int, default=1, help="Wrist 카메라 ID")
    parser.add_argument("--cam2", type=int, default=2, help="Top 카메라 ID")

    # Pi0
    parser.add_argument("--chunk-size", type=int, default=2, help="Pi0 액션 청크 스텝 수")
    parser.add_argument("--cycles", type=int, default=10, help="태스크당 최대 사이클")

    # STT
    parser.add_argument("--stt-backend", type=str, default="whisper",
                        choices=["whisper", "qwen"],
                        help="STT 백엔드 (whisper: faster-whisper, qwen: Qwen2.5-Omni)")
    parser.add_argument("--stt-model", type=str, default=None,
                        help="STT 모델명 (whisper: base/small/medium, qwen: Qwen/Qwen2.5-Omni-3B)")
    parser.add_argument("--stt-device", type=str, default=None,
                        help="STT 디바이스 (cuda/mps/cpu)")

    # Claude API
    parser.add_argument("--claude-model", type=str, default="claude-sonnet-4-20250514",
                        help="Claude 모델명")

    args = parser.parse_args()

    pipeline = VoiceClaudePipeline(args)
    pipeline.run()


if __name__ == "__main__":
    main()
