#!/usr/bin/env python3
"""
음성 명령 → Pi0 → DOBOT 전체 통합 스크립트

voice_module (STT) + chatbot_module (LLM) + pi0_dobot_client (VLA + Robot)

    python pi0_voice_client.py \\
        --server http://192.168.1.100:8000 \\
        --cam1 1 --cam2 2
"""

import sys
import os
import time
import argparse

# 같은 폴더의 모듈 import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from voice_module import VoiceSTT
from chatbot_module import ChatbotRouter, STOP_KEYWORDS
from pi0_dobot_client import Pi0Client, DobotController, CameraManager


class VoiceControlPipeline:
    """음성 → STT → LLM → Pi0 → DOBOT 전체 파이프라인"""

    def __init__(self, args):
        print(f"\n{'='*60}")
        print(f"  음성 명령 Dobot 제어 시스템")
        print(f"{'='*60}\n")

        # 1. STT 모듈
        print("[1/4] STT 모듈 로딩...")
        self.stt = VoiceSTT(
            model_name=args.stt_model,
            device=args.stt_device,
        )

        # 2. LLM 챗봇 모듈
        print("[2/4] LLM 모듈 로딩...")
        self.chatbot = ChatbotRouter(
            model_name=args.llm_model,
            device=args.llm_device,
        )

        # 3. Pi0 원격 클라이언트
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
        print(f"  명령을 말해주세요. '종료'로 끝낼 수 있습니다.")
        print(f"{'='*60}\n")

    def run(self):
        """메인 루프: 음성 대기 → 분류 → 실행 반복"""
        try:
            self.dobot.home()

            while True:
                # === 1단계: 음성 인식 ===
                text = self.stt.listen()
                if not text:
                    continue

                print(f"\n  [음성] \"{text}\"")

                # === 2단계: LLM 분류 ===
                result = self.chatbot.process(text)
                print(f"  [분류] type={result['type']}")

                if result["type"] == "stop":
                    print(f"  {result['response']}")
                    break

                elif result["type"] == "robot":
                    # 로봇 명령 → 바로 실행
                    print(f"  [명령] {result['command']}")
                    self._execute_robot_task(result["command"])

                elif result["type"] == "dialog":
                    # 대화 응답 출력
                    if result["response"]:
                        print(f"  [응답] {result['response']}")

                    # 물체 제안이 있으면 음성으로 확인
                    if result["suggest_object"]:
                        print(f"  [제안] {result['suggest_object']}을(를) 가져다 드릴까요?")
                        print(f"         '응' 또는 '네'로 확인해주세요.")

                        confirm_text = self.stt.listen()
                        if confirm_text and self._is_confirm(confirm_text):
                            cmd_result = self.chatbot.confirm_suggestion(result["suggest_object"])
                            print(f"  [확인] {cmd_result['response']}")
                            self._execute_robot_task(cmd_result["command"])
                        else:
                            print(f"  [취소] 알겠습니다.")

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
                cur, tgt = self.dobot.execute(delta)
                print(
                    f"  Cycle {cycle+1} [{i+1}/{len(actions)}] "
                    f"Δ[{delta[0]:+.1f},{delta[1]:+.1f},{delta[2]:+.1f}] "
                    f"G:{'ON' if self.dobot.grip_on else 'OFF'} "
                    f"{dt_ms:.0f}ms"
                )

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
    parser = argparse.ArgumentParser(description="음성 명령 → Pi0 → DOBOT 제어")

    # Pi0 서버
    parser.add_argument("--server", type=str, required=True,
                        help="Pi0 서버 URL (예: http://192.168.1.100:8000)")

    # DOBOT
    parser.add_argument("--port", type=str, default=None, help="DOBOT 시리얼 포트")
    parser.add_argument("--cam1", type=int, default=1, help="Top 카메라 ID")
    parser.add_argument("--cam2", type=int, default=2, help="Front 카메라 ID")

    # Pi0
    parser.add_argument("--chunk-size", type=int, default=2, help="Pi0 액션 청크 스텝 수")
    parser.add_argument("--cycles", type=int, default=20, help="태스크당 최대 사이클")

    # STT 모델
    parser.add_argument("--stt-model", type=str, default="Qwen/Qwen2.5-Omni-7B",
                        help="STT 모델명")
    parser.add_argument("--stt-device", type=str, default=None,
                        help="STT 디바이스 (cuda/mps/cpu)")

    # LLM 모델
    parser.add_argument("--llm-model", type=str, default="Qwen/Qwen3-8B",
                        help="LLM 모델명")
    parser.add_argument("--llm-device", type=str, default=None,
                        help="LLM 디바이스 (cuda/mps/cpu)")

    args = parser.parse_args()

    pipeline = VoiceControlPipeline(args)
    pipeline.run()


if __name__ == "__main__":
    main()
