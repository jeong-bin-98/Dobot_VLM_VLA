#!/usr/bin/env python3
"""
2단계 대화 경로 분류 + 커맨드 생성

STT 결과(한국어 텍스트) → Qwen 3.0으로 robot/dialog 분류 → 영어 프롬프트 or 대화 응답

    python chatbot_module.py
"""

import sys
import json
import re

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
except ImportError:
    print("pip install transformers torch")
    sys.exit(1)

# Pi0-FAST 학습 시 사용한 영어 프롬프트 (글자 하나까지 동일해야 함)
COMMAND_MAP = {
    "과자":       "pick up the snack and hand it over",
    "칸초":       "pick up the snack and hand it over",
    "간식":       "pick up the snack and hand it over",
    "음료":       "pick up the drink and hand it over",
    "피크닉":     "pick up the drink and hand it over",
    "물":         "pick up the drink and hand it over",
    "연필":       "pick up the pencil and hand it over",
    "지우개":     "pick up the eraser and hand it over",
    "휴지":       "pick up the tissue and hand it over",
    "스트레스":   "pick up the stress ball and hand it over",
    "스트레스볼": "pick up the stress ball and hand it over",
}

STOP_KEYWORDS = {"종료", "그만", "멈춰", "스톱", "stop", "끝", "정지"}

SYSTEM_PROMPT = """너는 DOBOT Magician 로봇 팔의 음성 명령 분류기야.
사용 가능한 물체: 과자, 음료, 연필, 지우개, 휴지, 스트레스볼

[분류 규칙]
- 물건을 가져다달라는 요청 → type: "robot", object에 물체명
- 일상 대화/감정 표현 → type: "dialog", 상황에 맞는 물체 제안

[STT 오인식 보정 — LLM이 맥락으로 판단]
- "과장 줘"는 로봇 명령 맥락에서 "과자 줘"의 오인식일 수 있음
- "음뇨"→"음료", "연삐"→"연필" 등 음성인식 오류 감안하여 판단
- 단, "과장님 전화해"처럼 명확한 대화 맥락이면 dialog 처리

[상황-물체 매핑]
- 피곤해/스트레스 받아 → 스트레스볼 제안
- 배고프다/출출해 → 과자 제안
- 목마르다 → 음료 제안
- 글 쓸 거 있어/필기해야 해 → 연필 제안
- 틀렸어/지워야 해 → 지우개 제안
- 코 풀어야 해/닦아야 해 → 휴지 제안

반드시 JSON으로만 응답:
{"type":"robot","object":"과자","response":null,"suggest_object":null}
또는
{"type":"dialog","object":null,"response":"힘드시군요.","suggest_object":"스트레스볼"}"""


class ChatbotRouter:
    """한국어 텍스트 → robot/dialog 분류 → 영어 프롬프트 or 대화 응답"""

    def __init__(self, model_name="Qwen/Qwen3-8B", device=None):
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

        print(f"  [LLM] 모델 로딩: {model_name} ({self.device})")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
            device_map=self.device,
        )
        self.model.eval()
        print(f"  [LLM] 모델 로드 완료")

    def process(self, korean_text: str) -> dict:
        """
        한국어 텍스트를 분류하고 적절한 응답/커맨드를 반환.

        Args:
            korean_text: STT에서 받은 한국어 텍스트

        Returns:
            {
                "type": "robot" | "dialog" | "stop",
                "command": "pick up the snack and hand it over" | None,
                "response": "대화 응답" | None,
                "suggest_object": "과자" | None,
                "raw_text": "원본 한국어"
            }
        """
        # 종료 키워드 체크
        if korean_text.strip() in STOP_KEYWORDS:
            return {
                "type": "stop",
                "command": None,
                "response": "시스템을 종료합니다.",
                "suggest_object": None,
                "raw_text": korean_text,
            }

        # Qwen 3.0으로 분류
        llm_result = self._classify(korean_text)

        # 결과 처리
        result = {
            "type": llm_result.get("type", "dialog"),
            "command": None,
            "response": llm_result.get("response"),
            "suggest_object": llm_result.get("suggest_object"),
            "raw_text": korean_text,
        }

        if result["type"] == "robot":
            obj = llm_result.get("object", "")
            # COMMAND_MAP에서 영어 프롬프트 매칭
            result["command"] = self._get_command(obj)
            if result["command"] is None:
                result["type"] = "dialog"
                result["response"] = f"'{obj}'은(는) 사용 가능한 물체가 아닙니다. 사용 가능: 과자, 음료, 연필, 지우개, 휴지, 스트레스볼"

        return result

    def confirm_suggestion(self, suggest_object: str) -> dict:
        """
        dialog에서 제안한 물체를 사용자가 확인한 경우 커맨드 반환.

        Args:
            suggest_object: 제안된 물체명 (한국어)

        Returns:
            {"type": "robot", "command": "...", ...}
        """
        command = self._get_command(suggest_object)
        return {
            "type": "robot",
            "command": command,
            "response": f"{suggest_object}을(를) 가져다 드릴게요.",
            "suggest_object": None,
            "raw_text": f"(확인: {suggest_object})",
        }

    def _classify(self, text: str) -> dict:
        """Qwen 3.0으로 텍스트 분류"""
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ]

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.3,
                do_sample=True,
            )

        response = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True,
        )

        return self._parse_json(response)

    def _parse_json(self, text: str) -> dict:
        """LLM 출력에서 JSON 추출"""
        # JSON 블록 찾기
        match = re.search(r'\{[^{}]*\}', text)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

        # 파싱 실패 시 기본값
        return {
            "type": "dialog",
            "object": None,
            "response": text.strip()[:100],
            "suggest_object": None,
        }

    @staticmethod
    def _get_command(object_name: str) -> str | None:
        """한국어 물체명 → COMMAND_MAP에서 영어 프롬프트 반환"""
        if not object_name:
            return None

        # 정확히 일치
        if object_name in COMMAND_MAP:
            return COMMAND_MAP[object_name]

        # 부분 매칭 (예: "스트레스볼" in "스트레스볼 좀")
        for key, cmd in COMMAND_MAP.items():
            if key in object_name or object_name in key:
                return cmd

        return None


# 단독 실행: 키보드 입력 테스트
if __name__ == "__main__":
    print("=" * 50)
    print("  ChatbotRouter 키보드 테스트")
    print("  '종료'로 끝내기")
    print("=" * 50)

    router = ChatbotRouter()

    while True:
        user_input = input("\n입력> ").strip()
        if not user_input:
            continue

        result = router.process(user_input)
        print(f"\n  type:     {result['type']}")
        print(f"  command:  {result['command']}")
        print(f"  response: {result['response']}")
        print(f"  suggest:  {result['suggest_object']}")

        if result["type"] == "stop":
            break

        # dialog에서 물체 제안이 있으면 확인 시뮬레이션
        if result["type"] == "dialog" and result["suggest_object"]:
            confirm = input(f"\n  {result['suggest_object']}을(를) 가져다 드릴까요? (y/n): ").strip()
            if confirm.lower() == "y":
                cmd_result = router.confirm_suggestion(result["suggest_object"])
                print(f"  → command: {cmd_result['command']}")
