"""
태스크 정규화 모듈

다양한 한국어/영어 사용자 입력을 학습 시 사용한 표준 태스크 문자열로 변환합니다.
의존성 없이 키워드 매칭 + difflib 퍼지 매칭으로 동작합니다.

사용법:
    normalizer = TaskNormalizer()
    canonical = normalizer.normalize("빨간 컵 집어줘")
    # -> "pick up the red cup"
"""

import json
from difflib import get_close_matches
from pathlib import Path
from typing import Optional

# 태스크 레지스트리: 학습에 사용된 정확한 문자열
# 태스크를 추가하려면 여기에 항목을 추가하세요.
TASK_REGISTRY = [
    {
        "canonical": "pick up the red cup",
        "keywords_en": ["pick", "grab", "grasp", "get", "take", "lift", "red cup"],
        "keywords_ko": ["집", "잡", "줍", "들", "가져", "빨간", "컵"],
    },
    # 태스크 추가 예시:
    # {
    #     "canonical": "place the cup on the plate",
    #     "keywords_en": ["place", "put", "set", "drop", "release", "plate"],
    #     "keywords_ko": ["놓", "내려", "두", "올려", "접시"],
    # },
]


class TaskNormalizer:
    def __init__(self, config_path: Optional[str] = None):
        if config_path and Path(config_path).exists():
            with open(config_path) as f:
                self.tasks = json.load(f)["tasks"]
        else:
            self.tasks = TASK_REGISTRY

        # 모든 canonical 문자열 목록
        self.canonical_list = [t["canonical"] for t in self.tasks]

    def normalize(self, user_input: str) -> str:
        """사용자 입력을 표준 태스크 문자열로 변환."""
        text = user_input.strip().lower()

        # 1) 정확히 일치하면 바로 반환
        if text in self.canonical_list:
            return text

        # 2) 키워드 매칭 (점수 기반)
        best_task = None
        best_score = 0

        for task in self.tasks:
            score = 0
            # 영어 키워드
            for kw in task["keywords_en"]:
                if kw.lower() in text:
                    score += 1
            # 한국어 키워드 (부분 문자열 매칭 — 교착어 대응)
            for kw in task["keywords_ko"]:
                if kw in text:
                    score += 1

            if score > best_score:
                best_score = score
                best_task = task["canonical"]

        if best_score >= 1:
            print(f"  [TaskNorm] \"{user_input}\" -> \"{best_task}\" (score: {best_score})")
            return best_task

        # 3) 퍼지 매칭 (오타 대응)
        all_keywords = []
        for task in self.tasks:
            all_keywords.extend(task["keywords_en"])
        matches = get_close_matches(text, all_keywords, n=1, cutoff=0.6)
        if matches:
            for task in self.tasks:
                if matches[0] in task["keywords_en"]:
                    print(f"  [TaskNorm] \"{user_input}\" ~> \"{task['canonical']}\" (fuzzy: {matches[0]})")
                    return task["canonical"]

        # 4) 매칭 실패 — 원본 그대로 전달 (Pi0가 자체 해석)
        print(f"  [TaskNorm] \"{user_input}\" -> 매칭 없음, 원본 전달")
        return user_input
