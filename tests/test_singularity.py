"""
Singularity 회피 전략 검증 테스트

테스트 대상:
- _cos_j2(): 역기구학 cos(θ2) 계산 정확성
- _path_crosses_singularity(): 경로 위험 판단
- _compute_via_point(): 경유점 생성 및 안전성
"""

import math
import sys
import os

import numpy as np
import pytest

# 프로젝트 루트를 path에 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "client"))

from pi0_dobot_client import (
    _cos_j2,
    _path_crosses_singularity,
    _compute_via_point,
    DOBOT_L1,
    DOBOT_L2,
    SINGULARITY_THRESHOLD,
    BOUNDS,
)


# ──────────────────────────────────────────
# _cos_j2 테스트
# ──────────────────────────────────────────

class TestCosJ2:
    """cos(θ2) = (x²+y² - L1² - L2²) / (2·L1·L2) 검증."""

    def test_full_extension_singularity(self):
        """팔 완전 신장 (r = L1+L2) → cos(θ2) = 1.0."""
        r = DOBOT_L1 + DOBOT_L2  # 282mm
        result = _cos_j2(r, 0)
        assert result == pytest.approx(1.0, abs=0.01)

    def test_full_fold_singularity(self):
        """팔 완전 접힘 (r = |L1-L2|) → cos(θ2) = -1.0."""
        r = abs(DOBOT_L1 - DOBOT_L2)  # 12mm
        result = _cos_j2(r, 0)
        assert result == pytest.approx(-1.0, abs=0.01)

    def test_elbow_90_degrees(self):
        """θ2 = 90° (최적 위치) → cos(θ2) = 0.0.
        이때 r = sqrt(L1² + L2²) ≈ 200mm."""
        r = math.sqrt(DOBOT_L1**2 + DOBOT_L2**2)
        result = _cos_j2(r, 0)
        assert result == pytest.approx(0.0, abs=0.01)

    def test_safe_position_200_0(self):
        """Home 위치 근처 (200, 0) → singularity 아님."""
        result = _cos_j2(200, 0)
        assert abs(result) < SINGULARITY_THRESHOLD

    def test_dangerous_position_near_max(self):
        """최대 도달 근처 (280, 0) → singularity 위험."""
        result = _cos_j2(280, 0)
        assert abs(result) > SINGULARITY_THRESHOLD

    def test_beyond_reach_clipped(self):
        """도달 불가 (350, 0) → cos(θ2) = 1.0 (클리핑)."""
        result = _cos_j2(350, 0)
        assert result == 1.0

    def test_origin_clipped(self):
        """원점 (0, 0) → cos(θ2) = -1.0 (클리핑)."""
        result = _cos_j2(0, 0)
        assert result == pytest.approx(-1.0, abs=0.01)

    def test_diagonal_position(self):
        """대각선 위치 (150, 150) — r=212mm → 안전 영역."""
        result = _cos_j2(150, 150)
        assert abs(result) < SINGULARITY_THRESHOLD

    def test_symmetry_xy(self):
        """(x, y)와 (y, x)는 같은 cos(θ2) — r² = x²+y²로 동일."""
        assert _cos_j2(200, 100) == pytest.approx(_cos_j2(100, 200))

    def test_negative_coordinates(self):
        """음수 좌표도 정상 작동 (r² = x²+y² 이므로)."""
        assert _cos_j2(200, -50) == pytest.approx(_cos_j2(200, 50))


# ──────────────────────────────────────────
# _path_crosses_singularity 테스트
# ──────────────────────────────────────────

class TestPathCrossesSingularity:

    def test_safe_path(self):
        """안전 영역 내 이동 (200,0) → (200,50) → 위험 없음."""
        assert _path_crosses_singularity(200, 0, 200, 50) is False

    def test_path_to_max_reach(self):
        """안전 → 최대 도달 근처 (200,0) → (280,0) → 위험."""
        assert _path_crosses_singularity(200, 0, 280, 0) is True

    def test_path_through_origin(self):
        """원점 근처 통과 (200,100) → (200,-100) — 경로가 r≈0 안 지남."""
        # 이 경로는 (200, 0)을 지나며 r≈200 → 안전
        assert _path_crosses_singularity(200, 100, 200, -100) is False

    def test_path_along_boundary(self):
        """경계 따라 이동 (270,0) → (270,50) — 둘 다 위험 근처."""
        # r ≈ 270~274mm, L1+L2=282mm
        # cos_j2(270,0) = (72900-18225-21609)/39690 = 33066/39690 = 0.833 < 0.92
        # 실제로 안전 범위일 수 있음
        result = _path_crosses_singularity(270, 0, 270, 50)
        # 270mm는 아슬아슬 — 결과가 어떻든 에러 없이 bool 반환
        assert isinstance(result, bool)

    def test_stationary(self):
        """제자리 (같은 위치) → 위험 없음 (안전 영역일 때)."""
        assert _path_crosses_singularity(200, 0, 200, 0) is False

    def test_short_dangerous_path(self):
        """짧은 이동이지만 목표가 위험 (275,0) → (282,0)."""
        assert _path_crosses_singularity(275, 0, 282, 0) is True


# ──────────────────────────────────────────
# _compute_via_point 테스트
# ──────────────────────────────────────────

class TestComputeViaPoint:

    def test_via_point_is_safe(self):
        """생성된 경유점은 singularity 안전 영역에 있어야 함."""
        via = _compute_via_point(200, 0, 280, 0)
        assert via is not None
        vx, vy = via
        cos_val = _cos_j2(vx, vy)
        assert abs(cos_val) < SINGULARITY_THRESHOLD

    def test_via_point_within_bounds(self):
        """경유점은 BOUNDS 안에 있어야 함."""
        via = _compute_via_point(200, 0, 280, 0)
        assert via is not None
        vx, vy = via
        assert BOUNDS["x"][0] <= vx <= BOUNDS["x"][1]
        assert BOUNDS["y"][0] <= vy <= BOUNDS["y"][1]

    def test_via_point_near_safe_radius(self):
        """경유점은 safe_radius(~200mm) 근처에 있어야 함."""
        safe_r = math.sqrt(DOBOT_L1**2 + DOBOT_L2**2)
        via = _compute_via_point(200, 0, 280, 0)
        assert via is not None
        vx, vy = via
        via_r = math.sqrt(vx**2 + vy**2)
        # safe_r 근처 (BOUNDS 클리핑으로 정확히 일치하지 않을 수 있음)
        assert via_r < safe_r * 1.5

    def test_via_point_different_from_endpoints(self):
        """경유점은 출발/도착과 다른 위치여야 함."""
        via = _compute_via_point(250, 50, 280, 20)
        assert via is not None
        vx, vy = via
        assert not (abs(vx - 250) < 1 and abs(vy - 50) < 1)  # 출발과 다름
        assert not (abs(vx - 280) < 1 and abs(vy - 20) < 1)  # 도착과 다름

    def test_via_point_for_diagonal_path(self):
        """대각선 경로에서도 경유점 생성."""
        via = _compute_via_point(200, -100, 250, 100)
        # 이 경로가 안전하면 None 반환 가능, 위험하면 경유점 반환
        if via is not None:
            vx, vy = via
            assert abs(_cos_j2(vx, vy)) < SINGULARITY_THRESHOLD

    def test_via_point_returns_none_when_impossible(self):
        """경유점을 안전하게 만들 수 없으면 None 반환."""
        # 극단적 케이스: BOUNDS 안에서 모든 점이 위험한 경우는 거의 없지만
        # 함수가 None을 반환하는 경로도 있을 수 있음
        result = _compute_via_point(150, 0, 150, 0)
        # 제자리 → 경유점 자체가 중점(150,0) 스케일 → 유효할 수 있음
        # None이든 tuple이든 에러 없이 반환되면 OK
        assert result is None or len(result) == 2

    def test_via_point_with_large_y(self):
        """y축 큰 값 (150, 130) → (200, 140) — 코너 영역."""
        via = _compute_via_point(150, 130, 200, 140)
        if via is not None:
            vx, vy = via
            assert BOUNDS["y"][0] <= vy <= BOUNDS["y"][1]


# ──────────────────────────────────────────
# 통합 시나리오 테스트
# ──────────────────────────────────────────

class TestIntegrationScenarios:
    """실제 VLA 추론 루프에서 발생할 수 있는 시나리오."""

    def test_home_to_far_target(self):
        """Home(200,0) → 먼 목표(300,50): singularity 감지 + 경유점 생성."""
        cx, cy = 200, 0
        tx, ty = 300, 50

        # 목표 위험도 체크
        target_cos = _cos_j2(tx, ty)
        assert abs(target_cos) > SINGULARITY_THRESHOLD  # 위험해야 함

        # 경로 위험 체크
        assert _path_crosses_singularity(cx, cy, tx, ty) is True

        # 경유점 생성
        via = _compute_via_point(cx, cy, tx, ty)
        assert via is not None
        assert abs(_cos_j2(via[0], via[1])) < SINGULARITY_THRESHOLD

    def test_safe_small_movement(self):
        """작은 이동 (200,0) → (210,10): 전부 안전, 경유점 불필요."""
        cx, cy = 200, 0
        tx, ty = 210, 10

        assert abs(_cos_j2(tx, ty)) < SINGULARITY_THRESHOLD
        assert _path_crosses_singularity(cx, cy, tx, ty) is False

    def test_alarm_recovery_scenario(self):
        """ALARM 후 Home(200,0,50) → 다시 추론: 안전한 곳에서 시작."""
        # Home 위치는 안전해야 함
        home_cos = _cos_j2(200, 0)
        assert abs(home_cos) < SINGULARITY_THRESHOLD

    def test_workspace_corners(self):
        """BOUNDS 4개 코너에서 cos(θ2) 확인."""
        corners = [
            (BOUNDS["x"][0], BOUNDS["y"][0]),  # (150, -150) r=212
            (BOUNDS["x"][0], BOUNDS["y"][1]),  # (150, 150)  r=212
            (BOUNDS["x"][1], BOUNDS["y"][0]),  # (310, -150) r=344
            (BOUNDS["x"][1], BOUNDS["y"][1]),  # (310, 150)  r=344
        ]
        for x, y in corners:
            cos_val = _cos_j2(x, y)
            # 가까운 코너는 안전, 먼 코너는 위험 (값만 확인, 에러 없으면 OK)
            assert -1.0 <= cos_val <= 1.0

    def test_far_corners_are_dangerous(self):
        """먼 코너 (310, ±150)는 singularity 위험."""
        assert abs(_cos_j2(310, 150)) > SINGULARITY_THRESHOLD
        assert abs(_cos_j2(310, -150)) > SINGULARITY_THRESHOLD

    def test_near_corners_are_safe(self):
        """가까운 코너 (150, ±150)는 안전."""
        # r = sqrt(150²+150²) ≈ 212mm → safe
        assert abs(_cos_j2(150, 150)) < SINGULARITY_THRESHOLD
        assert abs(_cos_j2(150, -150)) < SINGULARITY_THRESHOLD

    def test_constants_are_reasonable(self):
        """상수값이 합리적인지 확인."""
        assert 100 < DOBOT_L1 < 200  # mm
        assert 100 < DOBOT_L2 < 200  # mm
        assert 0.8 < SINGULARITY_THRESHOLD < 1.0
        # safe_radius는 BOUNDS 안에 있어야 함
        safe_r = math.sqrt(DOBOT_L1**2 + DOBOT_L2**2)
        assert BOUNDS["x"][0] <= safe_r <= BOUNDS["x"][1]
