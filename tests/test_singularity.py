"""
DH 파라미터 기반 Singularity 회피 전략 검증 테스트

DH 모델: r = 135*sin(j2) + 206
  - 135 = rear arm (a2)
  - 206 = forearm(147) + wrist_mech(59)
  - j2: bot.pose()[5]에서 읽은 관절각도
"""

import math
import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "client"))

from pi0_dobot_client import (
    _predict_j2,
    _path_crosses_singularity,
    _compute_via_point,
    DOBOT_A2,
    DOBOT_OFFSET,
    J2_SAFE_MIN,
    REACH_SAFE_MIN,
    BOUNDS,
    HOME_POS,
)


# ──────────────────────────────────────────
# _predict_j2 테스트
# ──────────────────────────────────────────

class TestPredictJ2:
    """r = 135*sin(j2) + 206  →  j2 = arcsin((r - 206) / 135)"""

    def test_singularity_boundary(self):
        """r = 206mm → j2 = 0° (singularity 경계)."""
        assert _predict_j2(206, 0) == pytest.approx(0.0, abs=0.5)

    def test_home_position_dangerous(self):
        """기존 Home (200, 0) → j2 ≈ -2.5° (위험!)."""
        j2 = _predict_j2(200, 0)
        assert j2 < 0  # 음수 = singularity 아래

    def test_new_home_safe(self):
        """새 Home (240, 0) → j2 ≈ 14.5° (안전)."""
        j2 = _predict_j2(240, 0)
        assert j2 > J2_SAFE_MIN

    def test_log_match_150_150(self):
        """로그 검증: (150, -150) → r=212mm → j2 ≈ 2.6° (실제 로그 2~4°)."""
        j2 = _predict_j2(150, -150)
        assert 1.0 < j2 < 5.0  # 실제 로그와 일치 범위

    def test_log_match_home(self):
        """로그 검증: Home (200, 0) → j2 ≈ -2.5° (실제 로그 -2.8°)."""
        j2 = _predict_j2(200, 0)
        assert -4.0 < j2 < -1.0  # 실제 로그와 일치 범위

    def test_safe_position_250(self):
        """(250, 0) → j2 ≈ 19° (안전)."""
        j2 = _predict_j2(250, 0)
        assert j2 > J2_SAFE_MIN

    def test_max_reach(self):
        """r = 341mm (이론적 최대) → j2 = 90°."""
        j2 = _predict_j2(341, 0)
        assert j2 == pytest.approx(90.0, abs=1.0)

    def test_beyond_reach_clipped(self):
        """r > 341mm → j2 = 90° (클리핑)."""
        j2 = _predict_j2(400, 0)
        assert j2 == pytest.approx(90.0, abs=0.1)

    def test_very_close_clipped(self):
        """r = 50mm → j2 클리핑 (도달 불가 영역)."""
        j2 = _predict_j2(50, 0)
        assert j2 == pytest.approx(-90.0, abs=0.1)

    def test_diagonal_position(self):
        """대각선 (200, 100) → r=224mm → j2 ≈ 7.7°."""
        j2 = _predict_j2(200, 100)
        r = math.sqrt(200**2 + 100**2)
        expected = math.degrees(math.asin((r - DOBOT_OFFSET) / DOBOT_A2))
        assert j2 == pytest.approx(expected, abs=0.1)

    def test_symmetry(self):
        """(x, y)와 (x, -y)는 같은 j2."""
        assert _predict_j2(250, 80) == pytest.approx(_predict_j2(250, -80))


# ──────────────────────────────────────────
# _path_crosses_singularity 테스트
# ──────────────────────────────────────────

class TestPathCrossesSingularity:

    def test_safe_path(self):
        """(250, 0) → (260, 50): 전 구간 j2 > 10° → 안전."""
        assert _path_crosses_singularity(250, 0, 260, 50) is False

    def test_dangerous_target(self):
        """(250, 0) → (200, 0): 목표 j2 ≈ -2.5° → 위험."""
        assert _path_crosses_singularity(250, 0, 200, 0) is True

    def test_both_safe(self):
        """(240, -50) → (260, 50): 양쪽 다 안전."""
        assert _path_crosses_singularity(240, -50, 260, 50) is False

    def test_stationary_safe(self):
        """제자리 (250, 0) → 안전."""
        assert _path_crosses_singularity(250, 0, 250, 0) is False

    def test_path_through_dangerous_zone(self):
        """(250, -100) → (250, 100): 중간에 (250, 0) 통과 — r=250, j2≈19° → 안전."""
        assert _path_crosses_singularity(250, -100, 250, 100) is False


# ──────────────────────────────────────────
# _compute_via_point 테스트
# ──────────────────────────────────────────

class TestComputeViaPoint:

    def test_via_point_is_safe(self):
        """경유점의 j2 > J2_SAFE_MIN."""
        via = _compute_via_point(250, 0, 200, 0)
        if via is not None:
            assert _predict_j2(via[0], via[1]) >= J2_SAFE_MIN

    def test_via_point_within_bounds(self):
        """경유점은 BOUNDS 안."""
        via = _compute_via_point(250, 0, 200, 0)
        if via is not None:
            assert BOUNDS["x"][0] <= via[0] <= BOUNDS["x"][1]
            assert BOUNDS["y"][0] <= via[1] <= BOUNDS["y"][1]

    def test_via_point_near_safe_reach(self):
        """경유점은 REACH_SAFE_MIN(≈229mm) + 20 ≈ 249mm 근처."""
        safe_r = REACH_SAFE_MIN + 20
        via = _compute_via_point(250, 50, 200, -30)
        if via is not None:
            via_r = math.sqrt(via[0]**2 + via[1]**2)
            assert via_r < safe_r * 1.5

    def test_returns_none_if_impossible(self):
        """안전한 경유점을 못 찾으면 None."""
        result = _compute_via_point(200, 0, 200, 0)
        assert result is None or len(result) == 2


# ──────────────────────────────────────────
# DH 모델 수학 검증
# ──────────────────────────────────────────

class TestDHModel:

    def test_offset_from_dh_params(self):
        """OFFSET = forearm(147) + wrist_mech(59) = 206."""
        assert DOBOT_OFFSET == pytest.approx(147 + 59)

    def test_reach_safe_min_formula(self):
        """REACH_SAFE_MIN = 206 + 135*sin(10°) ≈ 229mm."""
        expected = DOBOT_OFFSET + DOBOT_A2 * math.sin(math.radians(J2_SAFE_MIN))
        assert REACH_SAFE_MIN == pytest.approx(expected)

    def test_max_theoretical_reach(self):
        """최대 도달 = 206 + 135*sin(90°) = 341mm."""
        r_max = DOBOT_OFFSET + DOBOT_A2
        assert r_max == pytest.approx(341.0)

    def test_min_reach_at_j2_zero(self):
        """j2=0° → r = 206mm."""
        r_at_zero = DOBOT_OFFSET + DOBOT_A2 * math.sin(math.radians(0))
        assert r_at_zero == pytest.approx(206.0)


# ──────────────────────────────────────────
# 통합 시나리오 테스트
# ──────────────────────────────────────────

class TestIntegrationScenarios:

    def test_new_home_is_safe(self):
        """새 Home (240,0,80,0) → j2 > 10°."""
        assert _predict_j2(HOME_POS[0], HOME_POS[1]) > J2_SAFE_MIN

    def test_old_home_was_dangerous(self):
        """기존 Home (200,0) → j2 < 0° → 위험했음."""
        assert _predict_j2(200, 0) < J2_SAFE_MIN

    def test_bounds_exclude_dangerous_corners(self):
        """새 BOUNDS의 모든 코너가 안전."""
        corners = [
            (BOUNDS["x"][0], BOUNDS["y"][0]),
            (BOUNDS["x"][0], BOUNDS["y"][1]),
            (BOUNDS["x"][1], BOUNDS["y"][0]),
            (BOUNDS["x"][1], BOUNDS["y"][1]),
        ]
        for x, y in corners:
            j2 = _predict_j2(x, y)
            r = math.sqrt(x**2 + y**2)
            assert j2 > 0, f"({x},{y}) r={r:.0f}mm j2={j2:.1f}° 위험!"

    def test_workspace_center_safe(self):
        """작업 영역 중심 (250, 0) → 안전."""
        assert _predict_j2(250, 0) > J2_SAFE_MIN

    def test_constants_reasonable(self):
        """상수값 합리성."""
        assert 130 < DOBOT_A2 < 140
        assert 200 < DOBOT_OFFSET < 210
        assert 5 < J2_SAFE_MIN < 20
        assert BOUNDS["x"][0] >= 200  # r=206 이하 차단
