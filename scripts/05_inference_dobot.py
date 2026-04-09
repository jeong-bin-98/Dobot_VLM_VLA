#!/usr/bin/env python3
"""
Pi0-FAST 추론 — Dobot Magician 실시간 구동

학습된 Pi0 모델로 카메라 관측 -> 액션 예측 -> Dobot 실행을 반복합니다.

    python 05_inference_dobot.py \
        --model_path ./outputs/pi0fast_dobot/checkpoints/last/pretrained_model \
        --cam1 0 --cam2 1 \
        --task "pick up the red cup"
"""

import os
import sys
import time
import math
import argparse
from pathlib import Path
from typing import Optional

try:
    import cv2
    import json
    import numpy as np
    import torch
    import pydobot
    from serial.tools import list_ports
    from safetensors.torch import load_file
except ImportError as e:
    print(f"Missing dependency: {e}")
    sys.exit(1)

IMG_W, IMG_H = 640, 480

# Safety bounds (DobotController와 동일)
BOUNDS = {
    "x": (150, 310),
    "y": (-150, 150),
    "z": (-30, 150),
    "r": (-90, 90),
}
# Dobot Magician 기구학 상수
DOBOT_L1 = 135.0   # mm — rear arm 길이
DOBOT_L2 = 147.0   # mm — forearm 길이
SINGULARITY_THRESHOLD = 0.92  # |cos(θ2)| > 이 값이면 singularity 위험
ALARM_POS_THRESHOLD = 5.0  # mm — move_to 후 오차가 이보다 크면 ALARM 판정


def _cos_j2(x, y):
    """목표점 (x,y)에서의 cos(θ2) 계산. |값| → 1이면 singularity."""
    r_sq = x ** 2 + y ** 2
    denom = 2 * DOBOT_L1 * DOBOT_L2
    return float(np.clip((r_sq - DOBOT_L1 ** 2 - DOBOT_L2 ** 2) / denom, -1.0, 1.0))


def _path_crosses_singularity(cx, cy, tx, ty, n_samples=5):
    """직선 경로 상 n개 지점을 샘플링하여 singularity 관통 여부 판단."""
    for t in np.linspace(0, 1, n_samples):
        px = cx + t * (tx - cx)
        py = cy + t * (ty - cy)
        if abs(_cos_j2(px, py)) > SINGULARITY_THRESHOLD:
            return True
    return False


def _compute_via_point(cx, cy, tx, ty):
    """Singularity 영역을 우회하는 경유점 계산."""
    safe_r = math.sqrt(DOBOT_L1 ** 2 + DOBOT_L2 ** 2)

    mx, my = (cx + tx) / 2, (cy + ty) / 2
    mid_r = math.sqrt(mx ** 2 + my ** 2)

    if mid_r > 1e-3:
        scale = safe_r / mid_r
        vx, vy = mx * scale, my * scale
    else:
        dx, dy = tx - cx, ty - cy
        path_len = math.sqrt(dx ** 2 + dy ** 2)
        if path_len > 1e-3:
            vx, vy = -dy / path_len * safe_r, dx / path_len * safe_r
        else:
            vx, vy = safe_r, 0

    vx = float(np.clip(vx, *BOUNDS["x"]))
    vy = float(np.clip(vy, *BOUNDS["y"]))

    if abs(_cos_j2(vx, vy)) > SINGULARITY_THRESHOLD:
        return None
    return (vx, vy)


class ModelNormalizer:
    """모델 학습 시 사용된 정규화 통계를 로드하여 state 정규화 / action 역정규화 수행."""

    def __init__(self, model_path: str):
        import glob
        model_path = str(model_path)

        # Input: state 정규화 (mean/std)
        pre_files = sorted(glob.glob(os.path.join(model_path, "policy_preprocessor_step_*_normalizer_processor.safetensors")))
        if not pre_files:
            raise FileNotFoundError(
                f"Preprocessor not found in {model_path}. "
                f"Expected: policy_preprocessor_step_*_normalizer_processor.safetensors"
            )
        data = load_file(pre_files[0])
        self.state_mean = data["observation.state.mean"].numpy()
        self.state_std = data["observation.state.std"].numpy()
        self.state_std = np.where(self.state_std < 1e-6, 1.0, self.state_std)
        print(f"  Preprocessor loaded: state mean={self.state_mean}, std={self.state_std}")

        # Output: action 역정규화 (mean/std)
        post_files = sorted(glob.glob(os.path.join(model_path, "policy_postprocessor_step_*_unnormalizer_processor.safetensors")))
        if not post_files:
            raise FileNotFoundError(
                f"Postprocessor not found in {model_path}. "
                f"Expected: policy_postprocessor_step_*_unnormalizer_processor.safetensors"
            )
        data = load_file(post_files[0])
        self.action_mean = data["action.mean"].numpy()
        self.action_std = data["action.std"].numpy()
        self.action_std = np.where(self.action_std < 1e-6, 1.0, self.action_std)
        print(f"  Postprocessor loaded: action mean={self.action_mean}, std={self.action_std}")

    def normalize_state(self, raw: np.ndarray) -> np.ndarray:
        return (np.array(raw, dtype=np.float32) - self.state_mean) / self.state_std

    def unnormalize_action(self, norm: np.ndarray) -> np.ndarray:
        return norm * self.action_std + self.action_mean


def find_dobot_port() -> Optional[str]:
    for p in list_ports.comports():
        if any(chip in p.description for chip in ("CH340", "CP210")):
            return p.device
    for p in list_ports.comports():
        if "usbserial" in p.device:
            return p.device
    return None


def load_policy(model_path: str, device: str):
    """학습된 Pi0-FAST 모델 로드."""
    try:
        from lerobot.common.policies.pi0_fast.modeling_pi0fast import PI0FastPolicy
        policy = PI0FastPolicy.from_pretrained(model_path)
    except (ImportError, Exception):
        from lerobot.common.policies.pi0.modeling_pi0 import PI0Policy
        policy = PI0Policy.from_pretrained(model_path)

    policy.eval()
    policy.to(device)
    print(f"  Model loaded: {model_path}")
    print(f"  Device: {device}")
    return policy


def capture_image(cap: cv2.VideoCapture) -> Optional[torch.Tensor]:
    """카메라 프레임을 모델 입력 텐서로 변환. [1, 3, H, W] float32, 0-1 범위."""
    ret, frame = cap.read()
    if not ret:
        return None
    frame = cv2.resize(frame, (IMG_W, IMG_H))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
    return tensor.unsqueeze(0)


def get_state(bot: pydobot.Dobot) -> torch.Tensor:
    """Dobot 현재 상태를 [1, 5] 텐서로 반환. [x, y, z, r, grip]"""
    try:
        pose = bot.pose()
        state = [pose[0], pose[1], pose[2], pose[3], 0.0]
    except Exception:
        state = [0.0, 0.0, 0.0, 0.0, 0.0]
    return torch.tensor([state], dtype=torch.float32)


def execute_action(bot: pydobot.Dobot, action: list) -> pydobot.Dobot:
    """예측된 delta action을 Dobot에 실행 (cos(θ2) singularity 감지 + 경유점 회피).

    ALARM 복구 시 bot이 재생성되므로, 반환된 bot을 사용해야 합니다.
    """
    pose = bot.pose()
    cur = [pose[0], pose[1], pose[2], pose[3]]
    tx = float(np.clip(cur[0] + action[0], *BOUNDS["x"]))
    ty = float(np.clip(cur[1] + action[1], *BOUNDS["y"]))
    tz = float(np.clip(cur[2] + action[2], *BOUNDS["z"]))
    tr = float(np.clip(cur[3] + action[3], *BOUNDS["r"]))
    grip = action[4] > 0.5

    # 1단계: cos(θ2) 기반 singularity 감지
    target_cos = _cos_j2(tx, ty)
    target_dangerous = abs(target_cos) > SINGULARITY_THRESHOLD
    path_dangerous = _path_crosses_singularity(cur[0], cur[1], tx, ty)

    if target_dangerous or path_dangerous:
        via = _compute_via_point(cur[0], cur[1], tx, ty)
        if via:
            print(f"  >> Singularity 회피: 경유점 ({via[0]:.0f}, {via[1]:.0f})mm "
                  f"[cos(θ2): 목표={target_cos:+.2f}]")
            bot.move_to(via[0], via[1], tz, tr, wait=True)

        if target_dangerous:
            safe_r = math.sqrt(DOBOT_L1 ** 2 + DOBOT_L2 ** 2)
            dist = math.sqrt(tx ** 2 + ty ** 2)
            if dist > 1e-3:
                scale = safe_r / dist
                tx, ty = float(tx * scale), float(ty * scale)
                tx = float(np.clip(tx, *BOUNDS["x"]))
                ty = float(np.clip(ty, *BOUNDS["y"]))
            print(f"  >> 목표 스케일다운: ({tx:.0f}, {ty:.0f})mm")

    # 2단계: 최종 목표로 이동
    bot.move_to(tx, ty, tz, tr, wait=True)

    # 3단계: ALARM 감지 + j2 관절각도 사후 검증
    try:
        actual = bot.pose()
        error = math.sqrt((actual[0] - tx) ** 2 + (actual[1] - ty) ** 2 + (actual[2] - tz) ** 2)
        if error > ALARM_POS_THRESHOLD:
            print(f"  >> ALARM 감지 (오차 {error:.1f}mm), 복구 중...")
            bot = _clear_alarm(bot)
            return bot
        j2 = actual[5]
        if j2 < 5 or j2 > 80:
            print(f"  >> j2={j2:.1f}deg — singularity 근접 경고")
    except Exception:
        pass

    try:
        bot.suck(grip)
    except Exception:
        try:
            bot.grip(grip)
        except Exception:
            pass

    return bot


def _clear_alarm(bot: pydobot.Dobot) -> pydobot.Dobot:
    """ALARM 발생 시 시리얼 재연결로 복구 후 안전 위치로 이동."""
    import gc
    port = None
    try:
        ser = getattr(bot, 'ser', None)
        if ser:
            port = ser.port
            if ser.is_open:
                ser.close()
    except Exception:
        pass

    # 기존 객체 완전 제거
    try:
        del bot
    except Exception:
        pass
    gc.collect()

    print(f"  >> 알람 해제 중... (4초 대기)")
    time.sleep(4)

    try:
        new_bot = pydobot.Dobot(port=port, verbose=False)
        time.sleep(1)
        new_bot.speed(100, 100)
        new_bot.move_to(200, 0, 50, 0, wait=True)
        new_bot.speed(150, 150)
        pose = new_bot.pose()
        print(f"  >> ALARM 복구 완료: ({pose[0]:.0f},{pose[1]:.0f},{pose[2]:.0f})")
        return new_bot
    except Exception as e:
        print(f"  >> ALARM 복구 실패: {e}")
        # 최후 수단: 새 연결 시도
        try:
            return pydobot.Dobot(port=port, verbose=False)
        except Exception:
            raise RuntimeError(f"Dobot 재연결 불가: {e}")


def main():
    parser = argparse.ArgumentParser(description="Pi0 Inference on Dobot Magician")
    parser.add_argument("--model_path", type=str, required=True, help="학습된 모델 체크포인트 경로")
    parser.add_argument("--port", type=str, default=None, help="Dobot 시리얼 포트")
    parser.add_argument("--cam1", type=int, default=0, help="Wrist 카메라 ID (데이터 수집과 동일)")
    parser.add_argument("--cam2", type=int, default=1, help="Top 카메라 ID (데이터 수집과 동일)")
    parser.add_argument("--task", type=str, default="pick up the red cup", help="태스크 설명")
    parser.add_argument("--max_steps", type=int, default=50, help="최대 스텝 수")
    parser.add_argument("--device", type=str, default=None, help="추론 디바이스 (cuda/mps/cpu)")
    args = parser.parse_args()

    # 디바이스 자동 선택
    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    # 태스크 정규화
    from task_normalizer import TaskNormalizer
    task_normalizer = TaskNormalizer()
    task = task_normalizer.normalize(args.task)

    # 모델 + 정규화 통계 로드
    policy = load_policy(args.model_path, device)
    normalizer = ModelNormalizer(args.model_path)

    # Dobot 연결
    port = args.port or find_dobot_port()
    if not port:
        print("Dobot을 찾을 수 없습니다")
        return

    print(f"  Dobot 연결 중: {port}")
    time.sleep(3)
    bot = pydobot.Dobot(port=port, verbose=False)
    time.sleep(1)
    bot.speed(150, 150)
    print(f"  Dobot connected")

    # 카메라 연결
    cap1 = cv2.VideoCapture(args.cam1, cv2.CAP_AVFOUNDATION)
    cap2 = cv2.VideoCapture(args.cam2, cv2.CAP_AVFOUNDATION)
    if not cap1.isOpened() or not cap2.isOpened():
        print("카메라 연결 실패")
        return

    print(f"\n{'='*60}")
    print(f"  Pi0 Inference -- {task}")
    print(f"  Max steps: {args.max_steps}")
    print(f"  [Q] 중지  [SPACE] 시작/일시정지")
    print(f"{'='*60}\n")

    running = False
    step = 0

    while True:
        # 카메라 프리뷰
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        if ret1 and ret2:
            display = np.hstack([frame1, frame2])
            status = f"RUNNING step {step}/{args.max_steps}" if running else "PAUSED"
            color = (0, 0, 255) if running else (0, 255, 0)
            cv2.putText(display, status, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.imshow("Pi0 Inference", display)

        key = cv2.waitKey(30) & 0xFF
        if key == ord("q"):
            break
        elif key == ord(" "):
            running = not running
            print(f"  {'Started' if running else 'Paused'}")
            continue

        if not running or step >= args.max_steps:
            continue

        # 관측 수집 (cam1=wrist, cam2=top — 데이터 수집과 동일)
        img_wrist = capture_image(cap1)
        img_top = capture_image(cap2)
        raw_state = get_state(bot)

        if img_wrist is None or img_top is None:
            continue

        # State 정규화 (학습 시 mean/std로 정규화됨)
        norm_state = normalizer.normalize_state(raw_state.numpy().flatten())
        state_t = torch.tensor(norm_state, dtype=torch.float32).unsqueeze(0)

        # 모델 추론 (키 이름은 학습 데이터와 일치해야 함)
        observation = {
            "observation.images.wrist": img_wrist.to(device),
            "observation.images.top": img_top.to(device),
            "observation.state": state_t.to(device),
            "task": task,
        }

        with torch.no_grad():
            action = policy.select_action(observation)

        # 첫 번째 액션 추출 + 역정규화
        action_np = action[0].cpu().numpy() if action.dim() > 1 else action.cpu().numpy()
        action_list = normalizer.unnormalize_action(action_np[:5]).tolist()

        print(f"  Step {step + 1}: delta=({action_list[0]:+.1f}, {action_list[1]:+.1f}, "
              f"{action_list[2]:+.1f}) grip={'ON' if action_list[4] > 0.5 else 'OFF'}")

        # Dobot 실행 (ALARM 복구 시 bot이 재생성될 수 있음)
        bot = execute_action(bot, action_list)
        step += 1
        time.sleep(0.1)

    # 정리
    bot.suck(False)
    bot.close()
    cap1.release()
    cap2.release()
    cv2.destroyAllWindows()
    print(f"\n  완료: {step} steps 실행")


if __name__ == "__main__":
    main()
