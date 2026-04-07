#!/usr/bin/env python3
"""
Pi0-FAST 추론 — Dobot Magician 실시간 구동

학습된 Pi0 모델로 카메라 관측 -> 액션 예측 -> Dobot 실행을 반복합니다.

    python 05_inference_dobot.py \
        --model_path ./outputs/pi0fast_dobot/checkpoints/last/pretrained_model \
        --cam1 0 --cam2 1 \
        --task "pick up the red cup"
"""

import sys
import time
import argparse
from pathlib import Path
from typing import Optional

try:
    import cv2
    import numpy as np
    import torch
    import pydobot
    from serial.tools import list_ports
except ImportError as e:
    print(f"Missing dependency: {e}")
    sys.exit(1)

IMG_W, IMG_H = 640, 480


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


def execute_action(bot: pydobot.Dobot, action: list):
    """예측된 delta action을 Dobot에 실행."""
    pose = bot.pose()
    new_x = pose[0] + action[0]
    new_y = pose[1] + action[1]
    new_z = pose[2] + action[2]
    new_r = pose[3] + action[3]
    grip = action[4] > 0.5

    bot.move_to(new_x, new_y, new_z, new_r, wait=True)
    try:
        bot.suck(grip)
    except Exception:
        try:
            bot.grip(grip)
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser(description="Pi0 Inference on Dobot Magician")
    parser.add_argument("--model_path", type=str, required=True, help="학습된 모델 체크포인트 경로")
    parser.add_argument("--port", type=str, default=None, help="Dobot 시리얼 포트")
    parser.add_argument("--cam1", type=int, default=0, help="Top 카메라 ID")
    parser.add_argument("--cam2", type=int, default=1, help="Front 카메라 ID")
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
    normalizer = TaskNormalizer()
    task = normalizer.normalize(task)

    # 모델 로드
    policy = load_policy(args.model_path, device)

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

        # 관측 수집
        img1 = capture_image(cap1)
        img2 = capture_image(cap2)
        state = get_state(bot)

        if img1 is None or img2 is None:
            continue

        # 모델 추론
        observation = {
            "observation.images.top": img1.to(device),
            "observation.images.front": img2.to(device),
            "observation.state": state.to(device),
            "task": task,
        }

        with torch.no_grad():
            action = policy.select_action(observation)

        # 첫 번째 액션만 실행 (n_action_steps=1)
        action_np = action[0].cpu().numpy() if action.dim() > 1 else action.cpu().numpy()
        action_list = action_np[:5].tolist()

        print(f"  Step {step + 1}: delta=({action_list[0]:+.1f}, {action_list[1]:+.1f}, "
              f"{action_list[2]:+.1f}) grip={'ON' if action_list[4] > 0.5 else 'OFF'}")

        # Dobot 실행
        execute_action(bot, action_list)
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
