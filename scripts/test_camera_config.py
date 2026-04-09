#!/usr/bin/env python3
"""
카메라 고정 설정 테스트 — 수동 노출/WB/포커스 확인용

    python scripts/test_camera_config.py --cam1 0 --cam2 1
    python scripts/test_camera_config.py --cam1 2 --cam2 4   (Windows)

[M] 수동/자동 토글  [+/-] 노출 조정  [W/S] 색온도 조정  [ESC] 종료
"""

import argparse
import cv2
import sys

IMG_W, IMG_H = 640, 480

# 고정 설정값 (pi0_dobot_client.py / 01_collect_data.py와 동일)
MANUAL_CONFIG = {
    "autofocus": 0,
    "auto_exposure": 1,      # 1=manual, 3=auto
    "exposure": -4,           # -1 ~ -13 (환경에 맞게)
    "auto_wb": 0,
    "wb_temperature": 5000,   # 4000 ~ 6500
    "gain": 0,
    "fps": 30,
}


def apply_config(cap, config, manual=True):
    """카메라에 설정 적용."""
    if manual:
        cap.set(cv2.CAP_PROP_AUTOFOCUS, config["autofocus"])
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, config["auto_exposure"])
        cap.set(cv2.CAP_PROP_EXPOSURE, config["exposure"])
        cap.set(cv2.CAP_PROP_AUTO_WB, config["auto_wb"])
        cap.set(cv2.CAP_PROP_WB_TEMPERATURE, config["wb_temperature"])
        cap.set(cv2.CAP_PROP_GAIN, config["gain"])
    else:
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)
        cap.set(cv2.CAP_PROP_AUTO_WB, 1)
    cap.set(cv2.CAP_PROP_FPS, config["fps"])


def read_config(cap):
    """카메라에서 실제 적용된 설정 읽기."""
    return {
        "autofocus": cap.get(cv2.CAP_PROP_AUTOFOCUS),
        "auto_exposure": cap.get(cv2.CAP_PROP_AUTO_EXPOSURE),
        "exposure": cap.get(cv2.CAP_PROP_EXPOSURE),
        "auto_wb": cap.get(cv2.CAP_PROP_AUTO_WB),
        "wb_temperature": cap.get(cv2.CAP_PROP_WB_TEMPERATURE),
        "gain": cap.get(cv2.CAP_PROP_GAIN),
        "fps": cap.get(cv2.CAP_PROP_FPS),
    }


def main():
    parser = argparse.ArgumentParser(description="카메라 고정 설정 테스트")
    parser.add_argument("--cam1", type=int, default=0, help="Wrist 카메라 ID")
    parser.add_argument("--cam2", type=int, default=1, help="Top 카메라 ID")
    args = parser.parse_args()

    cap1 = cv2.VideoCapture(args.cam1)
    cap2 = cv2.VideoCapture(args.cam2)

    if not cap1.isOpened() or not cap2.isOpened():
        print("카메라 열기 실패")
        sys.exit(1)

    for cap in [cap1, cap2]:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, IMG_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, IMG_H)

    config = dict(MANUAL_CONFIG)
    manual_mode = True

    for cap in [cap1, cap2]:
        apply_config(cap, config, manual=True)

    print("카메라 고정 설정 테스트")
    print("[M] 수동/자동 토글  [+/-] 노출 조정  [W/S] 색온도 조정  [ESC] 종료")
    print(f"초기 설정: {config}")

    while True:
        _, f1 = cap1.read()
        _, f2 = cap2.read()
        if f1 is None or f2 is None:
            continue

        # 현재 설정 오버레이
        mode_str = "MANUAL" if manual_mode else "AUTO"
        color = (0, 255, 0) if manual_mode else (0, 0, 255)

        info_lines = [
            f"Mode: {mode_str}",
            f"Exposure: {config['exposure']}",
            f"WB Temp: {config['wb_temperature']}K",
            f"Gain: {config['gain']}",
        ]

        for i, line in enumerate(info_lines):
            cv2.putText(f1, line, (10, 30 + i * 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.putText(f2, f"WRIST | {mode_str}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        combined = cv2.hconcat([f1, f2])
        cv2.imshow("Camera Config Test", combined)

        key = cv2.waitKey(30) & 0xFF

        if key == 27:  # ESC
            break

        elif key == ord('m'):
            manual_mode = not manual_mode
            for cap in [cap1, cap2]:
                apply_config(cap, config, manual=manual_mode)
            print(f"{'수동' if manual_mode else '자동'} 모드")

        elif key == ord('+') or key == ord('='):
            config["exposure"] = min(config["exposure"] + 1, -1)
            if manual_mode:
                for cap in [cap1, cap2]:
                    cap.set(cv2.CAP_PROP_EXPOSURE, config["exposure"])
            print(f"노출: {config['exposure']}")

        elif key == ord('-'):
            config["exposure"] = max(config["exposure"] - 1, -13)
            if manual_mode:
                for cap in [cap1, cap2]:
                    cap.set(cv2.CAP_PROP_EXPOSURE, config["exposure"])
            print(f"노출: {config['exposure']}")

        elif key == ord('w'):
            config["wb_temperature"] = min(config["wb_temperature"] + 500, 6500)
            if manual_mode:
                for cap in [cap1, cap2]:
                    cap.set(cv2.CAP_PROP_WB_TEMPERATURE, config["wb_temperature"])
            print(f"색온도: {config['wb_temperature']}K")

        elif key == ord('s'):
            config["wb_temperature"] = max(config["wb_temperature"] - 500, 2500)
            if manual_mode:
                for cap in [cap1, cap2]:
                    cap.set(cv2.CAP_PROP_WB_TEMPERATURE, config["wb_temperature"])
            print(f"색온도: {config['wb_temperature']}K")

        elif key == ord('p'):
            # 현재 실제 설정값 출력
            for name, cap in [("wrist", cap1), ("top", cap2)]:
                actual = read_config(cap)
                print(f"  {name}: {actual}")

    cap1.release()
    cap2.release()
    cv2.destroyAllWindows()
    print(f"\n최종 설정값 (코드에 반영할 값):")
    print(f'  exposure: {config["exposure"]}')
    print(f'  wb_temperature: {config["wb_temperature"]}')
    print(f'  gain: {config["gain"]}')


if __name__ == "__main__":
    main()
