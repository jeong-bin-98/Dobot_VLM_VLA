#!/usr/bin/env python3
"""
Pi0 WebSocket 추론 클라이언트 (DOBOT PC)

카메라 캡처 -> WS 전송 -> action 수신 -> DOBOT 실행.
연결 1회 수립 후 유지.

    python pi0_ws_client.py \\
        --server ws://192.168.1.100:8765/ws \\
        --port COM4 --cam1 0 --cam2 1 \\
        --task "pick up the red cup"
"""

import sys
import os
import time
import json
import base64
import argparse
import threading
from typing import Optional, List

import numpy as np

try:
    import cv2
    import pydobot
    from serial.tools import list_ports
    import websocket  # pip install websocket-client
except ImportError as e:
    print(f"err: Missing: {e}")
    print("Install: pip install opencv-python pydobot pyserial websocket-client")
    sys.exit(1)
# DOBOT safety bounds
BOUNDS = {
    "x": (150, 310),
    "y": (-150, 150),
    "z": (-30, 150),
    "r": (-90, 90),
}

# Gripper timing -- adjust on-site
GRIPPER_DELAY_S = 0.1
GRIPPER_WAIT_S = 0.5
GRIPPER_THRESHOLD = 0.5

IMG_W, IMG_H = 640, 480
# Camera capture
class CameraCapture:
    def __init__(self, cam1_id: int = 0, cam2_id: int = 1):
        self.cap1 = cv2.VideoCapture(cam1_id)
        self.cap2 = cv2.VideoCapture(cam2_id)

        # Fallback to V4L2
        if not self.cap1.isOpened():
            self.cap1 = cv2.VideoCapture(cam1_id, cv2.CAP_V4L2)
        if not self.cap2.isOpened():
            self.cap2 = cv2.VideoCapture(cam2_id, cv2.CAP_V4L2)

        for cap in (self.cap1, self.cap2):
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, IMG_W)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, IMG_H)

        print(f"   Cameras: {cam1_id}, {cam2_id}")

    def capture(self):
        _, f1 = self.cap1.read()
        _, f2 = self.cap2.read()
        return f1, f2

    def frame_to_b64(self, frame) -> str:
        """BGR frame -> base64 JPEG string."""
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return base64.b64encode(buf).decode("utf-8")

    def close(self):
        self.cap1.release()
        self.cap2.release()
        cv2.destroyAllWindows()
# DOBOT controller
class DobotControl:
    def __init__(self, port: Optional[str] = None):
        port = port or self._find_port()
        self.dobot = pydobot.Dobot(port=port, verbose=False)
        self.dobot.speed(150, 150)
        self.grip_on = False
        print(f"   DOBOT: {port}")

    def _find_port(self) -> str:
        for p in list_ports.comports():
            if any(c in p.description for c in ("CH340", "CP210")):
                return p.device
        ports = list_ports.comports()
        return ports[0].device if ports else "/dev/ttyUSB0"

    def get_pose(self) -> List[float]:
        r = self.dobot.pose()
        return [round(r[i], 2) for i in range(4)]

    def get_state(self) -> List[float]:
        """[x, y, z, r, grip] -- 5-dim state."""
        return self.get_pose() + [1.0 if self.grip_on else 0.0]

    def execute(self, delta) -> tuple:
        """Apply delta action. Returns (current_pos, target_pos)."""
        cur = self.get_pose()

        tx = float(np.clip(cur[0] + delta[0], *BOUNDS["x"]))
        ty = float(np.clip(cur[1] + delta[1], *BOUNDS["y"]))
        tz = float(np.clip(cur[2] + delta[2], *BOUNDS["z"]))
        tr = float(np.clip(cur[3] + delta[3], *BOUNDS["r"]))

        # Move first
        self.dobot.move_to(tx, ty, tz, tr, wait=True)

        # Gripper after move (TODO: on-site debug)
        new_grip = delta[4] > GRIPPER_THRESHOLD
        if new_grip != self.grip_on:
            self.grip_on = new_grip
            time.sleep(GRIPPER_DELAY_S)
            try:
                self.dobot.grip(self.grip_on)
                time.sleep(GRIPPER_WAIT_S)
            except Exception:
                try:
                    self.dobot.suck(self.grip_on)
                    time.sleep(GRIPPER_WAIT_S)
                except Exception:
                    pass
            print(f"    Grip: {'ON' if self.grip_on else 'OFF'}")

        return cur, [tx, ty, tz, tr]

    def home(self):
        self.dobot.move_to(200, 0, 50, 0, wait=True)
        self.grip_on = False
        try:
            self.dobot.grip(False)
        except Exception:
            pass

    def close(self):
        if self.dobot:
            try:
                self.dobot.grip(False)
                self.dobot.suck(False)
                self.dobot.close()
            except Exception:
                pass
# WebSocket streaming client
class Pi0StreamClient:
    """
    Half-duplex WebSocket inference loop.

    Flow per cycle:
      capture -> encode -> send(WS) -> recv(WS) -> execute -> repeat
                         ^^^^^^^^^^^^^^^^^^^^^^^^
                         persistent connection, no re-handshake
    """

    def __init__(self, server_url: str, port: str, cam1: int, cam2: int,
                 task: str, chunk_size: int = 1, max_cycles: int = 50):
        self.server_url = server_url
        self.task = task
        self.chunk_size = chunk_size
        self.max_cycles = max_cycles

        # Hardware
        self.cameras = CameraCapture(cam1, cam2)
        self.dobot = DobotControl(port)
        self.ws: Optional[websocket.WebSocket] = None

    def connect(self):
        """Establish persistent WebSocket connection."""
        print(f"\n  Connecting to {self.server_url} ...")
        self.ws = websocket.WebSocket()
        self.ws.connect(self.server_url, timeout=10)
        print(f"   WebSocket connected")

    def run(self):
        """Main inference loop."""
        print(f"""
 ============================================================
  Pi0 WebSocket Streaming Client
 ============================================================
  Server: {self.server_url}
  Task:   {self.task}
  Chunk:  {self.chunk_size} action(s) per inference
  [SPACE] 1회 추론   [A] Auto   [H] Home   [G] Grip   [ESC] Exit
 ============================================================
""")
        self.dobot.home()
        auto_mode = False
        cycle = 0

        try:
            while cycle < self.max_cycles:
                # Preview
                f1, f2 = self.cameras.capture()
                if f1 is not None and f2 is not None:
                    self._show_preview(f1, f2, cycle, auto_mode)

                key = cv2.waitKey(30 if not auto_mode else 1) & 0xFF

                if key == 27:  # ESC
                    break

                elif key == ord(" ") or auto_mode:
                    t_total = time.time()

                    # 1) Capture + encode
                    f1, f2 = self.cameras.capture()
                    if f1 is None or f2 is None:
                        continue

                    state = self.dobot.get_state()
                    payload = json.dumps({
                        "image_top": self.cameras.frame_to_b64(f1),
                        "image_wrist": self.cameras.frame_to_b64(f2),
                        "state": state,
                        "task": self.task,
                        "chunk_size": self.chunk_size,
                    })

                    # 2) Send -> 3) Receive (half-duplex over persistent WS)
                    t_ws = time.time()
                    self.ws.send(payload)
                    response = json.loads(self.ws.recv())
                    ws_ms = (time.time() - t_ws) * 1000

                    actions = response["actions"]
                    infer_ms = response["inference_ms"]

                    # Debug on first cycle
                    if cycle == 0:
                        print(f"\n  [DEBUG] state: {state}")
                        print(f"  [DEBUG] delta[0]: {[f'{v:+.1f}' for v in actions[0]]}")
                        print(f"  [DEBUG] infer: {infer_ms:.0f}ms, ws_round: {ws_ms:.0f}ms\n")

                    # 4) Execute actions
                    for i, delta in enumerate(actions):
                        cur, tgt = self.dobot.execute(delta)
                        total_ms = (time.time() - t_total) * 1000

                        print(
                            f"  [{cycle+1}] "
                            f"delta[{delta[0]:+.1f},{delta[1]:+.1f},{delta[2]:+.1f}] "
                            f"({cur[0]:.0f},{cur[1]:.0f},{cur[2]:.0f})"
                            f"->({tgt[0]:.0f},{tgt[1]:.0f},{tgt[2]:.0f}) "
                            f"G:{'ON' if self.dobot.grip_on else 'OFF'} "
                            f"infer:{infer_ms:.0f}ms ws:{ws_ms:.0f}ms total:{total_ms:.0f}ms"
                        )

                    cycle += 1

                elif key == ord("a"):
                    auto_mode = not auto_mode
                    print(f"\n  {'AUTO' if auto_mode else 'MANUAL'} mode")

                elif key == ord("h"):
                    self.dobot.home()
                    print("  Home")

                elif key == ord("g"):
                    self.dobot.grip_on = not self.dobot.grip_on
                    try:
                        self.dobot.dobot.grip(self.dobot.grip_on)
                        time.sleep(GRIPPER_WAIT_S)
                    except Exception:
                        pass
                    print(f"  Grip: {'ON' if self.dobot.grip_on else 'OFF'}")

                elif key == ord("t"):
                    auto_mode = False
                    print("\n  New task:")
                    new_task = input("  > ").strip()
                    if new_task:
                        self.task = new_task
                        print(f"  Task: {self.task}")

        except KeyboardInterrupt:
            print("\n  Interrupted.")
        except Exception as e:
            print(f"\n  err: {e}")
            import traceback
            traceback.print_exc()

        self.close()

    def _show_preview(self, f1, f2, cycle, auto_mode):
        pose = self.dobot.get_pose()
        color = (0, 0, 255) if auto_mode else (0, 255, 0)
        mode = "AUTO" if auto_mode else "MANUAL"

        cv2.putText(f1, f"TOP | WS | {mode} | {cycle}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(f1, f"X:{pose[0]:.0f} Y:{pose[1]:.0f} Z:{pose[2]:.0f}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(f1, f"Task: {self.task[:40]}",
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

        cv2.putText(f2, f"WRIST | Grip: {'ON' if self.dobot.grip_on else 'OFF'}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        combined = np.hstack([f1, f2])
        cv2.imshow("Pi0 WS Streaming", combined)

    def close(self):
        if self.ws:
            try:
                self.ws.close()
            except Exception:
                pass
        self.dobot.close()
        self.cameras.close()
        print("  Session closed.")
# Entry point
def main():
    parser = argparse.ArgumentParser(description="Pi0 WebSocket Streaming Client")
    parser.add_argument("--server", type=str, default="ws://localhost:8765/ws",
                        help="WebSocket server URL (e.g. ws://192.168.1.100:8765/ws)")
    parser.add_argument("--port", type=str, default=None, help="DOBOT serial port")
    parser.add_argument("--cam1", type=int, default=0, help="Top camera ID")
    parser.add_argument("--cam2", type=int, default=1, help="Wrist camera ID")
    parser.add_argument("--task", type=str, default="pick up the object",
                        help="Language instruction")
    parser.add_argument("--chunk-size", type=int, default=1,
                        help="Action steps per inference (1-2 recommended)")
    parser.add_argument("--cycles", type=int, default=50, help="Max cycles")
    args = parser.parse_args()

    client = Pi0StreamClient(
        server_url=args.server,
        port=args.port,
        cam1=args.cam1,
        cam2=args.cam2,
        task=args.task,
        chunk_size=args.chunk_size,
        max_cycles=args.cycles,
    )

    try:
        client.connect()
        client.run()
    except Exception as e:
        print(f"\n  err: {e}")
        import traceback
        traceback.print_exc()
        client.close()
if __name__ == "__main__":
    main()
