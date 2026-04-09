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
import math
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
# Safety bounds (DH 파라미터 기반 — r=206mm 이하 차단)
BOUNDS = {
    "x": (200, 300),
    "y": (-120, 120),
    "z": (-30, 150),
    "r": (-90, 90),
}
HOME_POS = (240, 0, 80, 0)

# Dobot Magician DH 파라미터 기반 기구학 상수
DOBOT_A2 = 135.0
DOBOT_OFFSET = 206.0
J2_SAFE_MIN = 10.0
REACH_SAFE_MIN = DOBOT_OFFSET + DOBOT_A2 * math.sin(math.radians(J2_SAFE_MIN))
ALARM_POS_THRESHOLD = 5.0


def _predict_j2(x, y):
    """목표 (x,y)에서의 j2 예측 (degrees). DH 파라미터 기반."""
    r = math.sqrt(x ** 2 + y ** 2)
    sin_j2 = (r - DOBOT_OFFSET) / DOBOT_A2
    sin_j2 = float(np.clip(sin_j2, -1.0, 1.0))
    return math.degrees(math.asin(sin_j2))


def _path_crosses_singularity(cx, cy, tx, ty, n_samples=5):
    """직선 경로 상 j2 < J2_SAFE_MIN 구간 관통 여부."""
    for t in np.linspace(0, 1, n_samples):
        px = cx + t * (tx - cx)
        py = cy + t * (ty - cy)
        if _predict_j2(px, py) < J2_SAFE_MIN:
            return True
    return False


def _compute_via_point(cx, cy, tx, ty):
    """j2 위험 영역을 우회하는 경유점 계산."""
    safe_r = REACH_SAFE_MIN + 20

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

    if _predict_j2(vx, vy) < J2_SAFE_MIN:
        return None
    return (vx, vy)

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
            # 학습/추론 일관성을 위한 고정 설정 (오토 비활성화)
            cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)         # 오토포커스 OFF
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)     # 수동 노출 (1=manual, 3=auto)
            cap.set(cv2.CAP_PROP_EXPOSURE, -4)         # 노출값 (환경에 맞게 조정: -1~-13)
            cap.set(cv2.CAP_PROP_AUTO_WB, 0)           # 오토 화이트밸런스 OFF
            cap.set(cv2.CAP_PROP_WB_TEMPERATURE, 5000) # 색온도 고정 (4000~6500)
            cap.set(cv2.CAP_PROP_GAIN, 0)              # 게인 최소 (노이즈 방지)
            cap.set(cv2.CAP_PROP_FPS, 30)              # FPS 고정

        print(f"   Cameras: {cam1_id}, {cam2_id} (수동 노출/WB/포커스 고정)")

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
    MAX_CONSECUTIVE_ALARMS = 3

    def __init__(self, port: Optional[str] = None):
        port = port or self._find_port()
        self.dobot = pydobot.Dobot(port=port, verbose=False)
        self.dobot.speed(150, 150)
        self.grip_on = False
        self._alarm_count = 0
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
        """DH 파라미터 기반 singularity 회피 + via-point."""
        cur = self.get_pose()

        tx = float(np.clip(cur[0] + delta[0], *BOUNDS["x"]))
        ty = float(np.clip(cur[1] + delta[1], *BOUNDS["y"]))
        tz = float(np.clip(cur[2] + delta[2], *BOUNDS["z"]))
        tr = float(np.clip(cur[3] + delta[3], *BOUNDS["r"]))

        # 1단계: DH 기반 j2 예측으로 singularity 감지
        target_j2 = _predict_j2(tx, ty)
        target_dangerous = target_j2 < J2_SAFE_MIN
        path_dangerous = _path_crosses_singularity(cur[0], cur[1], tx, ty)

        if target_dangerous or path_dangerous:
            via = _compute_via_point(cur[0], cur[1], tx, ty)
            if via:
                print(f"    >> Singularity 회피: 경유점 ({via[0]:.0f}, {via[1]:.0f})mm "
                      f"[j2 예측: 목표={target_j2:+.1f}°]")
                self.dobot.move_to(via[0], via[1], tz, tr, wait=True)

            if target_dangerous:
                dist = math.sqrt(tx ** 2 + ty ** 2)
                if dist > 1e-3:
                    scale = REACH_SAFE_MIN / dist
                    tx, ty = float(tx * scale), float(ty * scale)
                    tx = float(np.clip(tx, *BOUNDS["x"]))
                    ty = float(np.clip(ty, *BOUNDS["y"]))
                print(f"    >> 목표 스케일: ({tx:.0f}, {ty:.0f})mm [j2→{_predict_j2(tx, ty):+.1f}°]")

        # 2단계: 최종 목표로 이동
        self.dobot.move_to(tx, ty, tz, tr, wait=True)

        # 3단계: ALARM 감지 + 실제 j2 사후 검증
        try:
            actual = self.dobot.pose()
            error = math.sqrt((actual[0] - tx) ** 2 + (actual[1] - ty) ** 2 + (actual[2] - tz) ** 2)
            if error > ALARM_POS_THRESHOLD:
                self._alarm_count += 1
                if self._alarm_count >= self.MAX_CONSECUTIVE_ALARMS:
                    print(f"    >> ALARM {self._alarm_count}회 연속 — 복구 중단, 현재 위치에서 계속")
                    self._alarm_count = 0
                    return cur, [actual[0], actual[1], actual[2], actual[3]], True
                print(f"    >> ALARM ({self._alarm_count}/{self.MAX_CONSECUTIVE_ALARMS}, 오차 {error:.1f}mm), 복구 중...")
                self._clear_alarm()
                pose = self.get_pose()
                return cur, pose, True
            else:
                self._alarm_count = 0
            real_j2 = actual[5]
            if real_j2 < J2_SAFE_MIN:
                print(f"    >> j2={real_j2:.1f}° 위험 (< {J2_SAFE_MIN}°)")
        except Exception:
            pass

        # 4단계: Gripper after move
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

        return cur, [tx, ty, tz, tr], False

    def _clear_alarm(self):
        """ALARM/정지 상태에서 Dobot을 깨우는 복구 시퀀스 (타임아웃 10초)."""
        import gc
        import threading as _th
        from pydobot.dobot import Message, CommunicationProtocolIDs as IDs, ControlValues as CV

        port = None
        try:
            ser = getattr(self.dobot, 'ser', None)
            if ser:
                port = ser.port
                if ser.is_open:
                    try:
                        ser.dtr = False
                        ser.rts = False
                        time.sleep(0.5)
                        ser.dtr = True
                        ser.rts = True
                        time.sleep(1.0)
                        print(f"    >> DTR 토글 완료")
                    except Exception:
                        pass
                    ser.close()
        except Exception:
            pass

        try:
            del self.dobot
        except Exception:
            pass
        self.dobot = None
        gc.collect()

        print(f"    >> Dobot 복구 중... (3초 대기)")
        time.sleep(3)

        result = {"success": False}

        def _reconnect():
            try:
                self.dobot = pydobot.Dobot(port=port or self._find_port(), verbose=False)
                time.sleep(0.5)
                self.dobot._set_queued_cmd_stop_exec()
                time.sleep(0.1)
                msg = Message()
                msg.id = IDs.CLEAR_ALL_ALARMS_STATE
                msg.ctrl = CV.ONE
                self.dobot._send_command(msg)
                time.sleep(0.1)
                self.dobot._set_queued_cmd_clear()
                time.sleep(0.1)
                self.dobot._set_queued_cmd_start_exec()
                time.sleep(0.5)
                self.dobot.speed(100, 100)
                self.grip_on = False
                try:
                    self.dobot.move_to(*HOME_POS, wait=True)
                except Exception:
                    pass
                self.dobot.speed(150, 150)
                result["success"] = True
            except Exception as e:
                result["error"] = str(e)

        t = _th.Thread(target=_reconnect, daemon=True)
        t.start()
        t.join(timeout=10)

        if t.is_alive():
            print(f"    >> Dobot 복구 타임아웃 (10초) — 수동 리셋 필요 (전원 OFF/ON)")
        elif result["success"]:
            try:
                pose = self.get_pose()
                print(f"    >> Dobot 복구 완료: ({pose[0]:.0f},{pose[1]:.0f},{pose[2]:.0f})")
            except Exception:
                print(f"    >> Dobot 복구 완료 (위치 읽기 실패)")
        else:
            print(f"    >> Dobot 복구 실패: {result.get('error', 'unknown')}")

    def home(self):
        self.grip_on = False
        try:
            self.dobot.grip(False)
        except Exception:
            pass
        try:
            self.dobot.suck(False)
        except Exception:
            pass
        self.dobot.move_to(*HOME_POS, wait=True)

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
                    # cam1=wrist, cam2=top (수집 스크립트와 동일한 매핑)
                    payload = json.dumps({
                        "image_top": self.cameras.frame_to_b64(f2),
                        "image_wrist": self.cameras.frame_to_b64(f1),
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
                        cur, tgt, alarmed = self.dobot.execute(delta)
                        total_ms = (time.time() - t_total) * 1000

                        print(
                            f"  [{cycle+1}] "
                            f"delta[{delta[0]:+.1f},{delta[1]:+.1f},{delta[2]:+.1f}] "
                            f"({cur[0]:.0f},{cur[1]:.0f},{cur[2]:.0f})"
                            f"->({tgt[0]:.0f},{tgt[1]:.0f},{tgt[2]:.0f}) "
                            f"G:{'ON' if self.dobot.grip_on else 'OFF'} "
                            f"infer:{infer_ms:.0f}ms ws:{ws_ms:.0f}ms total:{total_ms:.0f}ms"
                        )
                        if alarmed:
                            print(f"  >> ALARM 복구됨 — 나머지 action 스킵, 새로 관측합니다")
                            break

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
            self.safe_shutdown()
            return
        except Exception as e:
            print(f"\n  err: {e}")
            import traceback
            traceback.print_exc()

        self.safe_shutdown()

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

    def safe_shutdown(self):
        """Ctrl+C 등 강제종료 시: 홈 복귀 → 그리퍼 해제 → 연결 해제."""
        print("\n  Safe shutdown...")
        try:
            print("  Homing...")
            self.dobot.home()
            print("  Home OK")
        except Exception as e:
            print(f"  Home failed: {e}")
        self.close()

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
    except KeyboardInterrupt:
        client.safe_shutdown()
    except Exception as e:
        print(f"\n  err: {e}")
        import traceback
        traceback.print_exc()
        client.safe_shutdown()
if __name__ == "__main__":
    main()
