#!/usr/bin/env python3
"""
DOBOT Magician - LeRobot v3.0 데이터 수집 (단일 팔)

리더/팔로워 없이 프레임 단위 수동 수집.
[S] 관측 캡처 -> 팔 이동 -> [E] 델타 기록

    python 01_collect_data.py --port COM4 --cam1 0 --cam2 1 \\
        --task "pick up the red cup" --save_dir ./dataset_v3
"""

import sys
import os
import time
import json
import shutil
import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

try:
    import cv2
    import numpy as np
    import pydobot
    from serial.tools import list_ports
    # import keyboard  # macOS에서 루트 권한 필요 → cv2.waitKey()로 대체
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install: pip install opencv-python pydobot keyboard pandas pyarrow pyserial")
    sys.exit(1)
# Constants

IMG_W, IMG_H = 640, 480
JPEG_QUALITY = 95
DEFAULT_FPS = 5
WRIST_STEP_DEG = 5.0
WRIST_RANGE = (-90.0, 90.0)
# Utility: Auto-detect DOBOT serial port

def find_dobot_port() -> Optional[str]:
    """Auto-detect DOBOT Magician serial port (CH340 or CP210x chipset)."""
    for p in list_ports.comports():
        if any(chip in p.description for chip in ("CH340", "CP210")):
            return p.device
    # macOS: usbserial 포트 찾기
    for p in list_ports.comports():
        if "usbserial" in p.device:
            return p.device
    return None
# LeRobot v3.0 Data Collector

class LeRobotV3Collector:
    """v3.0 데이터 수집기. [S]로 관측 캡처, 팔 이동, [E]로 델타 기록."""

    def __init__(
        self,
        port: str,
        cam1_id: int = 0,
        cam2_id: int = 1,
        save_dir: str = "dataset_v3",
        task: str = "pick_object",
        fps: int = DEFAULT_FPS,
    ):
        self.port = port
        self.cam1_id = cam1_id
        self.cam2_id = cam2_id
        self.save_dir = Path(save_dir).resolve()
        self.task = task
        self.fps = fps

        # Hardware handles
        self.dobot: Optional[pydobot.Dobot] = None
        self.cap1: Optional[cv2.VideoCapture] = None
        self.cap2: Optional[cv2.VideoCapture] = None

        # Camera feature names (LeRobot convention)
        self.cam1_name = "observation.images.wrist"
        self.cam2_name = "observation.images.top"

        # Episode tracking
        self.episode_index = 0
        self.frame_index = 0
        self.global_frame_index = 0

        # Current episode buffers
        self.episode_images_cam1: List[np.ndarray] = []
        self.episode_images_cam2: List[np.ndarray] = []
        self.episode_data: List[Dict[str, Any]] = []

        # Step state (between [S] and [E])
        self.current_image1: Optional[np.ndarray] = None
        self.current_image2: Optional[np.ndarray] = None
        self.current_state: Optional[List[float]] = None
        self.current_grip: bool = False
        self.step_started: bool = False

        # Gripper & wrist
        self.grip_on: bool = False
        self.wrist_angle: float = 0.0
        self.start_wrist_angle: float = 0.0

        # Global stats accumulators
        self.all_states: List[List[float]] = []
        self.all_actions: List[List[float]] = []

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def connect(self) -> bool:
        """Initialize DOBOT and cameras. """
        print(f"\n{'='*60}")
        print(f"  DOBOT LeRobot v3.0 -- Data Collector")
        print(f"{'='*60}\n")

        # DOBOT
        try:
            print(f"  DOBOT 연결 중: {self.port} (3초 대기)")
            time.sleep(3)
            self.dobot = pydobot.Dobot(port=self.port, verbose=False)
            time.sleep(1)
            self.dobot.speed(200, 200)
            print(f"  DOBOT connected: {self.port}")
        except Exception as e:
            print(f"  DOBOT connection failed: {e}")
            print(f"  >> 카메라 전용 모드로 계속합니다 (Dobot 조작 불가)")
            self.dobot = None

        # Cameras
        self.cap1 = cv2.VideoCapture(self.cam1_id, cv2.CAP_AVFOUNDATION)
        self.cap2 = cv2.VideoCapture(self.cam2_id, cv2.CAP_AVFOUNDATION)
        if not self.cap1.isOpened() or not self.cap2.isOpened():
            print("  Camera initialization failed")
            return False
        for cap in (self.cap1, self.cap2):
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, IMG_W)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, IMG_H)
        print(f"  Cameras: wrist={self.cam1_id}, top={self.cam2_id}")

        # Directory structure & metadata
        self._create_v3_structure()
        self._find_last_episode()
        self.wrist_angle = self._get_pose()[3]

        print(f"\n  Path: {self.save_dir}")
        print(f"  Task: {self.task}")
        print(f"  Next episode: {self.episode_index}")
        print(f"  Format: LeRobot v3.0 (images, relative paths)")

        # 미저장 에피소드 복구 확인
        self._check_recovery()

        return True

    def disconnect(self):
        """Release all hardware resources."""
        self._force_close_dobot()
        for cap in (self.cap1, self.cap2):
            if cap:
                cap.release()
        cv2.destroyAllWindows()

    # ------------------------------------------------------------------
    # v3.0 Directory Structure & Metadata
    # ------------------------------------------------------------------

    def _create_v3_structure(self):
        """Create LeRobot v3.0 directory layout and initial metadata."""
        dirs = [
            "meta",
            "meta/episodes/chunk-000",
            "data/chunk-000",
            f"images/{self.cam1_name}/chunk-000",
            f"images/{self.cam2_name}/chunk-000",
        ]
        for d in dirs:
            (self.save_dir / d).mkdir(parents=True, exist_ok=True)

        # --- meta/info.json ---
        info_path = self.save_dir / "meta" / "info.json"
        if not info_path.exists():
            info = {
                "codebase_version": "v3.0",
                "robot_type": "dobot_magician",
                "fps": self.fps,
                "video": False,
                "total_episodes": 0,
                "total_frames": 0,
                "total_tasks": 1,
                "features": {
                    self.cam1_name: {
                        "dtype": "image",
                        "shape": [3, IMG_H, IMG_W],
                        "names": ["channel", "height", "width"],
                    },
                    self.cam2_name: {
                        "dtype": "image",
                        "shape": [3, IMG_H, IMG_W],
                        "names": ["channel", "height", "width"],
                    },
                    "observation.state": {
                        "dtype": "float32",
                        "shape": [5],
                        "names": ["x", "y", "z", "r", "grip"],
                    },
                    "action": {
                        "dtype": "float32",
                        "shape": [5],
                        "names": ["delta_x", "delta_y", "delta_z", "delta_r", "grip"],
                    },
                    "index": {"dtype": "int64", "shape": [1], "names": ["index"]},
                    "timestamp": {"dtype": "float32", "shape": [1], "names": ["timestamp"]},
                    "frame_index": {"dtype": "int64", "shape": [1], "names": ["frame_index"]},
                    "episode_index": {"dtype": "int64", "shape": [1], "names": ["episode_index"]},
                    "task_index": {"dtype": "int64", "shape": [1], "names": ["task_index"]},
                },
            }
            with open(info_path, "w") as f:
                json.dump(info, f, indent=2)

        # --- meta/tasks.jsonl (v3.0 format) ---
        tasks_path = self.save_dir / "meta" / "tasks.jsonl"
        if not tasks_path.exists():
            with open(tasks_path, "w") as f:
                f.write(json.dumps({"task_index": 0, "task": self.task}) + "\n")

    def _find_last_episode(self):
        """Resume from the last episode index if data already exists."""
        data_dir = self.save_dir / "data" / "chunk-000"
        if data_dir.exists():
            eps = list(data_dir.glob("episode_*.parquet"))
            if eps:
                self.episode_index = max(int(e.stem.split("_")[1]) for e in eps) + 1

        info_path = self.save_dir / "meta" / "info.json"
        if info_path.exists():
            with open(info_path) as f:
                self.global_frame_index = json.load(f).get("total_frames", 0)

        # 기존 데이터 복원은 [1] 키로 수동 선택

    def _load_existing_stats(self):
        """기존 parquet에서 state/action 데이터를 읽어 stats 누적 버퍼에 복원."""
        data_dir = self.save_dir / "data" / "chunk-000"
        if not data_dir.exists():
            return
        eps = sorted(data_dir.glob("episode_*.parquet"))
        if not eps:
            return
        loaded = 0
        for ep_path in eps:
            try:
                df = pd.read_parquet(ep_path)
                for _, row in df.iterrows():
                    self.all_states.append(row["observation.state"])
                    self.all_actions.append(row["action"])
                loaded += 1
            except Exception:
                continue
        if loaded:
            print(f"  기존 데이터 복원: {loaded}개 에피소드, {len(self.all_states)}개 프레임")

    # ------------------------------------------------------------------
    # Robot Reconnection
    # ------------------------------------------------------------------

    def _disable_air(self):
        """공압(suck/grip) enableCtrl=0 전송 — 완전히 끊기."""
        if not self.dobot:
            return
        from pydobot.dobot import Message, CommunicationProtocolIDs as IDs, ControlValues as CV
        for cmd_id in (IDs.SET_GET_END_EFFECTOR_SUCTION_CUP, IDs.SET_GET_END_EFFECTOR_GRIPPER):
            try:
                msg = Message()
                msg.id = cmd_id
                msg.ctrl = CV.THREE
                msg.params = bytearray([0x00, 0x00])  # enableCtrl=0, on=0
                self.dobot._send_command(msg)
            except Exception:
                pass

    def go_home(self):
        """[Q] -- Home 위치(0,0,0,0)로 이동. 호밍 완료 후 사용."""
        if not self.dobot:
            print("  Dobot 미연결 -- Go Home 불가")
            return
        try:
            print("  Home 위치로 이동 중...")
            self.dobot.move_to(200, 0, 50, 0, wait=True)
            self.wrist_angle = 0.0
            print("  Home 위치 도착: (200, 0, 50, 0)")
        except Exception as e:
            print(f"  Go Home 실패: {e}")

    def home(self):
        """[A] -- 호밍 (리밋스위치 기반 원점 복귀)."""
        if not self.dobot:
            print("  Dobot 미연결 -- Home 불가")
            return
        from pydobot.dobot import Message, CommunicationProtocolIDs as IDs, ControlValues as CV
        import struct
        try:
            print("  호밍 중... (리밋스위치 원점 복귀)")
            # SET_HOME_CMD (id=31) — queued, 리밋스위치 찍고 원점 설정
            msg = Message()
            msg.id = IDs.SET_HOME_CMD
            msg.ctrl = CV.THREE
            msg.params = bytearray(4)  # reserved 4 bytes (0)
            self.dobot._send_command(msg, wait=True)
            self.wrist_angle = 0.0
            pose = self._get_pose()
            print(f"  호밍 완료: x={pose[0]:.1f} y={pose[1]:.1f} z={pose[2]:.1f} r={pose[3]:.1f}")
        except Exception as e:
            print(f"  호밍 실패: {e}")

    def clear_alarm(self):
        """[X] -- DOBOT 알람 해제 (빨간불 → 초록불). test_dobot.py와 동일한 방식."""
        port = self.port

        # 1. 기존 연결 완전 제거 (명령 전송 없이 시리얼만 닫기)
        if self.dobot:
            try:
                ser = getattr(self.dobot, 'ser', None)
                if ser and ser.is_open:
                    ser.close()
            except Exception:
                pass
            del self.dobot
            self.dobot = None
            import gc
            gc.collect()

        # 2. test_dobot.py와 동일하게 3초 대기
        print(f"  알람 해제 중... (3초 대기)")
        time.sleep(3)

        # 3. 새 연결 (생성자가 _set_queued_cmd_start_exec + _set_queued_cmd_clear 호출)
        try:
            self.dobot = pydobot.Dobot(port=port, verbose=False)
            time.sleep(1)
            self.dobot.speed(100, 100)
            self.grip_on = False
            pose = self._get_pose()
            print(f"  알람 해제 완료 (초록불 복구)")
            print(f"  현재 위치: x={pose[0]:.1f} y={pose[1]:.1f} z={pose[2]:.1f} r={pose[3]:.1f}")
        except Exception as e:
            self.dobot = None
            print(f"  알람 해제 실패: {e}")

    def _force_close_dobot(self):
        """DOBOT 연결을 안전하게 해제. 공압 OFF → 시리얼 강제 닫기."""
        if not self.dobot:
            return

        ser = getattr(self.dobot, 'ser', None)

        # 1단계: 공압 완전 차단 (enableCtrl=0)
        try:
            if ser:
                ser.write_timeout = 1
            self._disable_air()
        except Exception:
            pass

        # 2단계: 정상 종료 시도
        try:
            self.dobot.close()
        except Exception:
            # 3단계: close() hang 시 시리얼 강제 닫기
            try:
                if ser and ser.is_open:
                    ser.close()
            except Exception:
                pass

        self.dobot = None
        self.grip_on = False

    def reconnect_dobot(self):
        """[F] -- 런타임에서 DOBOT 재연결. 끊김/에러 시 사용."""
        print("  DOBOT 연결 해제 중...")
        self._force_close_dobot()
        time.sleep(1)

        # 포트 재탐색 (USB 재연결 시 포트 바뀔 수 있음)
        port = find_dobot_port()
        if not port:
            print("  DOBOT 포트를 찾을 수 없습니다. USB 확인 후 다시 [F]")
            return

        try:
            print(f"  DOBOT 재연결 중: {port} (3초 대기)")
            time.sleep(3)
            self.dobot = pydobot.Dobot(port=port, verbose=False)
            time.sleep(1)
            self.dobot.speed(200, 200)
            # 재연결 직후 공압 완전 차단
            self._disable_air()
            self.port = port
            self.grip_on = False
            self.wrist_angle = self._get_pose()[3]
            pose = self._get_pose()
            print(f"  DOBOT 재연결 성공: {port}")
            print(f"  현재 위치: x={pose[0]:.1f} y={pose[1]:.1f} z={pose[2]:.1f} r={pose[3]:.1f}")
        except Exception as e:
            self.dobot = None
            print(f"  DOBOT 재연결 실패: {e}. USB 뽑았다 꽂고 다시 [F]")

    # ------------------------------------------------------------------
    # Robot State
    # ------------------------------------------------------------------

    def _get_pose(self) -> List[float]:
        """Read current DOBOT pose [x, y, z, r]. Safe with fallback."""
        try:
            r = self.dobot.pose()
            return [round(r[i], 2) for i in range(4)] if r and len(r) >= 4 else [0, 0, 0, 0]
        except Exception:
            return [0, 0, 0, 0]

    def _capture_images(self):
        """Capture frames from both cameras."""
        r1, f1 = self.cap1.read()
        r2, f2 = self.cap2.read()
        return (f1 if r1 else None, f2 if r2 else None)

    # ------------------------------------------------------------------
    # Autosave / Recovery
    # ------------------------------------------------------------------

    def _autosave_dir(self) -> Path:
        return self.save_dir / ".autosave"

    def _autosave(self):
        """매 스텝 완료 시 현재 에피소드를 임시 저장."""
        tmp = self._autosave_dir()
        tmp.mkdir(parents=True, exist_ok=True)
        ep_str = f"{self.episode_index:06d}"

        # 이미지 저장
        for i, (img1, img2) in enumerate(zip(self.episode_images_cam1, self.episode_images_cam2)):
            for cam_name, img in [(self.cam1_name, img1), (self.cam2_name, img2)]:
                img_dir = tmp / "images" / cam_name / f"episode_{ep_str}"
                img_dir.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(img_dir / f"frame_{i:06d}.jpg"), img,
                            [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])

        # 데이터 저장
        meta = {
            "episode_index": self.episode_index,
            "frame_index": self.frame_index,
            "global_frame_index": self.global_frame_index,
            "grip_on": self.grip_on,
            "wrist_angle": self.wrist_angle,
            "episode_data": self.episode_data,
        }
        with open(tmp / "recovery.json", "w") as f:
            json.dump(meta, f)

    def _clear_autosave(self):
        """에피소드 저장/폐기 후 임시 파일 삭제."""
        tmp = self._autosave_dir()
        if tmp.exists():
            shutil.rmtree(tmp)

    def _check_recovery(self):
        """시작 시 복구 가능한 데이터가 있는지 확인. 있으면 플래그만 세팅."""
        tmp = self._autosave_dir()
        recovery_file = tmp / "recovery.json"
        self._has_recovery = False
        if not recovery_file.exists():
            return
        try:
            with open(recovery_file) as f:
                meta = json.load(f)
            n_steps = len(meta["episode_data"])
            ep_idx = meta["episode_index"]
            self._has_recovery = True
            self._recovery_meta = meta
            print(f"\n  !! 미저장 에피소드 발견: episode {ep_idx}, {n_steps} steps")
            print(f"     위치: {tmp}")
            print(f"     [1] 복구 / [D] 폐기")
        except Exception:
            self._clear_autosave()

    def resume_mode(self):
        """[1] -- 기존 데이터 이어서 수집 모드. stats를 기존 데이터 기반으로 누적."""
        # autosave 복구가 있으면 먼저 처리
        if self._has_recovery:
            self._recover_autosave(self._recovery_meta, self._autosave_dir())
            self._has_recovery = False
            self._recovery_meta = None
            return

        # 기존 parquet에서 stats 로드
        if self.all_states:
            print("  이미 이어서 수집 모드입니다.")
            return
        self._load_existing_stats()
        if self.all_states:
            print(f"  이어서 수집 모드 ON: episode {self.episode_index}부터 시작")
        else:
            print("  기존 데이터가 없습니다. 새로 수집합니다.")

    def _recover_autosave(self, meta: dict, tmp: Path):
        """임시 저장 데이터로부터 에피소드 복구."""
        print(f"\n  [복구 시작]")
        self.episode_index = meta["episode_index"]
        self.frame_index = meta["frame_index"]
        self.global_frame_index = meta["global_frame_index"]
        self.grip_on = meta["grip_on"]
        self.wrist_angle = meta["wrist_angle"]
        self.episode_data = meta["episode_data"]
        print(f"  - 에피소드 번호: {self.episode_index}")
        print(f"  - 스텝 수: {len(self.episode_data)}")
        print(f"  - 그리퍼: {'ON' if self.grip_on else 'OFF'}")
        print(f"  - 손목 각도: {self.wrist_angle:.1f}")

        ep_str = f"{self.episode_index:06d}"
        self.episode_images_cam1.clear()
        self.episode_images_cam2.clear()

        img_ok = 0
        img_fail = 0
        for i in range(len(self.episode_data)):
            for cam_name, buf in [(self.cam1_name, self.episode_images_cam1),
                                  (self.cam2_name, self.episode_images_cam2)]:
                img_path = tmp / "images" / cam_name / f"episode_{ep_str}" / f"frame_{i:06d}.jpg"
                if img_path.exists():
                    buf.append(cv2.imread(str(img_path)))
                    img_ok += 1
                else:
                    buf.append(np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8))
                    img_fail += 1
        print(f"  - 이미지 복구: {img_ok}장 성공, {img_fail}장 누락")

        # stats 버퍼에도 복구
        for row in self.episode_data:
            self.all_states.append(row["observation.state"])
            self.all_actions.append(row["action"])

        self.step_started = False
        print(f"\n  [복구 완료] 에피소드 {self.episode_index} — {len(self.episode_data)} steps")
        print(f"  >> [V] 저장 / [S] 이어서 수집 / [D] 폐기")

    # ------------------------------------------------------------------
    # Step-by-Step Recording
    # ------------------------------------------------------------------

    def start_step(self):
        """[S] -- Capture observation at the START of a demonstration step."""
        if self.step_started:
            print("  Step already in progress. Press [E] to end it first.")
            return

        img1, img2 = self._capture_images()
        if img1 is None:
            print("  Camera capture failed")
            return

        self.current_image1 = img1.copy()
        self.current_image2 = img2.copy()
        self.current_state = self._get_pose()
        self.current_grip = self.grip_on
        self.start_wrist_angle = self.wrist_angle
        self.step_started = True

        print(f"\n  >> Step {self.frame_index + 1} started | Pos: {self.current_state[:3]}")

    def end_step(self):
        """[E] -- Record delta action at the END of a demonstration step."""
        if not self.step_started:
            print("  Press [S] first to start a step.")
            return

        target = self._get_pose()

        # Delta = target - start (for x, y, z)
        delta = [round(target[i] - self.current_state[i], 2) for i in range(3)]
        delta.append(round(self.wrist_angle - self.start_wrist_angle, 2))  # Δr
        delta.append(1.0 if self.grip_on else 0.0)                         # grip

        # State at observation time
        state = self.current_state[:3] + [self.start_wrist_angle, 1.0 if self.current_grip else 0.0]

        # Buffer images
        self.episode_images_cam1.append(self.current_image1)
        self.episode_images_cam2.append(self.current_image2)

        # Buffer data row
        self.episode_data.append({
            "index": self.global_frame_index + self.frame_index,
            "episode_index": self.episode_index,
            "frame_index": self.frame_index,
            "timestamp": self.frame_index / self.fps,
            "task_index": 0,
            "observation.state": state,
            "action": delta,
        })

        self.all_states.append(state)
        self.all_actions.append(delta)

        print(f"  << Step {self.frame_index + 1} | "
              f"Delta: ({delta[0]:+.1f}, {delta[1]:+.1f}, {delta[2]:+.1f})")

        self.frame_index += 1
        self.step_started = False
        self.current_image1 = self.current_image2 = self.current_state = None

        # 매 스텝마다 임시 백업
        self._autosave()

    # ------------------------------------------------------------------
    # Gripper & Wrist Control
    # ------------------------------------------------------------------

    def toggle_grip(self):
        """[G] -- Toggle gripper on/off. ON=잡기, OFF=벌리기."""
        if not self.dobot:
            print("  Dobot 미연결 -- 그리퍼 제어 불가")
            return
        self.grip_on = not self.grip_on
        try:
            self.dobot.grip(self.grip_on)
            state = "CLOSE" if self.grip_on else "OPEN"
            print(f"  Gripper: {state}")
        except Exception as e1:
            try:
                self.dobot.suck(self.grip_on)
                state = "CLOSE" if self.grip_on else "OPEN"
                print(f"  Gripper (suck fallback): {state}")
            except Exception as e2:
                self.grip_on = not self.grip_on
                print(f"  Grip FAILED - grip: {e1}, suck: {e2}")

    def rotate_wrist(self, direction: int):
        """[Z]/[C] -- Rotate wrist by ±WRIST_STEP_DEG degrees."""
        self.wrist_angle = max(
            WRIST_RANGE[0],
            min(WRIST_RANGE[1], self.wrist_angle + direction * WRIST_STEP_DEG),
        )
        try:
            p = self._get_pose()
            self.dobot.move_to(p[0], p[1], p[2], self.wrist_angle, wait=False)
        except Exception:
            pass
        print(f"\r  Wrist: {self.wrist_angle:.1f} deg   ", end="")

    # ------------------------------------------------------------------
    # Episode Management
    # ------------------------------------------------------------------

    def save_episode(self):
        """[V] -- Save the current episode to disk in v3.0 format."""
        if not self.episode_data:
            print("  No data to save.")
            return

        ep_str = f"{self.episode_index:06d}"

        # --- Save images with RELATIVE paths ---
        for i, (img1, img2) in enumerate(zip(self.episode_images_cam1, self.episode_images_cam2)):
            for cam_name, img in [(self.cam1_name, img1), (self.cam2_name, img2)]:
                ep_dir = self.save_dir / "images" / cam_name / "chunk-000" / f"episode_{ep_str}"
                ep_dir.mkdir(parents=True, exist_ok=True)
                frame_path = ep_dir / f"frame_{i:06d}.jpg"
                cv2.imwrite(str(frame_path), img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                # Store RELATIVE path (v3.0 -- portable across machines)
                rel_path = f"images/{cam_name}/chunk-000/episode_{ep_str}/frame_{i:06d}.jpg"
                self.episode_data[i][cam_name] = {"path": rel_path}

        # --- Save episode parquet ---
        df = pd.DataFrame(self.episode_data)
        pq.write_table(
            pa.Table.from_pandas(df),
            self.save_dir / "data" / "chunk-000" / f"episode_{ep_str}.parquet",
        )

        # --- Update episodes metadata ---
        ep_meta = {
            "episode_index": self.episode_index,
            "task_index": 0,
            "length": len(self.episode_data),
            "dataset_from_index": self.global_frame_index,
            "dataset_to_index": self.global_frame_index + len(self.episode_data),
        }
        ep_meta_path = self.save_dir / "meta" / "episodes" / "chunk-000" / "episodes.parquet"
        if ep_meta_path.exists():
            existing = pd.read_parquet(ep_meta_path)
            new_df = pd.concat([existing, pd.DataFrame([ep_meta])], ignore_index=True)
        else:
            new_df = pd.DataFrame([ep_meta])
        pq.write_table(pa.Table.from_pandas(
            new_df[["episode_index", "task_index", "length", "dataset_from_index", "dataset_to_index"]]
        ), ep_meta_path)

        # --- Update info.json ---
        info_path = self.save_dir / "meta" / "info.json"
        with open(info_path) as f:
            info = json.load(f)
        info["total_episodes"] = self.episode_index + 1
        info["total_frames"] = self.global_frame_index + len(self.episode_data)
        with open(info_path, "w") as f:
            json.dump(info, f, indent=2)

        # --- Update stats.json ---
        self._update_stats()

        print(f"\n  Episode {self.episode_index} -- {len(self.episode_data)} frames")

        # Reset for next episode
        self.global_frame_index += len(self.episode_data)
        self.episode_index += 1
        self.frame_index = 0
        self.episode_images_cam1.clear()
        self.episode_images_cam2.clear()
        self.episode_data.clear()
        self._clear_autosave()

    def _update_stats(self):
        """Recompute and save meta/stats.json from accumulated data."""
        if not self.all_states:
            return

        s = np.array(self.all_states)
        a = np.array(self.all_actions)
        stats = {
            "observation.state": {
                "min": s.min(0).tolist(),
                "max": s.max(0).tolist(),
                "mean": s.mean(0).tolist(),
                "std": s.std(0).tolist(),
            },
            "action": {
                "min": a.min(0).tolist(),
                "max": a.max(0).tolist(),
                "mean": a.mean(0).tolist(),
                "std": a.std(0).tolist(),
            },
            # ImageNet normalization defaults
            self.cam1_name: {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
            self.cam2_name: {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
        }
        with open(self.save_dir / "meta" / "stats.json", "w") as f:
            json.dump(stats, f, indent=2)

    def replay_episode(self):
        """[R] -- Replay the current (unsaved) episode on the robot."""
        if not self.episode_data:
            print("  No data to replay.")
            return

        if not self.dobot:
            print("  Dobot 미연결 -- 리플레이 불가")
            return
        print(f"\n  Replaying {len(self.episode_data)} steps... [Q] to stop")
        s0 = self.episode_data[0]["observation.state"]
        self.dobot.move_to(s0[0], s0[1], s0[2], s0[3], wait=True)
        time.sleep(0.5)

        for i, step in enumerate(self.episode_data):
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            cur = self._get_pose()
            d = step["action"]
            try:
                self.dobot.grip(d[4] > 0.5)
            except Exception:
                pass
            self.dobot.move_to(cur[0] + d[0], cur[1] + d[1], cur[2] + d[2], cur[3] + d[3], wait=True)
            print(f"\r     Step {i + 1}: delta=({d[0]:+.0f}, {d[1]:+.0f}, {d[2]:+.0f})", end="")
            time.sleep(0.3)
        print("\n  Replay complete.")

    def undo_step(self):
        """[W] -- 마지막 스텝 1개 취소."""
        if not self.episode_data:
            print("  취소할 스텝이 없습니다.")
            return
        self.episode_data.pop()
        self.episode_images_cam1.pop()
        self.episode_images_cam2.pop()
        self.all_states.pop()
        self.all_actions.pop()
        self.frame_index -= 1
        self.step_started = False
        print(f"  마지막 스텝 취소 | 남은 스텝: {len(self.episode_data)}")

    def discard_episode(self):
        """[D] -- Discard the current episode data."""
        n = len(self.episode_data)
        self.frame_index = 0
        self.episode_images_cam1.clear()
        self.episode_images_cam2.clear()
        self.episode_data.clear()
        self.step_started = False
        self._clear_autosave()
        self._has_recovery = False
        self._recovery_meta = None
        print(f"  에피소드 전체 폐기: {n} steps discarded.")

    # ------------------------------------------------------------------
    # Preview & Main Loop
    # ------------------------------------------------------------------

    def _show_preview(self):
        """Render live camera preview with status overlay."""
        img1, img2 = self._capture_images()
        if img1 is None:
            return

        # 포즈는 5프레임마다 갱신 (시리얼 통신 병목 방지)
        if not hasattr(self, '_cached_pose') or self.frame_counter % 5 == 0:
            self._cached_pose = self._get_pose()
        self.frame_counter = getattr(self, 'frame_counter', 0) + 1
        pose = self._cached_pose
        status, color = ("REC", (0, 0, 255)) if self.step_started else ("READY", (0, 255, 0))

        cv2.putText(img1, f"CAM1 | Ep {self.episode_index} | {status}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(img1, f"X:{pose[0]:.1f} Y:{pose[1]:.1f} Z:{pose[2]:.1f}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        if self.step_started and self.current_state:
            d = [pose[i] - self.current_state[i] for i in range(3)]
            cv2.putText(img1, f"Delta: ({d[0]:+.1f}, {d[1]:+.1f}, {d[2]:+.1f})",
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        cv2.putText(img2, f"CAM2 | Frame {self.frame_index} | v3.0",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(img2, f"Grip: {'ON' if self.grip_on else 'OFF'} | Wrist: {self.wrist_angle:.1f}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 2)

        combined = np.hstack([img1, img2])
        cv2.putText(combined,
                    "S:start E:end G:grip Z/C:wrist V:save R:replay W:undo D:discard Q:home A:homing F:reconn X:alarm",
                    (10, combined.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        cv2.imshow("LeRobot v3.0 Collector", combined)

    def run(self):
        """Main interactive collection loop."""
        print("""
 ============================================================
  DOBOT LeRobot v3.0 -- Data Collection
 ============================================================
  [S] 스텝 시작 (관측 캡처)    [E] 스텝 종료 (델타 기록)
  [G] 그리퍼 열기/닫기         [Z]/[C] 손목 회전 (좌/우)
  [V] 에피소드 저장            [R] 에피소드 리플레이
  [W] 마지막 스텝 취소         [D] 에피소드 전체 폐기
  [Q] Home 위치 이동           [A] 호밍 (리밋스위치 원점)
  [F] DOBOT 재연결             [X] 알람 해제 (빨간불→초록불)
  [1] 이어서 수집 (기존 데이터 기반으로 stats 누적)
  [ESC] 종료
  * 매 스텝 자동 백업 — 끊겨도 다음 실행 시 복구 가능
 ============================================================
""")
        while True:
            self._show_preview()
            key = cv2.waitKey(30) & 0xFF

            if key == ord("s"):
                self.start_step(); time.sleep(0.3)
            elif key == ord("e"):
                self.end_step(); time.sleep(0.3)
            elif key == ord("g"):
                self.toggle_grip(); time.sleep(0.3)
            elif key == ord("z"):
                self.rotate_wrist(-1); time.sleep(0.15)
            elif key == ord("c"):
                self.rotate_wrist(+1); time.sleep(0.15)
            elif key == ord("r"):
                self.replay_episode(); time.sleep(0.3)
            elif key == ord("v"):
                self.save_episode(); time.sleep(0.3)
            elif key == ord("w"):
                self.undo_step(); time.sleep(0.3)
            elif key == ord("d"):
                self.discard_episode(); time.sleep(0.3)
            elif key == ord("f"):
                self.reconnect_dobot(); time.sleep(0.3)
            elif key == ord("x"):
                self.clear_alarm(); time.sleep(0.3)
            elif key == ord("1"):
                self.resume_mode(); time.sleep(0.3)
            elif key == ord("a"):
                self.home(); time.sleep(0.3)
            elif key == ord("q"):
                self.go_home(); time.sleep(0.3)
            elif key == 27:
                if self.episode_data:
                    print(f"  Unsaved: {len(self.episode_data)} steps. [ESC] again or [V] to save.")
                    time.sleep(1)
                    k2 = cv2.waitKey(1000) & 0xFF
                    if k2 == 27:
                        break
                else:
                    break

        self.disconnect()
# Entry Point

def main():
    parser = argparse.ArgumentParser(
        description="DOBOT Magician -- LeRobot v3.0 Data Collection (Single-Arm Sequential)",
    )
    parser.add_argument("--port", type=str, default=None, help="DOBOT serial port (auto-detect if omitted)")
    parser.add_argument("--cam1", type=int, default=0, help="Top camera device ID")
    parser.add_argument("--cam2", type=int, default=1, help="Wrist camera device ID")
    parser.add_argument("--task", type=str, default="pick_object", help="Task description for this dataset")
    parser.add_argument("--save_dir", type=str, default="dataset_v3", help="Output dataset directory")
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS, help="Recording FPS")
    args = parser.parse_args()

    port = args.port or find_dobot_port()
    if not port:
        print("DOBOT not found. 카메라 전용 모드로 시작합니다.")
        port = "NONE"

    collector = LeRobotV3Collector(
        port=port,
        cam1_id=args.cam1,
        cam2_id=args.cam2,
        save_dir=args.save_dir,
        task=args.task,
        fps=args.fps,
    )

    if collector.connect():
        try:
            collector.run()
        except KeyboardInterrupt:
            print("\n  Interrupted.")
        finally:
            collector.disconnect()
if __name__ == "__main__":
    main()
