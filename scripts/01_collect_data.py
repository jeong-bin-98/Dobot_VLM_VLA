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
import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

try:
    import cv2
    import numpy as np
    import pydobot
    from serial.tools import list_ports
    import keyboard
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
    ports = list_ports.comports()
    return ports[0].device if ports else None
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
        self.cam1_name = "observation.images.top"
        self.cam2_name = "observation.images.wrist"

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
            self.dobot = pydobot.Dobot(port=self.port, verbose=False)
            self.dobot.speed(200, 200)
            print(f"  DOBOT connected: {self.port}")
        except Exception as e:
            print(f"  DOBOT connection failed: {e}")
            return False

        # Cameras
        self.cap1 = cv2.VideoCapture(self.cam1_id)
        self.cap2 = cv2.VideoCapture(self.cam2_id)
        if not self.cap1.isOpened() or not self.cap2.isOpened():
            print("  Camera initialization failed")
            return False
        for cap in (self.cap1, self.cap2):
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, IMG_W)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, IMG_H)
        print(f"  Cameras: top={self.cam1_id}, wrist={self.cam2_id}")

        # Directory structure & metadata
        self._create_v3_structure()
        self._find_last_episode()
        self.wrist_angle = self._get_pose()[3]

        print(f"\n  Path: {self.save_dir}")
        print(f"  Task: {self.task}")
        print(f"  Next episode: {self.episode_index}")
        print(f"  Format: LeRobot v3.0 (images, relative paths)")
        return True

    def disconnect(self):
        """Release all hardware resources."""
        if self.dobot:
            try:
                self.dobot.suck(False)
                self.dobot.close()
            except Exception:
                pass
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
        delta = [round(target- self.current_state[i], 2) for i in range(3)]
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

    # ------------------------------------------------------------------
    # Gripper & Wrist Control
    # ------------------------------------------------------------------

    def toggle_grip(self):
        """[G] -- Toggle gripper on/off."""
        self.grip_on = not self.grip_on
        try:
            self.dobot.grip(self.grip_on)
        except Exception:
            try:
                self.dobot.suck(self.grip_on)
            except Exception:
                pass
        print(f"  Grip: {'ON' if self.grip_on else 'OFF'}")

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

        print(f"\n  Replaying {len(self.episode_data)} steps... [Q] to stop")
        s0 = self.episode_data[0]["observation.state"]
        self.dobot.move_to(s0[0], s0[1], s0[2], s0[3], wait=True)
        time.sleep(0.5)

        for i, step in enumerate(self.episode_data):
            if keyboard.is_pressed("q"):
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

    def discard_episode(self):
        """[D] -- Discard the current episode data."""
        n = len(self.episode_data)
        self.frame_index = 0
        self.episode_images_cam1.clear()
        self.episode_images_cam2.clear()
        self.episode_data.clear()
        self.step_started = False
        print(f"  {n} steps discarded.")

    # ------------------------------------------------------------------
    # Preview & Main Loop
    # ------------------------------------------------------------------

    def _show_preview(self):
        """Render live camera preview with status overlay."""
        img1, img2 = self._capture_images()
        if img1 is None:
            return

        pose = self._get_pose()
        status, color = ("REC", (0, 0, 255)) if self.step_started else ("READY", (0, 255, 0))

        cv2.putText(img1, f"CAM1 | Ep {self.episode_index} | {status}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(img1, f"X:{pose[0]:.1f} Y:{pose[1]:.1f} Z:{pose[2]:.1f}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        if self.step_started and self.current_state:
            d = [pose- self.current_state[i] for i in range(3)]
            cv2.putText(img1, f"Delta: ({d[0]:+.1f}, {d[1]:+.1f}, {d[2]:+.1f})",
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        cv2.putText(img2, f"CAM2 | Frame {self.frame_index} | v3.0",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(img2, f"Grip: {'ON' if self.grip_on else 'OFF'} | Wrist: {self.wrist_angle:.1f}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 2)

        combined = np.hstack([img1, img2])
        cv2.putText(combined,
                    "[S]tart [E]nd [G]rip [Z/C]Wrist [R]eplay [V]Save [D]iscard [H]ome [ESC]",
                    (10, combined.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        cv2.imshow("LeRobot v3.0 Collector", combined)

    def run(self):
        """Main interactive collection loop."""
        print("""
 ============================================================
  DOBOT LeRobot v3.0 -- Data Collection
 ============================================================
  [S] Start Step    [E] End Step      [G] Toggle Gripper
  [Z] / [C] Wrist   [R] Replay        [V] Save Episode
  [D] Discard       [H] Home          [ESC] Exit
 ============================================================
""")
        while True:
            self._show_preview()
            key = cv2.waitKey(30) & 0xFF

            if key == ord("s") or keyboard.is_pressed("s"):
                self.start_step(); time.sleep(0.3)
            elif key == ord("e") or keyboard.is_pressed("e"):
                self.end_step(); time.sleep(0.3)
            elif key == ord("g") or keyboard.is_pressed("g"):
                self.toggle_grip(); time.sleep(0.3)
            elif key == ord("z") or keyboard.is_pressed("z"):
                self.rotate_wrist(-1); time.sleep(0.15)
            elif key == ord("c") or keyboard.is_pressed("c"):
                self.rotate_wrist(+1); time.sleep(0.15)
            elif key == ord("r") or keyboard.is_pressed("r"):
                self.replay_episode(); time.sleep(0.3)
            elif key == ord("v") or keyboard.is_pressed("v"):
                self.save_episode(); time.sleep(0.3)
            elif key == ord("d") or keyboard.is_pressed("d"):
                self.discard_episode(); time.sleep(0.3)
            elif key == ord("h") or keyboard.is_pressed("h"):
                self.dobot.move_to(200, 0, 50, 0, wait=True)
                self.wrist_angle = 0.0
            elif key == 27 or keyboard.is_pressed("esc"):
                if self.episode_data:
                    print(f"  Unsaved: {len(self.episode_data)} steps. [ESC] again or [V] to save.")
                    time.sleep(1)
                    if keyboard.is_pressed("esc"):
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
        print("DOBOT not found. Specify --port explicitly.")
        return

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
