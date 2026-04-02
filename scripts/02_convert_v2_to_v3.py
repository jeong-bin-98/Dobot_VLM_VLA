#!/usr/bin/env python3
"""
LeRobot v2.x -> v3.0 변환기

상대 경로, tasks.jsonl, episodes.parquet 생성.
비연속 에피소드 인덱스 자동 수정 지원.

    python 02_convert_v2_to_v3.py --input_dir ./dataset --output_dir ./dataset_v3 \\
        --task "pick up bottle" --fix_indices
"""

import os
import sys
import json
import shutil
import argparse
from pathlib import Path
from typing import Dict, List

try:
    import cv2
    import numpy as np
    import pyarrow.parquet as pq
    import pyarrow as pa
    import pandas as pd
except ImportError as e:
    print(f"Missing: {e}")
    print("Install: pip install opencv-python pyarrow pandas numpy")
    sys.exit(1)
class LeRobotV2toV3Converter:
    """v2.x -> v3.0 변환. 상대 경로, tasks.jsonl, episodes.parquet 생성."""

    def __init__(self, input_dir: str, output_dir: str, task_name: str = "pick_object", verbose: bool = True):
        self.input_dir = Path(input_dir).resolve()
        self.output_dir = Path(output_dir).resolve()
        self.task_name = task_name
        self.verbose = verbose
        self.camera_names: List[str] = []
        self.episodes_info: List[Dict] = []

    def log(self, msg: str, level: str = "info"):
        if self.verbose:
            prefix = {"info": "", "ok": "OK:", "warn": "WARN:", "error": "ERR:", "fix": "FIX:", "convert": ">>"}
            print(f"  {prefix.get(level, '')} {msg}")

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    def detect_cameras(self):
        """Auto-detect camera feature names from input dataset."""
        for parent in ("videos", "images"):
            parent_dir = self.input_dir / parent
            if parent_dir.exists():
                for item in parent_dir.iterdir():
                    if item.is_dir() and item.name.startswith("observation.images"):
                        if item.name not in self.camera_names:
                            self.camera_names.append(item.name)
        if self.camera_names:
            self.log(f"Detected cameras: {self.camera_names}", "info")

    def detect_task_from_input(self) -> str:
        """Try to detect task name from existing metadata files."""
        # Check tasks.jsonl (v2.x format)
        tasks_jsonl = self.input_dir / "meta" / "tasks.jsonl"
        if tasks_jsonl.exists():
            with open(tasks_jsonl) as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        if "task" in data:
                            return data["task"]
                    except (json.JSONDecodeError, KeyError):
                        pass

        # Check tasks.parquet
        tasks_pq = self.input_dir / "meta" / "tasks.parquet"
        if tasks_pq.exists():
            df = pd.read_parquet(tasks_pq)
            if df.index.name == "task" and len(df.index) > 0:
                val = str(df.index[0])
                if val and val != "0":
                    return val
            if "task" in df.columns:
                return str(df["task"].iloc[0])

        return self.task_name

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_indices(self) -> Dict:
        """Validate episode index consistency in the input dataset."""
        print(f"\n{'='*60}")
        print("  INDEX VALIDATION")
        print(f"{'='*60}\n")

        result = {"parquet_files": {}, "issues": [], "recommendations": []}
        data_dir = self.input_dir / "data" / "chunk-000"
        if not data_dir.exists():
            result["issues"].append("data/chunk-000 not found")
            return result

        parquet_files = sorted(data_dir.glob("episode_*.parquet"))

        for pf in parquet_files:
            file_idx = int(pf.stem.split("_")[1])
            try:
                df = pd.read_parquet(pf)
                content_idx = int(df["episode_index"].iloc[0]) if "episode_index" in df.columns else file_idx
                result["parquet_files"][pf.name] = {
                    "file_index": file_idx, "content_index": content_idx, "frames": len(df),
                }
                if file_idx != content_idx:
                    self.log(f"{pf.name}: file={file_idx}, content={content_idx} -- MISMATCH", "warn")
                    result["issues"].append(f"{pf.name}: filename != content index")
                else:
                    self.log(f"{pf.name}: index={file_idx}, frames={len(df)}", "ok")
            except Exception as e:
                result["issues"].append(f"{pf.name}: Error -- {e}")

        if result["parquet_files"]:
            indices = sorted(v["content_index"] for v in result["parquet_files"].values())
            expected = list(range(len(indices)))
            if indices != expected:
                result["issues"].append(f"Index gap: {indices} -> should be {expected}")
                result["recommendations"].append("Run with --fix_indices")

        print(f"\n  Summary: {len(result['parquet_files'])} files, {len(result['issues'])} issues")
        return result

    # ------------------------------------------------------------------
    # Conversion
    # ------------------------------------------------------------------

    def convert(self, fix_indices: bool = False, fps: int = 10):
        """Main conversion: v2.x -> v3.0."""
        print(f"\n{'='*60}")
        print("  CONVERTING v2.x -> v3.0")
        print(f"{'='*60}")
        print(f"  Input:  {self.input_dir}")
        print(f"  Output: {self.output_dir}")
        print(f"{'='*60}\n")

        self.detect_cameras()
        if not self.camera_names:
            self.log("No cameras detected", "error")
            return

        if self.task_name == "pick_object":
            detected = self.detect_task_from_input()
            if detected != "pick_object":
                self.task_name = detected
        self.log(f"Task: {self.task_name}", "info")

        self._create_output_structure()
        episode_mapping = self._get_episode_mapping(fix_indices)
        self._process_episodes(episode_mapping, fps)

        self._generate_info_json(fps)
        self._generate_episodes_parquet()
        self._generate_tasks_jsonl()
        self._generate_stats_json()

        total_frames = sum(ep["length"] for ep in self.episodes_info)
        print(f"\n{'='*60}")
        print(f"  CONVERSION COMPLETE")
        print(f"{'='*60}")
        print(f"  Episodes: {len(self.episodes_info)}")
        print(f"  Total frames: {total_frames}")
        print(f"  Task: {self.task_name}")
        print(f"  RELATIVE paths (portable)")
        print(f"{'='*60}")

    def _create_output_structure(self):
        dirs = [
            self.output_dir / "meta",
            self.output_dir / "meta" / "episodes" / "chunk-000",
            self.output_dir / "data" / "chunk-000",
        ]
        for cam in self.camera_names:
            dirs.append(self.output_dir / "images" / cam / "chunk-000")
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)

    def _get_episode_mapping(self, fix_indices: bool) -> Dict[int, int]:
        data_dir = self.input_dir / "data" / "chunk-000"
        parquet_files = sorted(data_dir.glob("episode_*.parquet"))

        indices = []
        for pf in parquet_files:
            try:
                df = pd.read_parquet(pf)
                idx = int(df["episode_index"].iloc[0]) if "episode_index" in df.columns else int(pf.stem.split("_")[1])
                indices.append(idx)
            except Exception:
                indices.append(int(pf.stem.split("_")[1]))
        indices = sorted(set(indices))

        if fix_indices:
            mapping = {old: new for new, old in enumerate(indices)}
            self.log(f"Index mapping: {mapping}", "fix")
        else:
            mapping = {idx: idx for idx in indices}
        return mapping

    def _process_episodes(self, episode_mapping: Dict[int, int], fps: int):
        data_dir = self.input_dir / "data" / "chunk-000"
        parquet_files = sorted(data_dir.glob("episode_*.parquet"))
        global_frame_idx = 0

        for pf in parquet_files:
            df = pd.read_parquet(pf)
            old_idx = int(df["episode_index"].iloc[0]) if "episode_index" in df.columns else int(pf.stem.split("_")[1])
            new_idx = episode_mapping.get(old_idx, old_idx)

            self.log(f"Processing episode {old_idx} -> {new_idx}", "convert")
            frame_count = self._extract_or_copy_frames(old_idx, new_idx)
            self._update_parquet(df, new_idx, global_frame_idx)

            self.episodes_info.append({
                "episode_index": new_idx,
                "task_index": 0,
                "length": frame_count,
                "dataset_from_index": global_frame_idx,
                "dataset_to_index": global_frame_idx + frame_count,
            })
            global_frame_idx += frame_count

    def _extract_or_copy_frames(self, old_idx: int, new_idx: int) -> int:
        old_ep, new_ep = f"{old_idx:06d}", f"{new_idx:06d}"
        frame_count = 0

        for cam in self.camera_names:
            output_dir = self.output_dir / "images" / cam / "chunk-000" / f"episode_{new_ep}"
            output_dir.mkdir(parents=True, exist_ok=True)

            # Try video first (v2.x stored MP4 videos)
            video_path = self.input_dir / "videos" / cam / "chunk-000" / f"episode_{old_ep}.mp4"
            if video_path.exists():
                cap = cv2.VideoCapture(str(video_path))
                idx = 0
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    cv2.imwrite(str(output_dir / f"frame_{idx:06d}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    idx += 1
                cap.release()
                frame_count = max(frame_count, idx)
                self.log(f"  {cam}: Extracted {idx} frames from video", "ok")
            else:
                # Try copying from images/
                src_dir = self.input_dir / "images" / cam / "chunk-000" / f"episode_{old_ep}"
                if src_dir.exists():
                    frames = sorted(src_dir.glob("frame_*.jpg"))
                    for i, src in enumerate(frames):
                        shutil.copy2(src, output_dir / f"frame_{i:06d}.jpg")
                    frame_count = max(frame_count, len(frames))
                    self.log(f"  {cam}: Copied {len(frames)} frames", "ok")

        return frame_count

    def _update_parquet(self, df: pd.DataFrame, new_idx: int, global_start: int):
        new_ep = f"{new_idx:06d}"

        df["episode_index"] = new_idx
        df["frame_index"] = range(len(df))
        df["index"] = range(global_start, global_start + len(df))

        if "task_index" not in df.columns:
            df["task_index"] = 0
        if "task" in df.columns:
            df = df.drop(columns=["task"])

        # RELATIVE paths (v3.0 key feature)
        for cam in self.camera_names:
            df[cam] = [
                {"path": f"images/{cam}/chunk-000/episode_{new_ep}/frame_{i:06d}.jpg"}
                for i in range(len(df))
            ]

        pq.write_table(
            pa.Table.from_pandas(df),
            self.output_dir / "data" / "chunk-000" / f"episode_{new_ep}.parquet",
        )

    def _generate_info_json(self, fps: int):
        total_frames = sum(ep["length"] for ep in self.episodes_info)
        features = {}
        for cam in self.camera_names:
            features[cam] = {"dtype": "image", "shape": [3, 480, 640], "names": ["channel", "height", "width"]}

        features.update({
            "observation.state": {"dtype": "float32", "shape": [5], "names": ["x", "y", "z", "r", "grip"]},
            "action": {"dtype": "float32", "shape": [5], "names": ["delta_x", "delta_y", "delta_z", "delta_r", "grip"]},
            "index": {"dtype": "int64", "shape": [1], "names": ["index"]},
            "timestamp": {"dtype": "float32", "shape": [1], "names": ["timestamp"]},
            "frame_index": {"dtype": "int64", "shape": [1], "names": ["frame_index"]},
            "episode_index": {"dtype": "int64", "shape": [1], "names": ["episode_index"]},
            "task_index": {"dtype": "int64", "shape": [1], "names": ["task_index"]},
        })

        info = {
            "codebase_version": "v3.0",
            "robot_type": "dobot_magician",
            "total_frames": total_frames,
            "total_episodes": len(self.episodes_info),
            "total_tasks": 1,
            "fps": fps,
            "video": False,
            "features": features,
        }
        with open(self.output_dir / "meta" / "info.json", "w") as f:
            json.dump(info, f, indent=2)

    def _generate_episodes_parquet(self):
        df = pd.DataFrame(self.episodes_info)
        df = df[["episode_index", "task_index", "length", "dataset_from_index", "dataset_to_index"]]
        pq.write_table(
            pa.Table.from_pandas(df),
            self.output_dir / "meta" / "episodes" / "chunk-000" / "episodes.parquet",
        )
        self.log("Generated episodes.parquet", "ok")

    def _generate_tasks_jsonl(self):
        """Generate tasks.jsonl (v3.0 format: one JSON object per line)."""
        tasks_path = self.output_dir / "meta" / "tasks.jsonl"
        with open(tasks_path, "w") as f:
            f.write(json.dumps({"task_index": 0, "task": self.task_name}) + "\n")
        self.log(f"Generated tasks.jsonl (task='{self.task_name}')", "ok")

    def _generate_stats_json(self):
        all_states, all_actions = [], []
        data_dir = self.output_dir / "data" / "chunk-000"
        for pf in sorted(data_dir.glob("episode_*.parquet")):
            df = pd.read_parquet(pf)
            if "observation.state" in df.columns:
                all_states.extend(df["observation.state"].tolist())
            if "action" in df.columns:
                all_actions.extend(df["action"].tolist())

        stats = {}
        if all_states:
            arr = np.array(all_states)
            stats["observation.state"] = {
                "min": arr.min(0).tolist(), "max": arr.max(0).tolist(),
                "mean": arr.mean(0).tolist(), "std": arr.std(0).tolist(),
            }
        if all_actions:
            arr = np.array(all_actions)
            stats["action"] = {
                "min": arr.min(0).tolist(), "max": arr.max(0).tolist(),
                "mean": arr.mean(0).tolist(), "std": arr.std(0).tolist(),
            }
        for cam in self.camera_names:
            stats[cam] = {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}

        with open(self.output_dir / "meta" / "stats.json", "w") as f:
            json.dump(stats, f, indent=2)
def main():
    parser = argparse.ArgumentParser(
        description="LeRobot v2.x -> v3.0 Converter",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python 02_convert_v2_to_v3.py --input_dir ./dataset --output_dir ./dataset_v3
  python 02_convert_v2_to_v3.py --input_dir ./dataset --output_dir ./dataset_v3 --fix_indices
  python 02_convert_v2_to_v3.py --input_dir ./dataset --output_dir ./dataset_v3 --task "pick bottle"
  python 02_convert_v2_to_v3.py --input_dir ./dataset --validate_only
""",
    )
    parser.add_argument("--input_dir", type=str, required=True, help="Input dataset directory")
    parser.add_argument("--output_dir", type=str, default=None, help="Output dataset directory")
    parser.add_argument("--task", type=str, default="pick_object", help="Task name")
    parser.add_argument("--validate_only", action="store_true", help="Only validate indices")
    parser.add_argument("--fix_indices", action="store_true", help="Fix non-consecutive indices")
    parser.add_argument("--fps", type=int, default=10, help="FPS (default: 10)")
    parser.add_argument("--quiet", action="store_true", help="Reduce output")
    args = parser.parse_args()

    if not args.validate_only and not args.output_dir:
        print("--output_dir required for conversion")
        sys.exit(1)

    converter = LeRobotV2toV3Converter(
        input_dir=args.input_dir,
        output_dir=args.output_dir or args.input_dir,
        task_name=args.task,
        verbose=not args.quiet,
    )

    converter.detect_cameras()
    converter.validate_indices()

    if not args.validate_only:
        converter.convert(fix_indices=args.fix_indices, fps=args.fps)
if __name__ == "__main__":
    main()
