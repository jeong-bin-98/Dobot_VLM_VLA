#!/usr/bin/env python3
"""
LeRobot v3.0 데이터셋 검증 및 자동 수정

디렉토리 구조, info.json, episodes.parquet, tasks.jsonl, 이미지 경로 검증.

    python 03_validate_dataset.py --dataset_dir ./dataset_v3
    python 03_validate_dataset.py --dataset_dir ./dataset_v3 --fix
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List

try:
    import pyarrow.parquet as pq
    import pyarrow as pa
    import pandas as pd
except ImportError as e:
    print(f"Missing: {e}")
    print("Install: pip install pyarrow pandas")
    sys.exit(1)
class LeRobotV3Validator:
    """LeRobot v3.0 Dataset Validator with auto-fix capability."""

    def __init__(self, dataset_dir: str, verbose: bool = True):
        self.dataset_dir = Path(dataset_dir).resolve()
        self.verbose = verbose
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.fixes: List[str] = []

    def log(self, msg: str, level: str = "info"):
        if self.verbose:
            prefix = {"info": "", "ok": "OK:", "warn": "WARN:", "error": "ERR:", "fix": "FIX:"}
            print(f"  {prefix.get(level, '')} {msg}")

    # ------------------------------------------------------------------
    # Validate
    # ------------------------------------------------------------------

    def validate(self) -> bool:
        """Run all validation checks. """
        print(f"\n{'='*60}")
        print(f"  LeRobot v3.0 Dataset Validator")
        print(f"{'='*60}")
        print(f"  Dataset: {self.dataset_dir}\n")

        self._check_directory_structure()
        self._check_info_json()
        self._check_episodes_parquet()
        self._check_tasks()
        self._check_data_parquets()
        self._check_image_paths()

        self._print_summary()
        return len(self.errors) == 0

    def _check_directory_structure(self):
        print("  Checking directory structure...")
        has_images = (self.dataset_dir / "images").exists()
        has_videos = (self.dataset_dir / "videos").exists()

        if has_images and not has_videos:
            self.log("images/ directory found (v3.0)", "ok")
        elif has_videos and not has_images:
            self.log("videos/ only -- this is v2.x format! Use 02_convert_v2_to_v3.py first.", "error")
            self.errors.append("v2.x format (videos/ only). Convert first.")
        elif has_videos and has_images:
            self.log("Both videos/ and images/ -- mixed format", "warn")
            self.warnings.append("Mixed format: both videos/ and images/ exist")

        for req in ("meta", "data/chunk-000"):
            if (self.dataset_dir / req).exists():
                self.log(f"{req}/", "ok")
            else:
                self.log(f"{req}/ -- MISSING", "error")
                self.errors.append(f"Missing directory: {req}")

    def _check_info_json(self):
        print("\n  Checking meta/info.json...")
        info_path = self.dataset_dir / "meta" / "info.json"
        if not info_path.exists():
            self.log("info.json -- MISSING", "error")
            self.errors.append("Missing meta/info.json")
            return

        with open(info_path) as f:
            info = json.load(f)

        version = info.get("codebase_version", "unknown")
        if "3" in str(version):
            self.log(f"codebase_version: {version}", "ok")
        else:
            self.log(f"codebase_version: {version} -- expected v3.x", "warn")
            self.warnings.append(f"codebase_version={version}")

        video = info.get("video")
        if video is False:
            self.log("video: false", "ok")
        elif video is True:
            self.log("video: true -- should be false for v3.0", "error")
            self.errors.append("info.json: video should be false")
        else:
            self.log("video field -- MISSING", "error")
            self.errors.append("info.json: missing 'video' field")

        features = info.get("features", {})
        self.log(f"features: {len(features)} defined", "ok")

        for feat_name, feat_info in features.items():
            if "names" not in feat_info:
                self.log(f"Feature '{feat_name}' missing 'names'", "error")
                self.errors.append(f"Feature '{feat_name}' missing 'names'")
            if "images" in feat_name and feat_info.get("dtype") == "video":
                self.log(f"{feat_name}: dtype=video -- should be 'image'", "error")
                self.errors.append(f"Feature '{feat_name}': dtype should be 'image'")

    def _check_episodes_parquet(self):
        print("\n  Checking episodes metadata...")
        ep_path = self.dataset_dir / "meta" / "episodes" / "chunk-000" / "episodes.parquet"
        if not ep_path.exists():
            self.log("episodes.parquet -- MISSING", "error")
            self.errors.append("Missing meta/episodes/chunk-000/episodes.parquet")
            return

        df = pd.read_parquet(ep_path)
        self.log(f"episodes.parquet: {len(df)} episodes", "ok")

        required_cols = ["episode_index", "task_index", "length", "dataset_from_index", "dataset_to_index"]
        for col in required_cols:
            if col not in df.columns:
                self.log(f"Column '{col}' -- MISSING", "warn")
                self.warnings.append(f"Missing column: {col}")

        if "episode_index" in df.columns:
            indices = df["episode_index"].tolist()
            if indices != list(range(len(indices))):
                self.log("Episode indices not consecutive", "warn")
                self.warnings.append(f"Non-consecutive indices: {indices}")

    def _check_tasks(self):
        """Check for tasks.jsonl (v3.0) or tasks.parquet (legacy)."""
        print("\n  Checking tasks metadata...")

        jsonl_path = self.dataset_dir / "meta" / "tasks.jsonl"
        pq_path = self.dataset_dir / "meta" / "tasks.parquet"

        if jsonl_path.exists():
            with open(jsonl_path) as f:
                lines = [json.loads(l.strip()) for l in f if l.strip()]
            if lines and "task" in lines[0]:
                self.log(f"tasks.jsonl: '{lines[0]['task']}' (v3.0 format)", "ok")
            else:
                self.log("tasks.jsonl exists but format is unexpected", "warn")
                self.warnings.append("tasks.jsonl format issue")
        elif pq_path.exists():
            df = pd.read_parquet(pq_path)
            if df.index.name == "task":
                self.log(f"tasks.parquet: task as INDEX (legacy but functional)", "ok")
            elif "task" in df.columns:
                self.log("tasks.parquet: task as COLUMN -- non-standard", "warn")
                self.warnings.append("tasks.parquet: task should be INDEX or use tasks.jsonl")
            else:
                self.log("tasks.parquet: no task field found", "error")
                self.errors.append("tasks metadata: missing task field")
        else:
            self.log("No tasks metadata found (tasks.jsonl or tasks.parquet)", "error")
            self.errors.append("Missing task metadata")

    def _check_data_parquets(self):
        print("\n  Checking data parquets...")
        data_dir = self.dataset_dir / "data" / "chunk-000"
        if not data_dir.exists():
            return

        parquet_files = sorted(data_dir.glob("*.parquet"))
        self.log(f"Found {len(parquet_files)} parquet files", "ok")

        for pf in parquet_files[:3]:
            df = pd.read_parquet(pf)
            ep_idx = df["episode_index"].iloc[0] if "episode_index" in df.columns else "?"
            self.log(f"  {pf.name}: {len(df)} frames, episode_index={ep_idx}", "ok")

        if len(parquet_files) > 3:
            self.log(f"  ... and {len(parquet_files) - 3} more", "info")

    def _check_image_paths(self):
        print("\n  Checking image paths...")
        data_dir = self.dataset_dir / "data" / "chunk-000"
        if not data_dir.exists():
            return

        parquet_files = sorted(data_dir.glob("*.parquet"))
        if not parquet_files:
            return

        df = pd.read_parquet(parquet_files[0])
        img_cols = [c for c in df.columns if "images" in c]

        for col in img_cols[:2]:
            first = df[col].iloc[0]
            if isinstance(first, dict) and "path" in first:
                path_str = first["path"]
                if path_str.startswith("/"):
                    full = Path(path_str)
                    if full.exists():
                        self.log(f"{col}: Absolute paths valid", "ok")
                    else:
                        self.log(f"{col}: Absolute path not found: {path_str}", "error")
                        self.errors.append(f"Image not found: {path_str}")
                else:
                    full = self.dataset_dir / path_str
                    if full.exists():
                        self.log(f"{col}: Relative paths valid", "ok")
                    else:
                        self.log(f"{col}: Relative path not found: {path_str}", "error")
                        self.errors.append(f"Image not found: {path_str}")

    # ------------------------------------------------------------------
    # Fix
    # ------------------------------------------------------------------

    def fix(self):
        """Apply automatic fixes for detected issues."""
        print(f"\n{'='*60}")
        print("  APPLYING FIXES")
        print(f"{'='*60}\n")

        self._fix_info_json()
        self._fix_episodes_parquet()
        self._fix_tasks()

        if self.fixes:
            print(f"\n  Applied {len(self.fixes)} fixes:")
            for f in self.fixes:
                print(f"    - {f}")
        else:
            print("  No fixes needed.")

    def _fix_info_json(self):
        info_path = self.dataset_dir / "meta" / "info.json"
        if not info_path.exists():
            return

        with open(info_path) as f:
            info = json.load(f)
        modified = False

        if info.get("video") is not False:
            info["video"] = False
            modified = True
            self.log("Set video: false", "fix")

        if "3" not in str(info.get("codebase_version", "")):
            info["codebase_version"] = "v3.0"
            modified = True
            self.log("Set codebase_version: v3.0", "fix")

        for feat_name, feat_info in info.get("features", {}).items():
            if "images" in feat_name and feat_info.get("dtype") == "video":
                feat_info["dtype"] = "image"
                modified = True
            if "names" not in feat_info:
                if "images" in feat_name:
                    feat_info["names"] = ["channel", "height", "width"]
                elif "state" in feat_name or "action" in feat_name:
                    shape = feat_info.get("shape", [5])
                    feat_info["names"] = ["x", "y", "z", "r", "grip"] if shape == [5] else [f"dim_{i}" for i in range(shape[0])]
                else:
                    feat_info["names"] = [feat_name]
                modified = True

        if modified:
            with open(info_path, "w") as f:
                json.dump(info, f, indent=2)
            self.fixes.append("Updated info.json")

    def _fix_episodes_parquet(self):
        data_dir = self.dataset_dir / "data" / "chunk-000"
        if not data_dir.exists():
            return

        parquet_files = sorted(data_dir.glob("*.parquet"))
        if not parquet_files:
            return

        episodes = []
        global_idx = 0
        for pf in parquet_files:
            df = pd.read_parquet(pf)
            ep_idx = int(df["episode_index"].iloc[0]) if "episode_index" in df.columns else int(pf.stem.split("_")[1])
            task_idx = int(df["task_index"].iloc[0]) if "task_index" in df.columns else 0
            length = len(df)
            episodes.append({
                "episode_index": ep_idx, "task_index": task_idx, "length": length,
                "dataset_from_index": global_idx, "dataset_to_index": global_idx + length,
            })
            global_idx += length

        ep_df = pd.DataFrame(episodes)
        ep_df = ep_df[["episode_index", "task_index", "length", "dataset_from_index", "dataset_to_index"]]

        ep_path = self.dataset_dir / "meta" / "episodes" / "chunk-000" / "episodes.parquet"
        ep_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pandas(ep_df), ep_path)
        self.fixes.append(f"Rebuilt episodes.parquet ({len(episodes)} episodes)")

    def _fix_tasks(self):
        """Ensure tasks.jsonl exists (v3.0 format)."""
        jsonl_path = self.dataset_dir / "meta" / "tasks.jsonl"
        if jsonl_path.exists():
            return

        task_name = "pick_object"

        # Try to extract from tasks.parquet
        pq_path = self.dataset_dir / "meta" / "tasks.parquet"
        if pq_path.exists():
            df = pd.read_parquet(pq_path)
            if df.index.name == "task" and len(df.index) > 0:
                task_name = str(df.index[0])
            elif "task" in df.columns:
                task_name = str(df["task"].iloc[0])

        with open(jsonl_path, "w") as f:
            f.write(json.dumps({"task_index": 0, "task": task_name}) + "\n")
        self.fixes.append(f"Created tasks.jsonl (task='{task_name}')")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def _print_summary(self):
        print(f"\n{'='*60}")
        print("  VALIDATION SUMMARY")
        print(f"{'='*60}\n")

        if not self.errors and not self.warnings:
            print("  Dataset is valid for LeRobot v3.0!")
        else:
            if self.errors:
                print(f"  ERRORS ({len(self.errors)}):")
                for e in self.errors:
                    print(f"    - {e}")
            if self.warnings:
                print(f"\n  WARNINGS ({len(self.warnings)}):")
                for w in self.warnings:
                    print(f"    - {w}")
            print("\n  Run with --fix to auto-fix issues.")
def main():
    parser = argparse.ArgumentParser(description="LeRobot v3.0 Dataset Validator")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Dataset directory")
    parser.add_argument("--fix", action="store_true", help="Apply automatic fixes")
    parser.add_argument("--quiet", action="store_true", help="Reduce output")
    args = parser.parse_args()

    validator = LeRobotV3Validator(args.dataset_dir, not args.quiet)
    is_valid = validator.validate()

    if args.fix:
        validator.fix()
        print("\n  Re-validating...")
        validator.errors.clear()
        validator.warnings.clear()
        validator.validate()
if __name__ == "__main__":
    main()
