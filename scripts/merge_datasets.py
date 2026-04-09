#!/usr/bin/env python3
"""
여러 데이터셋을 하나로 합치는 스크립트.
각 데이터셋의 task를 유지하면서 에피소드를 순서대로 합칩니다.

사용법:
    python scripts/merge_datasets.py \
        --datasets ./tissue_dataset_v1 ./snack_dataset_v1 \
        --output ./merged_dataset
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def merge_datasets(dataset_dirs: list[str], output_dir: str):
    output = Path(output_dir).resolve()
    if output.exists():
        shutil.rmtree(output)

    datasets = [Path(d).resolve() for d in dataset_dirs]

    # 검증
    for ds in datasets:
        if not (ds / "meta" / "info.json").exists():
            print(f"  에러: {ds}에 meta/info.json이 없습니다.")
            sys.exit(1)

    # 첫 번째 데이터셋의 info를 기본으로 사용
    with open(datasets[0] / "meta" / "info.json") as f:
        base_info = json.load(f)

    # task 수집 (중복 제거, 순서 유지)
    all_tasks = []
    task_to_index = {}
    for ds in datasets:
        tasks_path = ds / "meta" / "tasks.jsonl"
        if tasks_path.exists():
            with open(tasks_path) as f:
                for line in f:
                    t = json.loads(line)["task"]
                    if t not in task_to_index:
                        task_to_index[t] = len(all_tasks)
                        all_tasks.append(t)

    # 출력 디렉토리 구조 생성
    cam_names = ["observation.images.wrist", "observation.images.top"]
    dirs = [
        "meta", "meta/episodes/chunk-000", "data/chunk-000",
    ] + [f"images/{cam}/chunk-000" for cam in cam_names]
    for d in dirs:
        (output / d).mkdir(parents=True, exist_ok=True)

    # 에피소드 합치기
    global_ep_idx = 0
    global_frame_idx = 0
    all_ep_meta = []
    all_states = []
    all_actions = []

    for ds in datasets:
        # 이 데이터셋의 task
        ds_task = None
        tasks_path = ds / "meta" / "tasks.jsonl"
        if tasks_path.exists():
            with open(tasks_path) as f:
                ds_task = json.loads(f.readline())["task"]

        ds_task_idx = task_to_index.get(ds_task, 0)

        # 에피소드 파일 찾기
        data_dir = ds / "data" / "chunk-000"
        ep_files = sorted(data_dir.glob("episode_*.parquet"))

        for ep_file in ep_files:
            df = pd.read_parquet(ep_file)
            n_frames = len(df)

            # task_index 업데이트
            df["task_index"] = ds_task_idx

            # episode_index, index 리넘버링
            df["episode_index"] = global_ep_idx
            df["index"] = range(global_frame_idx, global_frame_idx + n_frames)
            df["frame_index"] = range(n_frames)

            # 이미지 경로를 상대경로로 변환 (lerobot v3.0 호환)
            ep_str = f"{global_ep_idx:06d}"
            for cam in cam_names:
                if cam in df.columns:
                    df[cam] = [{"path": f"images/{cam}/chunk-000/episode_{ep_str}/frame_{i:06d}.jpg"} for i in range(n_frames)]

            # 데이터 저장
            pq.write_table(
                pa.Table.from_pandas(df),
                output / "data" / "chunk-000" / f"episode_{ep_str}.parquet",
            )

            # 이미지 복사
            src_ep_str = ep_file.stem.split("_")[1]  # "000003" 등
            for cam in cam_names:
                src_img_dir = ds / "images" / cam / "chunk-000" / f"episode_{src_ep_str}"
                dst_img_dir = output / "images" / cam / "chunk-000" / f"episode_{ep_str}"
                if src_img_dir.exists():
                    shutil.copytree(src_img_dir, dst_img_dir)

            # stats 수집
            if "observation.state" in df.columns:
                for _, row in df.iterrows():
                    all_states.append(row["observation.state"])
                    all_actions.append(row["action"])

            # 에피소드 메타데이터
            all_ep_meta.append({
                "episode_index": global_ep_idx,
                "task_index": ds_task_idx,
                "length": n_frames,
                "dataset_from_index": global_frame_idx,
                "dataset_to_index": global_frame_idx + n_frames,
            })

            global_frame_idx += n_frames
            global_ep_idx += 1

        print(f"  {ds.name}: {len(ep_files)} 에피소드 추가 (task: {ds_task})")

    # 메타데이터 저장
    # episodes.parquet
    ep_df = pd.DataFrame(all_ep_meta)
    pq.write_table(
        pa.Table.from_pandas(ep_df),
        output / "meta" / "episodes" / "chunk-000" / "episodes.parquet",
    )

    # tasks.jsonl
    with open(output / "meta" / "tasks.jsonl", "w") as f:
        for task in all_tasks:
            f.write(json.dumps({"task_index": task_to_index[task], "task": task}) + "\n")

    # tasks.parquet
    tasks_df = pd.DataFrame(
        {"task_index": list(range(len(all_tasks)))},
        index=pd.Index(all_tasks, name="task"),
    )
    pq.write_table(pa.Table.from_pandas(tasks_df), output / "meta" / "tasks.parquet")

    # info.json
    base_info["total_episodes"] = global_ep_idx
    base_info["total_frames"] = global_frame_idx
    base_info["total_tasks"] = len(all_tasks)
    with open(output / "meta" / "info.json", "w") as f:
        json.dump(base_info, f, indent=2)

    # stats.json
    if all_states:
        s = np.array(all_states)
        a = np.array(all_actions)
        # ImageNet stats (lerobot 학습 시 use_imagenet_stats=True로 덮어쓰지만 키는 있어야 함)
        img_stats = {
            "mean": [[[0.485]], [[0.456]], [[0.406]]],
            "std": [[[0.229]], [[0.224]], [[0.225]]],
            "min": [[[0.0]], [[0.0]], [[0.0]]],
            "max": [[[1.0]], [[1.0]], [[1.0]]],
        }
        stats = {
            "observation.state": {
                "mean": s.mean(axis=0).tolist(),
                "std": s.std(axis=0).tolist(),
                "min": s.min(axis=0).tolist(),
                "max": s.max(axis=0).tolist(),
            },
            "action": {
                "mean": a.mean(axis=0).tolist(),
                "std": a.std(axis=0).tolist(),
                "min": a.min(axis=0).tolist(),
                "max": a.max(axis=0).tolist(),
            },
        }
        for cam in cam_names:
            stats[cam] = img_stats
        with open(output / "meta" / "stats.json", "w") as f:
            json.dump(stats, f, indent=2)

    print(f"\n  합치기 완료: {output}")
    print(f"  총 {global_ep_idx} 에피소드, {global_frame_idx} 프레임")
    print(f"  Tasks: {all_tasks}")


def main():
    parser = argparse.ArgumentParser(description="여러 데이터셋을 하나로 합치기")
    parser.add_argument("--datasets", nargs="+", required=True, help="합칠 데이터셋 경로들")
    parser.add_argument("--output", required=True, help="출력 데이터셋 경로")
    args = parser.parse_args()
    merge_datasets(args.datasets, args.output)


if __name__ == "__main__":
    main()
