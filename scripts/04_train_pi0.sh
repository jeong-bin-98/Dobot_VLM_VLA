#!/bin/bash
# Pi0-FAST 학습 스크립트
#
# 사전 준비:
#   pip install "lerobot[pi] @ git+https://github.com/huggingface/lerobot.git"
#
# 사용법:
#   bash 04_train_pi0.sh [mac|gpu]

MODE=${1:-mac}
DATASET_ROOT="./scripts/dataset_v3"

if [ "$MODE" = "mac" ]; then
    # ============================================================
    # Mac (Apple Silicon, 48GB+) — LoRA + 메모리 절약 모드
    # 느리지만 테스트 가능
    # ============================================================
    echo "Mac 모드 (MPS, 메모리 절약)"
    KMP_DUPLICATE_LIB_OK=TRUE lerobot-train \
        --dataset.repo_id=local/dataset_v3 \
        --dataset.root=${DATASET_ROOT} \
        --policy.type=pi0_fast \
        --policy.pretrained_path=lerobot/pi0_fast_base \
        --policy.dtype=float32 \
        --policy.gradient_checkpointing=true \
        --policy.push_to_hub=false \
        --policy.chunk_size=10 \
        --policy.n_action_steps=1 \
        --batch_size=1 \
        --steps=1000 \
        --output_dir=./outputs/pi0fast_dobot_mac \
        --policy.device=mps
else
    # ============================================================
    # GPU 서버 (A6000/A100 48GB+) — 풀 파인튜닝
    # ============================================================
    echo "GPU 서버 모드 (CUDA)"
    lerobot-train \
        --dataset.repo_id=local/dataset_v3 \
        --dataset.root=${DATASET_ROOT} \
        --policy.type=pi0_fast \
        --policy.pretrained_path=lerobot/pi0_fast_base \
        --policy.dtype=bfloat16 \
        --policy.gradient_checkpointing=true \
        --policy.chunk_size=10 \
        --policy.n_action_steps=1 \
        --batch_size=4 \
        --steps=100000 \
        --output_dir=./outputs/pi0fast_dobot \
        --policy.device=cuda
fi
