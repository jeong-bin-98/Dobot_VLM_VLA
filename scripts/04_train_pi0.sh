#!/bin/bash
# Pi0-FAST 학습 스크립트
#
# 사전 준비:
#   pip install "lerobot[pi] @ git+https://github.com/huggingface/lerobot.git"
#
# 사용법:
#   cd Dobot_VLM_VLA
#   bash scripts/04_train_pi0.sh [mac|gpu]

MODE=${1:-gpu}
DATASET_ROOT="./dataset_v3"

if [ "$MODE" = "mac" ]; then
    # ============================================================
    # Mac (Apple Silicon) — 테스트용
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
    # GPU 서버 (RTX A6000 x2, 48GB each) — 풀 파인튜닝
    # ============================================================
    echo "GPU 서버 모드 (RTX A6000 x2)"
    lerobot-train \
        --dataset.repo_id=local/dataset_v3 \
        --dataset.root=${DATASET_ROOT} \
        --policy.type=pi0_fast \
        --policy.pretrained_path=lerobot/pi0_fast_base \
        --policy.push_to_hub=false \
        --policy.dtype=bfloat16 \
        --policy.gradient_checkpointing=true \
        --policy.chunk_size=10 \
        --policy.n_action_steps=1 \
        --batch_size=4 \
        --steps=100000 \
        --output_dir=outputs/pi0fast_dobot
fi
