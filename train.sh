#!/bin/bash
# Pi0 / Pi0-FAST 학습 스크립트
#
# 단일 데이터셋 (pi0_fast + LoRA, 기본):
#   ./train.sh ./cup_dataset_v1 1 1000 outputs/cup_test
#
# 여러 데이터셋 합쳐서:
#   ./train.sh "./cup_dataset_v1 ./block_dataset_v1" 1 1000 outputs/multi_test
#
# 이어서 학습 (resume):
#   ./train.sh ./cup_dataset_v1 1 20000 outputs/cup_test resume
#
# Full fine-tuning (LoRA 없이):
#   ./train.sh ./cup_dataset_v1 1 20000 outputs/cup_full "" full
#
# Pi0 (flow-matching) + LoRA:
#   ./train.sh ./cup_dataset_v1 1 20000 outputs/cup_pi0 "" "" pi0
#
# Pi0 (flow-matching) + Full fine-tuning:
#   ./train.sh ./cup_dataset_v1 1 20000 outputs/cup_pi0_full "" full pi0

DATASETS="${1:-./dataset_v3}"
GPU="${2:-1}"
STEPS="${3:-100}"
OUTPUT="${4:-outputs/pi0fast_dobot_test}"
RESUME="${5:-}"
FULL_FT="${6:-}"      # "full" 입력 시 LoRA 없이 full fine-tuning
MODEL="${7:-pi0_fast}" # "pi0" 또는 "pi0_fast"

# Warmup: STEPS의 5%, 최소 10, 최대 500
WARMUP=$(( STEPS / 20 ))
[ "$WARMUP" -lt 10 ] && WARMUP=10
[ "$WARMUP" -gt 500 ] && WARMUP=500
# Decay: warmup 이후 구간
DECAY=$(( STEPS - WARMUP ))

# resume 옵션 설정
RESUME_FLAG=""
if [ "$RESUME" = "resume" ] || [ "$RESUME" = "true" ] || [ "$RESUME" = "1" ]; then
    RESUME_FLAG="--resume=true"
    echo "=== 이어서 학습 (resume) 모드 ==="
    echo "  체크포인트 경로: $OUTPUT"
    echo ""
fi

# 모델 타입 설정
if [ "$MODEL" = "pi0" ]; then
    POLICY_TYPE="pi0"
    PRETRAINED="lerobot/pi0_base"
else
    POLICY_TYPE="pi0_fast"
    PRETRAINED="lerobot/pi0fast-base"
fi

# LoRA / Full fine-tuning 옵션
PEFT_FLAGS=""
if [ "$FULL_FT" = "full" ]; then
    PEFT_FLAGS="--policy.use_peft=false"
    echo "=== ${POLICY_TYPE} | Full fine-tuning 모드 ==="
else
    PEFT_FLAGS="--peft.method_type=lora --peft.r=16 --peft.target_modules='[\"q_proj\",\"v_proj\",\"k_proj\",\"o_proj\"]'"
    echo "=== ${POLICY_TYPE} | LoRA fine-tuning 모드 (r=16) ==="
fi
echo ""

# 데이터셋이 여러 개인지 확인 (공백 구분)
read -ra DS_ARRAY <<< "$DATASETS"

if [ ${#DS_ARRAY[@]} -eq 1 ]; then
    # 단일 데이터셋
    DATASET_NAME=$(basename "${DS_ARRAY[0]}")
    CUDA_VISIBLE_DEVICES=$GPU eval lerobot-train \
        --dataset.repo_id="local/$DATASET_NAME" \
        --dataset.root="${DS_ARRAY[0]}" \
        --dataset.image_transforms.enable=true \
        --policy.type=$POLICY_TYPE \
        --policy.pretrained_path=$PRETRAINED \
        --policy.push_to_hub=false \
        --policy.dtype=bfloat16 \
        --policy.gradient_checkpointing=true \
        --policy.chunk_size=5 \
        --policy.n_action_steps=1 \
        $PEFT_FLAGS \
        --batch_size=4 \
        --steps="$STEPS" \
        --scheduler.type=cosine_decay_with_warmup \
        --scheduler.peak_lr=2.5e-5 \
        --scheduler.decay_lr=2.5e-6 \
        --scheduler.num_warmup_steps="$WARMUP" \
        --scheduler.num_decay_steps="$DECAY" \
        --save_freq=5000 \
        --output_dir="$OUTPUT" \
        $RESUME_FLAG
else
    # 여러 데이터셋 합쳐서 학습 (merge 후 학습)
    MERGED_DIR="./merged_dataset_tmp"
    MERGED_REPO="local/merged_dataset"
    trap 'echo ">>> 임시 데이터셋 정리: $MERGED_DIR"; rm -rf "$MERGED_DIR"' EXIT

    echo "=== 멀티 데이터셋 학습 ==="
    echo "  데이터셋: ${DS_ARRAY[*]}"
    echo "  GPU: $GPU | Steps: $STEPS"
    echo ""

    echo ">>> 데이터셋 합치는 중..."
    python scripts/merge_datasets.py --datasets "${DS_ARRAY[@]}" --output "$MERGED_DIR"

    if [ $? -ne 0 ]; then
        echo "데이터셋 합치기 실패!"
        exit 1
    fi

    echo ""
    echo ">>> 학습 시작..."
    CUDA_VISIBLE_DEVICES=$GPU eval lerobot-train \
        --dataset.repo_id="$MERGED_REPO" \
        --dataset.root="$MERGED_DIR" \
        --dataset.image_transforms.enable=true \
        --policy.type=$POLICY_TYPE \
        --policy.pretrained_path=$PRETRAINED \
        --policy.push_to_hub=false \
        --policy.dtype=bfloat16 \
        --policy.gradient_checkpointing=true \
        --policy.chunk_size=5 \
        --policy.n_action_steps=1 \
        $PEFT_FLAGS \
        --batch_size=4 \
        --steps="$STEPS" \
        --scheduler.type=cosine_decay_with_warmup \
        --scheduler.peak_lr=2.5e-5 \
        --scheduler.decay_lr=2.5e-6 \
        --scheduler.num_warmup_steps="$WARMUP" \
        --scheduler.num_decay_steps="$DECAY" \
        --save_freq=5000 \
        --output_dir="$OUTPUT" \
        $RESUME_FLAG
fi
