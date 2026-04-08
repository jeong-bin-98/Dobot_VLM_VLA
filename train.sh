#!/bin/bash
# Pi0-FAST 학습 스크립트
#
# 단일 데이터셋:
#   ./train.sh ./cup_dataset_v1 1 1000 outputs/cup_test
#
# 여러 데이터셋 합쳐서:
#   ./train.sh "./cup_dataset_v1 ./block_dataset_v1" 1 1000 outputs/multi_test
#
# 이어서 학습 (resume):
#   ./train.sh ./cup_dataset_v1 1 20000 outputs/cup_test resume

DATASETS="${1:-./dataset_v3}"
GPU="${2:-1}"
STEPS="${3:-100}"
OUTPUT="${4:-outputs/pi0fast_dobot_test}"
RESUME="${5:-}"

# resume 옵션 설정
RESUME_FLAG=""
if [ "$RESUME" = "resume" ] || [ "$RESUME" = "true" ] || [ "$RESUME" = "1" ]; then
    RESUME_FLAG="--resume=true"
    echo "=== 이어서 학습 (resume) 모드 ==="
    echo "  체크포인트 경로: $OUTPUT"
    echo ""
fi

# 데이터셋이 여러 개인지 확인 (공백 구분)
read -ra DS_ARRAY <<< "$DATASETS"

if [ ${#DS_ARRAY[@]} -eq 1 ]; then
    # 단일 데이터셋
    DATASET_NAME=$(basename "${DS_ARRAY[0]}")
    CUDA_VISIBLE_DEVICES=$GPU lerobot-train \
        --dataset.repo_id="local/$DATASET_NAME" \
        --dataset.root="${DS_ARRAY[0]}" \
        --policy.type=pi0_fast \
        --policy.pretrained_path=lerobot/pi0fast-base \
        --policy.push_to_hub=false \
        --policy.dtype=bfloat16 \
        --policy.gradient_checkpointing=true \
        --policy.chunk_size=5 \
        --policy.n_action_steps=1 \
        --peft.method_type=lora \
        --peft.r=16 \
        --peft.target_modules='["q_proj","v_proj","k_proj","o_proj"]' \
        --batch_size=4 \
        --steps="$STEPS" \
        --output_dir="$OUTPUT" \
        $RESUME_FLAG
else
    # 여러 데이터셋 합쳐서 학습 (merge 후 학습)
    MERGED_DIR="./merged_dataset_tmp"
    MERGED_REPO="local/merged_dataset"

    # merge용 인자 구성
    REPO_IDS=""
    ROOTS=""
    for ds in "${DS_ARRAY[@]}"; do
        name=$(basename "$ds")
        if [ -z "$REPO_IDS" ]; then
            REPO_IDS="local/$name"
            ROOTS="$ds"
        else
            REPO_IDS="$REPO_IDS,local/$name"
            ROOTS="$ROOTS,$ds"
        fi
    done

    echo "=== 멀티 데이터셋 학습 ==="
    echo "  데이터셋: ${DS_ARRAY[*]}"
    echo "  GPU: $GPU | Steps: $STEPS"
    echo ""

    # 기존 merged 데이터셋 삭제
    rm -rf "$MERGED_DIR"

    echo ">>> 데이터셋 합치는 중..."
    lerobot-edit-dataset \
        --new_repo_id "$MERGED_REPO" \
        --new_root "$MERGED_DIR" \
        --operation.type merge \
        --operation.repo_ids "[$REPO_IDS]" \
        --operation.roots "[$ROOTS]"

    if [ $? -ne 0 ]; then
        echo "데이터셋 합치기 실패!"
        exit 1
    fi

    echo ""
    echo ">>> 학습 시작..."
    CUDA_VISIBLE_DEVICES=$GPU lerobot-train \
        --dataset.repo_id="$MERGED_REPO" \
        --dataset.root="$MERGED_DIR" \
        --policy.type=pi0_fast \
        --policy.pretrained_path=lerobot/pi0fast-base \
        --policy.push_to_hub=false \
        --policy.dtype=bfloat16 \
        --policy.gradient_checkpointing=true \
        --policy.chunk_size=5 \
        --policy.n_action_steps=1 \
        --peft.method_type=lora \
        --peft.r=16 \
        --peft.target_modules='["q_proj","v_proj","k_proj","o_proj"]' \
        --batch_size=4 \
        --steps="$STEPS" \
        --output_dir="$OUTPUT" \
        $RESUME_FLAG
fi
