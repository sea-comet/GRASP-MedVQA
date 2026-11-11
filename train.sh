#!/usr/bin/env bash
# Train with the paper's settings.
#
# Usage: edit the settings below, then run
#     bash train.sh
# Extra arguments are passed on to train.py and override the values below,
# e.g.  bash train.sh --seed 1   (python train.py --help lists all options).

# ------------------------------------------------------------------ settings
MODEL=qwen                 # qwen: Qwen3-VL-8B-Instruct, internvl: InternVL3-8B
NUM_GPUS=4                 # 4 x A100-80GB
BATCH_SIZE=12              # rollouts per GPU (4 GPUs x 12 = 48)
DATA_ROOT=../data          # the unpacked data release
OUTPUT_DIR=outputs/$MODEL  # checkpoints and the final model
# SFT reasoning for wrong rollouts. "own" for both is the paper's method.
HYBRID_WRONG_ROW_CONTEXT=own  # own or correct_sibling: wrong rows of groups with some right rollouts
ALL_WRONG_CONTEXT=own         # own or selfreasoning: open-ended groups with no right rollout
# ---------------------------------------------------------------------------

set -euo pipefail        # stop at the first error
cd "$(dirname "$0")"     # so that relative paths are relative to this folder

if [ "$MODEL" = qwen ]; then
    BASE_MODEL=Qwen/Qwen3-VL-8B-Instruct
elif [ "$MODEL" = internvl ]; then
    BASE_MODEL=OpenGVLab/InternVL3-8B-hf
else
    echo "MODEL must be qwen or internvl" >&2
    exit 1
fi

export UNSLOTH_COMPILE_OVERWRITE=0                       # use unsloth_compiled_cache/ as is
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # reduce GPU memory fragmentation

# Each optimizer step: 48 rollouts x 2 accumulation steps = 12 questions x 8 rollouts.
accelerate launch --num_processes "$NUM_GPUS" train.py \
    --base_model "$BASE_MODEL" \
    --data_root "$DATA_ROOT" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size "$BATCH_SIZE" \
    --num_epochs 2 \
    --learning_rate 2e-5 \
    --gradient_accumulation_steps 2 \
    --num_generations 8 \
    --temperature 1.15 \
    --beta 0.02 \
    --epsilon 3e-4 --epsilon_high 4e-4 \
    --reward_weights 0.5 1.0 0.2 0.4 \
    --sft_lambda_max 0.4 --sft_gamma 1.0 \
    --correction_kappa 1.0 \
    --sft_hybrid_wrong_row_context "$HYBRID_WRONG_ROW_CONTEXT" \
    --sft_all_wrong_context "$ALL_WRONG_CONTEXT" \
    "$@"
