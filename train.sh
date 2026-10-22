#!/usr/bin/env bash
# The paper's configurations. Both use bf16 LoRA (r = 32, vision and language
# layers) and 12 questions x 8 rollouts per optimizer step
# (NUM_GPUS * BATCH_SIZE * GRAD_ACCUM = 96 rollouts).
#
#   MODEL=qwen     Qwen3-VL-8B-Instruct, 2 epochs (4 x A100-80GB, about 20 hours)
#   MODEL=internvl InternVL3-8B, 2 epochs (paper run: 3 x RTX 6000 Ada 48GB, BATCH_SIZE = 16)
#
#   DATA_ROOT=../data OUTPUT_DIR=outputs/qwen MODEL=qwen bash train.sh
#   DATA_ROOT=../data OUTPUT_DIR=outputs/internvl MODEL=internvl NUM_GPUS=3 bash train.sh
#
# Relative paths are resolved from this directory (code/); ../data is the
# unpacked data release.
set -euo pipefail
cd "$(dirname "$0")"

MODEL="${MODEL:-qwen}"
NUM_GPUS="${NUM_GPUS:-4}"
BATCH_SIZE="${BATCH_SIZE:-$((48 / NUM_GPUS))}"
DATA_ROOT="${DATA_ROOT:-../data}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/$MODEL}"

case "$MODEL" in
    qwen)     BASE_MODEL=Qwen/Qwen3-VL-8B-Instruct; NUM_EPOCHS=2 ;;
    internvl) BASE_MODEL=OpenGVLab/InternVL3-8B-hf; NUM_EPOCHS=2 ;;
    *) echo "MODEL must be qwen or internvl" >&2; exit 1 ;;
esac

export UNSLOTH_COMPILE_OVERWRITE=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

accelerate launch --num_processes "$NUM_GPUS" train.py \
    --data_root "$DATA_ROOT" \
    --output_dir "$OUTPUT_DIR" \
    --base_model "$BASE_MODEL" \
    --num_epochs "$NUM_EPOCHS" \
    --learning_rate 2e-5 \
    --batch_size "$BATCH_SIZE" \
    --gradient_accumulation_steps 2 \
    --num_generations 8 \
    --temperature 1.15 \
    --beta 0.02 \
    --epsilon 3e-4 --epsilon_high 4e-4 \
    --reward_weights 0.5 1.0 0.2 0.4 \
    --sft_lambda_max 0.4 --sft_gamma 1.0 \
    --correction_kappa 1.0 \
    "$@"
