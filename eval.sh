#!/usr/bin/env bash
# Evaluate a trained model on the in-domain and out-of-domain test splits.
#
# Usage: edit the settings below, then run
#     bash eval.sh
# Extra arguments are passed on to evaluate.py, e.g.  bash eval.sh --limit 100
# (python evaluate.py --help lists all options).

# ------------------------------------------------------------------ settings
MODEL_PATH=outputs/qwen/final_merged   # merged model written by train.sh
DATA_ROOT=../data                      # the unpacked data release
OUTPUT_DIR=eval_results/qwen           # per-split CSVs and summary.json
NUM_GPUS=4                             # 4 x A100-80GB
# ---------------------------------------------------------------------------

set -euo pipefail        # stop at the first error
cd "$(dirname "$0")"     # so that relative paths are relative to this folder

IN_DOMAIN="validation slake_test vqa_rad_test pathvqa_test"
OUT_OF_DOMAIN="omnimed_ct_test omnimed_xray_test omnimed_mri_test omnimed_us_test
               omnimed_oct_test omnimed_derm_test omnimed_fundus_test omnimed_micro_test"

# 512x512 images, as in training; images already converted are skipped.
python prepare_images.py --root "$DATA_ROOT" --splits $IN_DOMAIN $OUT_OF_DOMAIN

accelerate launch --num_processes "$NUM_GPUS" evaluate.py \
    --model_path "$MODEL_PATH" \
    --data_root "$DATA_ROOT" \
    --output_dir "$OUTPUT_DIR" \
    --splits $IN_DOMAIN $OUT_OF_DOMAIN \
    "$@"
