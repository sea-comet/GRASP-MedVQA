# GRASP: Group-Adaptive Hybrid RL–SFT Post-Training for Medical VQA without Reasoning Annotations

Wanqi Dong, Wanyue Dong, Jingze Ge, Mehul Motani

Paper: arXiv link coming soon. Data (MedTerse): https://huggingface.co/datasets/seacomet/medterse-vqa

Training and evaluation code for the paper's models: Qwen3-VL-8B-Instruct
(main model) and InternVL3-8B, fine-tuned with LoRA on short-answer medical VQA
with the same objective. Baselines and ablation launchers are not included.
The paper calls the RL branch GSPO-EGC and the state-adaptive correction below
the extreme-group correction (EGC).

## Files

| file | content |
|---|---|
| `train.sh` | launcher with the paper's configurations (`MODEL=qwen` or `internvl`) |
| `train.py` | model, LoRA, data and trainer setup |
| `trainer.py` | the objective, as a subclass of the Unsloth GRPO trainer |
| `internvl.py` | InternVL3 image routing and RoPE setting; the objective is unchanged |
| `objective.py` | the objective's math (SFT weights, correction, normalizers) |
| `rewards.py` | format, correctness, ClinicalBERTScore and BiomedCLIP rewards |
| `format_utils.py` | output format parsing and reasoning repair |
| `data.py`, `prepare_images.py` | data loading and the 512x512 image cache |
| `eval.sh`, `evaluate.py` | evaluation on the in-domain and out-of-domain test splits |

## Objective

For each question b, G = 8 responses `<REASONING> r </REASONING><answer> a </answer>`
are sampled (temperature 1.15) and scored with

    R = 0.5 R_format + 1.0 R_correct + 0.2 R_ClinicalBERTScore + 0.4 R_BiomedCLIP

(`rewards.py`). Let p_b be the fraction of the G answers that are correct.

1. **GSPO.** Group-relative advantages A = (R - mean_group R) / std_batch R, a
   sequence-level importance ratio clipped to [1 - 3e-4, 1 + 4e-4], and a KL
   penalty (beta = 0.02) to the base model.
2. **State-adaptive correction.** In an all-wrong or all-correct group every
   rollout gets the same correctness reward, so A carries no correctness
   contrast (only the format and similarity rewards vary). These groups get

       A_total = A + kappa (s_t / G) (p_b - mu_t),      kappa = 1

   where mu_t is an EMA (decay 0.99) of the batch accuracy and s_t an EMA of
   RMS(A), both updated once per rollout batch. Mixed groups keep A.
3. **Group-adaptive answer SFT.** Every rollout row also gets a cross-entropy
   term on the target `<REASONING> c </REASONING><answer> a* </answer><eos>`,
   where c is the row's own sampled reasoning (repaired to 8-40 words, or
   borrowed from another rollout when unusable) and a* the gold answer. Only
   `<answer> a* </answer><eos>` is supervised. Row weight:

       lambda(p_b) = lambda_max (1 - p_b)^gamma,        lambda_max = 0.4, gamma = 1

4. **Loss.** `L = L_GSPO / EMA(RMS(A_total)) + sum_rows lambda(p_b) NLL_row / (N_tokens * NLL_ref)`,
   where NLL_ref is the mean answer NLL of the first 20 micro-batches (then
   frozen, floor 0.5). The KL coefficient is rescaled so that its effective value stays
   beta.

## Setup

    conda create -n grpo python=3.11 && conda activate grpo
    pip install -r requirements.txt

`import unsloth` writes `unsloth_compiled_cache/UnslothGRPOTrainer.py` into the
working directory, generated from the installed TRL; `trainer.py` subclasses it
without modifying it, so keep the pinned versions.

Models downloaded at first use: `Qwen/Qwen3-VL-8B-Instruct` or
`OpenGVLab/InternVL3-8B-hf`, `microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224`,
`emilyalsentzer/Bio_ClinicalBERT`.

## Data

The data release ([`seacomet/medterse-vqa`](https://huggingface.co/datasets/seacomet/medterse-vqa)
on the Hugging Face Hub) holds `train/qa.jsonl` and the evaluation splits as
text only, plus `build_images.py`, which downloads the images from the original
sources and verifies their hashes. Download it to `data/` next to this
repository (`GRASP-MedVQA/` and `data/` side by side) and build the images:

    huggingface-cli download seacomet/medterse-vqa --repo-type dataset --local-dir data
    cd data && python build_images.py --root .

`--data_root` must then contain `train/qa.jsonl`, one question per line with at
least the fields

    {"question_id": "...", "image_id": "<sha256 of the image file>",
     "image_file": "train/images/<image_id>.png",
     "question": "...", "answer": "short gold answer"}

and the image files under `image_file`. Build the 512x512 cache once, from
this repository:

    python prepare_images.py --root ../data --splits train

## Training

Set the model (`MODEL=qwen` or `MODEL=internvl`) and paths at the top of
`train.sh`, then run from this repository (`train.sh` resolves relative paths
from its own directory)

    bash train.sh

Both models train for 2 epochs on 4 x A100-80GB with 12 rollouts per GPU and 2
accumulation steps: 12 questions x 8 rollouts per optimizer step, 1,074 steps
per epoch on 12,893 questions.

For InternVL3, `internvl.py` passes a one-row-per-image grid through the
upstream chunked forward so that each chunk receives its own images, removes it
before InternVL's forward, and switches the dynamic-NTK RoPE to the default
type (identical below 32,768 tokens). Its `<img>`, `</img>` and `<IMG_CONTEXT>`
tokens are suppressed during sampling, as Qwen's vision tokens are.

Training resumes from the newest checkpoint in
`OUTPUT_DIR`; the running statistics mu_t, s_t and the loss normalizers are
saved with every checkpoint. The LoRA adapter and a merged 16-bit model are
written to `OUTPUT_DIR/final_lora` and `OUTPUT_DIR/final_merged`.

| hyperparameter | value |
|---|---|
| LoRA rank / alpha / dropout | 32 / 32 / 0, vision and language layers |
| optimizer | AdamW 8-bit, lr 2e-5, cosine, warmup 10%, weight decay 0.1, grad-norm clip 0.5 |
| rollouts | G = 8, temperature 1.15, max prompt 512, max completion 256 tokens |
| GSPO | sequence-level ratio, epsilon 3e-4 / 4e-4, beta 0.02, one update per batch |
| answer SFT | lambda_max 0.4, gamma 1, reasoning context 8-40 words, NLL_ref over 20 micro-batches |
| correction | kappa 1, EMA decay 0.99 |
| reward weights | format 0.5, correctness 1.0, ClinicalBERTScore 0.2, BiomedCLIP 0.4 |
| seed | 42 (LoRA initialization, data order, sampling) |

## Evaluation

Set the model and paths at the top of `eval.sh`, then run

    bash eval.sh

It builds the 512x512 images of the test splits (once) and evaluates the model
on the in-domain splits (`validation`, `slake_test`, `vqa_rad_test`,
`pathvqa_test`) and the eight out-of-domain OmniMedVQA-V2 modality splits
(`omnimed_*_test`) on 4 x A100-80GB. Decoding is greedy with the training prompt
and at most 256 new tokens. The prediction is the single `<answer>` span,
correct if it equals the gold answer after normalization (lower case, number
words to digits, whitespace collapsed, punctuation removed).

`OUTPUT_DIR` receives one CSV per split (question, gold, prediction, full
response, correctness) and `summary.json`: accuracy per split and, for each
group, pooled accuracy over all its questions, macro average over its splits,
accuracy on yes/no and on open questions, and balanced yes/no accuracy (mean of
the accuracies on gold "yes" and gold "no").

## License

The code in this directory is released under the MIT License (`LICENSE`).

The Unsloth-generated trainer is not distributed here; it is produced locally
by `import unsloth` and is under the LGPL-3.0-or-later of Unsloth Zoo
(Unsloth itself is Apache-2.0).
