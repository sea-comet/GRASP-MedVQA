"""Train Qwen3-VL-8B-Instruct or InternVL3-8B (bf16 LoRA) with the hybrid GSPO + answer-SFT objective.

    accelerate launch --num_processes 4 train.py --data_root ../data --output_dir outputs/run
    accelerate launch --num_processes 4 train.py --base_model OpenGVLab/InternVL3-8B-hf ...

See train.sh for the configurations used in the paper.
"""

import argparse
import os

# Use the Unsloth-generated trainer as it is on disk; must precede `import unsloth`.
os.environ.setdefault("UNSLOTH_COMPILE_OVERWRITE", "0")

from unsloth import FastVisionModel  # noqa: E402  (must be imported before trl / transformers)
import torch  # noqa: E402
from datasets import Image  # noqa: E402
from transformers import AutoConfig  # noqa: E402
from transformers.trainer_utils import get_last_checkpoint  # noqa: E402

from data import load_split, make_conversation  # noqa: E402
from rewards import REWARD_FUNCS  # noqa: E402
from trainer import GRPOConfig, HybridGRPOTrainer, configure_method  # noqa: E402

# Rollout lengths vary, and every new shape recompiles the fused GRPO loss
# (compiled with fullgraph=True); the default limits stop a multi-epoch run.
torch._dynamo.config.recompile_limit = 4096
torch._dynamo.config.accumulated_recompile_limit = 65536

MAX_SEQ_LENGTH = 4096

# Image placeholder tokens. A generated one would misalign the image features
# in the next forward pass, so they are never sampled. InternVL3's language
# model shares Qwen's vocabulary and adds its own three.
QWEN_VISUAL_TOKENS = ["<|vision_start|>", "<|vision_end|>", "<|image_pad|>", "<|video_pad|>"]
INTERNVL_VISUAL_TOKENS = ["<img>", "</img>", "<IMG_CONTEXT>"]
SUPPORTED_MODEL_TYPES = ("qwen3_vl", "internvl")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", required=True, help="Directory holding train/qa.jsonl and the 512px image cache")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--base_model", default="Qwen/Qwen3-VL-8B-Instruct",
                   help="Qwen/Qwen3-VL-8B-Instruct or OpenGVLab/InternVL3-8B-hf")
    # Optimization
    p.add_argument("--learning_rate", type=float, default=2e-5)
    p.add_argument("--num_epochs", type=int, default=2)
    p.add_argument("--batch_size", type=int, default=12, help="Rollouts per GPU per micro-step")
    p.add_argument("--gradient_accumulation_steps", type=int, default=2)
    p.add_argument("--max_grad_norm", type=float, default=0.5)
    p.add_argument("--lora_rank", type=int, default=32)
    p.add_argument("--seed", type=int, default=42, help="LoRA initialization, data order and sampling")
    # GSPO
    p.add_argument("--num_generations", type=int, default=8)
    p.add_argument("--temperature", type=float, default=1.15)
    p.add_argument("--beta", type=float, default=0.02, help="KL coefficient to the base model")
    p.add_argument("--epsilon", type=float, default=3e-4)
    p.add_argument("--epsilon_high", type=float, default=4e-4)
    p.add_argument("--max_prompt_length", type=int, default=512)
    p.add_argument("--max_completion_length", type=int, default=256)
    p.add_argument("--reward_weights", type=float, nargs=4, default=[0.5, 1.0, 0.2, 0.4],
                   help="format, correctness, ClinicalBERTScore, BiomedCLIP")
    # Answer SFT and correction
    p.add_argument("--sft_lambda_max", type=float, default=0.4)
    p.add_argument("--sft_gamma", type=float, default=1.0)
    p.add_argument("--correction_kappa", type=float, default=1.0)
    p.add_argument("--sft_target_template", default="<answer>{answer}</answer>")
    # Bookkeeping
    p.add_argument("--save_steps", type=int, default=100)
    p.add_argument("--save_total_limit", type=int, default=3)
    p.add_argument("--report_to", default="none", help="e.g. wandb")
    return p.parse_args()


def main():
    args = parse_args()
    model_type = AutoConfig.from_pretrained(args.base_model).model_type
    if model_type not in SUPPORTED_MODEL_TYPES:
        raise ValueError(f"{args.base_model} is {model_type!r}; supported: {SUPPORTED_MODEL_TYPES}")
    internvl = model_type == "internvl"
    if internvl:
        from internvl import InternVLHybridGRPOTrainer as Trainer, use_default_rope
    else:
        Trainer = HybridGRPOTrainer
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    model, tokenizer = FastVisionModel.from_pretrained(
        model_name=args.base_model,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=torch.bfloat16,
        load_in_4bit=False,
        load_in_16bit=True,
        fast_inference=False,
        device_map={"": torch.cuda.current_device()},
    )
    if internvl:
        use_default_rope(model, MAX_SEQ_LENGTH)
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=True,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=args.lora_rank,
        lora_alpha=args.lora_rank,
        lora_dropout=0,
        bias="none",
        random_state=args.seed,
        use_rslora=False,
        loftq_config=None,
        use_gradient_checkpointing="unsloth",
    )

    # Map the text with undecoded image paths, then switch to lazy decoding.
    train_dataset = load_split(args.data_root, "train")
    train_dataset = train_dataset.map(make_conversation, with_indices=True, num_proc=8, load_from_cache_file=False)
    train_dataset = train_dataset.cast_column("image", Image(decode=True))

    visual_tokens = QWEN_VISUAL_TOKENS + (INTERNVL_VISUAL_TOKENS if internvl else [])
    suppress = tokenizer.tokenizer.convert_tokens_to_ids(visual_tokens)
    if tokenizer.tokenizer.convert_ids_to_tokens(suppress) != visual_tokens:
        raise ValueError(f"Visual control tokens missing from the vocabulary: {visual_tokens}")
    config = GRPOConfig(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        optim="adamw_8bit",
        bf16=True,
        logging_steps=1,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_epochs,
        max_grad_norm=args.max_grad_norm,
        num_generations=args.num_generations,
        temperature=args.temperature,
        generation_kwargs={"suppress_tokens": suppress},
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        # GSPO: sequence-level importance ratio, one update per rollout batch.
        importance_sampling_level="sequence",
        loss_type="grpo",
        steps_per_generation=1,
        epsilon=args.epsilon,
        epsilon_high=args.epsilon_high,
        beta=args.beta,
        scale_rewards="batch",
        reward_weights=args.reward_weights,
        mask_truncated_completions=False,
        log_completions=False,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        report_to=args.report_to,
        seed=args.seed,
        data_seed=args.seed,
    )
    configure_method(
        config,
        sft_lambda_max=args.sft_lambda_max,
        sft_gamma=args.sft_gamma,
        correction_kappa=args.correction_kappa,
        sft_target_template=args.sft_target_template,
    )

    trainer = Trainer(
        model=model,
        args=config,
        processing_class=tokenizer,
        reward_funcs=REWARD_FUNCS,
        train_dataset=train_dataset,
    )
    last = get_last_checkpoint(args.output_dir) if os.path.isdir(args.output_dir) else None
    trainer.train(resume_from_checkpoint=last)

    if trainer.is_world_process_zero():
        adapter_dir = os.path.join(args.output_dir, "final_lora")
        model.save_pretrained(adapter_dir)
        tokenizer.save_pretrained(adapter_dir)
        model.save_pretrained_merged(os.path.join(args.output_dir, "final_merged"), tokenizer,
                                     save_method="merged_16bit")
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()


if __name__ == "__main__":
    main()
