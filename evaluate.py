"""Evaluate a merged model on the in-domain and out-of-domain test splits.

    accelerate launch --num_processes 4 evaluate.py --model_path outputs/qwen/final_merged --data_root ../data

Greedy decoding with the training prompt. The prediction is the single
<answer> span, scored by normalized exact match against the gold answer.
Writes <output_dir>/<split>.csv per split and <output_dir>/summary.json.
"""

import argparse
import json
import os
from pathlib import Path
import statistics

# Reuse unsloth_compiled_cache/ instead of regenerating it; set before `import unsloth`.
os.environ.setdefault("UNSLOTH_COMPILE_OVERWRITE", "0")

from unsloth import FastVisionModel  # noqa: E402  (must be imported before transformers)
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
from datasets import Image  # noqa: E402
from torch.utils.data import DataLoader, Dataset, DistributedSampler  # noqa: E402
from tqdm import tqdm  # noqa: E402
from transformers import AutoConfig, AutoModelForImageTextToText, InternVLProcessor, Qwen3VLProcessor  # noqa: E402

from data import load_split, make_conversation  # noqa: E402
from rewards import _predictions, normalize_answer  # noqa: E402

IN_DOMAIN = ("validation", "slake_test", "vqa_rad_test", "pathvqa_test")
OUT_OF_DOMAIN = tuple(f"omnimed_{m}_test" for m in ("ct", "xray", "mri", "us", "oct", "derm", "fundus", "micro"))

MAX_SEQ_LENGTH = 4096
# Image placeholder tokens, never sampled (as in train.py).
QWEN_VISUAL_TOKENS = ["<|vision_start|>", "<|vision_end|>", "<|image_pad|>", "<|video_pad|>"]
INTERNVL_VISUAL_TOKENS = ["<img>", "</img>", "<IMG_CONTEXT>"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True, help="Merged model directory (train.py writes final_merged)")
    p.add_argument("--data_root", required=True, help="Data release with the 512px image cache (prepare_images.py)")
    p.add_argument("--output_dir", default="eval_results")
    p.add_argument("--splits", nargs="+", default=list(IN_DOMAIN + OUT_OF_DOMAIN))
    p.add_argument("--batch_size", type=int, default=16, help="Questions per GPU per batch")
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--limit", type=int, default=None, help="Only the first N questions of each split (quick check)")
    return p.parse_args()


def load_model(model_path):
    """Model, processor (left padding) and the token ids to suppress."""

    model_type = AutoConfig.from_pretrained(model_path).model_type
    device_map = {"": torch.cuda.current_device()}
    if model_type == "qwen3_vl":
        model, _ = FastVisionModel.from_pretrained(
            model_name=model_path, max_seq_length=MAX_SEQ_LENGTH, load_in_4bit=False, device_map=device_map
        )
        FastVisionModel.for_inference(model)
        # AutoProcessor returns a bare tokenizer for unsloth-merged Qwen3-VL directories.
        processor = Qwen3VLProcessor.from_pretrained(model_path)
        visual_tokens = QWEN_VISUAL_TOKENS
    elif model_type == "internvl":
        model = AutoModelForImageTextToText.from_pretrained(
            model_path, dtype=torch.bfloat16, device_map=device_map
        ).eval()
        # Also a bare tokenizer from AutoProcessor: unsloth saves the class as _Unsloth_Patched_InternVLProcessor.
        processor = InternVLProcessor.from_pretrained(model_path)
        visual_tokens = QWEN_VISUAL_TOKENS + INTERNVL_VISUAL_TOKENS
    else:
        raise ValueError(f"{model_path} is {model_type!r}; supported: qwen3_vl, internvl")
    processor.tokenizer.padding_side = "left"
    return model, processor, processor.tokenizer.convert_tokens_to_ids(visual_tokens)


class EvalSplit(Dataset):
    def __init__(self, data_root, split, limit=None):
        dataset = load_split(data_root, split).cast_column("image", Image(decode=True))
        if limit is not None:
            dataset = dataset.select(range(min(limit, len(dataset))))
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        row = self.dataset[index]
        answer = str(row["answer"])
        return {
            "index": index,
            "question_id": row["question_id"],
            "question": row["question"],
            "answer": answer,
            "question_type": "yes_no" if normalize_answer(answer) in {"yes", "no"} else "open",
            "prompt": make_conversation(row, index)["prompt"],
            "image": row["image"],
        }


def generate(model, processor, batch, suppress_tokens, max_new_tokens):
    texts = [processor.apply_chat_template(item["prompt"], tokenize=False, add_generation_prompt=True) for item in batch]
    inputs = processor(
        text=texts, images=[item["image"] for item in batch], return_tensors="pt", padding=True
    ).to(model.device)
    with torch.inference_mode():
        output = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True, suppress_tokens=suppress_tokens
        )
    new_tokens = output[:, inputs["input_ids"].size(1):]
    return processor.batch_decode(new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)


def evaluate_split(model, processor, suppress_tokens, args, split, rank, world_size):
    dataset = EvalSplit(args.data_root, split, args.limit)
    # Each rank takes every world_size-th question; the sampler pads with repeats, removed below.
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False) if world_size > 1 else None
    loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, collate_fn=list, num_workers=4)
    rows = []
    for batch in tqdm(loader, desc=split, disable=rank != 0):
        responses = generate(model, processor, batch, suppress_tokens, args.max_new_tokens)
        for item, response, span in zip(batch, responses, _predictions(responses)):
            prediction = (span or "").strip()
            rows.append({
                "index": item["index"],
                "question_id": item["question_id"],
                "question": item["question"],
                "answer": item["answer"],
                "question_type": item["question_type"],
                "prediction": prediction,
                "correct": bool(prediction) and normalize_answer(prediction) == normalize_answer(item["answer"]),
                "response": response,
            })
    if world_size > 1:
        gathered = [None] * world_size
        dist.all_gather_object(gathered, rows)
        rows = [row for part in gathered for row in part]
    if rank != 0:
        return
    by_index = {row["index"]: row for row in rows}
    table = pd.DataFrame([by_index[i] for i in range(len(dataset))]).drop(columns="index")
    table.to_csv(Path(args.output_dir) / f"{split}.csv", index=False)
    print(f"{split}: {100 * table['correct'].mean():.2f}% of {len(table)} questions", flush=True)


def summarize(output_dir):
    """Accuracy per split, plus pooled / macro / per-question-type accuracy per group."""

    tables = {}
    for split in IN_DOMAIN + OUT_OF_DOMAIN:
        path = Path(output_dir) / f"{split}.csv"
        if path.is_file():
            tables[split] = pd.read_csv(path, keep_default_na=False)
    accuracy = lambda table: 100 * table["correct"].mean()  # noqa: E731
    summary = {split: accuracy(table) for split, table in tables.items()}
    for group, splits in (("in_domain", IN_DOMAIN), ("out_of_domain", OUT_OF_DOMAIN)):
        if not all(split in tables for split in splits):
            continue
        pooled = pd.concat([tables[split] for split in splits])
        yes_no = pooled[pooled["question_type"] == "yes_no"]
        summary[f"{group}_pooled"] = accuracy(pooled)
        summary[f"{group}_macro"] = statistics.mean(summary[split] for split in splits)
        summary[f"{group}_yes_no"] = accuracy(yes_no)
        summary[f"{group}_open"] = accuracy(pooled[pooled["question_type"] == "open"])
        # Mean of the accuracies on gold "yes" and gold "no", which a constant answer cannot inflate.
        gold = yes_no["answer"].str.strip().str.lower()
        summary[f"{group}_yes_no_balanced"] = statistics.mean(accuracy(yes_no[gold == label]) for label in ("yes", "no"))
    # NaN (no question of that kind, e.g. with --limit) is left out.
    summary = {name: round(value, 2) for name, value in summary.items() if value == value}
    Path(output_dir, "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    for name, value in summary.items():
        print(f"{name:32s} {value:6.2f}")


def main():
    args = parse_args()
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    model, processor, suppress_tokens = load_model(args.model_path)
    # Start the process group only after loading: unsloth's loader runs collectives
    # that can deadlock once a group exists.
    if world_size > 1:
        dist.init_process_group("nccl")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    for split in args.splits:
        evaluate_split(model, processor, suppress_tokens, args, split, rank, world_size)
    if rank == 0:
        summarize(args.output_dir)
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
