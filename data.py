"""Training data: one ``qa.jsonl`` per split plus a 512x512 image cache.

Expected layout (see README):

    <root>/train/qa.jsonl            one JSON object per question:
                                     {"question_id", "image_id", "image_file",
                                      "question", "answer", ...}
    <root>/_training_images_512/     <image_id>.png, built by prepare_images.py
"""

from __future__ import annotations

from pathlib import Path

from format_utils import (
    MAX_REASONING_WORDS,
    MIN_REASONING_WORDS,
    REASONING_END,
    REASONING_START,
    SOLUTION_END,
    SOLUTION_START,
)

CACHE_DIRNAME = "_training_images_512"


def load_split(root: str | Path, split: str = "train"):
    """The split with an ``image`` column pointing at the 512px cache (not decoded)."""

    from datasets import Image, load_dataset

    root = Path(root).resolve()
    dataset = load_dataset("json", data_files=str(root / split / "qa.jsonl"), split="train")
    paths = [str(root / CACHE_DIRNAME / f"{image_id}.png") for image_id in dataset["image_id"]]
    missing = [path for path in paths if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} cached images are missing (first: {missing[0]}); run prepare_images.py"
        )
    return dataset.add_column("image", paths).cast_column("image", Image(decode=False))


def make_conversation(example, idx):
    text = (
        f"{example['question']}. Also first provide your reasoning in "
        f"{MIN_REASONING_WORDS} to {MAX_REASONING_WORDS} words"
        f" on how you solve the question between {REASONING_START} and {REASONING_END}"
        f" and then your final answer between {SOLUTION_START} and (put a single yes or no for close-ended question"
        f" and one to three words for open-ended questions) {SOLUTION_END}"
    )
    return {
        "prompt": [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]},
        ],
        "image": example["image"],
        "answer": example["answer"],
        # Row index; the BiomedCLIP reward caches image features under it.
        "sample_id": idx,
    }
