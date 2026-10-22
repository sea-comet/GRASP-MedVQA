"""The four rollout rewards, combined with weights 0.5 / 1.0 / 0.2 / 0.4.

  format        <= 2.5  five 0.5-point format items minus penalties
  correctness   2.0     normalized exact match of the <answer> span
  ClinicalBERT  [0, 1]  BERTScore F1 of answer vs. gold with Bio_ClinicalBERT
  BiomedCLIP    [0, 1]  cosine similarity of the image and the reasoning text

TRL passes every dataset column to the reward functions as a keyword list, so
``answer``, ``image`` and ``sample_id`` arrive aligned with ``completions``;
``rollout_truncated`` is added by the trainer.
"""

from __future__ import annotations

from collections import OrderedDict
import re
import string

import torch

from format_utils import (
    MAX_REASONING_WORDS,
    MIN_REASONING_WORDS,
    REASONING_END,
    REASONING_START,
    REPETITION_NGRAM_SIZE,
    SOLUTION_END,
    SOLUTION_START,
    analyze_completion_format,
)

BIOMEDCLIP_MODEL_NAME = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
CLINICAL_BERT_MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"
_ANSWER_PATTERN = f"{SOLUTION_START}(.*?){SOLUTION_END}"


def _content(completion) -> str:
    if isinstance(completion, list):
        return completion[0].get("content", "") if completion else ""
    return completion or ""


def _predictions(completions) -> list[str | None]:
    """The single <answer> span of each completion, or None."""

    spans = [re.findall(_ANSWER_PATTERN, _content(c), re.DOTALL) for c in completions]
    return [span[0] if len(span) == 1 else None for span in spans]


# --------------------------------------------------------------------------- format
def formatting_reward_func(completions, answer=None, rollout_truncated=None, **kwargs) -> list[float]:
    """0.5 each for: reasoning tags, reasoning length (8-40 words), answer tags,
    answer length (1 word for yes/no, else 1-3), strict ordered format.
    Penalties: repetition -1, truncation -1, empty reasoning block -2,
    degenerate "addCriterion" loop -2."""

    answers = answer or [""] * len(completions)
    truncated = rollout_truncated or [False] * len(completions)
    scores = []
    for index, completion in enumerate(completions):
        text = _content(completion)
        item = analyze_completion_format(
            text,
            truncated=bool(truncated[index]),
            is_close_ended=str(answers[index]).strip().casefold() in {"yes", "no"},
            min_reasoning_words=MIN_REASONING_WORDS,
            max_reasoning_words=MAX_REASONING_WORDS,
            ngram_size=REPETITION_NGRAM_SIZE,
        )
        score = 0.5 * sum(
            bool(item[key])
            for key in (
                "reasoning_tag_valid",
                "reasoning_length_valid",
                "solution_tag_valid",
                "solution_length_valid",
                "strict_full_format_valid",
            )
        )
        if item["repetition_detected"]:
            score -= 1.0
        if item["truncated"]:
            score -= 1.0
        if item["reasoning_tag_valid"] and not item["reasoning_nonempty"]:
            score -= 2.0
        if text:
            removal = text.replace("addCriterion", "").replace("\n", "")
            if (len(text) - len(removal)) / len(text) >= 0.5:
                score -= 2.0
        scores.append(score)
    return scores


# ---------------------------------------------------------------------- correctness
_NUMBER_WORDS = {
    "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
}


def normalize_answer(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"\b(one|two|three|four|five|six|seven|eight|nine|ten)\b",
                  lambda m: _NUMBER_WORDS[m.group(1)], text)
    text = re.sub(r"\s+", " ", text)
    return text.translate(str.maketrans("", "", string.punctuation))


def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    return [
        2.0 if p is not None and normalize_answer(str(a)) == normalize_answer(p) else 0.0
        for p, a in zip(_predictions(completions), answer)
    ]


# --------------------------------------------------------------------- ClinicalBERT
_clinical_bert_state: dict = {}


def clinical_bert_score_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    from bert_score import BERTScorer

    predictions = [(p or "").strip() for p in _predictions(completions)]
    golds = [str(a).strip() for a in answer]
    scores = [0.0] * len(predictions)
    rows = [i for i, p in enumerate(predictions) if p]
    if not rows:
        return scores
    device = f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu"
    if _clinical_bert_state.get("device") != device:
        # Layer 9 of the 12-layer BERT-base encoder.
        _clinical_bert_state["scorer"] = BERTScorer(
            model_type=CLINICAL_BERT_MODEL_NAME, num_layers=9, device=device
        )
        _clinical_bert_state["device"] = device
    _, _, f1 = _clinical_bert_state["scorer"].score(
        [predictions[i] for i in rows], [golds[i] for i in rows]
    )
    for i, value in zip(rows, f1.tolist()):
        scores[i] = max(0.0, float(value))
    return scores


# ----------------------------------------------------------------------- BiomedCLIP
_biomedclip_state: dict = {}
_image_cache: OrderedDict = OrderedDict()
_IMAGE_CACHE_SIZE = 4096


def _load_biomedclip():
    device = torch.device(f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu")
    if _biomedclip_state.get("device") == device:
        return _biomedclip_state
    import open_clip

    model, preprocess = open_clip.create_model_from_pretrained(
        BIOMEDCLIP_MODEL_NAME, device=device, precision="fp16" if device.type == "cuda" else "fp32"
    )
    model.eval().requires_grad_(False)
    _biomedclip_state.update(
        model=model,
        preprocess=preprocess,
        tokenizer=open_clip.get_tokenizer(BIOMEDCLIP_MODEL_NAME),
        device=device,
        dtype=next(model.parameters()).dtype,
    )
    _image_cache.clear()
    return _biomedclip_state


def _reasoning(completion) -> str:
    matches = re.findall(f"{REASONING_START}(.*?){REASONING_END}", _content(completion), re.DOTALL)
    return matches[0].strip() if len(matches) == 1 else ""


def biomedclip_reasoning_reward_func(completions, image=None, images=None, sample_id=None, **kwargs) -> list[float]:
    raw = image if image is not None else images
    unwrap = lambda img: (img[0] if img else None) if isinstance(img, (list, tuple)) else img  # noqa: E731
    images = [unwrap(img) for img in (raw or [None] * len(completions))]
    sample_ids = sample_id or [None] * len(completions)
    texts = [_reasoning(c) for c in completions]
    scores = [0.0] * len(completions)
    rows = [i for i, (t, img) in enumerate(zip(texts, images)) if t and img is not None]
    if not rows:
        return scores

    state = _load_biomedclip()
    model, device = state["model"], state["device"]

    # Every rollout of a group shares one image; encode each image once and
    # keep the features across steps, keyed by the dataset row.
    features, pending, missing = {}, {}, []
    for i in rows:
        key = ("sample", str(sample_ids[i])) if sample_ids[i] is not None else ("object", id(images[i]))
        if key in _image_cache:
            _image_cache.move_to_end(key)
            features[i] = _image_cache[key]
        elif key in pending:
            pending[key].append(i)
        else:
            pending[key] = [i]
            missing.append((key, images[i].convert("RGB")))
    if missing:
        pixels = torch.stack([state["preprocess"](img) for _, img in missing])
        with torch.inference_mode():
            encoded = torch.nn.functional.normalize(
                model.encode_image(pixels.to(device=device, dtype=state["dtype"])).float(), dim=-1
            )
        for (key, _), feature in zip(missing, encoded):
            _image_cache[key] = feature.detach()
            while len(_image_cache) > _IMAGE_CACHE_SIZE:
                _image_cache.popitem(last=False)
            for i in pending[key]:
                features[i] = feature

    tokens = state["tokenizer"]([texts[i] for i in rows])
    tokens = {k: v.to(device) for k, v in tokens.items()} if isinstance(tokens, dict) else tokens.to(device)
    with torch.inference_mode():
        encoded_text = torch.nn.functional.normalize(model.encode_text(tokens).float(), dim=-1)
    similarity = (torch.stack([features[i] for i in rows]) * encoded_text).sum(dim=-1).clamp(0.0, 1.0)
    for i, value in zip(rows, similarity.tolist()):
        scores[i] = float(value)
    return scores


REWARD_FUNCS = [
    formatting_reward_func,
    correctness_reward_func,
    clinical_bert_score_reward_func,
    biomedclip_reasoning_reward_func,
]
