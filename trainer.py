"""GSPO + group-adaptive answer SFT + state-adaptive advantage correction.

Subclasses the GRPO trainer that unsloth 2026.2.1 generates for trl 0.24.0
(``unsloth_compiled_cache/UnslothGRPOTrainer.py``, not modified). Per optimizer step:

1. Sample G rollouts per question and score them (rewards.py); upstream turns
   the weighted rewards into group-relative advantages A.
2. Correction: all-wrong and all-correct groups have no correctness contrast,
   so they get A_total = A + kappa * g_b (objective.py); other groups keep A.
3. GSPO loss on A_total, divided by an EMA of RMS(A_total).
4. Answer SFT on the same rows. Target:
   <REASONING> c </REASONING><answer> gold </answer><eos>, where c is the
   rollout's reasoning (borrowed from another rollout if it has none).
   Cross-entropy on the answer and EOS only, weighted per row by lambda(p_b)
   and divided by a reference NLL that is frozen after the first steps.

Two options choose c for wrong rollouts, whose own reasoning usually argues for
another answer. Both are off by default, which is the paper's method. They change
only c, which is not supervised:
- sft_mixed_wrong_row_context="correct_sibling": in a mixed group (some rollouts
  right, some wrong), a wrong row uses the reasoning of a random correct rollout.
- sft_all_wrong_context="selfreasoning": in an open-ended all-wrong group, the policy
  is told the gold answer and writes what it sees; every row uses that text.
"""

from __future__ import annotations

from collections import deque
from contextlib import nullcontext
import copy
import json
import os
from pathlib import Path
import random
from typing import Any

from accelerate.utils import gather_object
import torch
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence

import unsloth_compiled_cache.UnslothGRPOTrainer as _upstream

from data import make_selfreasoning_prompt
from format_utils import (
    MAX_REASONING_WORDS,
    MIN_REASONING_WORDS,
    REASONING_END,
    REASONING_START,
    REPETITION_NGRAM_SIZE,
    SOLUTION_END,
    SOLUTION_START,
    selfreasoning_from_reply,
    reasoning_context_from_completion,
)
from objective import (
    ReferenceScale,
    compensated_kl_beta,
    group_flags,
    sft_row_weights,
    state_adaptive_correction,
    update_ema,
    update_positive_ema,
    update_reference_scale,
)
from rewards import normalize_answer

GRPOConfig = _upstream.UnslothGRPOConfig

METHOD_DEFAULTS = {
    # lambda(p) = sft_lambda_max * (1 - p)^sft_gamma
    "sft_lambda_max": 0.4,
    "sft_gamma": 1.0,
    # A_total = A + correction_kappa * g_b
    "correction_kappa": 1.0,
    "sft_target_template": f"{SOLUTION_START}{{answer}}{SOLUTION_END}",
    "sft_max_completion_length": 128,
    "sft_min_reasoning_words": MIN_REASONING_WORDS,
    "sft_max_reasoning_words": MAX_REASONING_WORDS,
    "sft_repetition_ngram_size": REPETITION_NGRAM_SIZE,
    # c of the wrong rows of a mixed group (some rollouts right, some wrong):
    # "own" keeps their reasoning, "correct_sibling" uses a correct rollout's.
    "sft_mixed_wrong_row_context": "own",
    # c of an open-ended all-wrong group: "own" keeps each row's reasoning,
    # "selfreasoning" uses a text the policy writes when told the gold answer.
    "sft_all_wrong_context": "own",
    "selfreasoning_temperature": 0.7,
    # Recent usable reasonings, kept across steps: the SFT context of last
    # resort when no rollout in the batch has one.
    "sft_donor_bank_size": 256,
    # SFT normalizer: mean NLL of the first N micro-batches, then frozen; at least the floor.
    "sft_reference_steps": 20,
    "sft_normalizer_floor": 0.5,
    # Decay of the EMAs for mu_t, s_t and the GSPO normalizer.
    "ema_decay": 0.99,
    "ema_epsilon": 1e-4,
    "correctness_reward_name": "correctness_reward_func",
    # Pad rollouts to a multiple of this length, equal on all ranks, to limit recompiles.
    "rollout_length_bucket_size": 32,
}


def configure_method(config: Any, **overrides: Any) -> Any:
    unknown = sorted(set(overrides) - set(METHOD_DEFAULTS))
    if unknown:
        raise TypeError(f"Unknown options: {unknown}")
    values = {**METHOD_DEFAULTS, **overrides}
    if "{answer}" not in str(values["sft_target_template"]):
        raise ValueError("sft_target_template must contain {answer}.")
    if float(values["sft_gamma"]) <= 0 or float(values["correction_kappa"]) <= 0:
        raise ValueError("sft_gamma and correction_kappa must be positive.")
    if values["sft_mixed_wrong_row_context"] not in ("own", "correct_sibling"):
        raise ValueError('sft_mixed_wrong_row_context must be "own" or "correct_sibling".')
    if values["sft_all_wrong_context"] not in ("own", "selfreasoning"):
        raise ValueError('sft_all_wrong_context must be "own" or "selfreasoning".')
    if float(values["selfreasoning_temperature"]) <= 0:
        raise ValueError("selfreasoning_temperature must be positive.")
    for name, value in values.items():
        setattr(config, name, value)
    return config


# Patches to the generated module (its file stays unchanged):
# - the loss functions multiply the loss by the GSPO normalizer. The fused loss
#   computes its gradient in the forward pass, so scaling its return value
#   would not change the gradient;
# - pad rounds rollout lengths up to a bucket.
_ORIGINAL_COMPUTE_LOSS = _upstream.grpo_compute_loss
_ORIGINAL_ACCUMULATED_LOSS = _upstream.grpo_accumulated_loss
_ORIGINAL_PAD = _upstream.pad
_ACTIVE_CONTEXT: dict[str, Any] | None = None
_ACTIVE_BUCKET_SIZE = 0


def _bucketed_pad(tensors, padding_value=0, padding_side="right"):
    """Upstream pad, then pad further to a multiple of the bucket size (max length over ranks)."""

    padded = _ORIGINAL_PAD(tensors, padding_value=padding_value, padding_side=padding_side)
    bucket = int(_ACTIVE_BUCKET_SIZE)
    if bucket <= 1 or padded.ndim == 0:
        return padded
    length = torch.tensor(padded.size(-1), dtype=torch.long, device=padded.device)
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(length, op=torch.distributed.ReduceOp.MAX)
    extra = (int(length.item()) + bucket - 1) // bucket * bucket - padded.size(-1)
    if extra <= 0:
        return padded
    return F.pad(padded, (extra, 0) if padding_side == "left" else (0, extra), value=padding_value)


def _scaled_compute_loss(ref, new, old, sampling_per_token_logps, input_ids, mask, beta, advantages, **kwargs):
    result = _ORIGINAL_COMPUTE_LOSS(
        ref, new, old, sampling_per_token_logps, input_ids, mask, beta, advantages, **kwargs
    )
    scale = kwargs.get("gspo_loss_scale", 1.0)
    if isinstance(scale, torch.Tensor):
        scale = scale.to(device=result[0].device, dtype=result[0].dtype)
    return (result[0] * scale, *result[1:])


# Both upstream loss paths (fused and slow) receive gspo_loss_scale, set in compute_loss.
def _accumulated_loss(*args, **kwargs):
    trainer = kwargs.get("trainer") or (args[0] if args else None)
    context = getattr(trainer, "_objective_context", None)
    if context:
        kwargs.update(context)
    return _ORIGINAL_ACCUMULATED_LOSS(*args, **kwargs)


def _compute_loss_slow(*args, **kwargs):
    if _ACTIVE_CONTEXT:
        kwargs.update(_ACTIVE_CONTEXT)
    return _scaled_compute_loss(*args, **kwargs)


_upstream.grpo_compute_loss = _scaled_compute_loss
_upstream.grpo_compute_loss_slow = _compute_loss_slow
_upstream.grpo_accumulated_loss = _accumulated_loss
_upstream.pad = _bucketed_pad


class HybridGRPOTrainer(_upstream.UnslothGRPOTrainer):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        for name, default in METHOD_DEFAULTS.items():
            setattr(self, name, getattr(self.args, name, default))
        if self.correctness_reward_name not in self.reward_func_names:
            raise ValueError(f"Missing reward {self.correctness_reward_name!r}: {self.reward_func_names}")
        self._gspo_ema: float | None = None
        self._sft_reference: ReferenceScale | None = None
        # mu_t and s_t; saved in checkpoints so a resumed run continues them.
        self._correction_state: dict[str, Any] = {"mu": None, "scale": None}
        self._rollout_cache: dict[str, Any] | None = None
        self._objective_context: dict[str, Any] | None = None
        self._donor_bank: deque[str] = deque(maxlen=max(int(self.sft_donor_bank_size), 1))

    # ------------------------------------------------------------------ utils
    @staticmethod
    def _all_reduce_sum(values: torch.Tensor) -> torch.Tensor:
        values = values.clone()
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(values, op=torch.distributed.ReduceOp.SUM)
        return values

    def _global_rms(self, values: torch.Tensor) -> float:
        values = values.detach().float()
        stats = self._all_reduce_sum(torch.stack([values.square().sum(), values.new_tensor(float(values.numel()))]))
        return torch.sqrt(stats[0] / stats[1].clamp(min=1.0)).item()

    def _global_fraction(self, count: int, total: int, device) -> float:
        stats = self._all_reduce_sum(torch.tensor([float(count), float(total)], device=device))
        return (stats[0] / stats[1].clamp(min=1.0)).item()

    def _log(self, mode: str, name: str, value: float) -> None:
        self._metrics[mode][name].append(float(value))

    # -------------------------------------------------------------- rollouts
    def _calculate_rewards(self, inputs, prompts, completions, completion_ids_list):
        eos_ids = self.eos_token_id if isinstance(self.eos_token_id, (list, tuple, set)) else [self.eos_token_id]
        terminal = {int(t) for t in [*eos_ids, self.pad_token_id] if t is not None}
        truncated = [len(ids) > 0 and int(ids[-1]) not in terminal for ids in completion_ids_list]
        for row, flag in zip(inputs, truncated):
            row["rollout_truncated"] = flag  # read by the format reward
        try:
            rewards = super()._calculate_rewards(inputs, prompts, completions, completion_ids_list)
        finally:
            for row in inputs:
                row.pop("rollout_truncated", None)

        # `rewards` is gathered over all ranks: [num_rollouts_global, num_rewards].
        values = rewards[:, self.reward_func_names.index(self.correctness_reward_name)]
        valid = torch.isfinite(values)
        is_correct = valid & values.gt(0.0)
        flags = group_flags(is_correct, valid, num_generations=int(self.num_generations))
        start = self.accelerator.process_index * len(completions)
        local = slice(start, start + len(completions))
        completion_texts = [
            c[0].get("content", "") if isinstance(c, list) and c else str(c or "") for c in completions
        ]
        self._rollout_cache = {
            "group_all_wrong": flags.all_wrong[local].clone(),
            "group_all_correct": flags.all_correct[local].clone(),
            "group_valid": flags.valid[local].clone(),
            "group_correct_fraction": flags.correct_fraction[local].clone(),
            "is_correct": is_correct[local].clone(),
            "completion_texts": completion_texts,
            "process_start": start,
            "batch_accuracy": is_correct[valid].float().mean().item() if bool(valid.any()) else None,
        }
        if self.sft_mixed_wrong_row_context == "correct_sibling":
            # A group's rollouts can sit on several GPUs.
            self._rollout_cache["global_completion_texts"] = gather_object(completion_texts)
            self._rollout_cache["global_is_correct"] = is_correct.tolist()
        mode = "train" if self.model.training else "eval"
        per_group = lambda t: t.view(-1, int(self.num_generations))[:, 0].float().mean().item()  # noqa: E731
        self._log(mode, "groups/all_wrong", per_group(flags.all_wrong))
        self._log(mode, "groups/all_correct", per_group(flags.all_correct))
        self._log(mode, "groups/mean_correct_fraction", flags.correct_fraction.mean().item())
        return rewards

    def _generate_and_score_completions(self, inputs):
        global _ACTIVE_BUCKET_SIZE
        previous, _ACTIVE_BUCKET_SIZE = _ACTIVE_BUCKET_SIZE, int(self.rollout_length_bucket_size)
        try:
            output = super()._generate_and_score_completions(inputs)
        finally:
            _ACTIVE_BUCKET_SIZE = previous
        cache, self._rollout_cache = self._rollout_cache, None
        if cache is None:
            raise RuntimeError("Rollout state was not produced together with the rewards.")
        for key in ("group_all_wrong", "group_all_correct", "group_valid", "group_correct_fraction"):
            output[key] = cache[key]

        # Shift the normalized advantages: added to raw rewards, group centering would remove it.
        relative = output["advantages"]
        shift = self._state_adaptive_shift(output, relative, cache["batch_accuracy"])
        output["relative_advantages"] = relative.clone()
        output["total_advantages"] = relative + shift.to(device=relative.device, dtype=relative.dtype)
        output["advantages"] = output["total_advantages"]

        # Outside the bucketed-pad block above: only some GPUs generate here, and
        # the bucketed pad would wait for the others forever.
        selfreasoning = None
        if self.sft_all_wrong_context == "selfreasoning" and self.model.training:
            selfreasoning = self._selfreasoning_contexts(inputs, cache)
        ids, attention, loss_mask = self._build_sft_targets(inputs, cache, selfreasoning, output["completion_ids"].device)
        output["sft_completion_ids"] = ids
        output["sft_completion_mask"] = attention
        output["sft_loss_mask"] = loss_mask
        return output

    def _state_adaptive_shift(self, output, relative: torch.Tensor, batch_accuracy) -> torch.Tensor:
        """kappa * g_b for this rank's rows; in training, also updates mu_t and s_t.

        s_t is an EMA rather than this batch's RMS: a batch with few mixed groups
        has a near-zero RMS, which would cancel the correction when it matters most.
        """

        mode = "train" if self.model.training else "eval"
        state = self._correction_state
        observed = self._global_rms(relative)  # all-reduce: must run on every rank
        if mode == "train":
            if batch_accuracy is not None:
                state["mu"] = update_ema(state["mu"], batch_accuracy, decay=self.ema_decay)
            state["scale"] = update_ema(state["scale"], observed, decay=self.ema_decay)
        mu = state["mu"] if state["mu"] is not None else (batch_accuracy or 0.0)
        mu = min(max(float(mu), 0.0), 1.0)
        scale = float(state["scale"] if state["scale"] is not None else observed)
        self._log(mode, "correction/mu", mu)
        self._log(mode, "correction/scale", scale)
        return state_adaptive_correction(
            output["group_all_wrong"],
            output["group_all_correct"],
            output["group_valid"],
            mu=mu,
            scale=scale * float(self.correction_kappa),
            num_generations=int(self.num_generations),
        )

    # ------------------------------------------------------------ SFT target
    def _assign_reasoning_contexts(self, contexts, is_correct):
        """Reasoning used as (unsupervised) SFT context for each row.

        The row's own reasoning if usable; otherwise one borrowed from its group,
        then from the batch (correct rollouts preferred in both), then from the
        cross-step bank; otherwise empty.
        """

        count = len(contexts)
        group_size = int(self.num_generations) if count % int(self.num_generations) == 0 else max(count, 1)
        correct = [bool(v) for v in is_correct.tolist()][:count] + [False] * max(0, count - len(is_correct))
        usable = [i for i, c in enumerate(contexts) if c.usable]

        def choose(pool, row):
            if not pool:
                return None
            preferred = [i for i in pool if correct[i]] or pool
            return contexts[preferred[row % len(preferred)]].text

        texts = []
        for index, context in enumerate(contexts):
            if context.usable:
                texts.append(context.text)
                continue
            group = index // group_size
            donor = choose([i for i in usable if i // group_size == group and i != index], index)
            donor = donor or choose([i for i in usable if i != index], index)
            if not donor and self._donor_bank:
                bank = list(self._donor_bank)
                donor = bank[index % len(bank)]
            texts.append(donor or "")
        for index in usable:
            self._donor_bank.append(contexts[index].text)
        return texts

    def _parsed_reasoning(self, text):
        return reasoning_context_from_completion(
            text,
            max_words=int(self.sft_max_reasoning_words),
            min_words=int(self.sft_min_reasoning_words),
            ngram_size=int(self.sft_repetition_ngram_size),
        )

    def _borrow_correct_sibling_contexts(self, texts, cache):
        """Give each wrong row of a mixed group the reasoning of a random correct
        rollout of its group, in place. Returns how many rows changed.

        The gold answer then follows reasoning that led to it. A row keeps its context
        if its group has no correct rollout with usable reasoning (all-wrong groups).
        The random choice depends only on the step and the row, not on the number of GPUs.
        """

        size = int(self.num_generations)
        start = cache["process_start"]
        step = int(self.state.global_step)
        correct = cache["global_is_correct"]
        parsed = {}
        borrowed = 0
        for index in range(len(texts)):
            row = start + index
            if correct[row]:
                continue
            candidates = []
            first = row // size * size
            for sibling in range(first, first + size):
                if not correct[sibling]:
                    continue
                if sibling not in parsed:
                    parsed[sibling] = self._parsed_reasoning(cache["global_completion_texts"][sibling])
                if parsed[sibling].usable:
                    candidates.append(parsed[sibling].text)
            if candidates:
                texts[index] = random.Random(f"{step}:{row}").choice(candidates)
                borrowed += 1
        return borrowed

    def _generate_selfreasoning_replies(self, inputs, rows):
        """Ask the current policy for one reply per row in ``rows``, without gradient."""

        if not rows:
            return []
        prompts = [
            make_selfreasoning_prompt(inputs[i]["prompt"], str(inputs[i]["question"]).strip(), str(inputs[i]["answer"]).strip())
            for i in rows
        ]
        images = [[inputs[i]["image"]] for i in rows]
        previous = self.generation_config
        self.generation_config = copy.deepcopy(previous)
        self.generation_config.temperature = float(self.selfreasoning_temperature)
        try:
            _, completion_ids, _, _ = self._generate_single_turn(prompts, images)
        finally:
            self.generation_config = previous
        tokenizer = getattr(self.processing_class, "tokenizer", self.processing_class)
        return tokenizer.batch_decode(completion_ids, skip_special_tokens=True)

    def _selfreasoning_contexts(self, inputs, cache):
        """{local row: text} for the open-ended all-wrong groups.

        The policy is told the gold answer and writes what it sees in the image. The text
        is only context: the loss stays on the answer, so the policy is not taught to write
        it, even if it invents a finding.

        Yes/no groups keep their own reasoning, because the text often contradicts a forced
        "yes" or "no". A reply shorter than sft_min_reasoning_words is dropped, and its group
        keeps its own reasoning too.

        A group can sit on two GPUs. The GPU holding its first row writes the text and shares
        it, so all rows of a group get the same text. Replies are saved to
        selfreasoning.rank<N>.jsonl, to check what they claim to see.
        """

        size = int(self.num_generations)
        start = cache["process_start"]
        owned = [
            index
            for index, row in enumerate(inputs)
            if (start + index) % size == 0
            and bool(cache["group_all_wrong"][index])  # already false for invalid groups
            and normalize_answer(str(row["answer"])) not in {"yes", "no"}
        ]
        replies = self._generate_selfreasoning_replies(inputs, owned)
        contexts = [selfreasoning_from_reply(reply, min_words=int(self.sft_min_reasoning_words)) for reply in replies]
        if owned:
            rank = self.accelerator.process_index
            path = Path(self.args.output_dir) / f"selfreasoning.rank{rank}.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                for index, reply, context in zip(owned, replies, contexts):
                    handle.write(json.dumps({
                        "step": int(self.state.global_step),
                        "question": inputs[index]["question"],
                        "gold": inputs[index]["answer"],
                        "reply": reply,
                        "context": context,
                    }, ensure_ascii=False) + "\n")

        results = gather_object([((start + index) // size, context) for index, context in zip(owned, contexts)])
        self._log("train", "sft/selfreasoning_groups", len(results))
        by_group = {group: context for group, context in results if context is not None}
        return {
            index: by_group[(start + index) // size]
            for index in range(len(inputs))
            if (start + index) // size in by_group
        }

    def _build_sft_targets(self, inputs, cache, selfreasoning, device):
        """SFT token ids, attention mask and loss mask, padded to sft_max_completion_length."""

        tokenizer = getattr(self.processing_class, "tokenizer", self.processing_class)
        contexts = [self._parsed_reasoning(text) for text in cache["completion_texts"]]
        reasoning_texts = self._assign_reasoning_contexts(contexts, cache["is_correct"])
        mode = "train" if self.model.training else "eval"
        count = len(reasoning_texts)
        if self.sft_mixed_wrong_row_context == "correct_sibling":
            borrowed = self._borrow_correct_sibling_contexts(reasoning_texts, cache)
            self._log(mode, "sft/sibling_row_fraction", self._global_fraction(borrowed, count, device))
        if selfreasoning is not None:
            for index, text in selfreasoning.items():
                reasoning_texts[index] = text
            self._log(mode, "sft/selfreasoning_row_fraction", self._global_fraction(len(selfreasoning), count, device))
        solutions = [str(self.sft_target_template).format(answer=str(row["answer"]).strip()) for row in inputs]

        encode = lambda texts: tokenizer(texts, add_special_tokens=False, padding=False)["input_ids"]  # noqa: E731
        opening = list(encode(REASONING_START))
        closing = list(encode(REASONING_END))
        eos = self.eos_token_id[0] if isinstance(self.eos_token_id, (list, tuple)) else self.eos_token_id
        eos_ids = [] if eos is None else [int(eos)]
        max_length = int(self.sft_max_completion_length)

        rows, attention, losses = [], [], []
        for reasoning, solution in zip(encode(reasoning_texts), encode(solutions)):
            fixed = len(opening) + len(closing) + len(solution) + len(eos_ids)
            if not solution or fixed > max_length:
                raise ValueError(f"SFT solution is empty or longer than sft_max_completion_length={max_length}.")
            reasoning = list(reasoning)[: max_length - fixed]
            ids = opening + reasoning + closing + list(solution) + eos_ids
            # Loss on the answer and EOS only; the reasoning and its tags are context.
            mask = [0] * (len(opening) + len(reasoning) + len(closing)) + [1] * (len(solution) + len(eos_ids))
            rows.append(torch.tensor(ids, dtype=torch.long, device=device))
            attention.append(torch.ones(len(ids), dtype=torch.long, device=device))
            losses.append(torch.tensor(mask, dtype=torch.long, device=device))

        ids = pad_sequence(rows, batch_first=True, padding_value=self.pad_token_id)
        attention = pad_sequence(attention, batch_first=True, padding_value=0)
        losses = pad_sequence(losses, batch_first=True, padding_value=0)
        extra = max_length - ids.size(1)
        if extra:
            ids = F.pad(ids, (0, extra), value=self.pad_token_id)
            attention = F.pad(attention, (0, extra), value=0)
            losses = F.pad(losses, (0, extra), value=0)
        return ids, attention, losses

    def _sft_row_nll(self, model, inputs):
        """Returns (per-row NLL summed over supervised tokens, gradient scale,
        mean NLL per supervised token over all ranks).

        Row sums times the gradient scale give a per-token mean over all ranks
        once DDP averages the gradients.
        """

        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        target_ids = inputs["sft_completion_ids"]
        length = target_ids.size(1)
        model_inputs = {
            "input_ids": torch.cat([prompt_ids, target_ids], dim=1),
            "attention_mask": torch.cat([prompt_mask, inputs["sft_completion_mask"]], dim=1),
            "use_cache": False,
        }
        for key in ("pixel_values", "image_grid_thw", "pixel_attention_mask", "image_sizes"):
            if inputs.get(key) is not None:
                model_inputs[key] = inputs[key]
        if "logits_to_keep" in getattr(self, "model_kwarg_keys", set()):
            model_inputs["logits_to_keep"] = length + 1

        previous = os.environ.get("UNSLOTH_RETURN_HIDDEN_STATES")
        os.environ["UNSLOTH_RETURN_HIDDEN_STATES"] = "0"  # return logits, not hidden states
        dtype = getattr(self, "_autocast_dtype", None)
        if dtype not in (torch.float16, torch.bfloat16):
            dtype = torch.bfloat16 if getattr(self.args, "bf16", False) else None
        device_type = prompt_ids.device.type
        autocast = (
            torch.amp.autocast(device_type=device_type, dtype=dtype)
            if device_type == "cuda" and dtype is not None
            else nullcontext()
        )
        try:
            with autocast:
                logits = model(**model_inputs).logits
        finally:
            if previous is None:
                os.environ.pop("UNSLOTH_RETURN_HIDDEN_STATES", None)
            else:
                os.environ["UNSLOTH_RETURN_HIDDEN_STATES"] = previous

        shifted = logits[:, -(length + 1) : -1, :]
        token_nll = F.cross_entropy(
            shifted.float().reshape(-1, shifted.size(-1)), target_ids.reshape(-1), reduction="none"
        ).view_as(target_ids)
        row_nll = (token_nll * inputs["sft_loss_mask"].to(token_nll.dtype)).sum(dim=1)
        stats = self._all_reduce_sum(torch.stack([row_nll.sum().detach(), inputs["sft_loss_mask"].sum().to(row_nll.dtype)]))
        tokens = stats[1].clamp(min=1.0)
        world = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
        return row_nll, float(world) / tokens, stats[0] / tokens

    # ------------------------------------------------------------------ loss
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        global _ACTIVE_CONTEXT
        mode = "eval" if self.control.should_evaluate else "train"
        update = bool(model.training and mode == "train")

        # GSPO branch, divided by an EMA of RMS(A_total).
        rms = self._global_rms(inputs["total_advantages"])
        if update:
            self._gspo_ema = update_positive_ema(self._gspo_ema, rms, decay=self.ema_decay, minimum=self.ema_epsilon)
        gspo_ema = self._gspo_ema if self._gspo_ema is not None else max(rms, self.ema_epsilon)
        gspo_scale = inputs["total_advantages"].new_tensor(1.0 / max(gspo_ema, self.ema_epsilon))
        context = {"gspo_loss_scale": gspo_scale}
        self._objective_context = _ACTIVE_CONTEXT = context
        beta = self.beta
        # Keep the effective KL coefficient at beta despite the normalizer.
        self.beta = compensated_kl_beta(beta, float(gspo_scale.item()))
        try:
            gspo_loss = super().compute_loss(
                model, inputs, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch
            )
        finally:
            self.beta = beta
            self._objective_context = _ACTIVE_CONTEXT = None

        # Answer-SFT branch: rows weighted by lambda(p_b), divided by the frozen reference NLL.
        row_nll, gradient_scale, global_nll = self._sft_row_nll(model, inputs)
        if update:
            self._sft_reference = update_reference_scale(
                self._sft_reference,
                global_nll.item(),
                warmup_steps=int(self.sft_reference_steps),
                minimum=self.ema_epsilon,
            )
        reference = self._sft_reference.value if self._sft_reference is not None else global_nll.item()
        normalizer = max(reference, float(self.sft_normalizer_floor), self.ema_epsilon)
        weights = sft_row_weights(
            inputs["group_correct_fraction"],
            inputs["group_valid"].to(inputs["group_correct_fraction"].device),
            lambda_max=float(self.sft_lambda_max),
            gamma=float(self.sft_gamma),
        ).to(device=row_nll.device, dtype=row_nll.dtype)
        sft_loss = (row_nll * weights).sum() * gradient_scale / normalizer
        sft_loss = sft_loss / max(int(self.current_gradient_accumulation_steps), 1)

        self._log(mode, "loss/gspo_ema", gspo_ema)
        self._log(mode, "loss/sft_nll", global_nll.item())
        self._log(mode, "loss/sft_reference", reference)
        self._log(mode, "loss/sft_row_weight", self.accelerator.gather(weights.detach().float()).mean().item())
        return gspo_loss + sft_loss

    # ----------------------------------------------------------- checkpoints
    # The method's running state (mu_t, s_t, GSPO EMA, SFT reference) goes into method_state.json.
    def _save_checkpoint(self, *args, **kwargs):
        result = super()._save_checkpoint(*args, **kwargs)
        if self.is_world_process_zero():
            path = Path(self.args.output_dir) / f"checkpoint-{self.state.global_step}" / "method_state.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            reference = self._sft_reference
            path.write_text(json.dumps({
                "correction": self._correction_state,
                "gspo_ema": self._gspo_ema,
                "sft_reference": None if reference is None else vars(reference),
            }, indent=2), encoding="utf-8")
        return result

    def _load_from_checkpoint(self, resume_from_checkpoint, *args, **kwargs):
        result = super()._load_from_checkpoint(resume_from_checkpoint, *args, **kwargs)
        path = Path(resume_from_checkpoint) / "method_state.json"
        if path.is_file():
            state = json.loads(path.read_text(encoding="utf-8"))
            self._correction_state = dict(state["correction"])
            self._gspo_ema = state["gspo_ema"]
            if state["sft_reference"] is not None:
                self._sft_reference = ReferenceScale(**state["sft_reference"])
        return result
