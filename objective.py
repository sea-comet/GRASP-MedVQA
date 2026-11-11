"""Math of the training objective, independent of the trainer.

Notation (per optimizer step, G rollouts per question):
  p_b   fraction of the G rollouts of question b whose answer is correct
  A     group-relative advantage, (r - mean_group(r)) / std_batch(r)
  mu_t  running accuracy (EMA of the batch's correct fraction)
  s_t   running advantage scale (EMA of RMS(A))
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class GroupFlags:
    """Statistics of each rollout's group (the same value for all G rollouts)."""

    all_wrong: torch.Tensor
    all_correct: torch.Tensor
    valid: torch.Tensor
    correct_fraction: torch.Tensor  # p_b


def group_flags(is_correct: torch.Tensor, is_valid: torch.Tensor, *, num_generations: int) -> GroupFlags:
    """A group with a non-finite correctness score is invalid: never all-wrong
    or all-correct, so a failed reward is not counted as a wrong answer."""

    if is_correct.ndim != 1 or is_correct.numel() % num_generations:
        raise ValueError("is_correct must be 1-D and split into whole groups.")
    correct = is_correct.bool().view(-1, num_generations)
    valid = is_valid.bool().view(-1, num_generations).all(dim=1)
    counts = correct.sum(dim=1)
    repeat = lambda values: values.repeat_interleave(num_generations)  # noqa: E731
    return GroupFlags(
        all_wrong=repeat(valid & counts.eq(0)),
        all_correct=repeat(valid & counts.eq(num_generations)),
        valid=repeat(valid),
        correct_fraction=repeat(correct.float().mean(dim=1)),
    )


def sft_row_weights(
    correct_fraction: torch.Tensor, valid: torch.Tensor, *, lambda_max: float, gamma: float
) -> torch.Tensor:
    """SFT weight lambda(p_b) = lambda_max * (1 - p_b)^gamma; invalid groups use p_b = 0.5."""

    p = torch.where(valid.bool(), correct_fraction.float(), torch.full_like(correct_fraction.float(), 0.5))
    return float(lambda_max) * (1.0 - p.clamp(0.0, 1.0)).pow(float(gamma))


def state_adaptive_correction(
    all_wrong: torch.Tensor,
    all_correct: torch.Tensor,
    valid: torch.Tensor,
    *,
    mu: float,
    scale: float,
    num_generations: int,
) -> torch.Tensor:
    """Advantage shift g_b = (s_t / G)(p_b - mu_t) for all-wrong and all-correct groups.

    All-wrong (p_b = 0): -s_t * mu_t / G. All-correct (p_b = 1): +s_t * (1 - mu_t) / G.
    Mixed groups already have a correctness contrast and get 0.
    """

    if not 0.0 <= float(mu) <= 1.0 or not math.isfinite(float(scale)) or float(scale) < 0.0:
        raise ValueError("mu must lie in [0, 1] and scale must be finite and non-negative.")
    ok = valid.bool()
    unit = float(scale) / int(num_generations)
    wrong = (all_wrong.bool() & ok).float()
    correct = (all_correct.bool() & ok).float()
    return unit * (1.0 - float(mu)) * correct - unit * float(mu) * wrong


def update_ema(old: float | None, observation: float, *, decay: float) -> float:
    if old is None:
        return float(observation)
    return float(decay) * float(old) + (1.0 - float(decay)) * float(observation)


def update_positive_ema(old: float | None, observation: float, *, decay: float, minimum: float) -> float:
    """EMA with the observation and the result floored at ``minimum``."""

    observation = max(float(observation), float(minimum))
    return max(update_ema(old, observation, decay=decay), float(minimum))


@dataclass(frozen=True)
class ReferenceScale:
    """SFT loss normalizer: mean SFT NLL over the first ``warmup_steps``
    micro-batches, then frozen.

    Not a running mean: that would shrink as the target is learned, and dividing
    by it would keep the loss of an already-learned target large.
    """

    value: float
    count: int
    frozen: bool


def update_reference_scale(
    state: ReferenceScale | None, observation: float, *, warmup_steps: int, minimum: float
) -> ReferenceScale:
    observation = max(float(observation), float(minimum))
    if state is None:
        return ReferenceScale(value=observation, count=1, frozen=1 >= int(warmup_steps))
    if state.frozen:
        return state
    count = state.count + 1
    value = (state.value * state.count + observation) / count
    return ReferenceScale(value=max(value, float(minimum)), count=count, frozen=count >= int(warmup_steps))


def compensated_kl_beta(beta: float, loss_scale: float, *, max_ratio: float = 100.0) -> float:
    """beta / loss_scale, capped at beta * max_ratio.

    The upstream loss contains beta * KL and is multiplied as a whole by
    loss_scale (the GSPO normalizer); this keeps the effective KL coefficient at beta.
    """

    if not math.isfinite(float(beta)) or float(beta) == 0.0:
        return float(beta)
    if not math.isfinite(float(loss_scale)) or float(loss_scale) <= 0.0:
        return float(beta) * float(max_ratio)
    return float(beta) * min(1.0 / float(loss_scale), float(max_ratio))
