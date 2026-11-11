"""InternVL3-8B support: the same objective plus two compatibility fixes.

InternVL3-8B-hf turns each image into one 448x448 tile, so ``pixel_values``
has one row per sample.

1. Images. The upstream trainer splits ``pixel_values`` into chunks only when
   ``image_grid_thw`` is given; otherwise every chunk gets no image and the
   policy silently trains text-only. We pass a dummy all-ones grid (one row per
   sample) and remove it before InternVL's forward, which would pass the
   unknown key on to its language model.
2. RoPE. The language model uses dynamic-NTK RoPE, which cannot be compiled.
   Below 32,768 tokens it equals the default RoPE, and our sequences stay within
   4,096, so the rotary modules are switched to the default type.
"""

from __future__ import annotations

import functools
from typing import Any

import torch

from trainer import HybridGRPOTrainer

_QWEN_SHAPED_KEYS = ("image_grid_thw", "pixel_attention_mask", "image_sizes")


def row_major_grid(num_rows: int, device: torch.device) -> torch.Tensor:
    """Dummy ``image_grid_thw``: one image row per sample."""

    return torch.ones((int(num_rows), 1), device=device, dtype=torch.long)


def use_default_rope(model: Any, max_seq_length: int) -> list[str]:
    """Switch dynamic-NTK rotary modules to the default type after checking that
    their frequencies match; returns the changed module names."""

    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

    changed = []
    for name, module in model.named_modules():
        rope_type = getattr(module, "rope_type", None)
        if not isinstance(rope_type, str) or "dynamic" not in rope_type:
            continue
        limit = getattr(module, "original_max_seq_len", None)
        if limit is None or max_seq_length > limit:
            raise ValueError(f"{name}: dynamic RoPE differs from default above {limit} tokens")
        default, scaling = ROPE_INIT_FUNCTIONS["default"](module.config, "cpu")
        current = module.inv_freq.detach().float().cpu()
        if not torch.allclose(current, default.float(), rtol=0, atol=1e-7) or float(
            getattr(module, "attention_scaling", 1.0)
        ) != float(scaling):
            raise ValueError(f"{name}: frequencies differ from default RoPE")
        module.rope_type = "default"
        changed.append(name)
    return changed


def _find_internvl_module(model: Any):
    """Descend PEFT / DDP / Unsloth wrappers to InternVLForConditionalGeneration."""

    seen, current = set(), model
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if type(current).__name__.startswith("InternVLForConditionalGeneration"):
            return current
        current = next(
            (getattr(current, a) for a in ("module", "base_model", "model")
             if getattr(current, a, None) is not None and id(getattr(current, a)) not in seen),
            None,
        )
    raise RuntimeError(f"No InternVLForConditionalGeneration inside {type(model).__name__}.")


class InternVLVisionRouter:
    """Wraps InternVL's forward to drop the Qwen-only keys (``image_grid_thw`` etc.),
    for every forward pass: reference, old policy, policy and SFT."""

    def __init__(self, model: Any):
        self.module = _find_internvl_module(model)
        original = self.module.forward
        self.stripped_calls = 0

        # Keep the original signature: upstream inspects it to decide whether to pass logits_to_keep.
        @functools.wraps(original)
        def routed_forward(*args: Any, **kwargs: Any):
            removed = [kwargs.pop(key) for key in _QWEN_SHAPED_KEYS if key in kwargs]
            if removed:
                self.stripped_calls += 1
            return original(*args, **kwargs)

        self.module.forward = routed_forward


class InternVLHybridGRPOTrainer(HybridGRPOTrainer):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if getattr(self.args, "use_liger_loss", False):
            raise NotImplementedError("use_liger_loss bypasses the InternVL vision router.")
        self.vision_router = InternVLVisionRouter(self.model)

    @staticmethod
    def _needs_grid(holder: dict[str, Any]) -> bool:
        return holder.get("pixel_values") is not None and holder.get("image_grid_thw") is None

    def _get_per_token_logps_and_entropies(self, model, input_ids, attention_mask, logits_to_keep, *args, **kwargs):
        if self._needs_grid(kwargs):
            kwargs["image_grid_thw"] = row_major_grid(input_ids.size(0), input_ids.device)
        return super()._get_per_token_logps_and_entropies(
            model, input_ids, attention_mask, logits_to_keep, *args, **kwargs
        )

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if self._needs_grid(inputs):
            # Copy, so the dummy grid is not written into the buffered rollout batch.
            inputs = dict(inputs)
            inputs["image_grid_thw"] = row_major_grid(inputs["prompt_ids"].size(0), inputs["prompt_ids"].device)
        return super().compute_loss(
            model, inputs, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch
        )
