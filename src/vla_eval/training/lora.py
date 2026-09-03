"""LoRA (parameter-efficient fine-tuning) helpers built on top of `peft`.

Used primarily to fine-tune large backbones (OpenVLA) on a single/few GPUs
by only updating a small number of injected low-rank adapter weights.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from vla_eval.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class LoRAConfig:
    rank: int = 32
    alpha: int = 64
    dropout: float = 0.05
    target_modules: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")


def apply_lora(model: Any, config: LoRAConfig) -> Any:
    """Wrap `model` with LoRA adapters using the given configuration.

    Returns the PEFT-wrapped model. Logs the number of trainable vs. total
    parameters for transparency.
    """
    from peft import LoraConfig as PeftLoraConfig
    from peft import get_peft_model

    peft_config = PeftLoraConfig(
        r=config.rank,
        lora_alpha=config.alpha,
        lora_dropout=config.dropout,
        target_modules=list(config.target_modules),
        task_type="CAUSAL_LM",
    )
    wrapped = get_peft_model(model, peft_config)
    trainable, total = _count_parameters(wrapped)
    logger.info(
        "lora_applied",
        trainable_params=trainable,
        total_params=total,
        trainable_pct=round(100 * trainable / max(total, 1), 4),
    )
    return wrapped


def _count_parameters(model: Any) -> tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total
