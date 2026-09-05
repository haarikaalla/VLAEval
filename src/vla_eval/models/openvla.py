"""Wrapper around OpenVLA (and OpenVLA-style) models from Hugging Face.

OpenVLA (https://openvla.github.io/) is a 7B-parameter open-source
vision-language-action model built on a Prismatic VLM backbone. This wrapper
loads it via `transformers.AutoModelForVision2Seq`, exposes the common
`VLAPolicy` interface, and supports parameter-efficient fine-tuning (LoRA)
through `peft`.

Loading the full 7B checkpoint requires a CUDA GPU with >=16GB VRAM (bf16) or
a quantized (4-bit/8-bit) configuration. For CI and local development without
a GPU, use `vla_eval.models.baseline.BaselineVLAModel` instead.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from vla_eval.core.exceptions import ModelLoadError
from vla_eval.core.logging import get_logger
from vla_eval.models.base import Action, BaseVLAModel, Observation

logger = get_logger(__name__)

DEFAULT_CHECKPOINT = "openvla/openvla-7b"


class OpenVLAModel(BaseVLAModel):
    """Adapter exposing an OpenVLA checkpoint through the `VLAPolicy` interface."""

    name = "openvla"

    def __init__(
        self,
        checkpoint: str = DEFAULT_CHECKPOINT,
        *,
        device: str = "cuda",
        action_dim: int = 7,
        load_in_4bit: bool = False,
        use_lora: bool = False,
        lora_rank: int = 32,
        hf_token: str | None = None,
        revision: str | None = None,
    ) -> None:
        super().__init__(device=device)
        self.checkpoint = checkpoint
        self.action_dim = action_dim
        self.load_in_4bit = load_in_4bit
        self.use_lora = use_lora
        self.lora_rank = lora_rank
        self.hf_token = hf_token
        self.revision = revision
        self._model: Any = None
        self._processor: Any = None
        self._loaded = False

    def load(self, path: str | None = None) -> None:
        """Load the base (or fine-tuned) checkpoint and processor from the Hub/disk."""
        try:
            import torch
            import transformers
            from transformers import AutoProcessor
        except ImportError as exc:
            raise ModelLoadError("transformers/torch are required to load OpenVLA models.") from exc

        auto_model_cls = getattr(transformers, "AutoModelForVision2Seq", None)
        if auto_model_cls is None:
            raise ModelLoadError(
                "Installed transformers package does not provide AutoModelForVision2Seq."
            )

        checkpoint = path or self.checkpoint
        logger.info("openvla_load_start", checkpoint=checkpoint, device=self.device)

        quantization_kwargs: dict[str, Any] = {}
        if self.load_in_4bit:
            from transformers import BitsAndBytesConfig

            quantization_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )

        self._processor = AutoProcessor.from_pretrained(
            checkpoint, trust_remote_code=True, token=self.hf_token, revision=self.revision
        )
        self._model = auto_model_cls.from_pretrained(
            checkpoint,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            token=self.hf_token,
            revision=self.revision,
            **quantization_kwargs,
        )
        if not self.load_in_4bit:
            self._model = self._model.to(self.device)

        if self.use_lora:
            self._attach_lora()

        self._loaded = True
        logger.info("openvla_load_complete", checkpoint=checkpoint)

    def _attach_lora(self) -> None:
        from peft import LoraConfig, get_peft_model

        lora_config = LoraConfig(
            r=self.lora_rank,
            lora_alpha=self.lora_rank * 2,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            task_type="CAUSAL_LM",
        )
        self._model = get_peft_model(self._model, lora_config)
        logger.info("openvla_lora_attached", rank=self.lora_rank)

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            self.load()

    def predict(self, observation: Observation) -> Action:
        import torch
        from PIL import Image

        self._ensure_loaded()
        self.eval_mode()

        image_hwc = np.transpose(observation.image, (1, 2, 0))
        image = Image.fromarray((image_hwc * 255).astype(np.uint8))
        prompt = f"In: What action should the robot take to {observation.instruction}?\nOut:"

        inputs = self._processor(prompt, image).to(self.device, dtype=torch.bfloat16)
        with torch.no_grad():
            action = self._model.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)
        return Action(values=np.asarray(action))

    def train_mode(self) -> None:
        self._ensure_loaded()
        self._model.train()

    def eval_mode(self) -> None:
        self._ensure_loaded()
        self._model.eval()

    def parameters(self) -> Any:
        self._ensure_loaded()
        return self._model.parameters()

    def save(self, path: str) -> None:
        self._ensure_loaded()
        self._model.save_pretrained(path)
        self._processor.save_pretrained(path)
