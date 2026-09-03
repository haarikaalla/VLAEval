"""Hydra-driven evaluation/benchmark entry point.

Usage:
    python scripts/evaluate.py model=baseline dataset=pusht
    python scripts/evaluate.py model=openvla_lora dataset=pusht evaluation.max_samples=200
"""

from __future__ import annotations

import hydra
from omegaconf import DictConfig, OmegaConf

from vla_eval.core.logging import configure_logging, get_logger
from vla_eval.data.preprocessing import make_synthetic_dataset, preprocess_dataset
from vla_eval.evaluation.benchmark import run_offline_benchmark
from vla_eval.evaluation.report import generate_markdown_report
from vla_eval.models.registry import create_model
from vla_eval.utils.device import resolve_device

logger = get_logger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    configure_logging()
    logger.info("evaluation_config", config=OmegaConf.to_yaml(cfg))

    device = resolve_device(cfg.training.device)

    if cfg.dataset.name == "synthetic":
        dataset = make_synthetic_dataset(
            num_samples=cfg.dataset.get("num_samples", 256),
            action_dim=cfg.dataset.get("action_dim", 7),
            state_dim=cfg.dataset.get("state_dim", 7),
        )
    else:
        dataset = preprocess_dataset(
            cfg.dataset.name,
            cache_dir=cfg.dataset.cache_dir,
            processed_dir=cfg.dataset.processed_dir,
            max_episodes=cfg.dataset.get("max_episodes"),
        )

    model_kwargs = {k: v for k, v in cfg.model.items() if k not in ("name", "type")}
    policy = create_model(cfg.model.type, device=device, **model_kwargs)

    result = run_offline_benchmark(policy, dataset, max_samples=cfg.evaluation.get("max_samples"))
    report_path = generate_markdown_report(result, cfg.evaluation.report_dir)

    logger.info(
        "evaluation_finished",
        composite_score=result.composite_score,
        mse=result.action_metrics.mse,
        report_path=str(report_path),
    )


if __name__ == "__main__":
    main()
