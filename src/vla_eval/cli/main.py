"""Typer-based CLI entry point (`vla-eval`).

Subcommands are grouped by concern: `data`, `train`, `evaluate`, `serve`.
For fully Hydra-driven experiment configs (multi-run sweeps, config
composition), see the scripts in `scripts/` which use `@hydra.main`.
"""

from __future__ import annotations

from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from vla_eval.core.logging import configure_logging, get_logger

app = typer.Typer(
    name="vla-eval",
    help="Train, benchmark, and evaluate Vision-Language-Action robotics models.",
    no_args_is_help=True,
)
data_app = typer.Typer(help="Dataset download/preprocessing commands.")
app.add_typer(data_app, name="data")

console = Console()
logger = get_logger(__name__)


@app.callback()
def _main() -> None:
    configure_logging()


# --------------------------------------------------------------------------
# Data commands
# --------------------------------------------------------------------------


@data_app.command("list")
def data_list() -> None:
    """List all registered public robotics datasets."""
    from vla_eval.data.registry import list_datasets

    table = Table(title="Registered Datasets")
    table.add_column("Name")
    table.add_column("Hub Repo ID")
    table.add_column("Task Type")
    table.add_column("Action Dim")
    table.add_column("License")
    for spec in list_datasets():
        table.add_row(
            spec.name, spec.hub_repo_id, spec.task_type, str(spec.action_dim), spec.license
        )
    console.print(table)


@data_app.command("download")
def data_download(
    name: Annotated[str, typer.Argument(help="Registered dataset name, e.g. 'pusht'.")],
    cache_dir: Annotated[str, typer.Option(help="Local cache directory.")] = "./data/cache",
) -> None:
    """Download a registered dataset from the Hugging Face Hub."""
    from vla_eval.core.config import get_settings
    from vla_eval.data.datasets import download_dataset

    settings = get_settings()
    result = download_dataset(name, cache_dir=cache_dir, hf_token=settings.hf_token)
    console.print(
        f"[green]Downloaded[/green] '{name}' -> {result.local_path} "
        f"(backend={result.backend}, bytes={result.num_bytes})"
    )


@data_app.command("preprocess")
def data_preprocess(
    name: Annotated[str, typer.Argument(help="Registered dataset name, e.g. 'pusht'.")],
    cache_dir: Annotated[str, typer.Option()] = "./data/cache",
    processed_dir: Annotated[str, typer.Option()] = "./data/processed",
    max_episodes: Annotated[int | None, typer.Option()] = None,
) -> None:
    """Preprocess a downloaded dataset into a training-ready format."""
    from vla_eval.data.preprocessing import preprocess_dataset

    dataset = preprocess_dataset(
        name, cache_dir=cache_dir, processed_dir=processed_dir, max_episodes=max_episodes
    )
    console.print(f"[green]Preprocessed[/green] '{name}': {len(dataset)} samples")


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------


@app.command("train")
def train(
    model: Annotated[
        str, typer.Option(help="Registered model name, e.g. 'baseline-cnn'.")
    ] = "baseline-cnn",
    dataset: Annotated[
        str, typer.Option(help="Registered dataset name, or 'synthetic'.")
    ] = "synthetic",
    epochs: Annotated[int, typer.Option()] = 10,
    batch_size: Annotated[int, typer.Option()] = 32,
    learning_rate: Annotated[float, typer.Option()] = 1e-4,
    device: Annotated[str, typer.Option()] = "cpu",
    checkpoint_dir: Annotated[str, typer.Option()] = "./models/checkpoints",
    run_name: Annotated[str | None, typer.Option()] = None,
) -> None:
    """Fine-tune/train a VLA model on a dataset with MLflow tracking."""
    from vla_eval.data.preprocessing import make_synthetic_dataset, preprocess_dataset
    from vla_eval.models.registry import create_model
    from vla_eval.training.callbacks import Callback, EarlyStopping, ModelCheckpoint
    from vla_eval.training.trainer import Trainer, TrainingConfig
    from vla_eval.utils.device import resolve_device

    resolved_device = resolve_device(device)
    ds = make_synthetic_dataset() if dataset == "synthetic" else preprocess_dataset(dataset)

    policy = create_model(model, device=resolved_device, action_dim=ds.spec.action_dim)
    config = TrainingConfig(
        run_name=run_name or f"{model}-{dataset}",
        num_epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        device=resolved_device,
        checkpoint_dir=checkpoint_dir,
    )
    callbacks: list[Callback] = [
        EarlyStopping(monitor="val_loss", patience=config.early_stopping_patience),
        ModelCheckpoint(checkpoint_dir=checkpoint_dir, monitor="val_loss"),
    ]
    trainer = Trainer(policy, config, callbacks=callbacks)
    result = trainer.fit(ds)
    console.print(
        f"[green]Training complete.[/green] Final checkpoint: {result['final_checkpoint']}"
    )


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------


@app.command("evaluate")
def evaluate(
    model: Annotated[str, typer.Option()] = "baseline-cnn",
    dataset: Annotated[str, typer.Option()] = "synthetic",
    checkpoint: Annotated[str | None, typer.Option(help="Path to a trained checkpoint.")] = None,
    device: Annotated[str, typer.Option()] = "cpu",
    max_samples: Annotated[int | None, typer.Option()] = None,
    report_dir: Annotated[str, typer.Option()] = "./reports/generated",
) -> None:
    """Run the offline benchmark suite for a model on a dataset and generate a report."""
    from vla_eval.data.preprocessing import make_synthetic_dataset, preprocess_dataset
    from vla_eval.evaluation.benchmark import run_offline_benchmark
    from vla_eval.evaluation.report import generate_markdown_report
    from vla_eval.models.registry import create_model
    from vla_eval.utils.device import resolve_device

    resolved_device = resolve_device(device)
    ds = make_synthetic_dataset() if dataset == "synthetic" else preprocess_dataset(dataset)

    policy = create_model(model, device=resolved_device, action_dim=ds.spec.action_dim)
    if checkpoint:
        policy.load(checkpoint)

    result = run_offline_benchmark(policy, ds, max_samples=max_samples)
    report_path = generate_markdown_report(result, report_dir)

    console.print(f"[green]Composite score:[/green] {result.composite_score}/100")
    console.print(f"Report written to {report_path}")


# --------------------------------------------------------------------------
# Serve
# --------------------------------------------------------------------------


@app.command("serve")
def serve(
    host: Annotated[
        str, typer.Option()
    ] = "0.0.0.0",  # nosec B104 -- intended for containerized deployment
    port: Annotated[int, typer.Option()] = 8000,
    reload: Annotated[bool, typer.Option()] = False,
) -> None:
    """Run the FastAPI backend with uvicorn."""
    import uvicorn

    uvicorn.run("vla_eval.api.main:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    app()
