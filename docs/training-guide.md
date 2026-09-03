# Training Guide

## 1. Datasets

Registered datasets live in `vla_eval.data.registry` (`DatasetSpec` entries). List them with:

```bash
vla-eval data list
```

Currently curated: `pusht`, `aloha_sim_insertion_human`, `aloha_sim_transfer_cube_human`, `xarm_lift_medium`, `berkeley_autolab_ur5`, `bridge_orig` — all sourced from `lerobot/*` repos on the Hugging Face Hub. A `synthetic` dataset is also available (generated in-memory, no network access) for CI and quick smoke tests.

### Data provenance

Every non-synthetic dataset is fetched at runtime from `https://huggingface.co/datasets/lerobot/<name>`, the standardized robot-learning dataset collection maintained by the Hugging Face [LeRobot](https://github.com/huggingface/lerobot) project. `berkeley_autolab_ur5` and `bridge_orig` are real-robot recordings originally released as part of [Open X-Embodiment](https://robotics-transformer-x.github.io/) and re-published in LeRobot format.

No dataset content is vendored in this repository — downloads land in `./data/cache/` (gitignored) and can be pinned to a specific Hub revision for reproducibility:

```python
from vla_eval.data.datasets import download_dataset

download_dataset("pusht", revision="a1b2c3d")  # pin to a commit SHA
```

Set `HF_TOKEN` in your `.env` to access gated or private repos. Respect each dataset's upstream license (MIT for the simulation sets, CC-BY-4.0 for the real-robot sets) when publishing derived models or results.

Download and preprocess a real dataset:

```bash
vla-eval data download pusht --cache-dir ./data/cache
vla-eval data preprocess pusht --cache-dir ./data/cache --processed-dir ./data/processed
```

Downloading prefers the `lerobot` package's dataset loader when installed (`pip install vla-eval[lerobot]`) and falls back to `huggingface_hub.snapshot_download` otherwise, with retries via `tenacity`.

## 2. Models

Registered models (`vla_eval.models.registry.list_models()`):

| Name | Config | Notes |
|---|---|---|
| `baseline-cnn` | `configs/model/baseline.yaml` | Small CNN+MLP baseline. Always available (no extra deps), fast on CPU. |
| `openvla` | `configs/model/openvla_lora.yaml` | Wraps `openvla/openvla-7b` via `transformers`. Supports 4-bit quantization (`load_in_4bit`) and LoRA fine-tuning. Requires a GPU for practical use. |
| `lerobot-act` | `configs/model/lerobot_act.yaml` | LeRobot's ACT (Action Chunking Transformer) policy via `lerobot.make_policy`. |
| `lerobot-diffusion` | — | LeRobot Diffusion Policy, same factory (`policy_type=diffusion`). |

## 3. Training

### Via the CLI (quick, low-friction)

```bash
vla-eval train --dataset pusht --model baseline-cnn --epochs 20 --batch-size 32 --learning-rate 1e-4
```

### Via Hydra (full experiment-config control)

```bash
python scripts/train.py dataset=pusht model=baseline training=default
python scripts/train.py model=openvla_lora training=lora_finetune dataset=pusht
python scripts/train.py -m training.learning_rate=1e-4,5e-5,1e-5   # multirun sweep
```

Config composition is defined in [`configs/config.yaml`](../configs/config.yaml):

```yaml
defaults:
  - dataset: pusht
  - model: baseline
  - training: default
  - evaluation: default
  - _self_
```

Override any leaf value on the command line, e.g. `training.num_epochs=50`, or swap an entire group, e.g. `dataset=aloha_sim_insertion`.

### LoRA fine-tuning

`configs/training/lora_finetune.yaml` + `configs/model/openvla_lora.yaml` enable LoRA via `vla_eval.training.lora.apply_lora`, which wraps `peft.get_peft_model` and logs trainable-parameter counts. Combine with `load_in_4bit: true` on the model config to fine-tune large models (e.g. OpenVLA-7B) on a single GPU.

## 4. Experiment tracking (MLflow)

Every `Trainer.fit()` call is wrapped in `vla_eval.training.mlflow_utils.mlflow_run`, which:

- Sets the tracking URI from `MLFLOW_TRACKING_URI` (default `http://localhost:5000`; the Docker Compose stack points this at the bundled `mlflow` service).
- Creates/reuses the experiment named by `MLFLOW_EXPERIMENT_NAME` (default `vla-eval`).
- Logs hyperparameters, per-epoch metrics, and the final checkpoint as an artifact.

For a zero-dependency local run without a running MLflow server, point at a local SQLite-backed store:

```bash
export MLFLOW_TRACKING_URI=sqlite:///./mlruns.db
```

> MLflow ≥2.x no longer supports the legacy `file://...mlruns` store by default (raises `MlflowException: filesystem tracking backend ... in maintenance mode`). Always use a database-backed URI (`sqlite:///...` or `postgresql://...`) or set `MLFLOW_ALLOW_FILE_STORE=true` to opt back in.

## 5. Evaluation & benchmarking

```bash
vla-eval evaluate --dataset pusht --model baseline-cnn --checkpoint ./models/checkpoints/final_model.pt
# or
python scripts/evaluate.py dataset=pusht model=baseline
```

This runs `run_offline_benchmark` (open-loop action-prediction metrics: MSE, per-dimension breakdown, latency) and writes a Markdown report with charts to `evaluation.report_dir` (default `./reports/generated`). Results are combined into a 0–100 `composite_score` (success rate + accuracy + speed) and can be submitted to the leaderboard via the API (`POST /api/v1/leaderboard`) or automatically when running evaluation jobs through the API/worker.

Closed-loop (simulator rollout) benchmarking is supported via `run_closed_loop_benchmark` for environments implementing the `SimulationEnv` protocol — see `vla_eval.evaluation.benchmark`.
