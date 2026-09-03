"""Benchmark runner: evaluates a `VLAPolicy` against a dataset offline
(open-loop action prediction) and measures inference latency/throughput.

Closed-loop (rollout) evaluation in simulation (e.g. LIBERO, SimplerEnv,
ManiSkill) is supported via the `SimulationEvaluator` extension point --
implement `rollout_episode` for a specific simulator to plug it in.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from vla_eval.core.logging import get_logger
from vla_eval.data.preprocessing import VLAEpisodeDataset
from vla_eval.evaluation.metrics import (
    ActionMetrics,
    LatencyMetrics,
    action_prediction_metrics,
    composite_score,
    latency_metrics,
)
from vla_eval.models.base import Observation
from vla_eval.models.base import VLAPolicy as VLAPolicyProtocol

logger = get_logger(__name__)


@dataclass
class BenchmarkResult:
    model_name: str
    dataset_name: str
    action_metrics: ActionMetrics
    latency: LatencyMetrics
    success_rate: float | None
    composite_score: float
    num_samples: int
    metadata: dict = field(default_factory=dict)


class SimulationEnv(Protocol):
    """Structural interface for a closed-loop rollout environment."""

    def reset(self) -> Observation: ...
    def step(self, action: np.ndarray) -> tuple[Observation, float, bool, dict]: ...


def run_offline_benchmark(
    policy: VLAPolicyProtocol,
    dataset: VLAEpisodeDataset,
    *,
    max_samples: int | None = None,
) -> BenchmarkResult:
    """Evaluate `policy` on `dataset` via open-loop action prediction.

    For each sample, feed the recorded observation to the policy and compare
    its predicted action against the ground-truth action taken by the
    demonstrator. This is a fast, deterministic proxy for closed-loop
    performance and does not require a simulator.
    """
    n = len(dataset) if max_samples is None else min(len(dataset), max_samples)
    predictions = []
    targets = []
    latencies: list[float] = []

    policy.reset()
    for i in range(n):
        sample = dataset[i]
        obs = Observation(
            image=sample["image"], state=sample["state"], instruction=sample["instruction"]
        )

        start = time.perf_counter()
        action = policy.predict(obs)
        latencies.append(time.perf_counter() - start)

        predictions.append(action.values)
        targets.append(sample["action"])

    predictions_arr = np.stack(predictions, axis=0)
    targets_arr = np.stack(targets, axis=0)

    action_metrics = action_prediction_metrics(predictions_arr, targets_arr)
    latency = latency_metrics(latencies)
    score = composite_score(action_metrics, latency, success=None)

    result = BenchmarkResult(
        model_name=policy.name,
        dataset_name=dataset.name,
        action_metrics=action_metrics,
        latency=latency,
        success_rate=None,
        composite_score=score,
        num_samples=n,
    )
    logger.info(
        "benchmark_complete",
        model=policy.name,
        dataset=dataset.name,
        mse=action_metrics.mse,
        score=score,
        num_samples=n,
    )
    return result


def run_closed_loop_benchmark(
    policy: VLAPolicyProtocol,
    env: SimulationEnv,
    *,
    num_episodes: int = 20,
    max_steps: int = 300,
) -> BenchmarkResult:
    """Evaluate `policy` via closed-loop rollouts in a simulation environment.

    `env` must implement `reset()` -> Observation and
    `step(action)` -> (Observation, reward, done, info), where
    `info.get("success")` indicates task completion.
    """
    successes: list[bool] = []
    latencies: list[float] = []
    all_actions: list[np.ndarray] = []

    for episode in range(num_episodes):
        obs = env.reset()
        policy.reset()
        done = False
        steps = 0
        episode_success = False

        while not done and steps < max_steps:
            start = time.perf_counter()
            action = policy.predict(obs)
            latencies.append(time.perf_counter() - start)
            all_actions.append(action.values)

            obs, _reward, done, info = env.step(action.values)
            episode_success = bool(info.get("success", False))
            steps += 1

        successes.append(episode_success)
        logger.debug("episode_complete", episode=episode, success=episode_success, steps=steps)

    latency = latency_metrics(latencies)
    sr = sum(successes) / max(len(successes), 1)

    # Without ground-truth actions in closed-loop mode, use a placeholder
    # ActionMetrics derived from action magnitude variance (informational only).
    actions_arr = np.stack(all_actions, axis=0) if all_actions else np.zeros((1, 1))
    dummy_metrics = ActionMetrics(
        mse=0.0, mae=0.0, max_error=0.0, per_dim_mse=[0.0] * actions_arr.shape[-1]
    )
    score = composite_score(dummy_metrics, latency, success=sr)

    return BenchmarkResult(
        model_name=policy.name,
        dataset_name=getattr(env, "name", "simulation"),
        action_metrics=dummy_metrics,
        latency=latency,
        success_rate=sr,
        composite_score=score,
        num_samples=num_episodes,
        metadata={"max_steps": max_steps},
    )
