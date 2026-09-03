"""Generate human-readable evaluation reports (Markdown + charts) from
benchmark results, suitable for archiving as MLflow/CI artifacts.
"""

from __future__ import annotations

from pathlib import Path

from jinja2 import Template

from vla_eval.core.logging import get_logger
from vla_eval.evaluation.benchmark import BenchmarkResult

logger = get_logger(__name__)

_REPORT_TEMPLATE = Template("""\
# VLA-Eval Benchmark Report

**Model:** {{ result.model_name }}
**Dataset:** {{ result.dataset_name }}
**Samples evaluated:** {{ result.num_samples }}
**Generated:** {{ generated_at }}

## Summary

| Metric | Value |
| --- | --- |
| Composite Score | {{ '%.2f'|format(result.composite_score) }} / 100 |
| Action MSE | {{ '%.6f'|format(result.action_metrics.mse) }} |
| Action MAE | {{ '%.6f'|format(result.action_metrics.mae) }} |
| Max Error | {{ '%.6f'|format(result.action_metrics.max_error) }} |
| Success Rate | {{ result.success_rate if result.success_rate is not none else 'N/A' }} |
| Mean Latency (ms) | {{ '%.3f'|format(result.latency.mean_ms) }} |
| p95 Latency (ms) | {{ '%.3f'|format(result.latency.p95_ms) }} |
| Throughput (Hz) | {{ '%.2f'|format(result.latency.throughput_hz) }} |

## Per-Dimension Action MSE

{% for v in result.action_metrics.per_dim_mse -%}
- dim[{{ loop.index0 }}]: {{ '%.6f'|format(v) }}
{% endfor %}

{% if chart_path %}
![Latency distribution]({{ chart_path }})
{% endif %}
""")


def generate_markdown_report(result: BenchmarkResult, output_dir: str | Path) -> Path:
    """Render a Markdown report for a single benchmark result."""
    from datetime import datetime, timezone

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    chart_path = generate_charts(result, output_dir)
    content = _REPORT_TEMPLATE.render(
        result=result,
        generated_at=datetime.now(timezone.utc).isoformat(),
        chart_path=chart_path.name if chart_path else None,
    )

    report_path = output_dir / f"report_{result.model_name}_{result.dataset_name}.md"
    report_path.write_text(content, encoding="utf-8")
    logger.info("report_generated", path=str(report_path))
    return report_path


def generate_charts(result: BenchmarkResult, output_dir: str | Path) -> Path | None:
    """Generate a simple bar chart of per-dimension action MSE."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    output_dir = Path(output_dir)
    fig, ax = plt.subplots(figsize=(6, 4))
    dims = list(range(len(result.action_metrics.per_dim_mse)))
    ax.bar(dims, result.action_metrics.per_dim_mse, color="#2563eb")
    ax.set_xlabel("Action dimension")
    ax.set_ylabel("MSE")
    ax.set_title(f"{result.model_name} on {result.dataset_name}: per-dim action MSE")
    fig.tight_layout()

    chart_path = output_dir / f"chart_{result.model_name}_{result.dataset_name}.png"
    fig.savefig(chart_path, dpi=120)
    plt.close(fig)
    return chart_path
