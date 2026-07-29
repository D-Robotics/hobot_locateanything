"""Write bounded activation-statistics artifacts and optional diagnostic plots."""

from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable


PNG_NAMES = (
    "activation_heatmap.png",
    "outlier_distribution.png",
    "scale_convergence.png",
    "graph_coverage.png",
)

CSV_COLUMNS = (
    "component",
    "graph_stage",
    "module_name",
    "kind",
    "min",
    "max",
    "absmax",
    "mean",
    "std",
    "p99_abs",
    "p999_abs",
    "observed_elements",
    "finite_elements",
    "nonfinite_count",
    "execution_count",
    "max_hit_rate",
    "clipping_range_abs",
    "clipping_rate",
    "clipping_rate_exact",
)


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _atomic_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def write_activation_statistics(
    output_dir: Path,
    rows: list[dict[str, Any]],
    *,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write complete numeric statistics; percentiles are diagnostic estimates."""

    output_dir = Path(output_dir)
    payload = {
        "schema_version": 2,
        "activation_point_count": len(rows),
        "statistics_scope": "finite activation elements; non-finite values counted separately",
        "standard_deviation": "population",
        "percentile_estimator": {
            "method": "fixed_log2_histogram",
            "bins": 512,
            "minimum_exponent": -32.0,
            "maximum_exponent": 32.0,
            "diagnostic_only": True,
        },
        "max_hit_rate_definition": (
            "fraction of finite activations equal to the observed absolute maximum; "
            "this is not a clipping metric"
        ),
        "clipping_rate_definition": (
            "fraction of finite absolute activations strictly above the finalized range"
        ),
        "clipping_rate_measurement": (
            "exact for same-stream absmax ranges; a smaller fixed range requires replay "
            "with that range supplied to ActivationTracker"
        ),
        "metadata": metadata or {},
        "activation_points": rows,
    }
    _atomic_json(output_dir / "activation_stats.json", payload)
    _atomic_csv(output_dir / "activation_stats.csv", rows)
    return payload


def _draw_activation_heatmap(plt: Any, rows: list[dict[str, Any]], path: Path) -> None:
    stages = sorted({str(row["graph_stage"]) for row in rows})
    module_scores: dict[str, float] = {}
    values: dict[tuple[str, str], float] = {}
    for row in rows:
        module = f"{row['component']}:{row['module_name']}"
        value = row.get("absmax")
        if value is None or not math.isfinite(float(value)):
            continue
        value = float(value)
        values[(module, str(row["graph_stage"]))] = value
        module_scores[module] = max(module_scores.get(module, 0.0), value)
    modules = [name for name, _ in sorted(
        module_scores.items(), key=lambda item: (-item[1], item[0])
    )[:80]]
    figure, axis = plt.subplots(figsize=(max(8, len(stages) * 0.75), max(4, len(modules) * 0.18)))
    if modules and stages:
        matrix = [
            [
                math.log10(max(values.get((module, stage), float("nan")), 1e-12))
                for stage in stages
            ]
            for module in modules
        ]
        image = axis.imshow(matrix, aspect="auto", interpolation="nearest", cmap="viridis")
        axis.set_xticks(range(len(stages)), stages, rotation=45, ha="right")
        axis.set_yticks(range(len(modules)), modules, fontsize=6)
        figure.colorbar(image, ax=axis, label="log10(absmax)")
    else:
        axis.text(0.5, 0.5, "No finite activation statistics", ha="center", va="center")
        axis.set_axis_off()
    axis.set_title("Activation range by graph stage")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _draw_outlier_distribution(plt: Any, rows: list[dict[str, Any]], path: Path) -> None:
    p99_ratios = []
    p999_ratios = []
    for row in rows:
        absmax = row.get("absmax")
        p99 = row.get("p99_abs")
        p999 = row.get("p999_abs")
        if absmax is None or float(absmax) <= 0:
            continue
        if p99 is not None:
            p99_ratios.append(float(p99) / float(absmax))
        if p999 is not None:
            p999_ratios.append(float(p999) / float(absmax))
    figure, axis = plt.subplots(figsize=(9, 5))
    bins = [index / 40 for index in range(41)]
    if p99_ratios:
        axis.hist(p99_ratios, bins=bins, alpha=0.65, label="p99_abs / absmax")
    if p999_ratios:
        axis.hist(p999_ratios, bins=bins, alpha=0.65, label="p999_abs / absmax")
    if p99_ratios or p999_ratios:
        axis.legend()
    else:
        axis.text(0.5, 0.5, "No finite percentile estimates", ha="center", va="center")
    axis.set_xlabel("Tail percentile divided by absmax")
    axis.set_ylabel("Activation point count")
    axis.set_title("Activation tail concentration (diagnostic estimates)")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _draw_scale_convergence(plt: Any, convergence: dict[str, Any], path: Path) -> None:
    figure, axis = plt.subplots(figsize=(9, 5))
    plotted = False
    for component, result in sorted(convergence.get("components", {}).items()):
        comparisons = result.get("vs_full", [])
        x = [entry.get("from_samples") for entry in comparisons]
        y = [entry.get("p95_relative_drift") for entry in comparisons]
        pairs = [(a, b) for a, b in zip(x, y) if a is not None and b is not None]
        if pairs:
            axis.plot(
                [pair[0] for pair in pairs],
                [pair[1] for pair in pairs],
                marker="o",
                label=f"{component} P95",
            )
            plotted = True
    if plotted:
        axis.legend()
    else:
        axis.text(0.5, 0.5, "No intermediate convergence checkpoint", ha="center", va="center")
    axis.set_xlabel("Calibration samples")
    axis.set_ylabel("Relative scale drift versus full set")
    axis.set_title("Scale convergence")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _draw_graph_coverage(plt: Any, coverage: dict[str, Any], path: Path) -> None:
    counts = coverage.get("stage_execution_counts", {})
    stages = list(coverage.get("expected_stages", []))
    values = [int(counts.get(stage, 0)) for stage in stages]
    figure, axis = plt.subplots(figsize=(max(9, len(stages) * 0.7), 5))
    if stages:
        colors = ["#2c7a4b" if value > 0 else "#b43c3c" for value in values]
        axis.bar(range(len(stages)), values, color=colors)
        axis.set_xticks(range(len(stages)), stages, rotation=45, ha="right")
    else:
        axis.text(0.5, 0.5, "No graph-stage coverage", ha="center", va="center")
    axis.set_ylabel("Graph executions")
    axis.set_title("Graph-stage coverage")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def generate_activation_report(
    output_dir: Path,
    rows: list[dict[str, Any]],
    convergence: dict[str, Any],
    coverage: dict[str, Any],
    *,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Always write JSON/CSV and generate plots when matplotlib is available."""

    output_dir = Path(output_dir)
    statistics = write_activation_statistics(output_dir, rows, metadata=metadata)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        _draw_activation_heatmap(plt, rows, output_dir / PNG_NAMES[0])
        _draw_outlier_distribution(plt, rows, output_dir / PNG_NAMES[1])
        _draw_scale_convergence(plt, convergence, output_dir / PNG_NAMES[2])
        _draw_graph_coverage(plt, coverage, output_dir / PNG_NAMES[3])
    except (ImportError, ModuleNotFoundError) as exc:
        skipped = {
            "schema_version": 1,
            "status": "skipped",
            "reason": f"matplotlib is unavailable: {exc}",
            "required_pngs": list(PNG_NAMES),
            "numeric_reports_written": ["activation_stats.json", "activation_stats.csv"],
        }
        _atomic_json(output_dir / "activation_report_skipped.json", skipped)
        _atomic_json(output_dir / "activation_report_status.json", skipped)
        return {"statistics": statistics, "plots": skipped}
    except Exception as exc:
        skipped = {
            "schema_version": 1,
            "status": "skipped",
            "reason": f"diagnostic plot generation failed: {type(exc).__name__}: {exc}",
            "required_pngs": list(PNG_NAMES),
            "numeric_reports_written": ["activation_stats.json", "activation_stats.csv"],
        }
        _atomic_json(output_dir / "activation_report_skipped.json", skipped)
        _atomic_json(output_dir / "activation_report_status.json", skipped)
        return {"statistics": statistics, "plots": skipped}

    status = {
        "schema_version": 1,
        "status": "generated",
        "pngs": list(PNG_NAMES),
        "numeric_reports_written": ["activation_stats.json", "activation_stats.csv"],
    }
    _atomic_json(output_dir / "activation_report_status.json", status)
    return {"statistics": statistics, "plots": status}
