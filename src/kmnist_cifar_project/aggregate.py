from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd

from .config import load_yaml
from .plotting import plot_metric_bars_by_dataset
from .utils import ensure_dir, flatten_dict, safe_name, save_json, timestamp


MAIN_METRICS = ["ARI", "NMI", "purity", "silhouette"]
PLOT_EXTENSIONS = {".png", ".jpg", ".jpeg", ".svg", ".pdf"}


def _read_run(run_dir: Path) -> dict | None:
    metrics_path = run_dir / "metrics.json"
    config_path = run_dir / "config.yaml"
    if not metrics_path.exists() or not config_path.exists():
        return None
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    cfg = load_yaml(config_path)
    flat_cfg = flatten_dict(cfg)
    row = {
        "run_name": metrics.get("run_name", run_dir.name),
        "run_dir": str(run_dir),
    }
    row.update(flat_cfg)
    for key in MAIN_METRICS:
        if key in (metrics.get("test") or {}):
            row[key] = metrics["test"][key]
    for key in MAIN_METRICS:
        if key in (metrics.get("train") or {}):
            row[f"train_{key}"] = metrics["train"][key]
    return row


def _run_rank(row: dict) -> tuple[int, float]:
    run_dir = Path(row["run_dir"])
    exact_name_match = int(run_dir.name == safe_name(row["run_name"]))
    try:
        mtime = run_dir.stat().st_mtime
    except OSError:
        mtime = 0.0
    return exact_name_match, mtime


def _deduplicate_runs(rows: list[dict]) -> list[dict]:
    selected: dict[str, dict] = {}
    for row in rows:
        key = str(row["run_name"])
        if key not in selected or _run_rank(row) > _run_rank(selected[key]):
            selected[key] = row
    return list(selected.values())


def _copy_plot_tree(src_dir: Path, dst_dir: Path) -> int:
    if not src_dir.exists():
        return 0
    count = 0
    for src_path in src_dir.rglob("*"):
        if not src_path.is_file() or src_path.suffix.lower() not in PLOT_EXTENSIONS:
            continue
        rel_path = src_path.relative_to(src_dir)
        dst_path = dst_dir / rel_path
        ensure_dir(dst_path.parent)
        shutil.copy2(src_path, dst_path)
        count += 1
    return count


def copy_run_plots_to_comparison(df: pd.DataFrame, output_dir: str | Path) -> pd.DataFrame:
    output_dir = Path(output_dir)
    run_plots_dir = ensure_dir(output_dir / "run_plots")
    rows = []
    for _, row in df.sort_values("run_name").iterrows():
        run_name = str(row["run_name"])
        run_dir = Path(str(row["run_dir"]))
        dst_dir = run_plots_dir / safe_name(run_name)
        copied_count = _copy_plot_tree(run_dir / "plots", dst_dir)
        rows.append(
            {
                "run_name": run_name,
                "run_dir": str(run_dir),
                "comparison_plot_dir": str(dst_dir),
                "copied_plot_count": copied_count,
            }
        )
    manifest = pd.DataFrame(rows)
    manifest.to_csv(output_dir / "copied_run_plots.csv", index=False, encoding="utf-8-sig")
    return manifest


def collect_runs(runs_dir: str | Path, run_dirs: list[Path] | None = None) -> pd.DataFrame:
    if run_dirs is None:
        candidates = sorted(Path(runs_dir).glob("*"))
    else:
        candidates = [Path(p) for p in run_dirs]
    rows = []
    for run_dir in candidates:
        if run_dir.is_dir():
            row = _read_run(run_dir)
            if row is not None:
                rows.append(row)
    rows = _deduplicate_runs(rows)
    return pd.DataFrame(rows)


def compare_runs(runs_dir: str | Path, output_dir: str | Path | None = None, run_dirs: list[Path] | None = None) -> Path:
    runs_dir = Path(runs_dir)
    if output_dir is None:
        root = runs_dir.parent if runs_dir.name == "runs" else runs_dir
        output_dir = root / "comparisons" / timestamp()
    output_dir = ensure_dir(output_dir)

    df = collect_runs(runs_dir, run_dirs=run_dirs)
    if df.empty:
        raise RuntimeError(f"No completed runs found under {runs_dir}")

    df.to_csv(output_dir / "all_runs_metrics.csv", index=False, encoding="utf-8-sig")
    plot_metric_bars_by_dataset(df, output_dir / "metric_bars", MAIN_METRICS)
    copied_plots = copy_run_plots_to_comparison(df, output_dir)

    sort_metric = "ARI" if "ARI" in df.columns else df.select_dtypes("number").columns[0]
    best = df.sort_values(sort_metric, ascending=False)
    best.to_csv(output_dir / f"runs_sorted_by_{sort_metric}.csv", index=False, encoding="utf-8-sig")
    if "dataset.name" in df.columns:
        best_by_dataset = (
            df.sort_values(sort_metric, ascending=False)
            .groupby("dataset.name", as_index=False)
            .head(3)
        )
        best_by_dataset.to_csv(output_dir / f"top3_by_dataset_{sort_metric}.csv", index=False, encoding="utf-8-sig")

    save_json(
        output_dir / "comparison_summary.json",
        {
            "runs_dir": str(runs_dir),
            "output_dir": str(output_dir),
            "num_runs": int(len(df)),
            "sort_metric": sort_metric,
            "copied_run_plot_count": int(copied_plots["copied_plot_count"].sum()),
        },
    )
    print(f"[compare] saved {len(df)} runs -> {output_dir}")
    return output_dir
