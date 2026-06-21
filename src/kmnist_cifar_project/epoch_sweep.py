from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from .aggregate import MAIN_METRICS
from .config import deep_merge, print_config
from .runner import run_experiment
from .utils import ensure_dir, safe_name, save_json, timestamp


def parse_epochs(raw: str | list[int] | tuple[int, ...]) -> list[int]:
    if isinstance(raw, (list, tuple)):
        epochs = [int(x) for x in raw]
    else:
        epochs = [int(x.strip()) for x in str(raw).split(",") if x.strip()]
    epochs = sorted(set(epochs))
    if not epochs:
        raise ValueError("At least one epoch value is required.")
    if any(x <= 0 for x in epochs):
        raise ValueError(f"Epoch values must be positive: {epochs}")
    return epochs


def _dataset_training_defaults(dataset_name: str) -> dict[str, Any]:
    normalized = dataset_name.replace("-", "").replace("_", "").upper()
    if normalized in {"CIFAR10", "CIFAR"}:
        return {
            "normalize": True,
            "augment_train": True,
            "random_crop_padding": 4,
            "random_horizontal_flip": True,
            "random_affine_degrees": 0,
            "random_translate": 0,
        }
    if normalized == "KMNIST":
        return {
            "normalize": True,
            "augment_train": True,
            "random_crop_padding": 0,
            "random_horizontal_flip": False,
            "random_affine_degrees": 10,
            "random_translate": 0.1,
        }
    raise ValueError(f"Unsupported epoch sweep dataset: {dataset_name}")


def _run_name(dataset_name: str, epoch: int, train_size: int, test_size: int, run_prefix: str | None) -> str:
    if run_prefix:
        return safe_name(f"{run_prefix}_e{epoch}")
    dataset_part = safe_name(dataset_name).lower()
    size_part = "full" if int(train_size) <= 0 and int(test_size) <= 0 else f"train{train_size}_test{test_size}"
    return safe_name(f"{dataset_part}_resnet18_e{epoch}_kmeans_{size_part}")


def _build_epoch_cfg(
    base_cfg: dict[str, Any],
    dataset_name: str,
    epoch: int,
    train_size: int,
    test_size: int,
    run_prefix: str | None,
) -> dict[str, Any]:
    dataset_defaults = _dataset_training_defaults(dataset_name)
    override = {
        "project": {
            "run_name": _run_name(dataset_name, epoch, train_size, test_size, run_prefix),
            "run_dir_mode": "name",
            "reuse_existing_run": True,
        },
        "dataset": {
            "name": dataset_name,
            "train_size": int(train_size),
            "test_size": int(test_size),
            **dataset_defaults,
        },
        "feature": {
            "kind": "cnn",
            "scale_before_cluster": True,
            "cnn": {
                "arch": "resnet18",
                "embedding_dim": 128,
                "dropout": 0.0,
                "use_batchnorm": True,
                "epochs": int(epoch),
                "optimizer": "sgd",
                "lr": 0.05,
                "weight_decay": 0.0005,
            },
        },
        "clustering": {
            "method": "kmeans",
            "n_clusters": 10,
        },
        "visualization": {
            "tsne": {
                "enabled": False,
            },
        },
    }
    return deep_merge(base_cfg, override)


def _expected_run_dir(cfg: dict[str, Any]) -> Path:
    return Path(cfg["project"]["output_dir"]) / "runs" / safe_name(cfg["project"]["run_name"])


def _read_epoch_result(run_dir: Path, epoch: int) -> dict[str, Any]:
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics.json for epoch={epoch}: {metrics_path}")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    row: dict[str, Any] = {
        "epoch": int(epoch),
        "run_name": metrics.get("run_name", run_dir.name),
        "run_dir": str(run_dir),
    }
    for metric in MAIN_METRICS:
        row[metric] = (metrics.get("test") or {}).get(metric)
        row[f"train_{metric}"] = (metrics.get("train") or {}).get(metric)

    history_path = run_dir / "training_history.csv"
    if history_path.exists():
        history = pd.read_csv(history_path)
        if not history.empty:
            final = history.sort_values("epoch").iloc[-1]
            for col in ["train_loss", "train_acc", "eval_acc"]:
                if col in final:
                    row[f"final_{col}"] = float(final[col])
    return row


def _plot_epoch_metrics(df: pd.DataFrame, output_dir: Path) -> Path:
    path = output_dir / "resnet_epoch_sweep_clustering_metrics.png"
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.2), sharex=True)
    axes = axes.flatten()
    plot_df = df.sort_values("epoch")
    for ax, metric in zip(axes, MAIN_METRICS):
        ax.plot(plot_df["epoch"], plot_df[metric], marker="o", linewidth=2)
        for _, row in plot_df.iterrows():
            value = row.get(metric)
            if pd.notna(value):
                ax.text(row["epoch"], value, f"{value:.3f}", fontsize=8, ha="center", va="bottom")
        ax.set_title(metric)
        ax.set_xlabel("ResNet18 training epochs")
        ax.set_ylabel("test score")
        ax.grid(True, alpha=0.3)
    fig.suptitle("ResNet18 epoch sweep: clustering metrics", y=1.02, fontsize=13)
    plt.tight_layout()
    plt.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_classifier_summary(df: pd.DataFrame, output_dir: Path) -> Path | None:
    cols = [c for c in ["final_train_acc", "final_eval_acc", "final_train_loss"] if c in df.columns]
    if not cols:
        return None
    path = output_dir / "resnet_epoch_sweep_classifier_summary.png"
    plot_df = df.sort_values("epoch")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    if "final_train_acc" in plot_df and "final_eval_acc" in plot_df:
        axes[0].plot(plot_df["epoch"], plot_df["final_train_acc"], marker="o", label="train acc")
        axes[0].plot(plot_df["epoch"], plot_df["final_eval_acc"], marker="o", label="test classifier acc")
        axes[0].set_ylabel("accuracy")
        axes[0].legend()
    else:
        axes[0].axis("off")
    axes[0].set_xlabel("ResNet18 training epochs")
    axes[0].set_title("Classifier accuracy")
    axes[0].grid(True, alpha=0.3)

    if "final_train_loss" in plot_df:
        axes[1].plot(plot_df["epoch"], plot_df["final_train_loss"], marker="o", color="#D55E00")
        axes[1].set_ylabel("loss")
    else:
        axes[1].axis("off")
    axes[1].set_xlabel("ResNet18 training epochs")
    axes[1].set_title("Final training loss")
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("ResNet18 epoch sweep: classifier summary", y=1.03, fontsize=13)
    plt.tight_layout()
    plt.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return path


def run_resnet_epoch_sweep(
    base_cfg: dict[str, Any],
    *,
    dataset_name: str,
    epochs: list[int],
    train_size: int,
    test_size: int,
    output_dir: str | Path | None,
    run_prefix: str | None,
    dry_run: bool,
    plot_only: bool,
) -> Path | None:
    sweep_cfgs = [
        _build_epoch_cfg(base_cfg, dataset_name, epoch, train_size, test_size, run_prefix)
        for epoch in epochs
    ]

    if dry_run:
        for cfg in sweep_cfgs:
            print(f"[epoch-sweep:dry-run] {cfg['project']['run_name']}")
            print_config(cfg)
        return None

    if output_dir is None:
        output_dir = Path(base_cfg["project"]["output_dir"]) / "sweeps" / f"resnet_epoch_{safe_name(dataset_name).lower()}_{timestamp()}"
    out_dir = ensure_dir(output_dir)

    rows = []
    run_dirs = []
    for epoch, cfg in zip(epochs, sweep_cfgs):
        print(f"[epoch-sweep] epoch={epoch}: {cfg['project']['run_name']}")
        run_dir = _expected_run_dir(cfg) if plot_only else run_experiment(cfg)
        if run_dir is None:
            continue
        run_dir = Path(run_dir)
        run_dirs.append(run_dir)
        rows.append(_read_epoch_result(run_dir, epoch))

    if not rows:
        raise RuntimeError("No epoch sweep results were collected.")

    df = pd.DataFrame(rows).sort_values("epoch")
    csv_path = out_dir / "resnet_epoch_sweep_metrics.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    metric_plot = _plot_epoch_metrics(df, out_dir)
    classifier_plot = _plot_classifier_summary(df, out_dir)
    save_json(
        out_dir / "resnet_epoch_sweep_summary.json",
        {
            "dataset": dataset_name,
            "epochs": epochs,
            "train_size": train_size,
            "test_size": test_size,
            "run_dirs": [str(p) for p in run_dirs],
            "metrics_csv": str(csv_path),
            "metric_plot": str(metric_plot),
            "classifier_plot": str(classifier_plot) if classifier_plot else None,
            "plot_only": bool(plot_only),
        },
    )
    print(f"[epoch-sweep] saved -> {out_dir}")
    return out_dir
