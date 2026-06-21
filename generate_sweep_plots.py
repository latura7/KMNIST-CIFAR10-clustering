from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kmnist_cifar_project.aggregate import MAIN_METRICS, collect_runs
from kmnist_cifar_project.config import load_yaml
from kmnist_cifar_project.evaluation import evaluate_clustering
from kmnist_cifar_project.utils import ensure_dir, safe_name, timestamp


def _metric_grid(
    df: pd.DataFrame,
    x_col: str,
    group_col: str,
    output_path: Path,
    title: str,
    xlabel: str,
) -> None:
    if df.empty:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5), sharex=True)
    axes = axes.flatten()
    for ax, metric in zip(axes, MAIN_METRICS):
        if metric not in df.columns:
            ax.axis("off")
            continue
        for group_name, group_df in df.groupby(group_col):
            plot_df = group_df.sort_values(x_col)
            ax.plot(plot_df[x_col], plot_df[metric], marker="o", label=str(group_name))
        ax.set_title(metric)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("score")
        ax.grid(True, alpha=0.3)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(4, len(labels)), bbox_to_anchor=(0.5, 0.98))
    fig.suptitle(title, y=1.04, fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _line_plot(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    group_col: str,
    output_path: Path,
    title: str,
    xlabel: str,
    ylabel: str,
) -> None:
    if df.empty or y_col not in df.columns:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4.8))
    for group_name, group_df in df.groupby(group_col):
        plot_df = group_df.sort_values(x_col)
        plt.plot(plot_df[x_col], plot_df[y_col], marker="o", label=str(group_name))
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=170)
    plt.close()


def _plot_ae_reconstruction_loss(ae_epoch_df: pd.DataFrame, output_dir: Path) -> Path | None:
    rows: list[dict] = []
    for dataset, group_df in ae_epoch_df.groupby("dataset.name"):
        longest = group_df.sort_values("epochs").iloc[-1]
        history_path = Path(str(longest["run_dir"])) / "training_history.csv"
        if not history_path.exists():
            continue
        history = pd.read_csv(history_path)
        if "epoch" not in history.columns or "reconstruction_loss" not in history.columns:
            continue
        for _, hrow in history.iterrows():
            rows.append(
                {
                    "dataset": dataset,
                    "source_run": longest["run_name"],
                    "epoch": int(hrow["epoch"]),
                    "reconstruction_loss": float(hrow["reconstruction_loss"]),
                }
            )

    if not rows:
        return None

    loss_df = pd.DataFrame(rows)
    loss_df.to_csv(output_dir / "ae_reconstruction_loss_curves.csv", index=False, encoding="utf-8-sig")
    path = output_dir / "ae_reconstruction_loss_curves.png"
    _line_plot(
        loss_df,
        "epoch",
        "reconstruction_loss",
        "dataset",
        path,
        "AutoEncoder reconstruction loss",
        "epoch",
        "reconstruction loss",
    )
    return path


def _test_metric_rows(df: pd.DataFrame) -> pd.DataFrame:
    keep = ["run_name", "run_dir", "dataset.name", "feature.kind", "clustering.method"]
    keep += [col for col in MAIN_METRICS if col in df.columns]
    return df[[col for col in keep if col in df.columns]].copy()


def _add_pca_explained_variance(df: pd.DataFrame) -> pd.DataFrame:
    values: list[float] = []
    for _, row in df.iterrows():
        pca_path = Path(str(row["run_dir"])) / "models" / "pca.joblib"
        if not pca_path.exists():
            values.append(float("nan"))
            continue
        try:
            pca = joblib.load(pca_path)
            values.append(float(np.sum(getattr(pca, "explained_variance_ratio_", [np.nan]))))
        except Exception:
            values.append(float("nan"))
    out = df.copy()
    out["explained_variance"] = values
    return out


def plot_completed_experiment_sweeps(runs_dir: Path, output_dir: Path) -> list[Path]:
    df = collect_runs(runs_dir)
    created: list[Path] = []
    if df.empty:
        return created

    # PCA component sweep: includes the main PCA baseline row plus added sweep rows.
    pca_df = df[
        (df.get("feature.kind") == "pca")
        & (df.get("clustering.method") == "kmeans")
        & df.get("feature.pca.n_components").notna()
    ].copy()
    if not pca_df.empty:
        pca_df["pca_components"] = pd.to_numeric(pca_df["feature.pca.n_components"], errors="coerce")
        pca_df = _add_pca_explained_variance(pca_df)
        cols = ["run_name", "dataset.name", "pca_components", "explained_variance"] + MAIN_METRICS
        pca_csv = output_dir / "pca_component_sweep_metrics.csv"
        pca_df[[col for col in cols if col in pca_df.columns]].sort_values(["dataset.name", "pca_components"]).to_csv(
            pca_csv, index=False, encoding="utf-8-sig"
        )
        p = output_dir / "pca_component_sweep_metrics.png"
        _metric_grid(pca_df, "pca_components", "dataset.name", p, "PCA component sweep", "number of PCA components")
        created.append(p)
        p = output_dir / "pca_component_sweep_explained_variance.png"
        _line_plot(
            pca_df,
            "pca_components",
            "explained_variance",
            "dataset.name",
            p,
            "PCA cumulative explained variance",
            "number of PCA components",
            "explained variance ratio sum",
        )
        created.append(p)

    # AE latent dimension sweep: hold epoch count at each dataset's main baseline.
    ae_df = df[(df.get("feature.kind") == "ae") & (df.get("clustering.method") == "kmeans")].copy()
    if not ae_df.empty:
        ae_df["latent_dim"] = pd.to_numeric(ae_df.get("feature.ae.latent_dim"), errors="coerce")
        ae_df["epochs"] = pd.to_numeric(ae_df.get("feature.ae.epochs"), errors="coerce")
        ae_df = ae_df.dropna(subset=["latent_dim", "epochs"])

        baseline_epochs = {"KMNIST": 5, "CIFAR10": 8}
        latent_parts = []
        for dataset, epochs in baseline_epochs.items():
            latent_parts.append(ae_df[(ae_df["dataset.name"] == dataset) & (ae_df["epochs"] == epochs)])
        ae_latent_df = pd.concat(latent_parts, ignore_index=True) if latent_parts else pd.DataFrame()
        if not ae_latent_df.empty:
            cols = ["run_name", "dataset.name", "latent_dim", "epochs"] + MAIN_METRICS
            ae_latent_df[[col for col in cols if col in ae_latent_df.columns]].sort_values(["dataset.name", "latent_dim"]).to_csv(
                output_dir / "ae_latent_dim_sweep_metrics.csv", index=False, encoding="utf-8-sig"
            )
            p = output_dir / "ae_latent_dim_sweep_metrics.png"
            _metric_grid(ae_latent_df, "latent_dim", "dataset.name", p, "AutoEncoder latent dimension sweep", "latent dimension")
            created.append(p)

        baseline_latent = {"KMNIST": 16, "CIFAR10": 64}
        epoch_parts = []
        for dataset, latent_dim in baseline_latent.items():
            epoch_parts.append(ae_df[(ae_df["dataset.name"] == dataset) & (ae_df["latent_dim"] == latent_dim)])
        ae_epoch_df = pd.concat(epoch_parts, ignore_index=True) if epoch_parts else pd.DataFrame()
        if not ae_epoch_df.empty:
            cols = ["run_name", "dataset.name", "latent_dim", "epochs"] + MAIN_METRICS
            ae_epoch_df[[col for col in cols if col in ae_epoch_df.columns]].sort_values(["dataset.name", "epochs"]).to_csv(
                output_dir / "ae_epoch_sweep_metrics.csv", index=False, encoding="utf-8-sig"
            )
            p = output_dir / "ae_epoch_sweep_metrics.png"
            _metric_grid(ae_epoch_df, "epochs", "dataset.name", p, "AutoEncoder epoch sweep", "epochs")
            created.append(p)
            p = _plot_ae_reconstruction_loss(ae_epoch_df, output_dir)
            if p is not None:
                created.append(p)

    return created


def _load_saved_features(run_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    path = run_dir / "features_train_test.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing saved feature file: {path}")
    data = np.load(path)
    return data["z_train"], data["y_train"], data["z_test"], data["y_test"]


def run_k_sweep(
    source_runs: list[Path],
    k_values: list[int],
    output_dir: Path,
    seed: int,
    n_init: int,
    max_iter: int,
    silhouette_sample_size: int,
) -> list[Path]:
    rows: list[dict] = []
    for run_dir in source_runs:
        if not run_dir.exists():
            print(f"[k-sweep] skip missing run: {run_dir}")
            continue
        cfg = load_yaml(run_dir / "config.yaml")
        z_train, y_train, z_test, y_test = _load_saved_features(run_dir)
        n_classes = int(max(np.max(y_train), np.max(y_test)) + 1)
        dataset = str(cfg["dataset"]["name"])
        source_name = str(cfg["project"]["run_name"])
        for k in k_values:
            print(f"[k-sweep] {source_name}: k={k}")
            model = KMeans(n_clusters=int(k), random_state=seed, n_init=n_init, max_iter=max_iter)
            train_clusters = model.fit_predict(z_train)
            test_clusters = model.predict(z_test)
            train_metrics = evaluate_clustering(z_train, y_train, train_clusters, n_classes, silhouette_sample_size, seed)
            test_metrics = evaluate_clustering(z_test, y_test, test_clusters, n_classes, silhouette_sample_size, seed + 1)
            row = {
                "source_run": source_name,
                "source_run_dir": str(run_dir),
                "dataset": dataset,
                "k": int(k),
            }
            row.update(test_metrics)
            for metric, value in train_metrics.items():
                row[f"train_{metric}"] = value
            rows.append(row)

    created: list[Path] = []
    if not rows:
        return created
    df = pd.DataFrame(rows)
    csv_path = output_dir / "k_sweep_resnet_full_metrics.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    p = output_dir / "k_sweep_resnet_full_metrics.png"
    _metric_grid(df, "k", "dataset", p, "K-means k sweep on saved ResNet18 full features", "number of clusters k")
    created.append(p)
    return created


def copy_to_report_figures(paths: list[Path], report_figures_dir: Path) -> None:
    report_figures_dir.mkdir(parents=True, exist_ok=True)
    for path in paths:
        if path.exists():
            shutil.copy2(path, report_figures_dir / path.name)


def parse_int_list(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate sweep plots for the final project report.")
    parser.add_argument("--runs-dir", default="final_project/outputs/runs")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--report-figures-dir", default="final_project/report_figures")
    parser.add_argument("--k-values", default="5,8,10,12,15,20")
    parser.add_argument("--k-source-run", action="append", default=[])
    parser.add_argument("--skip-k-sweep", action="store_true")
    parser.add_argument("--skip-completed-run-sweeps", action="store_true")
    parser.add_argument("--no-copy-report-figures", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-init", type=int, default=50)
    parser.add_argument("--max-iter", type=int, default=500)
    parser.add_argument("--silhouette-sample-size", type=int, default=2000)
    args = parser.parse_args()

    output_dir = ensure_dir(Path(args.output_dir) if args.output_dir else ROOT / "outputs" / "sweeps" / timestamp())
    created: list[Path] = []

    if not args.skip_completed_run_sweeps:
        created.extend(plot_completed_experiment_sweeps(Path(args.runs_dir), output_dir))

    if not args.skip_k_sweep:
        source_runs = [Path(x) for x in args.k_source_run]
        if not source_runs:
            source_runs = [
                ROOT / "outputs" / "runs" / "kmnist_resnet18_e20_kmeans_full",
                ROOT / "outputs" / "runs" / "cifar10_resnet18_e20_kmeans_full",
            ]
        created.extend(
            run_k_sweep(
                source_runs,
                parse_int_list(args.k_values),
                output_dir,
                args.seed,
                args.n_init,
                args.max_iter,
                args.silhouette_sample_size,
            )
        )

    if not args.no_copy_report_figures:
        copy_to_report_figures(created, Path(args.report_figures_dir))

    print(f"[sweeps] saved -> {output_dir}")
    for path in created:
        print(path)


if __name__ == "__main__":
    main()
