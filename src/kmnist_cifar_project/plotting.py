from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from matplotlib import font_manager


def _configure_matplotlib_fonts() -> None:
    preferred_fonts = [
        "Malgun Gothic",
        "Yu Gothic",
        "Meiryo",
        "Noto Sans CJK JP",
        "Noto Sans CJK KR",
        "Arial Unicode MS",
    ]
    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    for font_name in preferred_fonts:
        if font_name in available_fonts:
            matplotlib.rcParams["font.family"] = "sans-serif"
            matplotlib.rcParams["font.sans-serif"] = [font_name, "DejaVu Sans"]
            break
    matplotlib.rcParams["axes.unicode_minus"] = False


_configure_matplotlib_fonts()

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from .utils import ensure_dir


def _to_image_array(x):
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    if x.ndim == 3 and x.shape[0] in {1, 3}:
        x = np.transpose(x, (1, 2, 0))
    if x.ndim == 3 and x.shape[-1] == 1:
        x = x[:, :, 0]
    min_value = float(np.nanmin(x))
    max_value = float(np.nanmax(x))
    if min_value < 0.0 or max_value > 1.0:
        denom = max(max_value - min_value, 1e-8)
        x = (x - min_value) / denom
    return x


def plot_dataset_sample_grid(
    dataset,
    class_names: list[str],
    path: str | Path,
    n: int = 40,
    cols: int = 10,
    seed: int = 42,
    title: str = "dataset samples",
) -> None:
    if len(dataset) == 0:
        return
    rng = np.random.default_rng(seed)
    indices = np.arange(len(dataset))
    if len(indices) > n:
        indices = rng.choice(indices, size=int(n), replace=False)

    rows = int(np.ceil(len(indices) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.2, rows * 1.45))
    axes = np.array(axes).reshape(-1)

    for ax, idx in zip(axes, indices):
        image, label = dataset[int(idx)]
        arr = _to_image_array(image)
        ax.imshow(arr, cmap="gray" if arr.ndim == 2 else None)
        ax.set_title(class_names[int(label)][:12], fontsize=7)
        ax.axis("off")
    for ax in axes[len(indices):]:
        ax.axis("off")

    fig.suptitle(title, fontsize=11)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_class_distribution(labels: np.ndarray, class_names: list[str], path: str | Path, title: str) -> pd.DataFrame:
    counts = np.bincount(labels.astype(int), minlength=len(class_names))
    total = max(int(counts.sum()), 1)
    df = pd.DataFrame(
        {
            "label": np.arange(len(class_names)),
            "class_name": class_names,
            "count": counts.astype(int),
            "ratio": counts / total,
        }
    )
    plt.figure(figsize=(max(7, len(class_names) * 0.75), 4))
    plt.bar(df["class_name"], df["count"])
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.ylabel("count")
    plt.title(title)
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return df


def plot_training_history(history: list[dict], path: str | Path, title: str) -> None:
    if not history:
        return
    df = pd.DataFrame(history)
    plt.figure(figsize=(7, 4))
    for col in df.columns:
        if col != "epoch" and pd.api.types.is_numeric_dtype(df[col]):
            plt.plot(df["epoch"], df[col], marker="o", label=col)
    plt.title(title)
    plt.xlabel("epoch")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_reconstruction_grid(
    originals,
    reconstructions,
    labels,
    class_names: list[str],
    path: str | Path,
    title: str = "original vs reconstruction",
    n: int = 10,
) -> None:
    n = min(int(n), len(originals), len(reconstructions))
    if n <= 0:
        return
    fig, axes = plt.subplots(2, n, figsize=(n * 1.25, 3.0))
    axes = np.array(axes).reshape(2, n)
    for i in range(n):
        orig = _to_image_array(originals[i])
        recon = _to_image_array(reconstructions[i])
        axes[0, i].imshow(orig, cmap="gray" if orig.ndim == 2 else None)
        axes[0, i].set_title(class_names[int(labels[i])][:10], fontsize=7)
        axes[0, i].axis("off")
        axes[1, i].imshow(recon, cmap="gray" if recon.ndim == 2 else None, vmin=0, vmax=1)
        axes[1, i].set_title("recon", fontsize=7)
        axes[1, i].axis("off")
    axes[0, 0].set_ylabel("original", fontsize=8)
    axes[1, 0].set_ylabel("recon", fontsize=8)
    fig.suptitle(title, fontsize=11)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_embedding_2d(
    features: np.ndarray,
    labels: np.ndarray,
    class_names: list[str],
    path: str | Path,
    title: str,
    max_points: int,
    seed: int,
    reducer: str = "pca",
    perplexity: int = 30,
) -> None:
    if len(features) == 0:
        return
    if len(features) > max_points:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(features), size=int(max_points), replace=False)
        features_plot = features[idx]
        labels_plot = labels[idx]
    else:
        features_plot = features
        labels_plot = labels

    if features_plot.shape[1] > 2:
        if reducer == "tsne":
            z2 = TSNE(
                n_components=2,
                perplexity=min(int(perplexity), max(5, len(features_plot) // 3)),
                learning_rate="auto",
                init="pca",
                random_state=seed,
            ).fit_transform(features_plot)
        else:
            z2 = PCA(n_components=2, random_state=seed).fit_transform(features_plot)
    else:
        z2 = features_plot

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(z2[:, 0], z2[:, 1], c=labels_plot, s=8, alpha=0.75, cmap="tab10")
    plt.title(title)
    plt.xlabel("dim 1")
    plt.ylabel("dim 2")
    if class_names and len(np.unique(labels_plot)) <= len(class_names):
        handles, _ = scatter.legend_elements(num=len(class_names))
        plt.legend(handles, class_names, bbox_to_anchor=(1.04, 1), loc="upper left", fontsize=7)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_cluster_heatmap(table: pd.DataFrame, path: str | Path, title: str, normalize: bool = True) -> None:
    values = table.to_numpy(dtype=float)
    if normalize:
        row_sums = values.sum(axis=1, keepdims=True)
        values = np.divide(values, row_sums, out=np.zeros_like(values), where=row_sums != 0)

    plt.figure(figsize=(10, 5.5))
    plt.imshow(values, aspect="auto", cmap="viridis")
    plt.colorbar(label="ratio inside cluster" if normalize else "count")
    plt.xticks(range(len(table.columns)), table.columns, rotation=45, ha="right", fontsize=8)
    plt.yticks(range(len(table.index)), table.index, fontsize=8)
    plt.xlabel("true class")
    plt.ylabel("cluster")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_cluster_samples(
    dataset,
    cluster_labels: np.ndarray,
    class_names: list[str],
    path: str | Path,
    n_per_cluster: int,
    max_clusters: int,
    seed: int,
) -> None:
    clusters = sorted(np.unique(cluster_labels))[: int(max_clusters)]
    if not clusters:
        return
    rng = np.random.default_rng(seed)
    rows = len(clusters)
    cols = int(n_per_cluster)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.15, rows * 1.35))
    axes = np.array(axes).reshape(rows, cols)

    for r, cluster_id in enumerate(clusters):
        idx = np.where(cluster_labels == cluster_id)[0]
        if len(idx) > cols:
            idx = rng.choice(idx, size=cols, replace=False)
        for c in range(cols):
            ax = axes[r, c]
            ax.axis("off")
            if c >= len(idx):
                continue
            image, label = dataset[int(idx[c])]
            arr = _to_image_array(image)
            ax.imshow(arr, cmap="gray" if arr.ndim == 2 else None)
            if c == 0:
                ax.set_ylabel(f"C{cluster_id}", rotation=0, labelpad=16, fontsize=8)
            ax.set_title(class_names[int(label)][:10], fontsize=6)

    fig.suptitle("sample images by cluster", fontsize=11)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_mapped_class_grids(
    dataset,
    predicted_classes: np.ndarray,
    true_labels: np.ndarray,
    class_names: list[str],
    output_dir: str | Path,
    n_per_class: int,
    cols: int,
    seed: int,
) -> None:
    out = ensure_dir(output_dir)
    rng = np.random.default_rng(seed)
    for pred_label, pred_name in enumerate(class_names):
        indices = np.where(predicted_classes == pred_label)[0]
        if len(indices) == 0:
            continue
        if len(indices) > n_per_class:
            indices = rng.choice(indices, size=int(n_per_class), replace=False)
        rows = int(np.ceil(len(indices) / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.15, rows * 1.35))
        axes = np.array(axes).reshape(-1)
        true_counts = np.bincount(true_labels[indices].astype(int), minlength=len(class_names))
        dominant_true = class_names[int(true_counts.argmax())]
        shown_purity = true_counts.max() / max(len(indices), 1)

        for ax, idx in zip(axes, indices):
            image, true_label = dataset[int(idx)]
            arr = _to_image_array(image)
            ax.imshow(arr, cmap="gray" if arr.ndim == 2 else None)
            color = "green" if int(true_label) == pred_label else "red"
            ax.set_title(class_names[int(true_label)][:10], fontsize=6, color=color)
            ax.axis("off")
        for ax in axes[len(indices):]:
            ax.axis("off")
        fig.suptitle(
            f"mapped as {pred_name} | dominant true: {dominant_true} | shown purity: {shown_purity:.2f}",
            fontsize=10,
        )
        plt.tight_layout()
        plt.savefig(out / f"mapped_as_{pred_label}_{pred_name.replace('/', '_')}.png", dpi=160)
        plt.close()


def plot_mismatch_grid(
    dataset,
    predicted_classes: np.ndarray,
    true_labels: np.ndarray,
    class_names: list[str],
    path: str | Path,
    n: int,
    cols: int,
    seed: int,
) -> None:
    mismatch_idx = np.where(predicted_classes != true_labels)[0]
    if len(mismatch_idx) == 0:
        return
    rng = np.random.default_rng(seed)
    if len(mismatch_idx) > n:
        mismatch_idx = rng.choice(mismatch_idx, size=int(n), replace=False)
    rows = int(np.ceil(len(mismatch_idx) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.35, rows * 1.55))
    axes = np.array(axes).reshape(-1)
    for ax, idx in zip(axes, mismatch_idx):
        image, _ = dataset[int(idx)]
        true_label = int(true_labels[int(idx)])
        pred_label = int(predicted_classes[int(idx)])
        arr = _to_image_array(image)
        ax.imshow(arr, cmap="gray" if arr.ndim == 2 else None)
        ax.set_title(f"{class_names[true_label][:7]} -> {class_names[pred_label][:7]}", fontsize=6, color="red")
        ax.axis("off")
    for ax in axes[len(mismatch_idx):]:
        ax.axis("off")
    fig.suptitle("mismatched mapped class examples | true -> mapped", fontsize=10)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_metric_bars(df: pd.DataFrame, output_dir: str | Path, metrics: list[str]) -> None:
    out = ensure_dir(output_dir)
    for metric in metrics:
        if metric not in df.columns:
            continue
        plot_df = df.sort_values(metric, ascending=False).copy()
        labels = plot_df["run_name"].astype(str).tolist()
        plt.figure(figsize=(max(9, len(labels) * 0.55), 5))
        plt.bar(range(len(plot_df)), plot_df[metric])
        plt.xticks(range(len(plot_df)), labels, rotation=60, ha="right", fontsize=8)
        plt.ylabel(metric)
        plt.title(f"test {metric} by run")
        plt.grid(axis="y", alpha=0.25)
        plt.tight_layout()
        plt.savefig(out / f"test_{metric}_bar.png", dpi=160)
        plt.close()


def plot_metric_bars_by_dataset(df: pd.DataFrame, output_dir: str | Path, metrics: list[str]) -> None:
    if "dataset.name" not in df.columns:
        plot_metric_bars(df, output_dir, metrics)
        return

    out = ensure_dir(output_dir)
    grouped = list(df.groupby("dataset.name", dropna=False))
    for metric in metrics:
        if metric not in df.columns:
            continue
        fig, axes = plt.subplots(1, len(grouped), figsize=(max(7, len(grouped) * 6), 5), squeeze=False)
        axes = axes.reshape(-1)
        for ax, (dataset_name, dataset_df) in zip(axes, grouped):
            dataset_label = str(dataset_name)
            plot_df = dataset_df.sort_values(metric, ascending=False).copy()
            labels = plot_df["run_name"].astype(str).tolist()
            ax.bar(range(len(plot_df)), plot_df[metric])
            ax.set_xticks(range(len(plot_df)))
            ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=8)
            ax.set_ylabel(metric)
            ax.set_title(f"{dataset_label}")
            ax.grid(axis="y", alpha=0.25)
        fig.suptitle(f"test {metric} by run")
        plt.tight_layout()
        plt.savefig(out / f"test_{metric}_by_dataset_subplots.png", dpi=160)
        plt.close()
