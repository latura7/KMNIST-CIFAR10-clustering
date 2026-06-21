from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)


def clustering_purity(y_true: np.ndarray, cluster_labels: np.ndarray, n_classes: int) -> float:
    correct = 0
    for cluster_id in np.unique(cluster_labels):
        mask = cluster_labels == cluster_id
        counts = np.bincount(y_true[mask].astype(int), minlength=n_classes)
        correct += int(counts.max())
    return correct / len(y_true)


def cluster_majority_mapping(cluster_labels: np.ndarray, y_true: np.ndarray, n_classes: int) -> dict[int, int]:
    mapping: dict[int, int] = {}
    for cluster_id in np.unique(cluster_labels):
        mask = cluster_labels == cluster_id
        counts = np.bincount(y_true[mask].astype(int), minlength=n_classes)
        mapping[int(cluster_id)] = int(counts.argmax())
    return mapping


def map_clusters(cluster_labels: np.ndarray, mapping: dict[int, int]) -> np.ndarray:
    return np.array([mapping.get(int(label), -1) for label in cluster_labels], dtype=int)


def safe_silhouette(features: np.ndarray, cluster_labels: np.ndarray, sample_size: int, seed: int) -> float:
    unique = np.unique(cluster_labels)
    if len(unique) < 2 or len(unique) >= len(cluster_labels):
        return float("nan")
    if len(features) > sample_size:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(features), size=sample_size, replace=False)
        features = features[idx]
        cluster_labels = cluster_labels[idx]
    try:
        return float(silhouette_score(features, cluster_labels))
    except Exception:
        return float("nan")


def evaluate_clustering(
    features: np.ndarray,
    y_true: np.ndarray,
    cluster_labels: np.ndarray,
    n_classes: int,
    sample_size: int,
    seed: int,
) -> dict[str, float]:
    return {
        "ARI": float(adjusted_rand_score(y_true, cluster_labels)),
        "NMI": float(normalized_mutual_info_score(y_true, cluster_labels)),
        "purity": float(clustering_purity(y_true, cluster_labels, n_classes)),
        "silhouette": safe_silhouette(features, cluster_labels, sample_size, seed),
    }


def cluster_class_table(cluster_labels: np.ndarray, y_true: np.ndarray, class_names: list[str]) -> pd.DataFrame:
    table = pd.crosstab(
        pd.Series(cluster_labels, name="cluster"),
        pd.Series([class_names[int(y)] for y in y_true], name="true_class"),
    )
    return table


def mapping_summary(cluster_labels: np.ndarray, y_true: np.ndarray, class_names: list[str]) -> pd.DataFrame:
    rows = []
    n_classes = len(class_names)
    for cluster_id in sorted(np.unique(cluster_labels)):
        mask = cluster_labels == cluster_id
        counts = np.bincount(y_true[mask].astype(int), minlength=n_classes)
        majority = int(counts.argmax())
        rows.append(
            {
                "cluster": int(cluster_id),
                "mapped_class": class_names[majority],
                "cluster_size": int(mask.sum()),
                "majority_count": int(counts[majority]),
                "majority_ratio": float(counts[majority] / max(mask.sum(), 1)),
            }
        )
    return pd.DataFrame(rows)


def class_cluster_summary(cluster_labels: np.ndarray, y_true: np.ndarray, class_names: list[str]) -> pd.DataFrame:
    rows = []
    for class_id, class_name in enumerate(class_names):
        mask = y_true == class_id
        class_size = int(mask.sum())
        if class_size == 0:
            rows.append(
                {
                    "true_label": class_id,
                    "true_class": class_name,
                    "class_size": 0,
                    "dominant_cluster": None,
                    "dominant_cluster_count": 0,
                    "dominant_cluster_ratio": 0.0,
                    "num_clusters_present": 0,
                }
            )
            continue
        clusters = cluster_labels[mask].astype(int)
        unique_clusters, counts = np.unique(clusters, return_counts=True)
        max_idx = int(np.argmax(counts))
        rows.append(
            {
                "true_label": class_id,
                "true_class": class_name,
                "class_size": class_size,
                "dominant_cluster": int(unique_clusters[max_idx]),
                "dominant_cluster_count": int(counts[max_idx]),
                "dominant_cluster_ratio": float(counts[max_idx] / class_size),
                "num_clusters_present": int(len(unique_clusters)),
            }
        )
    return pd.DataFrame(rows)


def mapped_mismatch_pairs(mapped_pred: np.ndarray, y_true: np.ndarray, class_names: list[str]) -> pd.DataFrame:
    rows = []
    for true_id in range(len(class_names)):
        for pred_id in range(len(class_names)):
            if true_id == pred_id:
                continue
            count = int(((y_true == true_id) & (mapped_pred == pred_id)).sum())
            if count > 0:
                rows.append(
                    {
                        "true_label": true_id,
                        "true_class": class_names[true_id],
                        "mapped_label": pred_id,
                        "mapped_class": class_names[pred_id],
                        "count": count,
                    }
                )
    return pd.DataFrame(rows).sort_values("count", ascending=False).reset_index(drop=True) if rows else pd.DataFrame(
        columns=["true_label", "true_class", "mapped_label", "mapped_class", "count"]
    )
