from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

from .data import make_dataloaders
from .evaluation import (
    cluster_class_table,
    cluster_majority_mapping,
    class_cluster_summary,
    evaluate_clustering,
    map_clusters,
    mapped_mismatch_pairs,
    mapping_summary,
)
from .models import CifarResNet18, ConvAutoEncoder, MLPAutoEncoder, SmallCNN
from .plotting import (
    plot_cluster_heatmap,
    plot_cluster_samples,
    plot_class_distribution,
    plot_dataset_sample_grid,
    plot_embedding_2d,
    plot_mapped_class_grids,
    plot_mismatch_grid,
    plot_reconstruction_grid,
    plot_training_history,
)
from .utils import ensure_dir, flatten_dict, safe_name, save_json, save_yaml, seed_everything, select_device, timestamp


def _collect_flattened(loader) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for images, labels in tqdm(loader, desc="collect flattened", leave=False):
        xs.append(images.view(images.size(0), -1).numpy())
        ys.append(labels.numpy())
    return np.concatenate(xs, axis=0).astype(np.float32), np.concatenate(ys, axis=0).astype(int)


def _collect_dataset_labels(dataset) -> np.ndarray:
    return np.array([int(dataset[i][1]) for i in range(len(dataset))], dtype=int)


def _indices_digest(indices: np.ndarray) -> str:
    return hashlib.sha1(np.asarray(indices, dtype=np.int64).tobytes()).hexdigest()[:10]


def _dataset_view_dir(cfg: dict, loaders: dict, output_dir: Path, run_dir: Path | None = None) -> Path:
    spec = loaders["spec"]
    seed = int(cfg["project"]["seed"])
    train_digest = _indices_digest(loaders["train_indices"])
    test_digest = _indices_digest(loaders["test_indices"])
    name = safe_name(
        f"{spec.name}_train{len(loaders['train_set'])}_test{len(loaders['test_set'])}"
        f"_seed{seed}_tr{train_digest}_te{test_digest}"
    )
    overview_cfg = cfg["visualization"].get("dataset_overview", {})
    if not bool(overview_cfg.get("shared", True)):
        if run_dir is None:
            return output_dir / "dataset_overview" / name
        return run_dir / "plots" / "dataset_overview"
    root = Path(overview_cfg.get("output_dir") or (output_dir / "shared_dataset_views"))
    return root / name


def _save_dataset_overview(cfg: dict, loaders: dict, output_dir: Path, run_dir: Path | None = None) -> Path | None:
    overview_cfg = cfg["visualization"].get("dataset_overview", {})
    if not bool(overview_cfg.get("enabled", True)):
        return None
    spec = loaders["spec"]
    overview_dir = ensure_dir(_dataset_view_dir(cfg, loaders, output_dir, run_dir))
    n = int(overview_cfg.get("sample_count", 40))
    cols = int(overview_cfg.get("cols", 10))
    seed = int(cfg["project"]["seed"])

    required = [
        overview_dir / "train_samples.png",
        overview_dir / "test_samples.png",
        overview_dir / "train_class_distribution.png",
        overview_dir / "test_class_distribution.png",
        overview_dir / "train_class_distribution.csv",
        overview_dir / "test_class_distribution.csv",
        overview_dir / "train_indices.npy",
        overview_dir / "test_indices.npy",
        overview_dir / "dataset_view_manifest.json",
    ]
    if all(path.exists() for path in required):
        print(f"[dataset-view] reuse -> {overview_dir}")
        return overview_dir

    print(f"[dataset-view] save -> {overview_dir}")
    np.save(overview_dir / "train_indices.npy", loaders["train_indices"])
    np.save(overview_dir / "test_indices.npy", loaders["test_indices"])
    save_json(
        overview_dir / "dataset_view_manifest.json",
        {
            "dataset": spec.name,
            "input_shape": spec.input_shape,
            "num_classes": spec.num_classes,
            "class_names": spec.class_names,
            "train_size": len(loaders["train_set"]),
            "test_size": len(loaders["test_set"]),
            "seed": seed,
            "train_indices_digest": _indices_digest(loaders["train_indices"]),
            "test_indices_digest": _indices_digest(loaders["test_indices"]),
            "shared": bool(overview_cfg.get("shared", True)),
        },
    )

    if not (overview_dir / "train_samples.png").exists():
        plot_dataset_sample_grid(
            loaders["train_set"],
            spec.class_names,
            overview_dir / "train_samples.png",
            n=n,
            cols=cols,
            seed=seed,
            title=f"{spec.name} train samples",
        )
    if not (overview_dir / "test_samples.png").exists():
        plot_dataset_sample_grid(
            loaders["test_set"],
            spec.class_names,
            overview_dir / "test_samples.png",
            n=n,
            cols=cols,
            seed=seed + 1,
            title=f"{spec.name} test samples",
        )

    train_labels = _collect_dataset_labels(loaders["train_set"])
    test_labels = _collect_dataset_labels(loaders["test_set"])
    if not (overview_dir / "train_class_distribution.csv").exists() or not (overview_dir / "train_class_distribution.png").exists():
        plot_class_distribution(
            train_labels,
            spec.class_names,
            overview_dir / "train_class_distribution.png",
            f"{spec.name} train class distribution",
        ).to_csv(overview_dir / "train_class_distribution.csv", index=False, encoding="utf-8-sig")
    if not (overview_dir / "test_class_distribution.csv").exists() or not (overview_dir / "test_class_distribution.png").exists():
        plot_class_distribution(
            test_labels,
            spec.class_names,
            overview_dir / "test_class_distribution.png",
            f"{spec.name} test class distribution",
        ).to_csv(overview_dir / "test_class_distribution.csv", index=False, encoding="utf-8-sig")
    return overview_dir


@torch.no_grad()
def _save_autoencoder_reconstructions(model, loader, class_names: list[str], device: torch.device, path: Path, n: int = 10) -> None:
    model.eval()
    images, labels = next(iter(loader))
    images = images.to(device)
    reconstructions, _ = model(images)
    plot_reconstruction_grid(
        images.cpu(),
        reconstructions.cpu(),
        labels.cpu().numpy(),
        class_names,
        path,
        title="AutoEncoder original vs reconstruction",
        n=n,
    )


def _train_autoencoder(model, loader, cfg: dict, device: torch.device) -> list[dict[str, float]]:
    ae_cfg = cfg["feature"]["ae"]
    loss_name = str(ae_cfg["loss"]).lower()
    criterion = nn.BCELoss() if loss_name == "bce" else nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=float(ae_cfg["lr"]), weight_decay=float(ae_cfg["weight_decay"]))
    history = []
    for epoch in range(1, int(ae_cfg["epochs"]) + 1):
        model.train()
        total_loss = 0.0
        total_samples = 0
        for images, _ in tqdm(loader, desc=f"AE epoch {epoch}", leave=False):
            images = images.to(device)
            x_hat, _ = model(images)
            loss = criterion(x_hat, images)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * images.size(0)
            total_samples += images.size(0)
        history.append({"epoch": epoch, "reconstruction_loss": total_loss / total_samples})
    return history


@torch.no_grad()
def _extract_autoencoder_features(model, loader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    zs, ys = [], []
    for images, labels in tqdm(loader, desc="extract AE features", leave=False):
        z = model.encode(images.to(device))
        zs.append(z.cpu().numpy())
        ys.append(labels.numpy())
    return np.concatenate(zs, axis=0).astype(np.float32), np.concatenate(ys, axis=0).astype(int)


def _train_cnn(model, train_loader, eval_loader, cfg: dict, device: torch.device) -> list[dict[str, float]]:
    cnn_cfg = cfg["feature"]["cnn"]
    criterion = nn.CrossEntropyLoss()
    optimizer_name = str(cnn_cfg.get("optimizer", "adam")).lower()
    if optimizer_name == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=float(cnn_cfg["lr"]),
            momentum=0.9,
            weight_decay=float(cnn_cfg["weight_decay"]),
        )
    else:
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=float(cnn_cfg["lr"]),
            weight_decay=float(cnn_cfg["weight_decay"]),
        )

    history = []
    for epoch in range(1, int(cnn_cfg["epochs"]) + 1):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for images, labels in tqdm(train_loader, desc=f"CNN epoch {epoch}", leave=False):
            images, labels = images.to(device), labels.to(device)
            logits, _ = model(images)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * images.size(0)
            correct += int((logits.argmax(dim=1) == labels).sum().item())
            total += images.size(0)
        eval_acc = _evaluate_cnn_accuracy(model, eval_loader, device)
        history.append(
            {
                "epoch": epoch,
                "train_loss": total_loss / total,
                "train_acc": correct / total,
                "eval_acc": eval_acc,
            }
        )
    return history


@torch.no_grad()
def _evaluate_cnn_accuracy(model, loader, device: torch.device) -> float:
    model.eval()
    correct, total = 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits, _ = model(images)
        correct += int((logits.argmax(dim=1) == labels).sum().item())
        total += images.size(0)
    return correct / max(total, 1)


@torch.no_grad()
def _extract_cnn_features(model, loader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    zs, ys = [], []
    for images, labels in tqdm(loader, desc="extract CNN features", leave=False):
        z = model.encode(images.to(device))
        zs.append(z.cpu().numpy())
        ys.append(labels.numpy())
    return np.concatenate(zs, axis=0).astype(np.float32), np.concatenate(ys, axis=0).astype(int)


def _build_cnn_model(cfg: dict, spec) -> nn.Module:
    cnn_cfg = cfg["feature"]["cnn"]
    arch = str(cnn_cfg.get("arch", "small_cnn")).lower()
    embedding_dim = int(cnn_cfg["embedding_dim"])
    dropout = float(cnn_cfg.get("dropout", 0.0))
    if arch in {"small", "small_cnn", "cnn"}:
        channels = [int(x) for x in cnn_cfg.get("channels", [32, 64, 128])]
        return SmallCNN(
            spec.input_shape,
            spec.num_classes,
            embedding_dim,
            channels=channels,
            dropout=dropout,
            use_batchnorm=bool(cnn_cfg.get("use_batchnorm", True)),
        )
    if arch in {"resnet18", "cifar_resnet18"}:
        return CifarResNet18(spec.input_shape, spec.num_classes, embedding_dim, dropout=dropout)
    raise ValueError(f"Unsupported feature.cnn.arch: {cnn_cfg.get('arch')}")


def _foundation_preprocess(
    images: torch.Tensor,
    device: torch.device,
    input_size: int,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
) -> torch.Tensor:
    x = images.to(device, non_blocking=True)
    if x.shape[1] == 1:
        x = x.repeat(1, 3, 1, 1)
    if x.shape[-2:] != (input_size, input_size):
        x = F.interpolate(x, size=(input_size, input_size), mode="bicubic", align_corners=False)
    mean_t = torch.tensor(mean, device=device, dtype=x.dtype).view(1, 3, 1, 1)
    std_t = torch.tensor(std, device=device, dtype=x.dtype).view(1, 3, 1, 1)
    return (x - mean_t) / std_t


def _build_foundation_encoder(cfg: dict, device: torch.device):
    foundation_cfg = cfg["feature"].get("foundation", {})
    provider = str(foundation_cfg.get("provider", "dinov2")).lower()
    model_name = str(foundation_cfg.get("model", "dinov2_vits14"))
    input_size = int(foundation_cfg.get("input_size", 224))

    if provider in {"dinov2", "dino"}:
        model = torch.hub.load("facebookresearch/dinov2", model_name, trust_repo=True)
        model = model.to(device).eval()
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)

        @torch.no_grad()
        def encode(images: torch.Tensor) -> torch.Tensor:
            x = _foundation_preprocess(images, device, input_size, mean, std)
            z = model(x)
            if isinstance(z, dict):
                z = z.get("x_norm_clstoken", next(iter(z.values())))
            return z.float()

        return encode, {"provider": provider, "model": model_name, "input_size": input_size}

    raise ValueError(f"Unsupported feature.foundation.provider: {provider}")


@torch.no_grad()
def _extract_foundation_features(encode, loader, provider_name: str) -> tuple[np.ndarray, np.ndarray]:
    zs, ys = [], []
    for images, labels in tqdm(loader, desc=f"extract {provider_name} features", leave=False):
        z = encode(images)
        zs.append(z.cpu().numpy())
        ys.append(labels.numpy())
    return np.concatenate(zs, axis=0).astype(np.float32), np.concatenate(ys, axis=0).astype(int)


def _build_features(cfg: dict, loaders: dict, run_dir: Path, device: torch.device) -> dict[str, Any]:
    feature_kind = str(cfg["feature"]["kind"]).lower()
    spec = loaders["spec"]
    models_dir = ensure_dir(run_dir / "models")
    plots_dir = ensure_dir(run_dir / "plots")

    if feature_kind == "pca":
        x_train, y_train = _collect_flattened(loaders["train_eval"])
        x_test, y_test = _collect_flattened(loaders["test"])
        max_components = max(1, min(x_train.shape[0] - 1, x_train.shape[1]))
        n_components = min(int(cfg["feature"]["pca"]["n_components"]), max_components)
        pca = PCA(
            n_components=n_components,
            whiten=bool(cfg["feature"]["pca"]["whiten"]),
            random_state=int(cfg["project"]["seed"]),
        )
        z_train = pca.fit_transform(x_train).astype(np.float32)
        z_test = pca.transform(x_test).astype(np.float32)
        if cfg["project"].get("save_models", True):
            joblib.dump(pca, models_dir / "pca.joblib")
        return {"z_train": z_train, "y_train": y_train, "z_test": z_test, "y_test": y_test, "history": []}

    if feature_kind == "ae":
        ae_cfg = cfg["feature"]["ae"]
        if str(ae_cfg["model_type"]).lower() == "conv":
            model = ConvAutoEncoder(spec.input_shape, int(ae_cfg["latent_dim"]))
        else:
            model = MLPAutoEncoder(spec.input_shape, int(ae_cfg["latent_dim"]), [int(x) for x in ae_cfg["hidden_dims"]])
        model = model.to(device)
        history = _train_autoencoder(model, loaders["train"], cfg, device)
        z_train, y_train = _extract_autoencoder_features(model, loaders["train_eval"], device)
        z_test, y_test = _extract_autoencoder_features(model, loaders["test"], device)
        if cfg["project"].get("save_models", True):
            torch.save(model.state_dict(), models_dir / "autoencoder.pt")
        plot_training_history(history, plots_dir / "autoencoder_history.png", "AutoEncoder training")
        _save_autoencoder_reconstructions(
            model,
            loaders["test"],
            spec.class_names,
            device,
            plots_dir / "autoencoder_reconstruction_grid.png",
            n=10,
        )
        return {"z_train": z_train, "y_train": y_train, "z_test": z_test, "y_test": y_test, "history": history}

    if feature_kind == "cnn":
        model = _build_cnn_model(cfg, spec).to(device)
        history = _train_cnn(model, loaders["train"], loaders["test"], cfg, device)
        z_train, y_train = _extract_cnn_features(model, loaders["train_eval"], device)
        z_test, y_test = _extract_cnn_features(model, loaders["test"], device)
        if cfg["project"].get("save_models", True):
            torch.save(model.state_dict(), models_dir / "cnn.pt")
        plot_training_history(history, plots_dir / "cnn_history.png", "CNN classifier training")
        return {"z_train": z_train, "y_train": y_train, "z_test": z_test, "y_test": y_test, "history": history}

    if feature_kind in {"foundation", "pretrained"}:
        encode, manifest = _build_foundation_encoder(cfg, device)
        provider_name = f"{manifest['provider']}:{manifest['model']}"
        z_train, y_train = _extract_foundation_features(encode, loaders["train_eval"], provider_name)
        z_test, y_test = _extract_foundation_features(encode, loaders["test"], provider_name)
        save_json(run_dir / "models" / "foundation_feature_extractor.json", manifest)
        return {"z_train": z_train, "y_train": y_train, "z_test": z_test, "y_test": y_test, "history": []}

    raise ValueError(f"Unsupported feature.kind: {cfg['feature']['kind']}")


def _scale_features(cfg: dict, z_train: np.ndarray, z_test: np.ndarray, run_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    if not bool(cfg["feature"].get("scale_before_cluster", True)):
        return z_train, z_test
    scaler = StandardScaler()
    z_train_scaled = scaler.fit_transform(z_train).astype(np.float32)
    z_test_scaled = scaler.transform(z_test).astype(np.float32)
    if cfg["project"].get("save_models", True):
        joblib.dump(scaler, run_dir / "models" / "feature_scaler.joblib")
    return z_train_scaled, z_test_scaled


def _cluster(cfg: dict, z_train: np.ndarray, z_test: np.ndarray, run_dir: Path):
    clustering_cfg = cfg["clustering"]
    method = str(clustering_cfg["method"]).lower()
    n_clusters = int(clustering_cfg["n_clusters"])
    if method == "kmeans":
        cluster_train = z_train
        cluster_test = z_test
        model = KMeans(
            n_clusters=n_clusters,
            random_state=int(cfg["project"]["seed"]),
            n_init=int(clustering_cfg["kmeans"]["n_init"]),
            max_iter=int(clustering_cfg["kmeans"]["max_iter"]),
        )
    elif method == "gmm":
        # GaussianMixture covariance estimation is more numerically stable in float64.
        cluster_train = z_train.astype(np.float64, copy=False)
        cluster_test = z_test.astype(np.float64, copy=False)
        model = GaussianMixture(
            n_components=n_clusters,
            covariance_type=str(clustering_cfg["gmm"]["covariance_type"]),
            n_init=int(clustering_cfg["gmm"].get("n_init", 1)),
            reg_covar=float(clustering_cfg["gmm"]["reg_covar"]),
            max_iter=int(clustering_cfg["gmm"]["max_iter"]),
            random_state=int(cfg["project"]["seed"]),
        )
    else:
        raise ValueError(f"Unsupported clustering.method: {clustering_cfg['method']}")

    train_labels = model.fit_predict(cluster_train)
    test_labels = model.predict(cluster_test)
    if cfg["project"].get("save_models", True):
        joblib.dump(model, run_dir / "models" / f"{method}.joblib")
    return train_labels.astype(int), test_labels.astype(int)


def _save_cluster_outputs(
    run_dir: Path,
    split: str,
    y_true: np.ndarray,
    cluster_labels: np.ndarray,
    mapped_pred: np.ndarray,
    class_names: list[str],
) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "sample_index": np.arange(len(y_true)),
            "true_label": y_true.astype(int),
            "true_class": [class_names[int(y)] for y in y_true],
            "cluster": cluster_labels.astype(int),
            "mapped_pred_label": mapped_pred.astype(int),
            "mapped_pred_class": [class_names[int(y)] if int(y) >= 0 else "unmapped" for y in mapped_pred],
        }
    )
    df.to_csv(run_dir / f"clusters_{split}.csv", index=False, encoding="utf-8-sig")
    return df


def run_experiment(cfg: dict, dry_run: bool = False) -> Path | None:
    if dry_run:
        print("[dry-run] would run:", cfg["project"]["run_name"])
        print(pd.Series(flatten_dict(cfg)).to_string())
        return None

    seed_everything(int(cfg["project"]["seed"]))
    device = select_device(str(cfg["project"].get("device", "auto")))

    output_dir = ensure_dir(cfg["project"]["output_dir"])
    runs_dir = ensure_dir(output_dir / "runs")
    run_name = safe_name(cfg["project"]["run_name"])
    run_dir_mode = str(cfg["project"].get("run_dir_mode", "name")).lower()
    if run_dir_mode == "timestamp":
        run_dir = runs_dir / f"{timestamp()}_{run_name}"
    elif run_dir_mode == "name":
        run_dir = runs_dir / run_name
    else:
        raise ValueError(f"Unsupported project.run_dir_mode: {cfg['project']['run_dir_mode']}")

    if bool(cfg["project"].get("reuse_existing_run", True)) and (run_dir / "metrics.json").exists() and (run_dir / "config.yaml").exists():
        print(f"[reuse] {cfg['project']['run_name']} -> {run_dir}")
        return run_dir

    run_dir = ensure_dir(run_dir)
    ensure_dir(run_dir / "plots")
    ensure_dir(run_dir / "models")

    print(f"[run] {cfg['project']['run_name']} -> {run_dir}")
    print(f"[device] {device}")
    save_yaml(run_dir / "config.yaml", cfg)

    loaders = make_dataloaders(cfg)
    spec = loaders["spec"]
    print(f"[dataset] {spec.name}, input_shape={spec.input_shape}, train={len(loaders['train_set'])}, test={len(loaders['test_set'])}")
    dataset_view_dir = _save_dataset_overview(cfg, loaders, output_dir, run_dir)
    save_json(
        run_dir / "dataset_view.json",
        {
            "dataset_view_dir": str(dataset_view_dir) if dataset_view_dir is not None else None,
            "train_indices_digest": _indices_digest(loaders["train_indices"]),
            "test_indices_digest": _indices_digest(loaders["test_indices"]),
        },
    )

    feature_data = _build_features(cfg, loaders, run_dir, device)
    z_train_raw = feature_data["z_train"]
    z_test_raw = feature_data["z_test"]
    y_train = feature_data["y_train"]
    y_test = feature_data["y_test"]
    z_train, z_test = _scale_features(cfg, z_train_raw, z_test_raw, run_dir)

    train_clusters, test_clusters = _cluster(cfg, z_train, z_test, run_dir)

    mapping = cluster_majority_mapping(train_clusters, y_train, spec.num_classes)
    train_pred = map_clusters(train_clusters, mapping)
    test_pred = map_clusters(test_clusters, mapping)

    eval_cfg = cfg["evaluation"]
    metrics_train = evaluate_clustering(
        z_train,
        y_train,
        train_clusters,
        spec.num_classes,
        int(eval_cfg["silhouette_sample_size"]),
        int(cfg["project"]["seed"]),
    )
    metrics_test = evaluate_clustering(
        z_test,
        y_test,
        test_clusters,
        spec.num_classes,
        int(eval_cfg["silhouette_sample_size"]),
        int(cfg["project"]["seed"]) + 1,
    )

    run_meta = {
        "run_name": cfg["project"]["run_name"],
        "run_dir": str(run_dir),
        "dataset_view_dir": str(dataset_view_dir) if dataset_view_dir is not None else None,
        "dataset": spec.name,
        "feature": cfg["feature"]["kind"],
        "clustering": cfg["clustering"]["method"],
        "train": metrics_train,
        "test": metrics_test,
    }
    save_json(run_dir / "metrics.json", run_meta)
    pd.DataFrame([metrics_train]).to_csv(run_dir / "metrics_train.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([metrics_test]).to_csv(run_dir / "metrics_test.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([flatten_dict(cfg)]).to_csv(run_dir / "flat_config.csv", index=False, encoding="utf-8-sig")
    if feature_data.get("history"):
        pd.DataFrame(feature_data["history"]).to_csv(run_dir / "training_history.csv", index=False, encoding="utf-8-sig")

    _save_cluster_outputs(run_dir, "train", y_train, train_clusters, train_pred, spec.class_names)
    _save_cluster_outputs(run_dir, "test", y_test, test_clusters, test_pred, spec.class_names)

    train_table = cluster_class_table(train_clusters, y_train, spec.class_names)
    test_table = cluster_class_table(test_clusters, y_test, spec.class_names)
    train_table.to_csv(run_dir / "cluster_class_train_counts.csv", encoding="utf-8-sig")
    test_table.to_csv(run_dir / "cluster_class_test_counts.csv", encoding="utf-8-sig")
    mapping_summary(train_clusters, y_train, spec.class_names).to_csv(run_dir / "cluster_mapping_train.csv", index=False, encoding="utf-8-sig")
    class_cluster_summary(train_clusters, y_train, spec.class_names).to_csv(run_dir / "class_cluster_summary_train.csv", index=False, encoding="utf-8-sig")
    class_cluster_summary(test_clusters, y_test, spec.class_names).to_csv(run_dir / "class_cluster_summary_test.csv", index=False, encoding="utf-8-sig")
    mapped_mismatch_pairs(train_pred, y_train, spec.class_names).to_csv(run_dir / "mapped_mismatch_pairs_train.csv", index=False, encoding="utf-8-sig")
    mapped_mismatch_pairs(test_pred, y_test, spec.class_names).to_csv(run_dir / "mapped_mismatch_pairs_test.csv", index=False, encoding="utf-8-sig")

    if cfg["project"].get("save_features", True):
        np.savez_compressed(
            run_dir / "features_train_test.npz",
            z_train=z_train,
            y_train=y_train,
            train_clusters=train_clusters,
            z_test=z_test,
            y_test=y_test,
            test_clusters=test_clusters,
        )

    if cfg["project"].get("save_plots", True):
        plots_dir = ensure_dir(run_dir / "plots")
        vis_cfg = cfg["visualization"]
        max_points = int(vis_cfg["max_points_2d"])
        split_payloads = {
            "train": {
                "features": z_train,
                "true": y_train,
                "clusters": train_clusters,
                "seed": int(cfg["project"]["seed"]),
            },
            "test": {
                "features": z_test,
                "true": y_test,
                "clusters": test_clusters,
                "seed": int(cfg["project"]["seed"]) + 1,
            },
        }
        pca2_cfg = vis_cfg.get("feature_pca2", {})
        if bool(pca2_cfg.get("enabled", True)):
            for split in pca2_cfg.get("splits", ["train", "test"]):
                if split not in split_payloads:
                    continue
                payload = split_payloads[split]
                plot_embedding_2d(
                    payload["features"],
                    payload["true"],
                    spec.class_names,
                    plots_dir / f"{split}_feature_pca2_true_class.png",
                    f"{split} feature PCA 2D | true class",
                    max_points,
                    payload["seed"],
                    reducer="pca",
                )
                plot_embedding_2d(
                    payload["features"],
                    payload["clusters"],
                    [str(i) for i in range(int(cfg["clustering"]["n_clusters"]))],
                    plots_dir / f"{split}_feature_pca2_cluster.png",
                    f"{split} feature PCA 2D | cluster",
                    max_points,
                    payload["seed"],
                    reducer="pca",
                )
        if bool(vis_cfg["tsne"]["enabled"]):
            for split, payload in split_payloads.items():
                plot_embedding_2d(
                    payload["features"],
                    payload["true"],
                    spec.class_names,
                    plots_dir / f"{split}_feature_tsne_true_class.png",
                    f"{split} feature t-SNE | true class",
                    int(vis_cfg["tsne"]["sample_size"]),
                    payload["seed"] + 10,
                    reducer="tsne",
                    perplexity=int(vis_cfg["tsne"]["perplexity"]),
                )
                plot_embedding_2d(
                    payload["features"],
                    payload["clusters"],
                    [str(i) for i in range(int(cfg["clustering"]["n_clusters"]))],
                    plots_dir / f"{split}_feature_tsne_cluster.png",
                    f"{split} feature t-SNE | cluster",
                    int(vis_cfg["tsne"]["sample_size"]),
                    payload["seed"] + 20,
                    reducer="tsne",
                    perplexity=int(vis_cfg["tsne"]["perplexity"]),
                )
        plot_cluster_heatmap(train_table, plots_dir / "train_cluster_class_heatmap.png", "train cluster vs true class")
        plot_cluster_heatmap(test_table, plots_dir / "test_cluster_class_heatmap.png", "test cluster vs true class")
        plot_cluster_samples(
            loaders["test_set"],
            test_clusters,
            spec.class_names,
            plots_dir / "test_cluster_samples.png",
            int(vis_cfg["cluster_samples_per_cluster"]),
            int(vis_cfg["max_clusters_to_plot"]),
            int(cfg["project"]["seed"]),
        )
        mapped_cfg = vis_cfg.get("mapped_class_grid", {})
        if bool(mapped_cfg.get("enabled", True)):
            mapped_split = str(mapped_cfg.get("split", "test")).lower()
            if mapped_split == "train":
                mapped_dataset, mapped_pred, mapped_true = loaders["train_set"], train_pred, y_train
            else:
                mapped_dataset, mapped_pred, mapped_true = loaders["test_set"], test_pred, y_test
                mapped_split = "test"
            mapped_dir = ensure_dir(plots_dir / f"{mapped_split}_mapped_class_grids")
            plot_mapped_class_grids(
                mapped_dataset,
                mapped_pred,
                mapped_true,
                spec.class_names,
                mapped_dir,
                int(mapped_cfg.get("samples_per_class", 40)),
                int(mapped_cfg.get("cols", 10)),
                int(cfg["project"]["seed"]),
            )
            plot_mismatch_grid(
                mapped_dataset,
                mapped_pred,
                mapped_true,
                spec.class_names,
                plots_dir / f"{mapped_split}_mapped_mismatches_true_to_mapped.png",
                int(mapped_cfg.get("samples_per_class", 40)),
                int(mapped_cfg.get("cols", 10)),
                int(cfg["project"]["seed"]) + 3,
            )

    print("[metrics:test]", metrics_test)
    return run_dir
