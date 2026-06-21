from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kmnist_cifar_project.data import _dataset_name, _default_normalization, _make_full_dataset, make_dataloaders
from kmnist_cifar_project.runner import _build_cnn_model, _collect_flattened
from kmnist_cifar_project.utils import ensure_dir, select_device


def load_cfg(run_dir: Path) -> dict:
    with (run_dir / "config.yaml").open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def to_display_image(x: torch.Tensor | np.ndarray, cfg: dict) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        arr = x.detach().cpu().numpy()
    else:
        arr = np.asarray(x)

    if arr.ndim == 3:
        normalized = _dataset_name(cfg["dataset"]["name"])
        if bool(cfg["dataset"].get("normalize", False)):
            mean, std = _default_normalization(normalized)
            mean_arr = np.asarray(mean, dtype=np.float32).reshape(-1, 1, 1)
            std_arr = np.asarray(std, dtype=np.float32).reshape(-1, 1, 1)
            arr = arr * std_arr + mean_arr
        arr = np.clip(arr, 0.0, 1.0)
        if arr.shape[0] in {1, 3}:
            arr = np.transpose(arr, (1, 2, 0))
        if arr.ndim == 3 and arr.shape[-1] == 1:
            arr = arr[:, :, 0]
    return arr


def flat_to_display_image(x: np.ndarray, input_shape: tuple[int, int, int]) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32).reshape(input_shape)
    arr = np.clip(arr, 0.0, 1.0)
    if arr.shape[0] in {1, 3}:
        arr = np.transpose(arr, (1, 2, 0))
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[:, :, 0]
    return arr


def save_pca_reconstruction_grid(
    run_dir: Path,
    output_path: Path,
    components: list[int] | None = None,
    n_images: int = 8,
) -> None:
    cfg = load_cfg(run_dir)
    loaders = make_dataloaders(cfg)
    spec = loaders["spec"]
    seed = int(cfg["project"]["seed"])

    x_train, _ = _collect_flattened(loaders["train_eval"])
    x_test, y_test = _collect_flattened(loaders["test"])
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(x_test), size=min(n_images, len(x_test)), replace=False)
    x_original = x_test[chosen]
    y_original = y_test[chosen]

    reconstructions: list[tuple[str, np.ndarray]] = []
    if components:
        max_components = min(x_train.shape[0] - 1, x_train.shape[1])
        for n_components in components:
            n_components = int(n_components)
            if n_components < 1 or n_components > max_components:
                continue
            pca = PCA(n_components=n_components, random_state=seed, svd_solver="randomized")
            z_train = pca.fit_transform(x_train)
            z_test = pca.transform(x_test)
            x_recon = pca.inverse_transform(z_test)[chosen]
            evr = float(pca.explained_variance_ratio_.sum())
            reconstructions.append((f"PCA {n_components}\nEVR {evr:.2f}", x_recon))
    else:
        pca = joblib.load(run_dir / "models" / "pca.joblib")
        z_test = pca.transform(x_test)
        x_recon = pca.inverse_transform(z_test)[chosen]
        n_components = int(getattr(pca, "n_components_", cfg["feature"]["pca"]["n_components"]))
        evr = float(np.sum(getattr(pca, "explained_variance_ratio_", [np.nan])))
        reconstructions.append((f"PCA {n_components}\nEVR {evr:.2f}", x_recon))

    n_rows = 1 + len(reconstructions)
    n_cols = len(chosen)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(1.35 * n_cols, 1.55 * n_rows))
    axes = np.asarray(axes).reshape(n_rows, n_cols)

    for col in range(n_cols):
        image = flat_to_display_image(x_original[col], spec.input_shape)
        axes[0, col].imshow(image, cmap="gray" if image.ndim == 2 else None)
        axes[0, col].set_title(spec.class_names[int(y_original[col])][:10], fontsize=8)
        axes[0, col].axis("off")
    axes[0, 0].set_ylabel("original", fontsize=9)

    for row, (label, x_recon) in enumerate(reconstructions, start=1):
        for col in range(n_cols):
            image = flat_to_display_image(x_recon[col], spec.input_shape)
            axes[row, col].imshow(image, cmap="gray" if image.ndim == 2 else None)
            axes[row, col].axis("off")
        axes[row, 0].set_ylabel(label, fontsize=8)

    fig.suptitle(f"{spec.name} PCA inverse reconstruction", y=1.02, fontsize=12)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180)
    plt.close()


def selected_activation_layers(model: torch.nn.Module) -> list[tuple[str, torch.nn.Module]]:
    names = dict(model.named_modules())
    wanted = []
    if all(name in names for name in ["conv1", "layer1", "layer2", "layer3", "layer4"]):
        for name in ["conv1", "layer1", "layer2", "layer3", "layer4"]:
            wanted.append((name, names[name]))
        return wanted
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            wanted.append((name, module))
    return wanted[:5]


def normalize_map(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    mn = float(np.nanmin(x))
    mx = float(np.nanmax(x))
    if mx <= mn:
        return np.zeros_like(x)
    return (x - mn) / (mx - mn)


@torch.no_grad()
def save_cnn_activation_grid(
    run_dir: Path,
    output_path: Path,
    sample_index: int = 0,
    top_channels: int = 6,
    raw_index: int | None = None,
) -> None:
    cfg = load_cfg(run_dir)
    loaders = make_dataloaders(cfg)
    spec = loaders["spec"]
    device = select_device(cfg["project"].get("device", "auto"))

    model = _build_cnn_model(cfg, spec).to(device)
    state_path = run_dir / "models" / "cnn.pt"
    try:
        state = torch.load(state_path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(state_path, map_location=device)
    try:
        model.load_state_dict(state)
    except RuntimeError:
        # Older saved SmallCNN runs were created before BatchNorm was enabled in
        # the config surface. Rebuild to match those checkpoints for post-hoc plots.
        arch = str(cfg["feature"]["cnn"].get("arch", "small_cnn")).lower()
        has_conv_bias = any(key.startswith("features.") and key.endswith(".bias") for key in state)
        has_batchnorm = any(".running_mean" in key or ".running_var" in key for key in state)
        if arch in {"small", "small_cnn", "cnn"} and has_conv_bias and has_batchnorm:
            result = model.load_state_dict(state, strict=False)
            serious_missing = [key for key in result.missing_keys if not key.endswith(".bias")]
            if serious_missing:
                raise RuntimeError(f"Could not load checkpoint for activation plot. Missing keys: {serious_missing}")
        elif arch in {"small", "small_cnn", "cnn"} and has_conv_bias and not has_batchnorm:
            cfg["feature"]["cnn"]["use_batchnorm"] = False
            model = _build_cnn_model(cfg, spec).to(device)
            model.load_state_dict(state)
        else:
            raise
    model.eval()

    if raw_index is None:
        dataset = loaders["test_set"]
        sample_index = int(np.clip(sample_index, 0, len(dataset) - 1))
    else:
        dataset = _make_full_dataset(cfg["dataset"]["name"], cfg["dataset"], train=False)
        sample_index = int(np.clip(raw_index, 0, len(dataset) - 1))
    image, label = dataset[sample_index]
    x = image.unsqueeze(0).to(device)

    activations: dict[str, torch.Tensor] = {}
    hooks = []

    def make_hook(layer_name: str):
        def hook(_module, _inputs, output):
            activations[layer_name] = output.detach().cpu()

        return hook

    layers = selected_activation_layers(model)
    for layer_name, module in layers:
        hooks.append(module.register_forward_hook(make_hook(layer_name)))

    logits, _ = model(x)
    pred = int(logits.argmax(dim=1).item())

    for hook in hooks:
        hook.remove()

    n_layers = len(layers)
    n_cols = top_channels + 1
    fig, axes = plt.subplots(n_layers, n_cols, figsize=(1.55 * n_cols, 1.5 * n_layers))
    axes = np.asarray(axes).reshape(n_layers, n_cols)
    display = to_display_image(image, cfg)

    for row, (layer_name, _module) in enumerate(layers):
        axes[row, 0].imshow(display, cmap="gray" if display.ndim == 2 else None)
        if row == 0:
            axes[row, 0].set_title(f"true {spec.class_names[int(label)]}\npred {spec.class_names[pred]}", fontsize=8)
        axes[row, 0].set_ylabel(layer_name, fontsize=8)
        axes[row, 0].set_xticks([])
        axes[row, 0].set_yticks([])

        activation = activations[layer_name][0]
        if activation.ndim != 3:
            continue
        scores = activation.abs().mean(dim=(1, 2)).numpy()
        channel_ids = np.argsort(scores)[-top_channels:][::-1]
        for col, channel_id in enumerate(channel_ids, start=1):
            fmap = normalize_map(activation[int(channel_id)].numpy())
            axes[row, col].imshow(fmap, cmap="magma")
            axes[row, col].set_title(f"ch {int(channel_id)}", fontsize=7)
            axes[row, col].axis("off")
        for col in range(1 + len(channel_ids), n_cols):
            axes[row, col].axis("off")

    fig.suptitle(f"{spec.name} CNN activation maps: {cfg['project']['run_name']}", y=1.01, fontsize=12)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180)
    plt.close()


def parse_components(text: str | None) -> list[int] | None:
    if not text:
        return None
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate extra PCA reconstruction and CNN activation visualizations from saved runs.")
    parser.add_argument("--output-dir", default="final_project/report_figures")
    parser.add_argument("--pca-run", action="append", default=[])
    parser.add_argument("--cnn-run", action="append", default=[])
    parser.add_argument("--components", default=None, help="Comma-separated PCA components. If omitted, defaults are used for built-in runs.")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--raw-index", type=int, default=None, help="Use the same raw test-set index for CNN activation maps, ignoring run subsets.")
    args = parser.parse_args()

    output_dir = ensure_dir(Path(args.output_dir))
    if args.pca_run or args.cnn_run:
        pca_runs = args.pca_run
        cnn_runs = args.cnn_run
    else:
        pca_runs = [
            "final_project/outputs/runs/kmnist_pca10_kmeans",
            "final_project/outputs/runs/cifar10_pca100_kmeans",
        ]
        cnn_runs = [
            "final_project/outputs/runs/kmnist_cnn128_kmeans",
            "final_project/outputs/runs/kmnist_resnet18_e20_kmeans_full",
            "final_project/outputs/runs/cifar10_cnn256_wide_aug_kmeans",
            "final_project/outputs/runs/cifar10_resnet18_e20_kmeans_full",
        ]

    explicit_components = parse_components(args.components)
    for run in pca_runs:
        run_dir = Path(run)
        cfg = load_cfg(run_dir)
        dataset_name = _dataset_name(cfg["dataset"]["name"]).lower()
        if explicit_components is not None:
            components = explicit_components
        elif dataset_name == "kmnist":
            components = [2, 10, 50]
        else:
            components = [2, 20, int(cfg["feature"]["pca"]["n_components"])]
        out = output_dir / f"{cfg['project']['run_name']}_pca_reconstruction.png"
        save_pca_reconstruction_grid(run_dir, out, components=components)
        print(out)

    for run in cnn_runs:
        run_dir = Path(run)
        cfg = load_cfg(run_dir)
        out = output_dir / f"{cfg['project']['run_name']}_activation_maps.png"
        dataset_name = _dataset_name(cfg["dataset"]["name"])
        default_raw_index = {"KMNIST": 0, "CIFAR10": 3, "CIFAR": 3}.get(dataset_name)
        raw_index = args.raw_index if args.raw_index is not None else default_raw_index
        save_cnn_activation_grid(run_dir, out, sample_index=args.sample_index, raw_index=raw_index)
        print(out)


if __name__ == "__main__":
    main()
