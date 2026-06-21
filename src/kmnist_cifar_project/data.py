from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    input_shape: tuple[int, int, int]
    num_classes: int
    class_names: list[str]


KMNIST_CLASS_NAMES = ["お", "き", "す", "つ", "な", "は", "ま", "や", "れ", "を"]


def _dataset_name(name: str) -> str:
    return str(name).replace("-", "").replace("_", "").upper()


def _split_path(dataset_cfg: dict[str, Any], normalized_name: str, split: str, full_size: int, subset_size: int, seed: int) -> Path:
    split_dir = Path(dataset_cfg.get("split_dir", "final_project/splits"))
    split_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{normalized_name}_{split}_full{full_size}_size{subset_size}_seed{seed}.npy"
    return split_dir / filename


def _as_tuple(value, fallback: tuple[float, ...]) -> tuple[float, ...]:
    if value is None:
        return fallback
    if isinstance(value, (int, float)):
        return (float(value),)
    return tuple(float(x) for x in value)


def _default_normalization(normalized_name: str) -> tuple[tuple[float, ...], tuple[float, ...]]:
    if normalized_name in {"CIFAR10", "CIFAR"}:
        return (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
    if normalized_name == "KMNIST":
        return (0.5,), (0.5,)
    return (0.5,), (0.5,)


def _make_transform(normalized_name: str, dataset_cfg: dict[str, Any], train: bool = False) -> transforms.Compose:
    steps: list[Any] = []
    if train and bool(dataset_cfg.get("augment_train", False)):
        crop_padding = int(dataset_cfg.get("random_crop_padding") or 0)
        if crop_padding > 0:
            crop_size = 32 if normalized_name in {"CIFAR10", "CIFAR"} else 28
            steps.append(transforms.RandomCrop(crop_size, padding=crop_padding))
        if bool(dataset_cfg.get("random_horizontal_flip", False)):
            steps.append(transforms.RandomHorizontalFlip())
        affine_degrees = float(dataset_cfg.get("random_affine_degrees") or 0.0)
        translate = float(dataset_cfg.get("random_translate") or 0.0)
        if affine_degrees > 0 or translate > 0:
            translate_arg = (translate, translate) if translate > 0 else None
            steps.append(transforms.RandomAffine(degrees=affine_degrees, translate=translate_arg))

    steps.append(transforms.ToTensor())
    if bool(dataset_cfg.get("normalize", False)):
        default_mean, default_std = _default_normalization(normalized_name)
        mean = _as_tuple(dataset_cfg.get("mean"), default_mean)
        std = _as_tuple(dataset_cfg.get("std"), default_std)
        steps.append(transforms.Normalize(mean, std))
    return transforms.Compose(steps)


def _make_full_dataset(name: str, dataset_cfg: dict[str, Any], train: bool, size: int | None = None, augment: bool = False):
    normalized = _dataset_name(name)
    transform = _make_transform(normalized, dataset_cfg, train=train and augment)
    root = dataset_cfg["root"]
    download = bool(dataset_cfg["download"])
    if normalized == "KMNIST":
        return datasets.KMNIST(root=str(root), train=train, download=download, transform=transform)
    if normalized in {"CIFAR10", "CIFAR"}:
        return datasets.CIFAR10(root=str(root), train=train, download=download, transform=transform)
    if normalized in {"FAKE", "FAKEDATA"}:
        image_size = (1, 28, 28) if train else (1, 28, 28)
        return datasets.FakeData(size=int(size or 256), image_size=image_size, num_classes=10, transform=transform)
    raise ValueError(f"Unsupported dataset.name: {name}")


def _make_indices(dataset, size: int | None, seed: int) -> np.ndarray:
    if size is None or int(size) <= 0 or int(size) >= len(dataset):
        return np.arange(len(dataset))
    rng = np.random.default_rng(seed)
    return rng.choice(len(dataset), size=int(size), replace=False)


def _subset(dataset, size: int | None, seed: int, dataset_cfg: dict[str, Any], normalized_name: str, split: str) -> tuple[Subset, np.ndarray]:
    subset_size = len(dataset) if size is None or int(size) <= 0 or int(size) >= len(dataset) else int(size)
    use_fixed_subset = bool(dataset_cfg.get("fixed_subset", True))

    if use_fixed_subset:
        path = _split_path(dataset_cfg, normalized_name, split, len(dataset), subset_size, seed)
        if path.exists() and not bool(dataset_cfg.get("regenerate_subset", False)):
            indices = np.load(path)
            if np.any(indices >= len(dataset)):
                raise ValueError(f"Saved split has out-of-range indices: {path}")
        else:
            indices = _make_indices(dataset, size, seed)
            np.save(path, indices)
    else:
        indices = _make_indices(dataset, size, seed)

    return Subset(dataset, indices.astype(int).tolist()), indices.astype(int)


def _infer_spec(name: str, dataset) -> DatasetSpec:
    sample, _ = dataset[0]
    input_shape = tuple(sample.shape)
    normalized = _dataset_name(name)
    class_names = KMNIST_CLASS_NAMES if normalized == "KMNIST" else getattr(dataset.dataset if isinstance(dataset, Subset) else dataset, "classes", None)
    if not class_names:
        class_names = KMNIST_CLASS_NAMES if normalized == "KMNIST" else [str(i) for i in range(10)]
    return DatasetSpec(
        name=name,
        input_shape=input_shape,
        num_classes=len(class_names),
        class_names=[str(x) for x in class_names],
    )


def make_dataloaders(cfg: dict):
    dataset_cfg = cfg["dataset"]
    project_cfg = cfg["project"]
    seed = int(project_cfg["seed"])
    name = dataset_cfg["name"]
    train_size = int(dataset_cfg["train_size"])
    test_size = int(dataset_cfg["test_size"])
    normalized = _dataset_name(name)

    fake_size_train = train_size if normalized in {"FAKE", "FAKEDATA"} else None
    fake_size_test = test_size if normalized in {"FAKE", "FAKEDATA"} else None

    train_full = _make_full_dataset(name, dataset_cfg, True, fake_size_train, augment=False)
    train_full_aug = _make_full_dataset(name, dataset_cfg, True, fake_size_train, augment=True)
    test_full = _make_full_dataset(name, dataset_cfg, False, fake_size_test)

    train_set, train_indices = _subset(train_full, train_size, seed, dataset_cfg, normalized, "train")
    train_aug_set = Subset(train_full_aug, train_indices.astype(int).tolist())
    test_set, test_indices = _subset(test_full, test_size, seed + 1, dataset_cfg, normalized, "test")
    spec = _infer_spec(name, train_set)

    pin_memory = bool(dataset_cfg.get("pin_memory", False)) and torch.cuda.is_available()
    num_workers = int(project_cfg.get("num_workers", 0))

    train_loader = DataLoader(
        train_aug_set,
        batch_size=int(dataset_cfg["train_batch_size"]),
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    train_eval_loader = DataLoader(
        train_set,
        batch_size=int(dataset_cfg["eval_batch_size"]),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=int(dataset_cfg["eval_batch_size"]),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return {
        "train": train_loader,
        "train_eval": train_eval_loader,
        "test": test_loader,
        "train_set": train_set,
        "test_set": test_set,
        "train_indices": train_indices,
        "test_indices": test_indices,
        "spec": spec,
    }
