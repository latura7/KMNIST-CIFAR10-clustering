from __future__ import annotations

import argparse
import csv
import copy
from pathlib import Path
from typing import Any

import yaml


SKIP_PLAN_COLUMNS = {"", "enabled", "notes", "note", "comment", "comments"}


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return data


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def parse_value(raw: str) -> Any:
    if raw is None:
        return None
    text = str(raw).strip()
    if text == "":
        return None
    try:
        return yaml.safe_load(text)
    except Exception:
        return text


def set_by_dot_path(data: dict[str, Any], dot_path: str, value: Any) -> None:
    if dot_path == "run_name":
        dot_path = "project.run_name"
    parts = dot_path.split(".")
    cur = data
    for part in parts[:-1]:
        if part not in cur or not isinstance(cur[part], dict):
            cur[part] = {}
        cur = cur[part]
    cur[parts[-1]] = value


def overrides_to_dict(items: list[str] | None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"--set expects KEY=VALUE, got: {item}")
        key, raw_value = item.split("=", 1)
        value = parse_value(raw_value)
        set_by_dot_path(out, key.strip(), value)
    return out


def row_to_overrides(row: dict[str, str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, raw_value in row.items():
        key = (key or "").strip()
        if key in SKIP_PLAN_COLUMNS:
            continue
        value = parse_value(raw_value)
        if value is None:
            continue
        set_by_dot_path(out, key, value)
    return out


def is_enabled(row: dict[str, str]) -> bool:
    raw = row.get("enabled", "true")
    value = parse_value(raw)
    return bool(value)


def load_plan(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def build_cfg(base_cfg: dict[str, Any], row: dict[str, str] | None, cli_sets: list[str] | None) -> dict[str, Any]:
    cfg = copy.deepcopy(base_cfg)
    if row:
        cfg = deep_merge(cfg, row_to_overrides(row))
    cfg = deep_merge(cfg, overrides_to_dict(cli_sets))
    return cfg


def print_config(cfg: dict[str, Any]) -> None:
    print(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))


def make_parser(default_mode: str = "single") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run KMNIST/CIFAR-10 feature clustering experiments.")
    parser.add_argument("--mode", choices=["single", "batch", "compare", "resnet-epoch-sweep"], default=default_mode)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--plan", default=None, help="CSV plan. If provided, mode=batch is implied unless --mode is set.")
    parser.add_argument("--runs-dir", default="outputs/runs")
    parser.add_argument("--output-dir", default=None, help="Comparison output directory override.")
    parser.add_argument("--set", action="append", default=[], help="Override config with dot path, e.g. --set dataset.name=CIFAR10")
    parser.add_argument("--dry-run", action="store_true", help="Print expanded configs without running training.")
    parser.add_argument("--epoch-sweep-dataset", default="CIFAR10", help="Dataset for --mode resnet-epoch-sweep.")
    parser.add_argument("--epoch-sweep-epochs", default="20,50,100,200", help="Comma-separated epoch values.")
    parser.add_argument("--epoch-sweep-train-size", type=int, default=0, help="0 means full train split.")
    parser.add_argument("--epoch-sweep-test-size", type=int, default=0, help="0 means full test split.")
    parser.add_argument("--epoch-sweep-run-prefix", default=None, help="Optional run_name prefix. Omit to reuse existing standard run names.")
    parser.add_argument("--epoch-sweep-plot-only", action="store_true", help="Do not run training; only plot existing epoch sweep runs.")
    return parser
