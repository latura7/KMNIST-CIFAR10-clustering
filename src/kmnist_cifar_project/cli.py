from __future__ import annotations

from pathlib import Path

from .aggregate import compare_runs
from .config import build_cfg, is_enabled, load_plan, load_yaml, make_parser, print_config
from .epoch_sweep import parse_epochs, run_resnet_epoch_sweep
from .runner import run_experiment


def main(default_mode: str = "single") -> None:
    parser = make_parser(default_mode=default_mode)
    args = parser.parse_args()
    mode = args.mode
    if args.plan and mode == "single":
        mode = "batch"

    if mode == "compare":
        compare_runs(args.runs_dir, output_dir=args.output_dir)
        return

    base_cfg = load_yaml(args.config)

    if mode == "single":
        cfg = build_cfg(base_cfg, row=None, cli_sets=args.set)
        if args.dry_run:
            print_config(cfg)
            return
        run_experiment(cfg)
        return

    if mode == "resnet-epoch-sweep":
        cfg = build_cfg(base_cfg, row=None, cli_sets=args.set)
        run_resnet_epoch_sweep(
            cfg,
            dataset_name=args.epoch_sweep_dataset,
            epochs=parse_epochs(args.epoch_sweep_epochs),
            train_size=int(args.epoch_sweep_train_size),
            test_size=int(args.epoch_sweep_test_size),
            output_dir=args.output_dir,
            run_prefix=args.epoch_sweep_run_prefix,
            dry_run=bool(args.dry_run),
            plot_only=bool(args.epoch_sweep_plot_only),
        )
        return

    if not args.plan:
        raise ValueError("--plan is required for batch mode")

    rows = load_plan(args.plan)
    completed_dirs: list[Path] = []
    for i, row in enumerate(rows, 1):
        if not is_enabled(row):
            print(f"[batch] skip row {i}: disabled")
            continue
        cfg = build_cfg(base_cfg, row=row, cli_sets=args.set)
        print(f"[batch] row {i}/{len(rows)}: {cfg['project']['run_name']}")
        if args.dry_run:
            print_config(cfg)
            continue
        try:
            run_dir = run_experiment(cfg)
            if run_dir is not None:
                completed_dirs.append(Path(run_dir))
        except Exception as exc:
            if bool(base_cfg.get("batch", {}).get("continue_on_error", True)):
                print(f"[batch:error] row {i} failed: {exc}")
                continue
            raise

    if args.dry_run:
        return
    if completed_dirs and bool(base_cfg.get("batch", {}).get("auto_compare", True)):
        runs_dir = Path(base_cfg["project"]["output_dir"]) / "runs"
        compare_runs(runs_dir, output_dir=args.output_dir, run_dirs=completed_dirs)
