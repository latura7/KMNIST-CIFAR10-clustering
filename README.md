# KMNIST / CIFAR-10 Clustering Project

This project runs feature extraction and clustering experiments for the final project:

- Datasets: KMNIST, CIFAR-10
- Feature representations: PCA, AutoEncoder latent, CNN embedding
- Clustering methods: K-means, Gaussian Mixture Model
- Outputs: per-run config, metrics, labels, optional features/models, plots, and batch comparison plots

## Commands

Activate the environment first:

```powershell
conda activate KMNIST_CIFAR-10_env
```

Run one experiment from the default config:

```powershell
python final_project/run.py --config final_project/configs/base.yaml
```

Override selected values for a one-off run:

```powershell
python final_project/run.py --config final_project/configs/base.yaml --set dataset.name=CIFAR10 --set feature.kind=cnn --set clustering.method=gmm --set project.run_name=cifar10_cnn_gmm
```

Run the planned experiment table:

```powershell
python final_project/run.py --config final_project/configs/base.yaml --plan final_project/configs/experiments.csv
```

Only print the expanded configs without training:

```powershell
python final_project/run.py --config final_project/configs/base.yaml --plan final_project/configs/experiments.csv --dry-run
```

Rebuild comparison plots from saved runs:

```powershell
python final_project/compare_runs.py --runs-dir final_project/outputs/runs
```

## Output Layout

Each run is saved under:

```text
final_project/outputs/runs/<run_name>/
```

By default, `project.run_name` is the run cache key. If a completed run folder already contains `metrics.json` and `config.yaml`, the runner reuses it instead of training again. Delete that run folder manually when you want to force a rerun.

Typical files:

- `config.yaml`: exact config used for the run
- `metrics.json`, `metrics_train.csv`, `metrics_test.csv`
- `clusters_train.csv`, `clusters_test.csv`
- `cluster_class_train_counts.csv`, `cluster_class_test_counts.csv`
- `class_cluster_summary_train.csv`, `class_cluster_summary_test.csv`
- `mapped_mismatch_pairs_train.csv`, `mapped_mismatch_pairs_test.csv`
- `features_train_test.npz` if `project.save_features: true`
- `plots/*.png`
- `models/*` if `project.save_models: true`
- `dataset_view.json`: pointer to the shared fixed-subset overview used by the run

Shared fixed-subset overview files are saved once under:

```text
final_project/outputs/shared_dataset_views/<dataset_subset_key>/
```

Typical shared files:

- `train_samples.png`, `test_samples.png`
- `train_class_distribution.png`, `test_class_distribution.png`
- `train_class_distribution.csv`, `test_class_distribution.csv`
- `train_indices.npy`, `test_indices.npy`
- `dataset_view_manifest.json`

Batch comparison files are saved under:

```text
final_project/outputs/comparisons/<timestamp>/
```

Comparison metric bar plots use one figure per metric, with dataset subplots:

```text
final_project/outputs/comparisons/<timestamp>/metric_bars/
```

Run-specific plots are also copied into the comparison folder for easier report writing:

```text
final_project/outputs/comparisons/<timestamp>/run_plots/<run_name>/
```

The copy manifest is saved as:

```text
final_project/outputs/comparisons/<timestamp>/copied_run_plots.csv
```

Dataset subset indices are reused by default through:

```yaml
project:
  run_dir_mode: name
  reuse_existing_run: true

dataset:
  fixed_subset: true
  split_dir: final_project/splits
  regenerate_subset: false

visualization:
  dataset_overview:
    shared: true
    output_dir: final_project/outputs/shared_dataset_views
```

## Experiment Table Format

`configs/experiments.csv` uses dot-path columns. Each row starts from `base.yaml`, then overrides only the non-empty cells in that row.

Example columns:

- `dataset.name`
- `dataset.normalize`
- `dataset.augment_train`
- `dataset.random_crop_padding`
- `dataset.random_horizontal_flip`
- `dataset.random_affine_degrees`
- `feature.kind`
- `feature.pca.n_components`
- `feature.ae.latent_dim`
- `feature.cnn.arch`
- `feature.cnn.channels`
- `feature.cnn.embedding_dim`
- `feature.cnn.dropout`
- `feature.cnn.use_batchnorm`
- `feature.cnn.epochs`
- `feature.cnn.optimizer`
- `feature.cnn.lr`
- `feature.cnn.weight_decay`
- `clustering.method`
- `clustering.gmm.covariance_type`

Set `enabled` to `false` to skip a row.

CNN architecture values currently supported:

- `small_cnn`: configurable channels/dropout/BatchNorm, e.g. `[32,64,128]` or `[64,128,256]`
- `resnet18`: CIFAR-style ResNet18 trained from scratch, with 3x3 stride-1 first convolution and no ImageNet maxpool

## Design Rationale

The project uses a lightweight local experiment runner instead of a hosted experiment platform.

- `base.yaml`: default values and knobs that are likely to change
- `experiments.csv`: row-wise experiment plan for repeated runs
- per-run folder: exact config, metrics, labels, run-specific plots, optional features, optional models
- shared dataset view folder: fixed train/test subset samples and class distributions
- comparison folder: newly generated aggregated metrics and comparison plots

This mirrors common ML experiment-management patterns:

- config-driven runs and multi-runs
- one reusable artifact directory per named run
- a later aggregation step over saved run metadata

Tools such as Hydra, MLflow, and W&B implement richer versions of this idea. For this course project, CSV+YAML is easier to submit, inspect, and reproduce locally.
