# Experiment Operation Notes

This file records working decisions for experiment operation. It is not intended to be copied directly into the report.

## Current Objective

The current goal is not report writing. The goal is to raise the performance of the comparison groups and identify which changes actually move the metrics.

Each run folder already stores the exact config used, so the active CSV can be kept compact.

## Focused CNN Impact Result

Latest focused comparison used:

`final_project/outputs/comparisons/20260621_013525`

Delta file:

`final_project/outputs/comparisons/20260621_013525/cnn_impact_delta_vs_baseline.csv`

### KMNIST

Baseline:

- `kmnist_cnn128_kmeans`
- ARI 0.6492
- NMI 0.7100
- purity 0.8147
- silhouette 0.2374

Wide/dropout/affine:

- `kmnist_cnn256_wide_do025_kmeans`
- ARI 0.5886
- NMI 0.7225
- purity 0.7793
- silhouette 0.1923

Delta:

- ARI: -0.0606
- NMI: +0.0125
- purity: -0.0353
- silhouette: -0.0451

Decision:

- KMNIST wide/dropout/affine should not be kept for performance tuning.
- Keep `kmnist_cnn128_kmeans` as the current best KMNIST configuration.
- `kmnist_cnn128_gmm` can be kept only if a KMeans-vs-GMM comparison is needed, but the best KMNIST result is KMeans.

Working interpretation:

- KMNIST baseline CNN feature is already strong.
- Increasing classifier capacity and augmentation improved classifier accuracy, but did not improve KMeans-friendly feature geometry.

## CIFAR10

Baseline:

- `cifar10_cnn128_kmeans`
- ARI 0.3470
- NMI 0.4578
- purity 0.5833
- silhouette 0.1102

Wide/normalize/crop/flip/dropout:

- `cifar10_cnn256_wide_aug_kmeans`
- ARI 0.3527
- NMI 0.5177
- purity 0.6320
- silhouette 0.1340

Delta:

- ARI: +0.0057
- NMI: +0.0598
- purity: +0.0487
- silhouette: +0.0238

Decision:

- CIFAR10 benefits from stronger CNN representation and augmentation.
- Keep the CIFAR10 wide/augmentation row.
- Add ResNet18-from-scratch rows for CIFAR10 next.
- ResNet18 trained from scratch is still within the required CNN-based feature category.
- Pretrained ImageNet, SimCLR, SCAN, self-labeling, or contrastive training should be treated as optional extensions.

## Active CSV Policy

Use a single active plan:

`final_project/configs/experiments.csv`

The active plan should contain:

- Required baseline coverage for PCA / AE / CNN on both datasets.
- The current best KMNIST CNN configuration.
- CIFAR10 CNN improvement candidates.
- ResNet18 candidates for CIFAR10.

Avoid keeping many low-impact GMM/PCA/AE variants enabled during tuning. Completed old runs can still be analyzed from their saved config and metrics.

## ResNet18 Short-Run Result

Latest comparison:

`final_project/outputs/comparisons/20260621_015021`

### CIFAR10 CNN Candidates

| Run | ARI | NMI | Purity | Silhouette |
|---|---:|---:|---:|---:|
| `cifar10_cnn256_wide_aug_kmeans` | 0.3527 | 0.5177 | 0.6320 | 0.1340 |
| `cifar10_cnn256_wide_aug_gmm` | 0.3706 | 0.4809 | 0.6100 | 0.0751 |
| `cifar10_resnet18_e20_kmeans` | 0.3642 | 0.5543 | 0.6290 | 0.2177 |
| `cifar10_resnet18_e20_gmm` | 0.4536 | 0.5440 | 0.6587 | 0.1611 |

Decision:

- ResNet18 is worth keeping for CIFAR10.
- `cifar10_resnet18_e20_gmm` is best by ARI and purity.
- `cifar10_resnet18_e20_kmeans` is best by NMI and silhouette.
- Keep both KMeans and GMM for the full-data CIFAR10 check.

## Full-Data Follow-Up

Added rows to `final_project/configs/experiments.csv`:

- `kmnist_cnn128_kmeans_full`
- `kmnist_cnn128_gmm_full`
- `cifar10_resnet18_e20_kmeans_full`
- `cifar10_resnet18_e20_gmm_full`

These rows use `dataset.train_size=0` and `dataset.test_size=0`, which the code interprets as the full official split.

## Report-Oriented Plan Update

The active CSV was reorganized to make ResNet18 the main CNN feature extractor for the report.

Current intent:

- Use ResNet18 results as the main CNN-based feature result.
- Keep SmallCNN results only as comparison baselines.
- Remove weak SmallCNN variants from the active CSV when they are not useful for interpretation.
- Keep KMNIST `cnn256` and KMNIST ResNet18 rows to test whether increasing model capacity actually improves clustering.

Active interpretation to check:

- On KMNIST, the existing SmallCNN already creates a strong clustering-friendly representation.
- Wider SmallCNN previously improved train-side/classifier behavior but reduced test clustering metrics.
- If KMNIST ResNet18 also does not improve clustering metrics, report this as evidence that more model capacity does not automatically improve unsupervised cluster geometry.
- On CIFAR10, ResNet18 is worth treating as the main CNN feature because it improved the best CIFAR10 metrics in the short run.

## Latest Full-Data Result

Latest comparison:

`final_project/outputs/comparisons/20260621_022324`

### Best Results

| Dataset | Run | ARI | NMI | Purity | Silhouette |
|---|---|---:|---:|---:|---:|
| KMNIST | `kmnist_resnet18_e20_kmeans_full` | 0.9284 | 0.9238 | 0.9672 | 0.5426 |
| CIFAR10 | `cifar10_resnet18_e20_kmeans_full` | 0.6408 | 0.7252 | 0.7825 | 0.2747 |

### Interpretation

- Full train/test split strongly improved ResNet18 + KMeans on both datasets.
- KMNIST ResNet18 + KMeans is now clearly better than SmallCNN.
- CIFAR10 ResNet18 + KMeans is also clearly the best final candidate.
- Full split GMM underperformed KMeans despite using the same learned features.
- The GMM issue is therefore mostly a clustering/fitting issue, not a CNN feature issue.
- Full GMM produced very imbalanced clusters on CIFAR10, including very large mixed clusters.

## Active CSV After Pruning

The active CSV was pruned to keep only the report-useful comparison axis:

- Required PCA + KMeans baseline for each dataset.
- Required AE + KMeans baseline for each dataset.
- SmallCNN comparison rows that explain the CNN scaling result.
- ResNet18 + KMeans rows for 12k and full split.
- Minimal ResNet18 + GMM full-split rows to explain why GMM is weaker than KMeans.

Removed from active CSV:

- Weak CIFAR10 `cnn128` runs.
- Nonessential GMM rows that do not help explain the final conclusion.
- SmallCNN full KMNIST row, because the main full-split comparison is ResNet18.
