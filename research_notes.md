# Research Notes: Tuning vs. Newer Clustering Methods

This note separates two questions:

1. What can be improved while keeping the current project method?
   - PCA / AutoEncoder / CNN feature extraction
   - KMeans / GMM clustering
2. What requires moving to newer deep clustering methods?

## Current Best Results

Latest comparison used:

`final_project/outputs/comparisons/20260621_001257/all_runs_metrics.csv`

Current best runs:

| Dataset | Feature | Clustering | ARI | NMI | Purity |
|---|---|---:|---:|---:|---:|
| KMNIST | CNN embedding | KMeans | 0.648 | 0.709 | 0.815 |
| KMNIST | CNN embedding | GMM | 0.584 | 0.693 | 0.780 |
| CIFAR10 | CNN embedding | KMeans | 0.347 | 0.458 | 0.583 |

PCA and vanilla AE are much lower, especially on CIFAR10.

## Same-Method Tuning Candidates

These keep the current project method and do not add a new algorithm family.

### Preprocessing

- Implement and enable dataset normalization.
  - Current `base.yaml` has `dataset.normalize`, but `data.py` currently only applies `transforms.ToTensor()`.
  - CIFAR10 literature commonly uses mean `[0.4914, 0.4822, 0.4465]` and std `[0.2023, 0.1994, 0.2010]`.
- Add train-only augmentation for CNN / AE training.
  - CIFAR10: random crop 32 with padding 4, random horizontal flip.
  - KMNIST: small random affine / translation can help CNN generalization.
  - Evaluation loaders must stay deterministic.

### PCA

- Tune `feature.pca.n_components`.
  - KMNIST: try 20, 50, 100, 200.
  - CIFAR10: try 50, 100, 200, 512.
- Tune `feature.pca.whiten`.
  - Whitening can help KMeans/GMM when dimensions have very different variances.
- Tune `feature.scale_before_cluster`.
  - Already enabled. Keep it enabled for KMeans/GMM.

### AutoEncoder

- Tune latent dimension.
  - KMNIST MLP AE: 16, 32, 64.
  - CIFAR10 Conv AE: 64, 128, 256.
- Tune epochs.
  - Current KMNIST AE 5 epochs, CIFAR10 AE 8 epochs are small.
  - Try 20 to 50 epochs for serious comparison.
- Tune AE model type for KMNIST.
  - Current KMNIST uses MLP AE. Conv AE is likely a better image prior.
- Tune loss.
  - BCE can be reasonable for grayscale in [0, 1].
  - MSE is usually stable for CIFAR10 reconstruction.
- Important limitation:
  - Vanilla AE reconstruction loss does not force semantic class separation.
  - Better AE clustering would require DCN / DEC / IDEC style clustering loss, which is a method change.

### CNN Feature

- Increase CNN epochs.
  - KMNIST current 5 epochs, CIFAR10 current 8 epochs.
  - Try KMNIST 10, 20.
  - Try CIFAR10 20, 50.
- Tune embedding dimension.
  - 64, 128, 256.
- Tune optimizer and regularization.
  - Adam lr 1e-3 is fine for quick runs.
  - For longer CIFAR10 runs, SGD momentum 0.9 or AdamW can be tested.
  - Weight decay: 0, 1e-4, 5e-4.
- Add stronger CNN backbone as an optional current-method extension.
  - ResNet18 feature + KMeans/GMM still fits "CNN feature + clustering".
  - This is not SCAN/SPICE yet, but it is a much stronger feature extractor.
- Current `SmallCNN` already has 3 convolution blocks with channels 32/64/128 and BatchNorm.
  - Structure-level knobs are now exposed in `experiments.csv`.
  - Available config knobs: `feature.cnn.arch`, `feature.cnn.channels`, `feature.cnn.dropout`, `feature.cnn.use_batchnorm`, `feature.cnn.embedding_dim`, `feature.cnn.epochs`, optimizer/lr/weight_decay.
  - Dataset knobs are also exposed: normalization and train-only augmentation crop/flip/affine.
- GitHub/reference ranges checked:
  - KMNIST official simple CNN benchmark uses Conv2D 32, Conv2D 64, MaxPool, Dropout 0.25, Dense 128, Dropout 0.5, epochs 12.
  - CIFAR10 PyTorch-CIFAR uses CIFAR-specific ResNet18 with 3x3 first conv, channels 64/128/256/512, SGD lr 0.1, momentum 0.9, weight decay 5e-4, cosine schedule, epochs 200.
  - Lightning CIFAR10 ResNet18 tutorial modifies torchvision ResNet18 for 32x32 images by using 3x3 stride-1 conv1 and removing maxpool; it notes 20-30 epochs can reach around 92-93% and 40-50 epochs around 93-94%.
  - SCAN/SimCLR CIFAR10 uses ResNet18, feature dim 128, SGD momentum 0.9, weight decay 1e-4, batch size 512, epochs 500; this is beyond the current supervised-CNN baseline but useful as an upper compute/reference point.

### KMeans / GMM

- KMeans:
  - `n_init=30` was already reasonable.
  - Applied low-impact stability default: `n_init=50`.
  - Applied low-impact stability default: `max_iter=500`.
- GMM:
  - Tune `covariance_type`: `diag`, `tied`, `full`.
  - `full` can help but is slower and less stable in high dimensions.
  - Tune `reg_covar`: 1e-6, 1e-5, 1e-4, 1e-3.
  - Applied low-impact stability default: `n_init=3`, `max_iter=300`.
  - GMM is unlikely to beat KMeans consistently unless the feature space is close to Gaussian clusters.
- Number of clusters:
  - Main result should keep `n_clusters=10`.
  - For an extra analysis, try overclustering, e.g. 20, 30, 50, then map clusters to labels.
  - This changes the interpretation, so it should be reported separately.

### Data / Compute Scaling

- Current project uses 12k train / 3k test subsets.
- Full KMNIST/CIFAR10 train/test may improve stability, especially for CNN and AE.
- This is not a new method, but it increases compute and makes the comparison less directly matched to current runs.

## Literature / External References

### KMNIST Official Benchmarks

Official KMNIST repo:

https://github.com/rois-codh/kmnist

Useful reported classification baselines:

| Method | Kuzushiji-MNIST Accuracy |
|---|---:|
| 4-nearest-neighbor | 92.10% |
| PCA + 4-NN | 93.98% |
| Keras Simple CNN | 94.63% |
| PreActResNet-18 | 97.82% |
| PreActResNet-18 + Manifold Mixup | 98.83% |

These are supervised classification numbers, not clustering scores, but they show that current CNN feature quality is not near the dataset ceiling.

### CIFAR10 Deep Clustering References

SCAN:

https://github.com/wvangansbeke/Unsupervised-Classification

Reported model zoo for CIFAR10:

| Method | ACC | NMI | ARI |
|---|---:|---:|---:|
| SCAN-loss | 81.6 | 71.5 | 66.5 |
| Self-labeling | 88.3 | 79.7 | 77.2 |

SCAN CIFAR10 configs:

- SimCLR pretext: ResNet18, feature dim 128, 500 epochs, batch size 512, SGD lr 0.4, temperature 0.1.
  - https://raw.githubusercontent.com/wvangansbeke/Unsupervised-Classification/master/configs/pretext/simclr_cifar10.yml
- SCAN: 20 nearest neighbors, entropy weight 5.0, 50 epochs, Adam lr 1e-4.
  - https://raw.githubusercontent.com/wvangansbeke/Unsupervised-Classification/master/configs/scan/scan_cifar10.yml
- Self-label: confidence threshold 0.99, 200 epochs, batch size 1000.
  - https://raw.githubusercontent.com/wvangansbeke/Unsupervised-Classification/master/configs/selflabel/selflabel_cifar10.yml

SPICE:

https://github.com/niuchuangnn/SPICE

Reported CIFAR10 model zoo:

| Method | ACC | NMI | ARI |
|---|---:|---:|---:|
| SPICE-Self | 83.8 | 73.4 | 70.5 |
| SPICE | 92.6 | 86.5 | 85.2 |

CRLC paper:

https://openaccess.thecvf.com/content/ICCV2021/papers/Do_Clustering_by_Maximizing_Mutual_Information_Across_Views_ICCV_2021_paper.pdf

Reported CIFAR10:

| Method | ACC | NMI | ARI |
|---|---:|---:|---:|
| DCCM | 62.3 | 49.6 | 40.8 |
| IIC | 61.7 | - | - |
| PICA | 69.6 | 59.1 | 51.2 |
| DRC | 72.7 | 62.1 | 54.7 |
| CRLC | 79.9 | 67.9 | 63.4 |

BRB / DCN / DEC / IDEC:

https://proceedings.iclr.cc/paper_files/paper/2025/file/f91685b940d5032f0f0c247edbd72edd-Paper-Conference.pdf

Reported CIFAR10 with contrastive task and self-labeling:

| Method | ACC | NMI | ARI |
|---|---:|---:|---:|
| Pretraining + KMeans | 68.97 | 63.98 | 40.13 |
| DEC | 88.29 | 80.60 | 77.23 |
| DEC + BRB | 90.57 | 82.57 | 81.18 |
| IDEC + BRB | 90.72 | 83.26 | 81.81 |
| DCN + BRB | 91.23 | 83.66 | 82.42 |
| SCAN | 88.3 | 79.7 | 77.2 |
| SeCu | 93.0 | 86.1 | 85.7 |

Reported KMNIST deep clustering examples from the same BRB paper:

| Method | KMNIST ACC | KMNIST NMI | KMNIST ARI |
|---|---:|---:|---:|
| DEC+BRB+Pretrain | 65.97 | 64.64 | 50.05 |
| IDEC+BRB+Pretrain | 65.61 | 65.12 | 48.91 |
| DCN+BRB+Pretrain | 65.19 | 61.01 | 46.74 |

DeepCluster GitHub issue:

https://github.com/facebookresearch/deepcluster/issues/22

Useful observation:

- A user reported poor CIFAR10 DeepCluster behavior with `k=10` and learning rates from 0.001 to 0.05, with NMI below 0.1 across 200 epochs.
- This supports the idea that naive deep clustering / KMeans on weak features is not enough for CIFAR10.

## Practical Interpretation

Within the current project method, the strongest realistic improvements are:

1. Implement normalization and train-only augmentation.
2. Strengthen CNN features with more epochs and possibly ResNet18.
3. Tune PCA dimensions and whitening.
4. Tune AE latent size/epochs, but expect limited gains unless adding a clustering loss.
5. Tune GMM covariance/reg_covar, but expect smaller gains than feature improvements.

The large jumps in the literature are mostly not from simply using more KMeans/GMM iterations. They come from stronger representation learning:

- SimCLR / contrastive pretraining
- nearest-neighbor consistency
- pseudo-label self-labeling
- DEC / IDEC / DCN clustering losses
- stronger backbones such as ResNet18/ResNet34
