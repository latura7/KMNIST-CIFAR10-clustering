# Final Project 요구사항 대응 체크리스트

기준 문서: `C:\LIM\Machine learning\Final_Project_kor.pdf`  
검토 대상: `final_project/report.tex`, `final_project/configs/experiments.csv`, `final_project/src/kmnist_cifar_project`, `final_project/outputs`

상태 표기:
- `완료`: 보고서 본문 또는 표/그림에 직접 반영됨
- `부분`: 코드/output에는 있거나 일부 관점은 반영됐지만, 보고서 본문 반영이 제한적임
- `미반영`: 아직 반영되지 않음
- `해당 없음`: optional 예시 중 선택하지 않은 항목

## 1. Project Overview

| 요구사항 | 상태 | 대응 위치 및 근거 |
|---|---:|---|
| 이미지 데이터셋에 대해 여러 feature representation과 clustering method 비교 | 완료 | `report.tex` Introduction: “어떤 feature representation 과 clustering method 조합이 의미 있는 cluster 를 만드는지 비교” (`report.tex:70`) |
| KMNIST와 CIFAR-10 두 데이터셋 모두 사용 | 완료 | `report.tex:71`, Dataset table (`report.tex:91-98`), `experiments.csv`에 KMNIST 21개, CIFAR10 24개 enabled row |
| 두 데이터셋의 특성 차이를 관찰하고 결과 해석에 반영 | 완료 | Dataset sample figure caption (`report.tex:115`), Cross-Dataset Discussion (`report.tex:825-888`), Conclusion (`report.tex:898-902`) |
| 단순 최고 숫자 탐색이 아니라 평가 기준과 시각 자료로 의미 있는 cluster 판단 | 완료 | Evaluation Metrics (`report.tex:222-278`), KMNIST/CIFAR visual analysis (`report.tex:416-499`, `640-718`), DINOv2 optional analysis (`report.tex:721-823`) |

## 2. Datasets

| 요구사항 | 상태 | 대응 위치 및 근거 |
|---|---:|---|
| KMNIST 사용 | 완료 | Dataset table: KMNIST `1x28x28`, subset/full split (`report.tex:95-98`) |
| CIFAR-10 사용 | 완료 | Dataset table: CIFAR-10 `3x32x32`, subset/full split (`report.tex:95-98`) |
| 데이터셋 특성을 직접 확인하고 실험 설계/해석에 반영 | 완료 | Dataset examples figure (`report.tex:108-116`), augmentation 차이 설명 (`report.tex:207-209`), dataset별 결과 해석 (`report.tex:280-718`) |

## 3. Required Methods

### 3.1 Feature Representation

| 요구사항 | 상태 | 대응 위치 및 근거 |
|---|---:|---|
| PCA-based feature 포함 | 완료 | Feature Representations (`report.tex:124`), PCA sweep (`report.tex:129-150`), KMNIST PCA result table (`report.tex:299`), CIFAR-10 PCA result table (`report.tex:519`) |
| AutoEncoder latent feature 포함 | 완료 | Feature Representations (`report.tex:125`), AE sweep (`report.tex:155-180`), KMNIST AE result table (`report.tex:300`), CIFAR-10 AE result table (`report.tex:520`) |
| CNN-based feature 포함 | 완료 | Feature Representations (`report.tex:126`), CNN settings table (`report.tex:189-199`), KMNIST/CIFAR result tables (`report.tex:301-307`, `521-524`) |

### 3.2 Clustering Method

| 요구사항 | 상태 | 대응 위치 및 근거 |
|---|---:|---|
| K-means clustering 포함 | 완료 | Clustering Methods (`report.tex:218`), main result tables (`report.tex:299-306`, `519-523`) |
| Gaussian Mixture Model clustering 포함 | 완료 | Clustering Methods (`report.tex:219-220`), KMNIST GMM rows (`report.tex:304`, `307`), CIFAR-10 GMM row (`report.tex:524`) |
| K-means와 GMM 차이 분석 | 완료 | KMNIST K-means/GMM comparison (`report.tex:392-397`, `440-458`, `491-493`), CIFAR-10 comparison (`report.tex:610-614`, `662-679`, `705-709`), cross-dataset GMM section (`report.tex:870-888`) |

### 3.3 다양한 조합과 parameter/model structure 비교

| 요구사항 | 상태 | 대응 위치 및 근거 |
|---|---:|---|
| feature, clustering, parameter, model structure 등을 다양하게 조합 | 완료 | `experiments.csv` enabled row 45개: PCA/AE/CNN/foundation, K-means/GMM, PCA component sweep, AE latent/epoch sweep, CNN structure 비교 |
| 10개 이상의 실험 조합 권장 | 완료 | `experiments.csv` enabled row 45개 |
| 실험 결과를 보고서 표로 정리 | 완료 | KMNIST result table (`report.tex:292-307`), CIFAR-10 result table (`report.tex:513-524`), DINOv2 optional result table (`report.tex:765-774`) |

## 4. Optional Extensions

| 요구사항 또는 예시 | 상태 | 대응 위치 및 근거 |
|---|---:|---|
| Required Components 포함 후 추가 방법 적용 가능 | 완료 | Required 결과 후 DINOv2 optional section 추가 (`report.tex:721-823`) |
| 다른 feature extraction 방법 적용 | 완료 | DINOv2 ViT-S/14 frozen image encoder 적용 (`report.tex:723-757`), result table (`report.tex:765-774`) |
| 다른 clustering algorithm 적용 | 해당 없음 | optional 예시지만 이번 보고서에서는 선택하지 않음. Required clustering인 K-means/GMM에 집중 |
| visualization 방식 추가 | 완료 | DINOv2 structure image (`report.tex:738-741`), DINOv2 t-SNE (`report.tex:796-807`), DINOv2 heatmap (`report.tex:815-817`), CNN activation map (`report.tex:373-381`, `593-601`) |
| 추가 실험은 기존 방법과 비교하고 한계 설명 | 완료 | DINOv2 vs ResNet18 비교 (`report.tex:777-792`), optional limitation (`report.tex:820-823`), conclusion (`report.tex:910-918`) |

## 5. Visualization and Analysis

| 요구사항 또는 예시 | 상태 | 대응 위치 및 근거 |
|---|---:|---|
| LAB 시각화 방법을 적극 활용 | 완료 | Dataset samples, PCA/AE reconstruction, t-SNE, heatmap, activation map, mismatch examples 등 사용 |
| feature space visualization | 완료 | KMNIST selected t-SNE (`report.tex:405-413`), KMNIST final t-SNE (`report.tex:420-437`), CIFAR selected t-SNE (`report.tex:623-638`), CIFAR final t-SNE (`report.tex:642-659`), DINOv2 t-SNE (`report.tex:796-807`) |
| cluster별 image visualization | 부분 | 코드/output에는 `test_cluster_samples.png`, `test_mapped_class_grids/`가 생성됨. 예: `outputs/runs/kmnist_resnet18_e20_kmeans_full/plots/test_cluster_samples.png`, `outputs/runs/cifar10_resnet18_e20_kmeans_full/plots/test_cluster_samples.png`. PDF 본문에는 cluster별 sample grid 자체는 직접 삽입하지 않았고, mismatch image grid로 대체함 (`report.tex:497-499`, `716-718`) |
| class와 cluster 관계 분석 | 완료 | KMNIST heatmaps (`report.tex:440-458`), CIFAR heatmaps (`report.tex:662-679`), DINOv2 heatmap (`report.tex:815-817`) |
| 잘 분리된 class와 자주 섞이는 class 분석 | 완료 | KMNIST well/mixed classes (`report.tex:463-493`), CIFAR mixed classes (`report.tex:644`, `690-709`) |
| KMNIST와 CIFAR-10 결과 비교 | 완료 | Cross-Dataset Discussion (`report.tex:825-888`), selected metric bars (`report.tex:829-840`) |
| PCA, AE latent, CNN feature 차이 비교 | 완료 | KMNIST feature별 분석 (`report.tex:314-390`), CIFAR feature별 분석 (`report.tex:536-607`), cross-dataset summary (`report.tex:843-846`) |
| K-means와 GMM 차이 비교 | 완료 | KMNIST/CIFAR individual comparison 및 cross-dataset GMM section (`report.tex:392-397`, `610-614`, `870-888`) |
| 시각화가 단순 장식이 아니라 근거로 사용됨 | 완료 | PCA/AE reconstruction 해석 (`report.tex:326-342`, `548-562`), t-SNE/heatmap과 metric 연결 (`report.tex:420-443`, `634-638`, `777-817`) |

## 6. Evaluation

| 요구사항 | 상태 | 대응 위치 및 근거 |
|---|---:|---|
| 평가 지표를 스스로 선택 | 완료 | ARI, NMI, purity, silhouette 선택 (`report.tex:224`) |
| 여러 관점에서 비교 권장 | 완료 | ARI/NMI/purity/silhouette 수식과 설명 (`report.tex:224-276`), result tables에 네 지표 모두 포함 |
| 어떤 class가 자주 섞이는지 분석 | 완료 | KMNIST mismatch pairs (`report.tex:463-493`), CIFAR mismatch pairs (`report.tex:684-709`) |
| 같은 pipeline 추천 여부 분석 | 완료 | Cross-Dataset Discussion (`report.tex:829-846`), Conclusion (`report.tex:892-918`) |
| 평가 기준 선택 이유 설명 | 완료 | ARI/NMI/purity/silhouette 설명 및 수식 (`report.tex:224-276`), DINOv2에서 silhouette 해석 차이 설명 (`report.tex:789-792`) |
| 정량/정성 결과 경향 비교 | 완료 | KMNIST t-SNE와 ARI/NMI 일치 설명 (`report.tex:420-422`), CIFAR t-SNE/heatmap 설명 (`report.tex:634-638`), DINOv2 metric/visual 설명 (`report.tex:777-817`) |
| 최종적으로 가장 의미 있는 조합을 근거와 함께 설명 | 완료 | KMNIST best result (`report.tex:364-365`), CIFAR best required result (`report.tex:583`), final recommendation (`report.tex:892-918`) |

## 7. Main Questions to Answer

| 질문 | 상태 | 대응 위치 및 답 |
|---|---:|---|
| 1. KMNIST에서 가장 의미 있는 조합은? | 완료 | ResNet18 + K-means full. Table (`report.tex:306`), explanation (`report.tex:364-365`), conclusion (`report.tex:898`) |
| 2. CIFAR-10에서 가장 의미 있는 조합은? | 완료 | Required 기준 ResNet18 + K-means full (`report.tex:523`, `583`, `899`), optional 포함 시 DINOv2 + K-means가 더 높음 (`report.tex:910-918`) |
| 3. 두 데이터셋에서 같은 방법이 좋았는가? | 완료 | Required comparison에서는 둘 다 ResNet18 + K-means가 best (`report.tex:830`, `900`), dataset별 의미 차이 설명 (`report.tex:832-834`, `900-902`) |
| 4. PCA, AE, CNN feature 차이는? | 완료 | Setup sweeps (`report.tex:129-180`), KMNIST/CIFAR feature analysis (`report.tex:314-390`, `536-607`), summary (`report.tex:843-846`, `904-908`) |
| 5. K-means와 GMM 차이는? | 완료 | K-means vs GMM sections (`report.tex:392-397`, `610-614`, `870-888`) |
| 6. 정량 평가와 시각 분석이 같은 결론을 주었는가? | 완료 | KMNIST/CIFAR/DINOv2 visual interpretation (`report.tex:420-443`, `634-638`, `777-817`) |
| 7. 어떤 class가 잘 분리/자주 섞였는가? | 완료 | KMNIST class analysis (`report.tex:463-493`), CIFAR mismatch analysis (`report.tex:684-709`) |
| 8. 최종 추천 pipeline과 이유는? | 완료 | Equation and final recommendation (`report.tex:892-918`) |

## 8. Report Guideline 대응

| 권장 구성 | 상태 | 대응 위치 |
|---|---:|---|
| Introduction | 완료 | `report.tex:68-78` |
| Experimental Setup | 완료 | `report.tex:80-278` |
| Results on KMNIST | 완료 | `report.tex:280-499` |
| Results on CIFAR-10 | 완료 | `report.tex:502-718` |
| Cross-Dataset Discussion | 완료 | `report.tex:825-888` |
| Conclusion | 완료 | `report.tex:890-918` |
| Appendix / reproducibility | 완료 | `report.tex:920-941` |
| References | 완료 | `report.tex:943-947` |

## 9. Submission

| 제출 요구사항 | 상태 | 대응 위치 및 근거 |
|---|---:|---|
| 실행 가능한 code 제출 | 완료 | Python project: `final_project/run.py`, `final_project/compare_runs.py`, `final_project/generate_sweep_plots.py`, `final_project/src/kmnist_cifar_project/*`, config: `final_project/configs/experiments.csv` |
| 실험 과정과 결과 확인 가능 | 완료 | 각 run 폴더에 `config.yaml`, `metrics.json`, `features_train_test.npz`, `plots/` 저장. 보고서 재현성 설명 (`report.tex:922-941`) |
| Final Report PDF 제출 | 준비 완료 | `final_project/report.pdf` 빌드됨. 실제 제출 업로드는 사용자가 수행해야 함 |

## 부분 반영 또는 미반영 항목

| 항목 | 상태 | 설명 |
|---|---:|---|
| PDF 본문에 cluster별 sample grid 직접 삽입 | 부분 | run output에는 `test_cluster_samples.png`와 `test_mapped_class_grids/`가 존재하지만, 최종 PDF 본문에는 직접 삽입하지 않았음. 대신 mismatch example grid, heatmap, t-SNE로 분석 근거를 제시함 |
| Optional “다른 clustering algorithm 적용” | 해당 없음 | 과제 optional 예시 중 하나일 뿐이며, 이번 optional은 다른 feature extraction 방법(DINOv2)에 집중함 |
| 제출 행위 자체 | 미반영 | 코드와 PDF는 준비됐지만, LMS/이메일 등 실제 제출은 사용자가 해야 함 |

## 종합 판단

필수 요구사항은 모두 반영되어 있다.  
보고서에는 두 데이터셋, 세 feature representation, 두 clustering method, 10개 이상의 실험 조합, 정량 평가, 시각 분석, 최종 pipeline 추천이 포함되어 있다.  
주의할 점은 cluster별 sample grid가 output에는 있으나 PDF 본문에는 직접 들어가지 않았다는 점이다. 필요하면 해당 figure를 appendix나 각 dataset visual analysis에 추가하면 더 완전해진다.
