| run_name | teacher_model | student_model | dataset | kd_type | hidden_rep_loss_weight | ce_loss_weight | soft_target_loss_weight | beta_feat | T_param | teacher_best_val_accuracy (%) | student_best_val_accuracy (%) | baseline_student_best_val_accuracy (%) | improvement_over_baseline (%) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Run_1 | ResNet50 | ResNetTiny | timm/resisc45 | CosineEmbeddingLoss | 0.1 | 0.75 | 0.25 | 0 | 2 | 92.2698 | 81.0317 | 36.6825 | 44.3492 |
| Run_2 | ResNet50 | ResNetTiny | timm/resisc45 | CosineEmbeddingLoss | 0.3 | 0.75 | 0.25 | 0 | 2 | 92.2698 | 80.9841 | 36.6825 | 44.3016 |
| Run_3 | ResNet50 | ResNetTiny | timm/resisc45 | CosineEmbeddingLoss | 0.5 | 0.75 | 0.25 | 0 | 2 | 92.2698 | 81.5556 | 36.6825 | 44.8730 |
| Run_4 | ResNet50 | ResNetTiny | timm/resisc45 | CosineEmbeddingLoss | 0.7 | 0.75 | 0.25 | 0 | 2 | 92.2698 | 81.6825 | 36.6825 | 45.0000 |
| Run_5 | ResNet50 | ResNetTiny | timm/resisc45 | CosineEmbeddingLoss | 0.9 | 0.75 | 0.25 | 0 | 2 | 92.2698 | 82.0317 | 36.6825 | 45.3492 |
| Run_6 | ResNet50 | ResNetTiny | timm/resisc45 | CosineEmbeddingLoss | 0.25 | 0.1 | 0.25 |  | 2 | 92.2698 | 81.8889 | 36.6825 | 45.2063 |
| Run_7 | ResNet50 | ResNetTiny | timm/resisc45 | CosineEmbeddingLoss | 0.25 | 0.3 | 0.25 | 0 | 2 | 92.2698 | 81.2222 | 36.6825 | 44.5397 |
| Run_8 | ResNet50 | ResNetTiny | timm/resisc45 | CosineEmbeddingLoss | 0.25 | 0.5 | 0.25 | 0 | 2 | 92.2698 | 81.2540 | 36.6825 | 44.5714 |
| Run_9 | ResNet50 | ResNetTiny | timm/resisc45 | CosineEmbeddingLoss | 0.25 | 0.7 | 0.25 | 0 | 2 | 92.2698 | 80.8095 | 36.6825 | 44.1270 |
| Run_10 | ResNet50 | ResNetTiny | timm/resisc45 | CosineEmbeddingLoss | 0.25 | 0.9 | 0.25 | 0 | 2 | 92.2698 | 81.2063 | 36.6825 | 44.5238 |
| Run_11 | ResNet50 | ResNetTiny | timm/resisc45 | CrossEntropyLoss | 0.25 | 0.1 | 0.25 | 0 | 2 | 92.2698 | 81.0159 | 36.6825 | 44.3333 |
| Run_12 | ResNet50 | ResNetTiny | timm/resisc45 | CrossEntropyLoss | 0.25 | 0.9 | 0.25 | 0 | 2 | 92.2698 | 81.2540 | 36.6825 | 44.5714 |
| Run_13 | ResNet50 | ResNetTiny | timm/resisc45 | CrossEntropyLoss | 0.25 | 0.75 | 0.25 |  | 3 | 92.2698 | 81.9048 | 36.6825 | 45.2222 |
| Run_14 | ResNet50 | ResNetTiny | timm/resisc45 | CrossEntropyLoss | 0.25 | 0.75 | 0.25 |  | 5 | 92.2698 | 82.3651 | 36.6825 | 45.6825 |
| Run_15 | ResNet50 | ResNetTiny | timm/resisc45 | CrossEntropyLoss | 0.25 | 0.75 | 0.25 |  | 10 | 92.2698 | 82.9365 | 36.6825 | 46.2540 |
| Run_16 | ResNet50 | ResNetTiny | timm/resisc45 | RegressorMSE | 0.1 | 0.5 | 0.5 |  | 2 | 92.2698 | 81.3968 | 36.6825 | 44.7143 |
| Run_17 | ResNet50 | ResNetTiny | timm/resisc45 | RegressorMSE | 0.4 | 0.5 | 0.1 |  | 4 | 92.2698 | 82.3810 | 36.6825 | 45.6984 |
| Run_18 | ResNet50 | ResNetTiny | timm/resisc45 | KLDivLoss | 0.25 | 0.3 | 0.25 | 0.25 | 4 | 92.2698 | 82.2381 | 36.6825 | 45.5556 |
| Run_19 | ResNet50 | ResNetTiny | timm/resisc45 | KLDivLoss | 0.25 | 0.2 | 0.25 | 0.1 | 4 | 92.2698 | 82.2381 | 36.6825 | 45.5556 |
| Run_20 | ResNet50 | ResNetTiny | timm/resisc45 | KLDivLoss | 0.25 | 0.01 | 0.25 |  | 4 | 92.2698 | 82.4921 | 36.6825 | 45.8095 |
