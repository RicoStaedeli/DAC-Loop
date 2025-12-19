| run_name | teacher_model | student_model | dataset | kd_type | hidden_rep_loss_weight | ce_loss_weight | soft_target_loss_weight | beta_feat | T_param | teacher_best_val_accuracy (%) | student_best_val_accuracy (%) | baseline_student_best_val_accuracy (%) | improvement_over_baseline (%) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Run_1 | ResNet50 | ResNet10mini | timm/resisc45 | CosineEmbeddingLoss | 0.1 | 0.75 | 0.25 | 0 | 2 | 92.2698 | 83.0635 | 82.9683 | 0.0952 |
| Run_2 | ResNet50 | ResNet10mini | timm/resisc45 | CosineEmbeddingLoss | 0.3 | 0.75 | 0.25 | 0 | 2 | 92.2698 | 83.5873 | 82.9683 | 0.6190 |
| Run_3 | ResNet50 | ResNet10mini | timm/resisc45 | CosineEmbeddingLoss | 0.5 | 0.75 | 0.25 | 0 | 2 | 92.2698 | 83.6190 | 82.9683 | 0.6508 |
| Run_4 | ResNet50 | ResNet10mini | timm/resisc45 | CosineEmbeddingLoss | 0.7 | 0.75 | 0.25 | 0 | 2 | 92.2698 | 84.0476 | 82.9683 | 1.0794 |
| Run_5 | ResNet50 | ResNet10mini | timm/resisc45 | CosineEmbeddingLoss | 0.9 | 0.75 | 0.25 | 0 | 2 | 92.2698 | 84.0317 | 82.9683 | 1.0635 |
| Run_6 | ResNet50 | ResNet10mini | timm/resisc45 | CosineEmbeddingLoss | 0.25 | 0.1 | 0.25 |  | 2 | 92.2698 | 85.0476 | 82.9683 | 2.0794 |
| Run_7 | ResNet50 | ResNet10mini | timm/resisc45 | CosineEmbeddingLoss | 0.25 | 0.3 | 0.25 | 0 | 2 | 92.2698 | 84.0476 | 82.9683 | 1.0794 |
| Run_8 | ResNet50 | ResNet10mini | timm/resisc45 | CosineEmbeddingLoss | 0.25 | 0.5 | 0.25 | 0 | 2 | 92.2698 | 83.5556 | 82.9683 | 0.5873 |
| Run_9 | ResNet50 | ResNet10mini | timm/resisc45 | CosineEmbeddingLoss | 0.25 | 0.7 | 0.25 | 0 | 2 | 92.2698 | 83.6032 | 82.9683 | 0.6349 |
| Run_10 | ResNet50 | ResNet10mini | timm/resisc45 | CosineEmbeddingLoss | 0.25 | 0.9 | 0.25 | 0 | 2 | 92.2698 | 83.4286 | 82.9683 | 0.4603 |
| Run_11 | ResNet50 | ResNet10mini | timm/resisc45 | CrossEntropyLoss | 0.25 | 0.1 | 0.25 | 0 | 2 | 92.2698 | 85.0635 | 82.9683 | 2.0952 |
| Run_12 | ResNet50 | ResNet10mini | timm/resisc45 | CrossEntropyLoss | 0.25 | 0.9 | 0.25 | 0 | 2 | 92.2698 | 84.1429 | 82.9683 | 1.1746 |
| Run_13 | ResNet50 | ResNet10mini | timm/resisc45 | CrossEntropyLoss | 0.25 | 0.75 | 0.25 |  | 3 | 92.2698 | 85.5397 | 82.9683 | 2.5714 |
| Run_14 | ResNet50 | ResNet10mini | timm/resisc45 | CrossEntropyLoss | 0.25 | 0.75 | 0.25 |  | 5 | 92.2698 | 86.1429 | 82.9683 | 3.1746 |
| Run_15 | ResNet50 | ResNet10mini | timm/resisc45 | CrossEntropyLoss | 0.25 | 0.75 | 0.25 |  | 10 | 92.2698 | 86.6349 | 82.9683 | 3.6667 |
| Run_16 | ResNet50 | ResNet10mini | timm/resisc45 | RegressorMSE | 0.1 | 0.5 | 0.5 |  | 2 | 92.2698 | 84.9365 | 82.9683 | 1.9683 |
| Run_17 | ResNet50 | ResNet10mini | timm/resisc45 | RegressorMSE | 0.4 | 0.5 | 0.1 |  | 4 | 92.2698 | 85.7937 | 82.9683 | 2.8254 |
| Run_18 | ResNet50 | ResNet10mini | timm/resisc45 | KLDivLoss | 0.25 | 0.3 | 0.25 | 0.25 | 4 | 92.2698 | 85.9365 | 82.9683 | 2.9683 |
| Run_19 | ResNet50 | ResNet10mini | timm/resisc45 | KLDivLoss | 0.25 | 0.2 | 0.25 | 0.1 | 4 | 92.2698 | 85.5873 | 82.9683 | 2.6190 |
| Run_20 | ResNet50 | ResNet10mini | timm/resisc45 | KLDivLoss | 0.25 | 0.01 | 0.25 |  | 4 | 92.2698 | 85.6667 | 82.9683 | 2.6984 |
