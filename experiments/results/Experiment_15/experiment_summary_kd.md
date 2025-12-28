| run_name | teacher_model | student_model | dataset | kd_type | hidden_rep_loss_weight | ce_loss_weight | soft_target_loss_weight | beta_feat | T_param | teacher_best_val_accuracy (%) | student_best_val_accuracy (%) | baseline_student_best_val_accuracy (%) | improvement_over_baseline (%) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Run_1 | ResNet50 | ResNetTiny | timm/resisc45 | CrossEntropyLoss | 0.25 | 0.75 | 0.25 |  | 10 | 92.2698 | 82.9524 | 79.8889 | 3.0635 |
| Run_2 | ResNet50 | ResNetTiny | timm/resisc45 | CrossEntropyLoss | 0.25 | 0.75 | 0.25 |  | 10 | 92.2698 | 82.3968 | 79.8889 | 2.5079 |
| Run_3 | ResNet50 | ResNetTiny | timm/resisc45 | CrossEntropyLoss | 0.25 | 0.75 | 0.25 |  | 10 | 92.2698 | 82.2063 | 79.8889 | 2.3175 |
| Run_4 | ResNet50 | ResNetTiny | blanchon/EuroSAT_MSI | CrossEntropyLoss | 0.25 | 0.75 | 0.5 | 0 | 2 | 96.9630 | 96.3148 | 95.6667 | 0.6481 |
| Run_5 | ResNet50 | ResNetTiny | blanchon/EuroSAT_MSI | CrossEntropyLoss | 0.25 | 0.75 | 0.5 | 0 | 2 | 96.9630 | 96.5370 | 95.6667 | 0.8704 |
| Run_6 | ResNet50 | ResNetTiny | blanchon/EuroSAT_MSI | CrossEntropyLoss | 0.25 | 0.75 | 0.5 | 0 | 2 | 96.9630 | 96.3519 | 95.6667 | 0.6852 |
| Run_7 | ResNet50 | ResNet10mini | timm/resisc45 | CrossEntropyLoss | 0.25 | 0.75 | 0.25 |  | 10 | 92.2698 | 86.0000 | 82.9683 | 3.0317 |
| Run_8 | ResNet50 | ResNet10mini | timm/resisc45 | CrossEntropyLoss | 0.25 | 0.75 | 0.25 |  | 10 | 92.2698 | 85.6349 | 82.9683 | 2.6667 |
| Run_9 | ResNet50 | ResNet10mini | timm/resisc45 | CrossEntropyLoss | 0.25 | 0.75 | 0.25 |  | 10 | 92.2698 | 86.3651 | 82.9683 | 3.3968 |
| Run_10 | ResNet50 | ResNet10mini | blanchon/EuroSAT_MSI | KLDivLoss | 0.25 | 0.1 | 0.25 | 0 | 4 | 96.9630 | 96.9630 | 96.3519 | 0.6111 |
| Run_11 | ResNet50 | ResNet10mini | blanchon/EuroSAT_MSI | KLDivLoss | 0.25 | 0.1 | 0.25 | 0 | 4 | 96.9630 | 97.0741 | 96.3519 | 0.7222 |
| Run_12 | ResNet50 | ResNet10mini | blanchon/EuroSAT_MSI | KLDivLoss | 0.25 | 0.1 | 0.25 | 0 | 4 | 96.9630 | 97.0370 | 96.3519 | 0.6852 |
