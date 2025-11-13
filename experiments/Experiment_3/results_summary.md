| run_name | teacher_model | student_model | dataset | kd_type | hidden_rep_loss_weight | ce_loss_weight | soft_target_loss_weight | alpha_kd | beta_feat | T_param | teacher_best_val_accuracy (%) | student_best_val_accuracy (%) | baseline_student_best_val_accuracy (%) | improvement_over_baseline (%) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Run_1 | ResNet18 | TinyCNN | blanchon/EuroSAT_MSI | CosineEmbeddingLoss | 0.1 | 0.75 | 0.25 |  |  | 2 | 97.7593 | 87.2778 | 87.8704 | -0.5926 |
| Run_2 | ResNet18 | TinyCNN | blanchon/EuroSAT_MSI | CosineEmbeddingLoss | 0.3 | 0.75 | 0.25 |  |  | 2 | 97.7593 | 86.8889 | 87.8704 | -0.9815 |
| Run_3 | ResNet18 | TinyCNN | blanchon/EuroSAT_MSI | CosineEmbeddingLoss | 0.5 | 0.75 | 0.25 |  |  | 2 | 97.7593 | 87.3333 | 87.8704 | -0.5370 |
| Run_4 | ResNet18 | TinyCNN | blanchon/EuroSAT_MSI | CosineEmbeddingLoss | 0.7 | 0.75 | 0.25 |  |  | 2 | 97.7593 | 87.3704 | 87.8704 | -0.5000 |
| Run_5 | ResNet18 | TinyCNN | blanchon/EuroSAT_MSI | CosineEmbeddingLoss | 0.9 | 0.75 | 0.25 |  |  | 2 | 97.7593 | 87.5926 | 87.8704 | -0.2778 |
| Run_6 | ResNet18 | TinyCNN | blanchon/EuroSAT_MSI | CosineEmbeddingLoss | 0.25 | 0.1 | 0.25 |  |  | 2 | 97.7593 | 87.5556 | 87.8704 | -0.3148 |
| Run_7 | ResNet18 | TinyCNN | blanchon/EuroSAT_MSI | CosineEmbeddingLoss | 0.25 | 0.3 | 0.25 |  |  | 2 | 97.7593 | 87.3148 | 87.8704 | -0.5556 |
| Run_8 | ResNet18 | TinyCNN | blanchon/EuroSAT_MSI | CosineEmbeddingLoss | 0.25 | 0.5 | 0.25 |  |  | 2 | 97.7593 | 86.9630 | 87.8704 | -0.9074 |
| Run_9 | ResNet18 | TinyCNN | blanchon/EuroSAT_MSI | CosineEmbeddingLoss | 0.25 | 0.7 | 0.25 |  |  | 2 | 97.7593 | 86.8333 | 87.8704 | -1.0370 |
| Run_10 | ResNet18 | TinyCNN | blanchon/EuroSAT_MSI | CosineEmbeddingLoss | 0.25 | 0.9 | 0.25 |  |  | 2 | 97.7593 | 87.0556 | 87.8704 | -0.8148 |
| Run_11 | ResNet18 | TinyCNN | blanchon/EuroSAT_MSI | CrossEntropyLoss | 0.25 | 0.1 | 0.25 |  |  | 2 | 97.7593 | 86.0741 | 87.8704 | -1.7963 |
| Run_12 | ResNet18 | TinyCNN | blanchon/EuroSAT_MSI | CrossEntropyLoss | 0.25 | 0.3 | 0.25 |  |  | 2 | 97.7593 | 86.3704 | 87.8704 | -1.5000 |
| Run_13 | ResNet18 | TinyCNN | blanchon/EuroSAT_MSI | CrossEntropyLoss | 0.25 | 0.5 | 0.25 |  |  | 2 | 97.7593 | 86.7407 | 87.8704 | -1.1296 |
| Run_14 | ResNet18 | TinyCNN | blanchon/EuroSAT_MSI | CrossEntropyLoss | 0.25 | 0.7 | 0.25 |  |  | 2 | 97.7593 | 86.7963 | 87.8704 | -1.0741 |
| Run_15 | ResNet18 | TinyCNN | blanchon/EuroSAT_MSI | CrossEntropyLoss | 0.25 | 0.9 | 0.25 |  |  | 2 | 97.7593 | 87.0000 | 87.8704 | -0.8704 |
| Run_16 | ResNet18 | TinyCNN | blanchon/EuroSAT_MSI | CrossEntropyLoss | 0.25 | 0.75 | 0.1 |  |  | 2 | 97.7593 | 87.0370 | 87.8704 | -0.8333 |
| Run_17 | ResNet18 | TinyCNN | blanchon/EuroSAT_MSI | CrossEntropyLoss | 0.25 | 0.75 | 0.3 |  |  | 2 | 97.7593 | 86.6111 | 87.8704 | -1.2593 |
| Run_18 | ResNet18 | TinyCNN | blanchon/EuroSAT_MSI | CrossEntropyLoss | 0.25 | 0.75 | 0.5 |  |  | 2 | 97.7593 | 86.6481 | 87.8704 | -1.2222 |
| Run_19 | ResNet18 | TinyCNN | blanchon/EuroSAT_MSI | CrossEntropyLoss | 0.25 | 0.75 | 0.7 |  |  | 2 | 97.7593 | 86.2963 | 87.8704 | -1.5741 |
| Run_20 | ResNet18 | TinyCNN | blanchon/EuroSAT_MSI | CrossEntropyLoss | 0.25 | 0.75 | 0.9 |  |  | 2 | 97.7593 | 86.2407 | 87.8704 | -1.6296 |
| Run_21 | ResNet18 | TinyCNN | blanchon/EuroSAT_MSI | CrossEntropyLoss | 0.25 | 0.75 | 0.25 |  |  | 3 | 97.7593 | 86.2963 | 87.8704 | -1.5741 |
| Run_22 | ResNet18 | TinyCNN | blanchon/EuroSAT_MSI | CrossEntropyLoss | 0.25 | 0.75 | 0.25 |  |  | 5 | 97.7593 | 85.0000 | 87.8704 | -2.8704 |
| Run_23 | ResNet18 | TinyCNN | blanchon/EuroSAT_MSI | CrossEntropyLoss | 0.25 | 0.75 | 0.25 |  |  | 10 | 97.7593 | 85.0926 | 87.8704 | -2.7778 |
| Run_24 | ResNet18 | TinyCNN | blanchon/EuroSAT_MSI | KLDivLoss | 0.25 | 0.75 | 0.25 |  |  | 10 | 97.7593 | 84.7593 | 87.8704 | -3.1111 |
