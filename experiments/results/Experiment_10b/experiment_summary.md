| run_name | teacher_model | student_model | dataset | kd_type | hidden_rep_loss_weight | ce_loss_weight | soft_target_loss_weight | alpha_kd | beta_feat | T_param | teacher_best_val_accuracy (%) | student_best_val_accuracy (%) | baseline_student_best_val_accuracy (%) | improvement_over_baseline (%) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Run_1 | ResNet50 | ResNet18 | timm/resisc45 | CrossEntropyLoss | 0.25 | 0.75 | 0.25 | 0 | 0 | 10 | 92.6825 | 92.9524 | 89.9683 | 2.9841 |
| Run_2 | ResNet50 | ResNet18 | timm/resisc45 | CrossEntropyLoss | 0.25 | 0.75 | 0.25 | 0 | 0 | 10 | 92.6825 | 92.9048 | 89.9683 | 2.9365 |
| Run_3 | ResNet50 | ResNet18 | timm/resisc45 | CrossEntropyLoss | 0.25 | 0.75 | 0.25 | 0 | 0 | 10 | 92.6825 | 92.5873 | 89.9683 | 2.6190 |
