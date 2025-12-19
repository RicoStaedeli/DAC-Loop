| run_name | experiment_name | run_id | model | dataset | augmentation | indices_config | epochs | learning_rate | batch_size | weight_decay | scheduler | seed | best_epoch | best_train_loss | best_val_loss | best_train_accuracy (%) | best_val_accuracy (%) | best_val_MSE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Run_1 | Experiment_1a | 1 | ResNet18 | timm/resisc45 | False | calc=[] - bands=[B01, B02, B03] | 50 | 0.0003 | 256 | 1e-05 | cosine | 42 | 47 | 0.007080 | 0.341327 | 100.0000 | 89.8889 | 0.003410 |
| Run_2 | Experiment_1a | 2 | ResNet10 | timm/resisc45 | False | calc=[] - bands=[B01, B02, B03] | 50 | 0.0003 | 256 | 1e-05 | cosine | 42 | 47 | 0.039200 | 0.385419 | 99.8889 | 88.5556 | 0.003898 |
| Run_3 | Experiment_1a | 3 | TinyCNN | timm/resisc45 | False | calc=[] - bands=[B01, B02, B03] | 50 | 0.0003 | 256 | 1e-05 | cosine | 42 | 36 | 2.290286 | 2.281528 | 35.6667 | 36.6825 | 0.017228 |
| Run_4 | Experiment_1a | 4 | ResNet50 | timm/resisc45 | True | calc=[] - bands=[B01, B02, B03] | 50 | 0.0003 | 256 | 1e-05 | cosine | 42 | 48 | 0.227263 | 0.308174 | 92.6878 | 90.4444 | 0.003168 |
| Run_5 | Experiment_1a | 5 | ResNet50 | timm/resisc45 | True | calc=[] - bands=[B01, B02, B03] | 100 | 0.003 | 256 | 1e-05 | cosine | 42 | 94 | 0.116522 | 0.814795 | 96.1376 | 87.0476 | 0.004454 |
| Run_6 | Experiment_1a | 6 | ResNet50 | timm/resisc45 | True | calc=[] - bands=[B01, B02, B03] | 50 | 0.003 | 256 | 1e-05 | cosine | 42 | 49 | 0.505953 | 0.650509 | 83.7037 | 82.3016 | 0.005592 |
| Run_7 | Experiment_1a | 7 | ResNet50 | timm/resisc45 | True | calc=[] - bands=[B01, B02, B03] | 100 | 0.0003 | 256 | 1e-05 | cosine | 42 | 99 | 0.062632 | 0.278373 | 98.1534 | 92.2698 | 0.002699 |
| Run_8 | Experiment_1a | 8 | ResNet10mini | timm/resisc45 | False | calc=[] - bands=[B01, B02, B03] | 50 | 0.0003 | 256 | 1e-05 | cosine | 42 | 44 | 0.022253 | 0.545932 | 99.9683 | 82.9683 | 0.005401 |
