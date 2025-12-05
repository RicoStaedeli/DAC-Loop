| run_name | experiment_name | run_id | model | dataset | augmentation | indices_config | epochs | learning_rate | batch_size | weight_decay | scheduler | seed | best_epoch | best_train_loss | best_val_loss | best_train_accuracy (%) | best_val_accuracy (%) | best_val_MSE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Run_1 | Experiment_5 | 1 | ResNet18 | timm/resisc45 | True | calc=[] - bands=[B01, B02, B03] | 50 | 0.0003 | 128 | 1e-05 | cosine | 42 | 47 | 0.312045 | 0.285836 | 90.8042 | 91.3333 | 0.002906 |
| Run_2 | Experiment_5 | 2 | ResNet10 | timm/resisc45 | True | calc=[] - bands=[B01, B02, B03] | 50 | 0.0003 | 128 | 1e-05 | cosine | 42 | 49 | 0.522177 | 0.403941 | 84.6825 | 88.3175 | 0.004024 |
| Run_3 | Experiment_5 | 3 | TinyCNN | timm/resisc45 | True | calc=[] - bands=[B01, B02, B03] | 50 | 0.0003 | 128 | 1e-05 | cosine | 42 | 48 | 2.859013 | 2.551090 | 22.7037 | 29.9206 | 0.018245 |
| Run_4 | Experiment_5 | 4 | ResNet50 | timm/resisc45 | True | calc=[] - bands=[B01, B02, B03] | 50 | 0.0003 | 128 | 1e-05 | cosine | 42 | 50 | 0.193962 | 0.297681 | 93.6561 | 90.8889 | 0.002981 |
| Run_5 | Experiment_5 | 5 | ResNet50 | timm/resisc45 | True | calc=[] - bands=[B01, B02, B03] | 100 | 0.003 | 128 | 1e-05 | cosine | 42 | 83 | 0.079817 | 0.715278 | 97.2275 | 89.3016 | 0.003688 |
| Run_6 | Experiment_5 | 6 | ResNet50 | timm/resisc45 | True | calc=[] - bands=[B01, B02, B03] | 50 | 0.003 | 256 | 1e-05 | cosine | 42 | 49 | 0.517164 | 1.220733 | 83.5767 | 81.6190 | 0.005890 |
| Run_7 | Experiment_5 | 7 | ResNet50 | timm/resisc45 | True | calc=[] - bands=[B01, B02, B03] | 100 | 0.0003 | 128 | 1e-05 | cosine | 42 | 93 | 0.056677 | 0.283054 | 98.2275 | 92.6825 | 0.002625 |
| Run_8 | Experiment_5 | 9 | ResNet10mini | timm/resisc45 | False | calc=[] - bands=[B01, B02, B03] | 50 | 0.0003 | 256 | 1e-05 | cosine | 42 | 41 | 0.024147 | 0.550558 | 99.9418 | 83.3016 | 0.005407 |
