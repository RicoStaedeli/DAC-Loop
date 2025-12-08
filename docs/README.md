# Project Documentation
This document provides an overview of the R-L Net framework, instructions for generating plots, baseline results, and ablation studies.


## Baselines

In this section we provide baseline results for various tasks using the R-L Net framework.

### Baselines for Teacher-Student Learning

The baseline for teacher-student learning are the models trained from scratch without any teacher supervision.

**Results for Teacher-Student Learning on the EuroSAT dataset:**

| run_name | experiment_name | run_id | model    | dataset              | augmentation | indices_config                                                               | epochs | learning_rate | batch_size | weight_decay | scheduler | seed | best_epoch | best_train_loss | best_val_loss | best_train_accuracy (%) | best_val_accuracy (%) | best_val_MSE |
|----------|-----------------|--------|----------|----------------------|--------------|------------------------------------------------------------------------------|--------|---------------|------------|--------------|-----------|------|------------|-----------------|---------------|-------------------------|-----------------------|--------------|
| Run_1    | Experiment_1    | 1      | ResNet18 | blanchon/EuroSAT_MSI | False        | calc=[] - bands=[B01, B02, B03, B04, B05, B06, B07, B08, B8A, B09, B11, B12] | 50     | 0.0003        | 256        | 1e-05        | cosine    | 42   | 47         | 0.000555        | 0.086449      | 99.9938                 | 98.2778               | 0.002988     |
| Run_2    | Experiment_1    | 2      | ResNet18 | blanchon/EuroSAT_MSI | False        | calc=[] - bands=[B02, B03, B04]                                              | 50     | 0.0003        | 256        | 1e-05        | cosine    | 42   | 46         | 0.000678        | 0.141871      | 99.9938                 | 96.8519               | 0.005382     |
| Run_3    | Experiment_1    | 3      | ResNet18 | blanchon/EuroSAT_MSI | False        | calc=[NDVI, NDWI, NDMI] - bands=[B02, B03, B04, B07, B08]                    | 50     | 0.0003        | 256        | 1e-05        | cosine    | 42   | 50         | 0.000349        | 0.095817      | 99.9938                 | 97.9444               | 0.003487     |
| Run_4    | Experiment_1    | 4      | ResNet10 | blanchon/EuroSAT_MSI | False        | calc=[] - bands=[B01, B02, B03, B04, B05, B06, B07, B08, B8A, B09, B11, B12] | 50     | 0.0003        | 256        | 1e-05        | cosine    | 42   | 24         | 0.008954        | 0.068166      | 99.7407                 | 98.4074               | 0.002831     |
| Run_5    | Experiment_1    | 5      | ResNet10 | blanchon/EuroSAT_MSI | False        | calc=[] - bands=[B02, B03, B04]                                              | 50     | 0.0003        | 256        | 1e-05        | cosine    | 42   | 37         | 0.001027        | 0.182201      | 99.9815                 | 95.2963               | 0.007731     |
| Run_6    | Experiment_1    | 6      | ResNet10 | blanchon/EuroSAT_MSI | False        | calc=[NDVI, NDWI, NDMI] - bands=[B02, B03, B04, B07, B08]                    | 50     | 0.0003        | 256        | 1e-05        | cosine    | 42   | 47         | 0.000445        | 0.084619      | 100.0000                | 97.8333               | 0.003587     |
| Run_7    | Experiment_1    | 7      | TinyCNN  | blanchon/EuroSAT_MSI | False        | calc=[] - bands=[B01, B02, B03, B04, B05, B06, B07, B08, B8A, B09, B11, B12] | 50     | 0.0003        | 256        | 1e-05        | cosine    | 42   | 45         | 0.320458        | 0.337920      | 89.3951                 | 89.2222               | 0.016315     |
| Run_8    | Experiment_1    | 8      | TinyCNN  | blanchon/EuroSAT_MSI | False        | calc=[] - bands=[B02, B03, B04]                                              | 50     | 0.0003        | 256        | 1e-05        | cosine    | 42   | 44         | 0.658462        | 0.674295      | 77.1667                 | 76.7222               | 0.033131     |
| Run_9    | Experiment_1    | 9      | TinyCNN  | blanchon/EuroSAT_MSI | False        | calc=[NDVI, NDWI, NDMI] - bands=[B02, B03, B04, B07, B08]                    | 50     | 0.0003        | 256        | 1e-05        | cosine    | 42   | 39         | 0.380586        | 0.381733      | 87.4259                 | 87.8704               | 0.018571     |
| Run_10   | Experiment_1    | 10     | ResNet50 | blanchon/EuroSAT_MSI | False        | calc=[] - bands=[B01, B02, B03, B04, B05, B06, B07, B08, B8A, B09, B11, B12] | 50     | 0.0003        | 256        | 1e-05        | cosine    | 42   | 48         | 0.000205        | 0.171966      | 99.9938                 | 96.9630               | 0.005394     |
| Run_11   | Experiment_1    | 11     | ResNet50 | blanchon/EuroSAT_MSI | False        | calc=[] - bands=[B02, B03, B04]                                              | 50     | 0.0003        | 256        | 1e-05        | cosine    | 42   | 45         | 0.002114        | 0.512845      | 99.9753                 | 90.9074               | 0.015801     |
| Run_12   | Experiment_1    | 12     | ResNet50 | blanchon/EuroSAT_MSI | False        | calc=[NDVI, NDWI, NDMI] - bands=[B02, B03, B04, B07, B08]                    | 50     | 0.0003        | 256        | 1e-05        | cosine    | 42   | 43         | 0.000237        | 0.205801      | 100.0000                | 96.1296               | 0.006716     |

**Results for Teacher-Student Learning on the RESISC45 dataset:**

| run_name | experiment_name | run_id | model        | dataset       | augmentation | indices_config                  | epochs | learning_rate | batch_size | weight_decay | scheduler | seed | best_epoch | best_train_loss | best_val_loss | best_train_accuracy (%) | best_val_accuracy (%) | best_val_MSE |
|----------|-----------------|--------|--------------|---------------|--------------|---------------------------------|--------|---------------|------------|--------------|-----------|------|------------|-----------------|---------------|-------------------------|-----------------------|--------------|
| Run_1    | Experiment_5    | 1      | ResNet18     | timm/resisc45 | True         | calc=[] - bands=[B01, B02, B03] | 50     | 0.0003        | 128        | 1e-05        | cosine    | 42   | 47         | 0.312045        | 0.285836      | 90.8042                 | 91.3333               | 0.002906     |
| Run_2    | Experiment_5    | 2      | ResNet10     | timm/resisc45 | True         | calc=[] - bands=[B01, B02, B03] | 50     | 0.0003        | 128        | 1e-05        | cosine    | 42   | 49         | 0.522177        | 0.403941      | 84.6825                 | 88.3175               | 0.004024     |
| Run_3    | Experiment_5    | 3      | TinyCNN      | timm/resisc45 | True         | calc=[] - bands=[B01, B02, B03] | 50     | 0.0003        | 128        | 1e-05        | cosine    | 42   | 48         | 2.859013        | 2.551090      | 22.7037                 | 29.9206               | 0.018245     |
| Run_4    | Experiment_5    | 4      | ResNet50     | timm/resisc45 | True         | calc=[] - bands=[B01, B02, B03] | 50     | 0.0003        | 128        | 1e-05        | cosine    | 42   | 50         | 0.193962        | 0.297681      | 93.6561                 | 90.8889               | 0.002981     |
| Run_5    | Experiment_5    | 5      | ResNet50     | timm/resisc45 | True         | calc=[] - bands=[B01, B02, B03] | 100    | 0.003         | 128        | 1e-05        | cosine    | 42   | 83         | 0.079817        | 0.715278      | 97.2275                 | 89.3016               | 0.003688     |
| Run_6    | Experiment_5    | 6      | ResNet50     | timm/resisc45 | True         | calc=[] - bands=[B01, B02, B03] | 50     | 0.003         | 256        | 1e-05        | cosine    | 42   | 49         | 0.517164        | 1.220733      | 83.5767                 | 81.6190               | 0.005890     |
| Run_7    | Experiment_5    | 7      | ResNet50     | timm/resisc45 | True         | calc=[] - bands=[B01, B02, B03] | 100    | 0.0003        | 128        | 1e-05        | cosine    | 42   | 93         | 0.056677        | 0.283054      | 98.2275                 | 92.6825               | 0.002625     |
| Run_8    | Experiment_5    | 8      | ResNet18     | timm/resisc45 | False        | calc=[] - bands=[B01, B02, B03] | 50     | 0.0003        | 128        | 1e-05        | cosine    | 42   | 49         | 0.006621        | 0.345476      | 99.9947                 | 89.9683               | 0.003377     |
| Run_9    | Experiment_5    | 9      | ResNet10mini | timm/resisc45 | False        | calc=[] - bands=[B01, B02, B03] | 50     | 0.0003        | 256        | 1e-05        | cosine    | 42   | 41         | 0.024147        | 0.550558      | 99.9418                 | 83.3016               | 0.005407     |


**Additional baselines are found through research:**

| Model                         | Parameters (M) | Dataset  | Accuracy Theirs(%) | Accuracy Ours Baselines(%) | Source                                                                     |
|-------------------------------|----------------|----------|--------------------|----------------------------|----------------------------------------------------------------------------|
| ResNet18                      |                | RESISC45 | 92.10              | 91.33                      | [Source](https://arxiv.org/pdf/2507.10381)                                 |
| ResNet18                      |                | EuroSAT  | 98.06              | 98.27                      | [Source](https://arxiv.org/pdf/2507.10381)                                 |
| ResNet50 - InDomain           |                | RESISC45 | 96.80              | 92.68                      | [Source](http://arxiv.org/abs/1911.06721)                                  |
| ResNet50 - InDomain           |                | EuroSAT  | 99.20              | 96.96                      | [Source](http://arxiv.org/abs/1911.06721)                                  |
| ResNet50                      |                | EuroSAT  | 98.57              | 96.96                      | [Source](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8736785) |
| EfficientNetB0                | 4.7            | RESISC45 | 92.3               |                            | [Source](https://arxiv.org/pdf/2302.13028)                                 |
| EfficientNetB0 + Distillation | 4.7            | RESISC45 | 94.8               |                            | [Source](https://arxiv.org/pdf/2302.13028)                                 |
| EfficientNetB0 + Attention    | 9.4            | RESISC45 | 93.8               |                            | [Source](https://arxiv.org/pdf/2211.02820)                                 |



## Ablation Study
The teacher was in all stages end experiments the same: ResNet50 trained on RESISC45 for 50 epochs with a best validation accuracy of 92.68%. [Experiment 5, Run_7](../experiments/results/Experiment_5/Run_7)

### ResNet10small on RESISC45
| Model                        | Parameter | Validation Accuracy (%) | Run ID                                                           | Notes                                              |
|------------------------------|-----------|-------------------------|------------------------------------------------------------------|----------------------------------------------------|
| ResNet10small - Our Baseline | 2,098,677 | 83.30                   | [Experiment 5, Run_9](../experiments/results/Experiment_5/Run_1) | Train from scratch without a teacher for 50 epochs |
| ResNet10small + KD           | 2,098,677 | 86.07                   | [Experiment 6b, 23](../experiments/results/Experiment_6b/Run_23) | Train from scratch with a teacher for 50 epochs    |

### ResNet10 on RESISC45
| Model                                                | Parameter | Validation Accuracy (%) | Run ID                                                                          | Notes                                                       |
|------------------------------------------------------|-----------|-------------------------|---------------------------------------------------------------------------------|-------------------------------------------------------------|
| ResNet10 - Our Baseline                              | 4,921,197 | 88.31                   | [Experiment 5, Run_2](../experiments/results/Experiment_5/Run_2)                | Train from scratch without a teacher for 50 epochs          |
| ResNet10 + KD                                        | 4,921,197 | 90.98                   | [Experiment 6a, Run_40](../experiments/results/Experiment_6a/Run_40)            | Train from scratch with a teacher for 50 epochs             |
| ResNet10 - No KD + Pruning (38.43%) - No KD          | 3,030,044 | 90.22                   | [Experiment 8, Run_1](../experiments/results/Experiment_8/Run_1)                | Compress model ResNet10 - Baseline and fine-tune without KD |
| ResNet10 - No KD + Pruning (38.43%) + KD             | 3,030,044 | 91.16                   | [Experiment 8, Run_9](../experiments/results/Experiment_8/Run_9)                | Compress model ResNet10 - No KD and fine-tune with KD       |
| ResNet10 + KD + Pruning (38.43%) - No KD             | 3,030,044 | 89.02                   | [Experiment 8, Run_5](../experiments/results/Experiment_8/Run_5)                | Compress model ResNet10 - KD and fine-tune without KD       |
| ResNet10 + KD + Pruning (38.43%) + KD                | 3,030,044 | 91.00                   | [Experiment 7a, Run_10](../experiments/results/Experiment_7a/Run_10)            | Compress model ResNet10 - KD and fine-tune with KD          |
| ResNet10 - No KD + Pruning (65.71%) - No KD          | 1,687,563 | 89.52                   | [Experiment 8, Run_2](../experiments/results/Experiment_8/Run_2)                | Compress model ResNet10 - Baseline and fine-tune without KD |
| ResNet10 - No KD + Pruning (65.71%) + KD             | 1,687,563 | 89.62                   | [Experiment 8, Run_10](../experiments/results/Experiment_8/Run_10)              | Compress model ResNet10 - No KD and fine-tune with KD       |
| ResNet10 + KD + Pruning (65.71%) - No KD             | 1,687,563 | 89.02                   | [Experiment 8, Run_6](../experiments/results/Experiment_8/Run_6)                | Compress model ResNet10 - KD and fine-tune without KD       |
| ResNet10 + KD + Pruning (65.71%) + KD                | 1,687,563 | 89.97                   | [Experiment 7a, Run_11](../experiments/results/Experiment_7a/Run_11)            | Compress model ResNet10 - KD and fine-tune with KD          |
| ResNet10 + 2 iterations of KD aware compression loop | 3,030,044 | **91.33**               | [Experiment 9a, Run_3, Iteration 4](../experiments/results/Experiment_9a/Run_3) | Distillation-Aware compression loop with 2 iterations       |
| ResNet10 + 4 iterations of KD aware compression loop | 1,872,563 | **90.89**               | [Experiment 9a, Run_3, Iteration 4](../experiments/results/Experiment_9a/Run_3) | Distillation-Aware compression loop with 4 iterations       |

### ResNet18 on RESISC45
| Model                                                | Parameter  | Validation Accuracy (%) | Run ID                                                                        | Notes                                                       |
|------------------------------------------------------|------------|-------------------------|-------------------------------------------------------------------------------|-------------------------------------------------------------|
| ResNet18 - Our Baseline                              | 11,191,917 | 89.96                   | [Experiment 5, Run_8](../experiments/results/Experiment_5/Run_1)              | Train from scratch without a teacher for 50 epochs          |
| ResNet18 + KD                                        | 11,191,917 | 92.98                   | [Experiment 6, Run_23](../experiments/results/Experiment_6/Run_23)            | Train from scratch with a teacher for 50 epochs             |
| ResNet18 - No KD + Pruning (38.43%) - No KD          | 5,878,933  | 90.14                   | [Experiment 8, Run_3](../experiments/results/Experiment_8/Run_3)              | Compress model ResNet18 - Baseline and fine-tune without KD |
| ResNet18 - No KD + Pruning (38.43%) + KD             | 5,878,933  | 91.90                   | [Experiment 8, Run_11](../experiments/results/Experiment_8/Run_11)            | Compress model ResNet18 - No KD and fine-tune with KD       |
| ResNet18 + KD + Pruning (38.43%) - No KD             | 5,878,933  | 91.37                   | [Experiment 8, Run_7](../experiments/results/Experiment_8/Run_7)              | Compress model ResNet18 - KD and fine-tune without KD       |
| ResNet18 + KD + Pruning (38.43%) + KD                | 5,878,933  | 92.35                   | [Experiment 7, Run_10](../experiments/results/Experiment_7/Run_10)            | Compress model ResNet18 - KD and fine-tune with KD          |
| ResNet18 - No KD + Pruning (65.71%) - No KD          | 2,708,024  | 88.84                   | [Experiment 8, Run_4](../experiments/results/Experiment_8/Run_4)              | Compress model ResNet18 - Baseline and fine-tune without KD |
| ResNet18 - No KD + Pruning (65.71%) + KD             | 2,708,024  | 90.48                   | [Experiment 8, Run_12](../experiments/results/Experiment_8/Run_12)            | Compress model ResNet18 - No KD and fine-tune with KD       |
| ResNet18 + KD + Pruning (65.71%) - No KD             | 2,708,024  | 89.63                   | [Experiment 8, Run_8](../experiments/results/Experiment_8/Run_8)              | Compress model ResNet18 - KD and fine-tune without KD       |
| ResNet18 + KD + Pruning (65.71%) + KD                | 2,708,024  | 91.37                   | [Experiment 7, Run_11](../experiments/results/Experiment_7/Run_11)            | Compress model ResNet18 - KD and fine-tune with KD          |
| ResNet18 + 6 iterations of KD aware compression loop | 1'653'545  | **92.44**               | [Experiment 9, Run_5, Iteration 6](../experiments/results/Experiment_9/Run_5) | Distillation-Aware compression loop with 6 iterations       |
| ResNet18 + 9 iterations of KD aware compression loop | 885,489    | **91.10**               | [Experiment 9, Run_7, Iteration 9](../experiments/results/Experiment_9/Run_7) | Distillation-Aware compression loop with 9 iterations       |


### ResNet10 on EuroSAT
| Model                                       | Parameter | Validation Accuracy (%) | Run ID                                                                 | Notes                                                       |
|---------------------------------------------|-----------|-------------------------|------------------------------------------------------------------------|-------------------------------------------------------------|
| ResNet10 - Our Baseline                     | 4,921,197 | 98.40                   | [Experiment 1, Run_4](../experiments/results/Experiment_1/Run_4)       | Train from scratch without a teacher for 50 epochs          |
| ResNet10 + KD                               | 4,921,197 | 98.57                   | [Experiment 4a, Run_11](../experiments/results/Experiment_4a/Run_11)   | Train from scratch with a teacher for 50 epochs             |
| ResNet10 - No KD + Pruning (38.46%) - No KD | 3,020,831 | 98.04                   | [Experiment 12, Run_1](../experiments/results/Experiment_12/Run_1)     | Compress model ResNet10 - Baseline and fine-tune without KD |
| ResNet10 - No KD + Pruning (38.46%) + KD    | 3,020,831 | 98.19                   | [Experiment 12, Run_9](../experiments/results/Experiment_12/Run_9)     | Compress model ResNet10 - No KD and fine-tune with KD       |
| ResNet10 + KD + Pruning (38.46%) - No KD    | 3,020,831 | 98.20                   | [Experiment 12, Run_5](../experiments/results/Experiment_12/Run_5)     | Compress model ResNet10 - KD and fine-tune without KD       |
| ResNet10 + KD + Pruning (38.46%) + KD       | 3,020,831 | 98.31                   | [Experiment 11a, Run_2](../experiments/results/Experiment_11a/Run_2)   | Compress model ResNet10 - KD and fine-tune with KD          |
| ResNet10 - No KD + Pruning (65.75%) - No KD | 1'681'300 | 97.83                   | [Experiment 12, Run_2](../experiments/results/Experiment_12/Run_2)     | Compress model ResNet10 - Baseline and fine-tune without KD |
| ResNet10 - No KD + Pruning (65.75%) + KD    | 1'681'300 | 98.15                   | [Experiment 12, Run_10](../experiments/results/Experiment_12/Run_10)   | Compress model ResNet10 - No KD and fine-tune with KD       |
| ResNet10 + KD + Pruning (65.75%) - No KD    | 1'681'300 | 97.89                   | [Experiment 12, Run_6](../experiments/results/Experiment_12/Run_6)     | Compress model ResNet10 - KD and fine-tune without KD       |
| ResNet10 + KD + Pruning (65.75%) + KD       | 1,810,324 | 98.15                   | [Experiment 11a, Run_20](../experiments/results/Experiment_11a/Run_20) | Compress model ResNet10 - KD and fine-tune with KD          |

### ResNet18 on EuroSAT
| Model                                       | Parameter | Validation Accuracy (%) | Run ID                                                               | Notes                                                       |
|---------------------------------------------|-----------|-------------------------|----------------------------------------------------------------------|-------------------------------------------------------------|
| ResNet18 - Our Baseline                     | 4,921,197 | 98.27                   | [Experiment 1, Run_1](../experiments/results/Experiment_1/Run_1)     | Train from scratch without a teacher for 50 epochs          |
| ResNet18 + KD                               | 4,921,197 | 98.74                   | [Experiment 4, Run_3](../experiments/results/Experiment_4/Run_3)     | Train from scratch with a teacher for 50 epochs             |
| ResNet18 - No KD + Pruning (47.48%) - No KD | 5,871,078 | 98.06                   | [Experiment 12, Run_3](../experiments/results/Experiment_12/Run_3)   | Compress model ResNet18 - Baseline and fine-tune without KD |
| ResNet18 - No KD + Pruning 47.48%) + KD     | 5,871,078 | 98.30                   | [Experiment 12, Run_11](../experiments/results/Experiment_12/Run_11) | Compress model ResNet18 - No KD and fine-tune with KD       |
| ResNet18 + KD + Pruning (47.48%) - No KD    | 5,871,078 | 98.35                   | [Experiment 12, Run_7](../experiments/results/Experiment_12/Run_7)   | Compress model ResNet18 - KD and fine-tune without KD       |
| ResNet18 + KD + Pruning (46.03%) + KD       | 6,032,870 | 98.56                   | [Experiment 11, Run_2](../experiments/results/Experiment_11/Run_2)   | Compress model ResNet18 - KD and fine-tune with KD          |
| ResNet18 - No KD + Pruning (75.82%) - No KD | 2,703,657 | 98.07                   | [Experiment 12, Run_4](../experiments/results/Experiment_12/Run_4)   | Compress model ResNet18 - Baseline and fine-tune without KD |
| ResNet18 - No KD + Pruning (75.82%) + KD    | 2,703,657 | 98.04                   | [Experiment 12, Run_12](../experiments/results/Experiment_12/Run_12) | Compress model ResNet18 - No KD and fine-tune with KD       |
| ResNet18 + KD + Pruning (75.82%) - No KD    | 2,703,657 | 97.80                   | [Experiment 12, Run_8](../experiments/results/Experiment_12/Run_8)   | Compress model ResNet18 - KD and fine-tune without KD       |
| ResNet18 + KD + Pruning (75.0%) + KD        | 2,794,281 | 98.33                   | [Experiment 11, Run_20](../experiments/results/Experiment_11/Run_20) | Compress model ResNet18 - KD and fine-tune with KD          |

KD-Aware compression consistently outperforms standard pruning methods, achieving higher accuracy with significant parameter
reductions.