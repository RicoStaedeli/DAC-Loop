# Experiment Descriptions

Experiments are organized in the following manner:

```
experiments/
├── configs/            # Configuration files for the experiments
└── results/            # Results of the experiments
```

An experiment folder contains an `Experiment_Orchestrator_Config.toml` file and multiple `Run-xy.toml` files. The
orchestrator config defines which runs to execute. The parameters of the specific run are in the run config files.

An example experiment can be found in [Experiment_XY](configurations/Experiment_XY). The name of the experiment folder is
passed as an argument to the `runner.py` script.
To execute an experiment, use the following command:

```bash
uv run runner.py --gpu 0 --experiment Experiment_XY
```

---



## Experiment 1: Baselines Teacher & Students on EuroSAT

**Goal:** Train baseline models on EuroSAT dataset with different band combinations to establish performance benchmarks
for subsequent experiments.
ResNet50 and ResNet18 are used as teacher models, while ResNet10 and TinyCNN serve as student models.

**Description**

- Models: ResNet50 ResNet18 (teacher), ResNet10, TinyCNN (students)
- Dataset: EuroSAT (`blanchon/EuroSAT_MSI`)
- Baseline were generated for different Band combinations:
    1. All bands: ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]
    2. RGB bands: ["B02", "B03", "B04"]
    3. Vegetation indices + RGB bands: ["NDVI", "NDWI", "NDMI"]["B02", "B03", "B04", "B07", "B08"]


**Outcomes**

- Baseline results for all models and band combinations
- Even the smallest model (TinyCNN) achieves accuracy ~90.00% when trained with all bands
- The best band comination is using all bands, followed by vegetation indices + RGB bands, and finally RGB bands only.

**Next Steps**

- Try to use this big teacher model (ResNet50 and ResNet18) to guide the smaller models with teacher-student learning (
  Experiment 2)

### Runs


## Experiment 2: Teacher-Student learning (replaced by Experiment 3 and 4)

**Goal:** Train the smaller models (ResNet10 and TiniCNN) with teacher-student training. Use the best teacher model (
best validation loss) ResNet18 as a guiding teacher.
This experiment was later replaced by Experiment 3 and 4 which include hyperparameter tuning.

**Outcomes**

- Teacher-Student learning works in theory but did not improve the baseline results in practice

**Next Steps**

- Hyperparameter tuning for the student model training (Experiment 3 and 4)
- Try bigger models as teachers --> Do Baseline for ResNet50 (done in Experiment 1, Runs 10-12)
- Try ResNet18 as student and ResNet50 as teacher (Experiment 4)

### Runs


## Experiment 3: Hyperparameter-tuning - EuroSAT, TinyCNN, ResNet18

**Goal:** Train the smaller models (TinyCNN) with teacher-student training. Use the best teacher model (best validation
loss) ResNet18 as a guiding teacher.
In this experiment we will do hyperparameter tuning for the student model training. Also we use different knowledge
distillation methods such as CosineEmbeddingLoss, CrossEntropyLoss and KLDivergenceLoss.

[Results](results/Experiment_3/experiment_summary.md)

**Outcomes**

- No Hyperparameter combination was able to outperform the baseline TinyCNN model trained from scratch

**Next Steps**

- Try bigger models as students and Teacher (Experiment 4)

### Runs


## Experiment 4: Hyperparameter-tuning - EuroSAT, ResNet18, ResNet50

**Goal:** Train the smaller model (ResNet18) with teacher-student training. Use the best teacher model ( the best
validation accuracy) ResNet50 as a guiding teacher.
In this experiment we will do hyperparameter tuning for the student model training. Also, we use different knowledge
distillation methods such as CosineEmbeddingLoss, CrossEntropyLoss and KLDivergenceLoss.

[Results](results/Experiment_4/experiment_summary.md)

**Outcomes**

- The baseline of the Student model (ResNet18) could be beaten with Teacher-Student learning using ResNet50 as teacher.
- The student model achieved better performance than the teacher model. --> Dataset could be to simple and therefore we
  change it to a more complex one in future experiments.
- Build Baseline for models on `RESISC45` Dataset (Experiment 5)

### Runs


## Experiment 4a: Hyperparameter-tuning - EuroSAT, ResNet10, ResNet50

**Goal:** Train the smaller model (ResNet10) with teacher-student training. Use the best teacher model ( the best
validation accuracy) ResNet50 as a guiding teacher.
In this experiment we will do hyperparameter tuning for the student model training. Also, we use different knowledge
distillation methods such as CosineEmbeddingLoss, CrossEntropyLoss and KLDivergenceLoss.

[Results](results/Experiment_4a/experiment_summary.md)

**Outcomes**

- The baseline of the Student model (ResNet10) could be beaten with Teacher-Student learning using ResNet50 as teacher.
- The student model achieved better performance than the teacher model. --> Dataset could be to simple and therefore we
  change it to a more complex one in future experiments.
- Build Baseline for models on `RESISC45` Dataset (Experiment 5)

### Runs

|
## Experiment 5: Baseline Teacher & Student on RESISC45

**Goal:** In this experiment we will establish a baseline using the `RESISC45` dataset. We will train different models (
ResNet18, ResNet18, ResNet50, TinyCNN) on the RESISC45 dataset and evaluate their performance.

[Results](results/Experiment_5/experiment_summary.md)

### Runs


## Experiment 6: Hyperparameter-tuning - RESISC45, ResNet18, ResNet50

**Goal:** Train the smaller model (ResNet18) with teacher-student training. Use the best teacher model ( the best
validation accuracy) ResNet50 as a guiding teacher.
In this experiment we will do hyperparameter tuning for the student model training. Also we use different knowledge
distillation methods such as CosineEmbeddingLoss, CrossEntropyLoss and KLDivergenceLoss.
Dataset is the `RESISC45` dataset.

[Results](results/Experiment_6/experiment_summary.md)

### Runs


## Experiment 6a: Hyperparameter-tuning - RESISC45, ResNet10, ResNet50

**Goal:** Train the smaller model (ResNet10) with teacher-student training. Use the best teacher model ( the best
validation accuracy) ResNet50 as a guiding teacher.
In this experiment we will do hyperparameter tuning for the student model training. Also we use different knowledge
distillation methods such as CosineEmbeddingLoss, CrossEntropyLoss and KLDivergenceLoss.
Dataset is the `RESISC45` dataset.

[Results](results/Experiment_6a/experiment_summary.md)


### Runs


## Experiment 6b: Hyperparameter-tuning - RESISC45, ResNet10small, ResNet50

**Goal:** Train the smaller model (ResNet10small) with teacher-student training. Use the best teacher model ( the best
validation accuracy) ResNet50 as a guiding teacher.
In this experiment we will do hyperparameter tuning for the student model training. Also we use different knowledge
distillation methods such as CosineEmbeddingLoss, CrossEntropyLoss and KLDivergenceLoss.
Dataset is the `RESISC45` dataset.

[Results](results/Experiment_6b/experiment_summary.md)

### Runs


## Experiment 7: Compression - RESISC45, ResNet18, ResNet50

**Goal:** In this Experiment we will compress the ResNet18 model using pruning techniques. The compressed models will be
fine-tuned with knowledge distillation using the ResNet50 as teacher model. This experiment was done using the best
hyperparameters found in Experiment 5 and 6. Only Resic45 dataset is used.
All experiment use a ResNet18 student model trained with a ResNet50 teacher model. The model resulting form the
following was used: [Experiment 6, Run 23](results/Experiment_6/Run_23/config.toml)

[Results](results/Experiment_7/experiment_summary.md)

### Runs


## Experiment 7a: Compression - RESISC45, ResNet18, ResNet50

**Goal:** In this Experiment we will compress the ResNet18 model using pruning techniques. The compressed models will be
fine-tuned with knowledge distillation using the ResNet50 as teacher model. This experiment was done using the best
hyperparameters found in Experiment 5 and 6. Only Resisc45 dataset is used.
All experiment use a ResNet10 student model trained with a ResNet50 teacher model. The model resulting form the
following was used: [Experiment 6a, Run 40](results/Experiment_6a/Run_40/config.toml)

[Results](results/Experiment_7a/experiment_summary.md)

## Experiment 8: Ablation Experiments - EuroSAT, ResNet10, ResNet18

**Goal:** The goal of this experiment is to perform ablation studies to better understand the impact of the proposed
distillation aware pruning loop.
All ablation studies were done on the EuroSAT dataset using ResNet10 and ResNet18 as student models and ResNet50 as a
teacher.

Teacher model ResNet50: [Experiment 1, Run 10](results/Experiment_1/Run_10/config.toml)

| Run  | Description                                                                                                                                                                                                                                                                                                                                                                                            |
|------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [1]  | Train without any teacher KD in the complete loop to get a baseline - ResNet10<br>- EuroSAT<br>**- Stage 1:** No teacher --> [Experiment 1, Run 4](results/Experiment_1/Run_4/config.toml)<br>**- Stage 2:** Compression with hyperparameter from [Experiment 7, Run 2](results/Experiment_7/Run_2/config.toml)<br>**- Stage 3:** No Teacher                                                           |
| [2]  | Train without any teacher KD in the complete loop to get a baseline - ResNet10<br>- EuroSAT<br>**- Stage 1:** No teacher --> [Experiment 1, Run 4](results/Experiment_1/Run_4/config.toml)<br>**- Stage 2:** Compression with hyperparameter from [Experiment 7, Run 11](results/Experiment_7/Run_11/config.toml)<br>**- Stage 3:** No Teacher                                                         |
| [3]  | Train without any teacher KD in the complete loop to get a baseline - ResNet18<br>- EuroSAT<br>**- Stage 1:** No teacher --> [Experiment 1, Run 1](results/Experiment_1/Run_1/config.toml)<br>**- Stage 2:** Compression with hyperparameter from [Experiment 6, Run 2](results/Experiment_6/Run_2/config.toml)<br>**- Stage 3:** No Teacher                                                           |
| [4]  | Train without any teacher KD in the complete loop to get a baseline - ResNet18<br>- EuroSAT<br>**- Stage 1:** No teacher --> [Experiment 1, Run 1](results/Experiment_1/Run_1/config.toml)<br>**- Stage 2:** Compression with hyperparameter from [Experiment 6, Run 11](results/Experiment_6/Run_11/config.toml)<br>**- Stage 3:** No Teacher                                                         |
| [5]  | Train with teacher KD only in the first stage to see if kd in second stage is necessary - ResNet10<br>- EuroSAT<br>**- Stage 1:** Train with teacher --> [Experiment 4, Run 11](results/Experiment_4/Run_11/config.toml)<br>**- Stage 2:** Compression with hyperparameter from [Experiment 7, Run 2](results/Experiment_7/Run_2/config.toml)<br>**- Stage 3:** No Teacher                             |
| [6]  | Train with teacher KD only in the first stage to see if kd in second stage is necessary - ResNet10<br>- EuroSAT<br>**- Stage 1:** Train with teacher --> [Experiment 4, Run 11](results/Experiment_4/Run_11/config.toml)<br>**- Stage 2:** Compression with hyperparameter from [Experiment 7, Run 11](results/Experiment_7/Run_11/config.toml)<br>**- Stage 3:** No Teacher                           |
| [7]  | Train with teacher KD only in the first stage to see if kd in second stage is necessary - ResNet18<br>- EuroSAT<br>**- Stage 1:** Train with teacher --> [Experiment 3, Run 3](results/Experiment_3/Run_3/config.toml)<br>**- Stage 2:** Compression with hyperparameter from [Experiment 6, Run 2](results/Experiment_6/Run_2/config.toml)<br>**- Stage 3:** No Teacher                               |
| [8]  | Train with teacher KD only in the first stage to see if kd in second stage is necessary - ResNet18<br>- EuroSAT<br>**- Stage 1:** Train with teacher --> [Experiment 3, Run 3](results/Experiment_3/Run_3/config.toml)<br>**- Stage 2:** Compression with hyperparameter from [Experiment 6, Run 11](results/Experiment_6/Run_11/config.toml)<br>**- Stage 3:** No Teacher                             |
| [9]  | Train with teacher KD only in the second stage to see if kd in first stage is necessary - ResNet10<br>- EuroSAT<br>**- Stage 1:** No teacher --> [Experiment 1, Run 4](results/Experiment_1/Run_4/config.toml)<br>**- Stage 2:** Compression with hyperparameter from [Experiment 7, Run 2](results/Experiment_7/Run_2/config.toml)<br>**- Stage 3:** Train with teacher --> use model after pruning   |
| [10] | Train with teacher KD only in the second stage to see if kd in first stage is necessary - ResNet10<br>- EuroSAT<br>**- Stage 1:** No teacher --> [Experiment 1, Run 4](results/Experiment_1/Run_4/config.toml)<br>**- Stage 2:** Compression with hyperparameter from [Experiment 7, Run 11](results/Experiment_7/Run_11/config.toml)<br>**- Stage 3:** Train with teacher --> use model after pruning |
| [11] | Train with teacher KD only in the second stage to see if kd in first stage is necessary - ResNet18<br>- EuroSAT<br>**- Stage 1:** No teacher --> [Experiment 1, Run 1](results/Experiment_1/Run_1/config.toml)<br>**- Stage 2:** Compression with hyperparameter from [Experiment 6, Run 2](results/Experiment_6/Run_2/config.toml)<br>**- Stage 3:** Train with teacher --> use model after pruning   |
| [12] | Train with teacher KD only in the second stage to see if kd in first stage is necessary - ResNet18<br>- EuroSAT<br>**- Stage 1:** No teacher --> [Experiment 1, Run 1](results/Experiment_1/Run_1/config.toml)<br>**- Stage 2:** Compression with hyperparameter from [Experiment 6, Run 11](results/Experiment_6/Run_11/config.toml)<br>**- Stage 3:** Train with teacher --> use model after pruning |


### ResNet10 on EuroSAT
| Model                                       | Parameter | Validation Accuracy (%) | Run ID                                              | Notes                                                       |
|---------------------------------------------|-----------|-------------------------|-----------------------------------------------------|-------------------------------------------------------------|
| ResNet10 - Our Baseline                     | 4,921,197 | 98.40                   | [Experiment 1, Run_4](results/Experiment_1/Run_4)   | Train from scratch without a teacher for 50 epochs          |
| ResNet10 + KD                               | 4,921,197 | 98.57                   | [Experiment 4, Run_11](results/Experiment_4/Run_11) | Train from scratch with a teacher for 50 epochs             |
| ResNet10 - No KD + Pruning (38.46%) - No KD | 3,020,831 | 98.04                   | [Experiment 8, Run_1](results/Experiment_8/Run_1)   | Compress model ResNet10 - Baseline and fine-tune without KD |
| ResNet10 - No KD + Pruning (38.46%) + KD    | 3,020,831 | 98.19                   | [Experiment 8, Run_9](results/Experiment_8/Run_9)   | Compress model ResNet10 - No KD and fine-tune with KD       |
| ResNet10 + KD + Pruning (38.46%) - No KD    | 3,020,831 | 98.20                   | [Experiment 8, Run_5](results/Experiment_8/Run_5)   | Compress model ResNet10 - KD and fine-tune without KD       |
| ResNet10 + KD + Pruning (38.46%) + KD       | 3,020,831 | 98.31                   | [Experiment 7, Run_2](results/Experiment_7/Run_2)   | Compress model ResNet10 - KD and fine-tune with KD          |
| ResNet10 - No KD + Pruning (65.75%) - No KD | 1'681'300 | 97.83                   | [Experiment 8, Run_2](results/Experiment_8/Run_2)   | Compress model ResNet10 - Baseline and fine-tune without KD |
| ResNet10 - No KD + Pruning (65.75%) + KD    | 1'681'300 | 98.15                   | [Experiment 8, Run_10](results/Experiment_8/Run_10) | Compress model ResNet10 - No KD and fine-tune with KD       |
| ResNet10 + KD + Pruning (65.75%) - No KD    | 1'681'300 | 97.89                   | [Experiment 8, Run_6](results/Experiment_8/Run_6)   | Compress model ResNet10 - KD and fine-tune without KD       |
| ResNet10 + KD + Pruning (63.12%) + KD       | 1,810,324 | 98.17                   | [Experiment 7, Run_11](results/Experiment_7/Run_11) | Compress model ResNet10 - KD and fine-tune with KD          |

### ResNet18 on EuroSAT
| Model                                       | Parameter | Validation Accuracy (%) | Run ID                                              | Notes                                                       |
|---------------------------------------------|-----------|-------------------------|-----------------------------------------------------|-------------------------------------------------------------|
| ResNet18 - Our Baseline                     | 4,921,197 | 98.27                   | [Experiment 1, Run_1](results/Experiment_1/Run_1)   | Train from scratch without a teacher for 50 epochs          |
| ResNet18 + KD                               | 4,921,197 | 98.74                   | [Experiment 3, Run_3](results/Experiment_3/Run_3)   | Train from scratch with a teacher for 50 epochs             |
| ResNet18 - No KD + Pruning (47.48%) - No KD | 5,871,078 | 98.06                   | [Experiment 8, Run_3](results/Experiment_8/Run_3)   | Compress model ResNet18 - Baseline and fine-tune without KD |
| ResNet18 - No KD + Pruning 47.48%) + KD     | 5,871,078 | 98.30                   | [Experiment 8, Run_11](results/Experiment_8/Run_11) | Compress model ResNet18 - No KD and fine-tune with KD       |
| ResNet18 + KD + Pruning (47.48%) - No KD    | 5,871,078 | 98.35                   | [Experiment 8, Run_7](results/Experiment_8/Run_7)   | Compress model ResNet18 - KD and fine-tune without KD       |
| ResNet18 + KD + Pruning (46.03%) + KD       | 6,032,870 | 98.56                   | [Experiment 6, Run_2](results/Experiment_6/Run_2)   | Compress model ResNet18 - KD and fine-tune with KD          |
| ResNet18 - No KD + Pruning (75.82%) - No KD | 2,703,657 | 98.07                   | [Experiment 8, Run_4](results/Experiment_8/Run_4)   | Compress model ResNet18 - Baseline and fine-tune without KD |
| ResNet18 - No KD + Pruning (75.82%) + KD    | 2,703,657 | 98.04                   | [Experiment 8, Run_12](results/Experiment_8/Run_12) | Compress model ResNet18 - No KD and fine-tune with KD       |
| ResNet18 + KD + Pruning (75.82%) - No KD    | 2,703,657 | 97.80                   | [Experiment 8, Run_8](results/Experiment_8/Run_8)   | Compress model ResNet18 - KD and fine-tune without KD       |
| ResNet18 + KD + Pruning (75.0%) + KD        | 2,794,281 | 98.67                   | [Experiment 6, Run_11](results/Experiment_6/Run_11) | Compress model ResNet18 - KD and fine-tune with KD          |


## Experiment 8a: Ablation Experiments - RESISC45, ResNet10, ResNet18

**Goal:** The goal of this experiment is to perform ablation studies to better understand the impact of the proposed
distillation aware pruning loop.
All ablation studies were done on the Resisc45 dataset using ResNet10 and ResNet18 as student models and ResNet50 as a
teacher.

| Run  | Description                                                                                                                                                                                                                                                                                                                                                                       |
|------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [1]  | Train without any teacher KD in the complete loop to get a baseline - ResNet10<br>- Resisc45<br>**- Stage 1:** No teacher<br>**- Stage 2:** Compression with hyperparameter from [Experiment 7a, Run 10](results/Experiment_7a/Run_10/config.toml)<br>**- Stage 3:** No Teacher                                                                                                   |
| [2]  | Train without any teacher KD in the complete loop to get a baseline - ResNet10<br>- Resisc45<br>**- Stage 1:** No teacher<br>**- Stage 2:** Compression with hyperparameter from [Experiment 7a, Run 11](results/Experiment_7a/Run_11/config.toml)<br>**- Stage 3:** No Teacher                                                                                                   |
| [3]  | Train without any teacher KD in the complete loop to get a baseline - ResNet18<br>- Resisc45<br>**- Stage 1:** No teacher<br>**- Stage 2:** Compression with hyperparameter from [Experiment 7, Run 10](results/Experiment_7/Run_10/config.toml)<br>**- Stage 3:** No Teacher                                                                                                     |
| [4]  | Train without any teacher KD in the complete loop to get a baseline - ResNet18<br>- Resisc45<br>**- Stage 1:** No teacher<br>**- Stage 2:** Compression with hyperparameter from [Experiment 7, Run 11](results/Experiment_7/Run_11/config.toml)<br>**- Stage 3:** No Teacher                                                                                                     |
| [5]  | Train with teacher KD only in the first stage to see if kd in second stage is necessary - ResNet10<br>- Resisc45<br>**- Stage 1:** Train with teacher --> [Experiment 6a, Run 40](results/Experiment_6a/Run_40/config.toml)<br>**- Stage 2:** Compression with hyperparameter from [Experiment 7a, Run 10](results/Experiment_7a/Run_10/config.toml)<br>**- Stage 3:** No Teacher |
| [6]  | Train with teacher KD only in the first stage to see if kd in second stage is necessary - ResNet10<br>- Resisc45<br>**- Stage 1:** Train with teacher --> [Experiment 6a, Run 40](results/Experiment_6a/Run_40/config.toml)<br>**- Stage 2:** Compression with hyperparameter from [Experiment 7a, Run 11](results/Experiment_7a/Run_11/config.toml)<br>**- Stage 3:** No Teacher |
| [7]  | Train with teacher KD only in the first stage to see if kd in second stage is necessary - ResNet18<br>- Resisc45<br>**- Stage 1:** Train with teacher --> [Experiment 6, Run 23](results/Experiment_6/Run_23/config.toml)<br>**- Stage 2:** Compression with hyperparameter from [Experiment 7, Run 10](results/Experiment_7/Run_10/config.toml)<br>**- Stage 3:** No Teacher     |
| [8]  | Train with teacher KD only in the first stage to see if kd in second stage is necessary - ResNet18<br>- Resisc45<br>**- Stage 1:** Train with teacher --> [Experiment 6, Run 23](results/Experiment_6/Run_23/config.toml)<br>**- Stage 2:** Compression with hyperparameter from [Experiment 7, Run 11](results/Experiment_7/Run_11/config.toml)<br>**- Stage 3:** No Teacher     |
| [9]  | Train with teacher KD only in the second stage to see if kd in first stage is necessary - ResNet10<br>- Resisc45<br>**- Stage 1:** No teacher<br>**- Stage 2:** Compression with hyperparameter from [Experiment 7a, Run 10](results/Experiment_7a/Run_10/config.toml)<br>**- Stage 3:** Train with teacher --> use model after pruning                                           |
| [10] | Train with teacher KD only in the second stage to see if kd in first stage is necessary - ResNet10<br>- Resisc45<br>**- Stage 1:** No teacher<br>**- Stage 2:** Compression with hyperparameter from [Experiment 7a, Run 11](results/Experiment_7a/Run_11/config.toml)<br>**- Stage 3:** Train with teacher --> use model after pruning                                           |
| [11] | Train with teacher KD only in the second stage to see if kd in first stage is necessary - ResNet18<br>- Resisc45<br>**- Stage 1:** No teacher<br>**- Stage 2:** Compression with hyperparameter from [Experiment 7, Run 10](results/Experiment_7/Run_10/config.toml)<br>**- Stage 3:** Train with teacher --> use model after pruning                                             |
| [12] | Train with teacher KD only in the second stage to see if kd in first stage is necessary - ResNet18<br>- Resisc45<br>**- Stage 1:** No teacher<br>**- Stage 2:** Compression with hyperparameter from [Experiment 7, Run 11](results/Experiment_7/Run_11/config.toml)<br>**- Stage 3:** Train with teacher --> use model after pruning                                             |

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

[Results](results/Experiment_8/experiment_summary.md)



## Experiment 9: Loop Experiments - RESISC45, ResNet18

**Goal:** The goal of this experiment is to perform knowledge aware compression in multiple loops to see if further improvements can be achieved.
All experiments were done on the Resisc45 dataset using ResNet18 as student model and ResNet50 as a teacher.
Startingpoint for all experiments is the model from [Experiment 6, Run 23](results/Experiment_6/Run_23/config.toml).
Teacher: [Experiment 5, Run 7](results/Experiment_5/Run_7/config.toml)

[Results](results/Experiment_9/experiment_summary.md)

## Experiment 9a: Loop Experiments - RESISC45, ResNet10

**Goal:** The goal of this experiment is to perform knowledge aware compression in multiple loops to see if further improvements can be achieved.
All experiments were done on the Resisc45 dataset using ResNet10 as student model and ResNet50 as a teacher.
Startingpoint for all experiments is the model from [Experiment 6a, Run 40](results/Experiment_6a/Run_40/config.toml).
Teacher: [Experiment 5, Run 7](results/Experiment_5/Run_7/config.toml)

[Results](results/Experiment_9a/experiment_summary.md)


## Experiment 10: Seed Experiments for Experiment 1 Baselines EuroSAT

**Goal:** The goal of this experiment is to evaluate the stability of the baseline experiments from Experiment 1 on EuroSAT with different random seeds.

[Results](results/Experiment_10/experiment_summary.md)

### Runs


## Experiment 10a: Seed Experiments for Experiment 5 Baselines Resisc45

**Goal:** The goal of this experiment is to evaluate the stability of the baseline experiments from Experiment 5 on Resisc45 with different random seeds.

[Results](results/Experiment_10a/experiment_summary.md)

### Runs


## Experiment 10b: Seed Runs for Resisc45 and Resnet18

**Goal:** The goal of this experiment is to evaluate the stability of the baseline experiments from Experiment 8 on Resisc45 with different random seeds.
ResNet18.
 Seed ResNet18 RESISC:
- 1-3 KD --> [Exp 6, Run 23](results/Experiment_6/Run_23)
- 4-6 ResNet18 - No KD + Pruning (38.43%) - No KD --> [Exp 8, Run 3](results/Experiment_8/Run_3)
- 7-9 ResNet18 - No KD + Pruning (38.43%) + KD --> [Exp 8, Run 11](results/Experiment_8/Run_11)
- 10-12 ResNet18 + KD + Pruning (38.43%) - No KD --> [Exp 8, Run 7](results/Experiment_8/Run_7)
- 13-15 ResNet18 + KD + Pruning (38.43%) + KD --> [Exp 7, Run 10](results/Experiment_7/Run_10)
- 16-18 ResNet18 - No KD + Pruning (65.71%) - No KD --> [Exp 8, Run 4](results/Experiment_8/Run_4)
- 19-21 ResNet18 - No KD + Pruning (65.71%) + KD --> [Exp 8, Run 12](results/Experiment_8/Run_12)
- 22-24 ResNet18 + KD + Pruning (65.71%) - No KD --> [Exp 8, Run 8](results/Experiment_8/Run_8)
- 25-27 ResNet18 + KD + Pruning (65.71%) + KD --> [Exp 7, Run 11](results/Experiment_7/Run_11)
- 28-30 KD loop 6 iters --> [Exp 9, Run 5](results/Experiment_9/Run_5)
- 31-33 KD loop 9 iters --> [Exp 9, Run 7](results/Experiment_9/Run_7)
- 34-35 unstructured pruning -> [Exp 7, Run 19](results/Experiment_7/Run_19)

### Results

[Results](results/Experiment_10b/experiment_summary.md)



## Experiment 10c: Seed Experiments for Experiment 8 Ablation Study Resisc45 and Resnet10

**Goal:** The goal of this experiment is to evaluate the stability of the baseline experiments from Experiment 8 on Resisc45 with different random seeds.
ResNet10.
[Results](results/Experiment_10c/experiment_summary.md)

### Runs


## Experiment 11: Compression - EuroSAT, ResNet18, ResNet50

**Goal:** In this Experiment we will compress the ResNet18 model using pruning techniques. The compressed models will be
fine-tuned with knowledge distillation using the ResNet50 as teacher model. This experiment was done using the best
hyperparameters found in Experiment 5 and 6. Only EuroSAT dataset is used.
All experiment use a ResNet18 student model trained with a ResNet50 teacher model. The model resulting form the
following was used: [Experiment 4, Run 3](results/Experiment_4/Run_3/config.toml)
Teacher model ResNet50: [Experiment 1, Run 10](results/Experiment_1/Run_10/config.toml)

[Results](results/Experiment_11/experiment_summary.md)

### Runs


## Experiment 11a: Compression - EuroSAT, ResNet10, ResNet50

**Goal:** In this Experiment we will compress the ResNet18 model using pruning techniques. The compressed models will be
fine-tuned with knowledge distillation using the ResNet50 as teacher model. This experiment was done using the best
hyperparameters found in Experiment 5 and 6. Only EuroSAT dataset is used.
All experiment use a ResNet18 student model trained with a ResNet50 teacher model. The model resulting form the
following was used: [Experiment 4a, Run 11](results/Experiment_4a/Run_11/config.toml)
Teacher model ResNet50: [Experiment 1, Run 10](results/Experiment_1/Run_10/config.toml)

[Results](results/Experiment_11a/experiment_summary.md)

### Runs


### Runs