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
for subsequent experiments. Models used:
- ResNetTiny (1M)
- ResNet10mini (2M)
- ResNet10 (5M)
- ResNet18 (11M)
- ResNet50 (24M) → Used as teacher later

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
- The best band combination is using all bands, followed by vegetation indices + RGB bands, and finally RGB bands only.

**Results**

- [Results](results/Experiment_1/experiment_summary.md)

| Model        | Accuracy |
|--------------|----------|
| ResNetTiny   |          |
| ResNet10mini |          |
| ResNet10     |          |
| ResNet18     |          |
| ResNet50     |          |

---

## Experiment 1a: Baseline Teacher & Student on RESISC45

**Goal:** In this experiment we will establish a baseline using the `RESISC45` dataset. Models used:
- ResNetTiny (1M)
- ResNet10mini (2M)
- ResNet10 (5M)
- ResNet18 (11M)
- ResNet50 (24M) → Used as teacher later

**Outcomes**
- [Exp 1a Run 7](results/Experiment_1a/Run_7) has proven to achieve the best results on RESISC45 Dataset and will be used as Teacher model for all following experiments.

**Results**
- [Results](results/Experiment_1a/experiment_summary.md)

| Model        | Accuracy |
|--------------|----------|
| ResNetTiny   |          |
| ResNet10mini |          |
| ResNet10     |          |
| ResNet18     |          |
| ResNet50     |          |

---

## Experiment 2: KD Hyperparameter-tuning - EuroSAT, ResNetTiny, ResNet50

**Goal:** Train the smaller models (ResNetTiny) with teacher-student training. Use the best teacher model (best validation
loss) of ResNet50 as a guiding teacher.
In this experiment we will do hyperparameter tuning for the student model training. Also, we use different knowledge
distillation methods such as CosineEmbeddingLoss, CrossEntropyLoss and KLDivergenceLoss.


**Outcomes**

- 

**Results**
- [Results](results/Experiment_2/experiment_summary.md)

---

## Experiment 2a: KD Hyperparameter-tuning - RESISC45, ResNetTiny, ResNet50

**Goal:** Train the smaller models (ResNetTiny) with teacher-student training. Use the best teacher model (best validation
loss) of ResNet50 as a guiding teacher.
In this experiment we will do hyperparameter tuning for the student model training. Also, we use different knowledge
distillation methods such as CosineEmbeddingLoss, CrossEntropyLoss and KLDivergenceLoss.


**Outcomes**

- 

**Results**
- [Results](results/Experiment_2a/experiment_summary.md)

---



## Experiment 3: KD Hyperparameter-tuning - EuroSAT, ResNet18, ResNet50

**Goal:** Train the smaller model (ResNet18) with teacher-student training. Use the best teacher model ( the best
validation accuracy) ResNet50 as a guiding teacher.
In this experiment we will do hyperparameter tuning for the student model training. Also, we use different knowledge
distillation methods such as CosineEmbeddingLoss, CrossEntropyLoss and KLDivergenceLoss.


**Outcomes**

- The baseline of the Student model (ResNet18) could be beaten with Teacher-Student learning using ResNet50 as teacher.
- The student model achieved better performance than the teacher model. --> Dataset could be to simple and therefore we
  change it to a more complex one in future experiments.
- Build Baseline for models on `RESISC45` Dataset (Experiment 3a)

**Results**
- [Results](results/Experiment_3/experiment_summary.md)

--

## Experiment 3a: KD Hyperparameter-tuning - RESISC45, ResNet18, ResNet50

**Goal:** Train the smaller model (ResNet18) with teacher-student training. Use the best teacher model ( the best
validation accuracy) ResNet50 as a guiding teacher.
In this experiment we will do hyperparameter tuning for the student model training. Also, we use different knowledge
distillation methods such as CosineEmbeddingLoss, CrossEntropyLoss and KLDivergenceLoss.


**Outcomes**
- df

**Results**
- [Results](results/Experiment_3a/experiment_summary.md)

---

## Experiment 4: KD Hyperparameter-tuning - EuroSAT, ResNet10, ResNet50

**Goal:** Train the smaller model (ResNet18) with teacher-student training. Use the best teacher model ( the best
validation accuracy) ResNet50 as a guiding teacher.
In this experiment we will do hyperparameter tuning for the student model training. Also, we use different knowledge
distillation methods such as CosineEmbeddingLoss, CrossEntropyLoss and KLDivergenceLoss.


**Outcomes**
- df

**Results**
- [Results](results/Experiment_4/experiment_summary.md)

---

## Experiment 4a: KD Hyperparameter-tuning - RESISC45, ResNet10, ResNet50

**Goal:** Train the smaller model (ResNet18) with teacher-student training. Use the best teacher model ( the best
validation accuracy) ResNet50 as a guiding teacher.
In this experiment we will do hyperparameter tuning for the student model training. Also, we use different knowledge
distillation methods such as CosineEmbeddingLoss, CrossEntropyLoss and KLDivergenceLoss.


**Outcomes**
- df

**Results**
- [Results](results/Experiment_4a/experiment_summary.md)

---

## Experiment 5: KD Hyperparameter-tuning - EuroSAT, ResNet10mini, ResNet50

**Goal:** Train the smaller model (ResNet10mini) with teacher-student training. Use the best teacher model ( the best
validation accuracy) ResNet50 as a guiding teacher.
In this experiment we will do hyperparameter tuning for the student model training. Also, we use different knowledge
distillation methods such as CosineEmbeddingLoss, CrossEntropyLoss and KLDivergenceLoss.


**Outcomes**
- df

**Results**
- [Results](results/Experiment_5/experiment_summary.md)

---

## Experiment 5a: KD Hyperparameter-tuning - RESISC45, ResNet10mini, ResNet50

**Goal:** Train the smaller model (ResNet10mini) with teacher-student training. Use the best teacher model ( the best
validation accuracy) ResNet50 as a guiding teacher.
In this experiment we will do hyperparameter tuning for the student model training. Also, we use different knowledge
distillation methods such as CosineEmbeddingLoss, CrossEntropyLoss and KLDivergenceLoss.


**Outcomes**
- df

**Results**
- [Results](results/Experiment_5a/experiment_summary.md)

---

## Experiment 6: Pruning Hyperparameter-Tuning  - EuroSAT, ResNet18, ResNet50

**Goal:** In this Experiment we will compress the ResNet18 model using pruning techniques. The compressed models will be
fine-tuned with knowledge distillation using the ResNet50 as teacher model. This experiment was done using the best KD-hyperparameters 
found in Experiment 3. Only `EuroSAT` dataset is used.
All experiment use a ResNet18 student model trained with a ResNet50 teacher model. The model resulting form the
following was used: [Experiment 3, Run 3](results/Experiment_3/Run_3/config.toml) as starting point.

**Results**
- [Results](results/Experiment_6/experiment_summary.md)

---

## Experiment 6a: Pruning Hyperparameter-Tuning  - RESISC45, ResNet18, ResNet50

**Goal:** In this Experiment we will compress the ResNet18 model using pruning techniques. The compressed models will be
fine-tuned with knowledge distillation using the ResNet50 as teacher model. This experiment was done using the best KD-hyperparameters 
found in Experiment 3a. Only `RESISC45` dataset is used.
All experiment use a ResNet18 student model trained with a ResNet50 teacher model. The model resulting form the
following was used: [Experiment 3a, Run 23](results/Experiment_3a/Run_23/config.toml) as starting point.

**Results**
- [Results](results/Experiment_6a/experiment_summary.md)

---


## Experiment 7: Pruning Hyperparameter-Tuning  - EuroSAT, ResNet10, ResNet50

**Goal:** In this Experiment we will compress the ResNet18 model using pruning techniques. The compressed models will be
fine-tuned with knowledge distillation using the ResNet50 as teacher model. This experiment was done using the best KD-hyperparameters 
found in Experiment 4. Only `EuroSAT` dataset is used.
All experiment use a ResNet18 student model trained with a ResNet50 teacher model. The model resulting form the
following was used: [Experiment 4, Run 11](results/Experiment_11/Run_11/config.toml) as starting point.

**Results**
- [Results](results/Experiment_7/experiment_summary.md)

---

## Experiment 7a: Pruning Hyperparameter-Tuning  - RESISC45, ResNet10, ResNet50

**Goal:** In this Experiment we will compress the ResNet18 model using pruning techniques. The compressed models will be
fine-tuned with knowledge distillation using the ResNet50 as teacher model. This experiment was done using the best KD-hyperparameters 
found in Experiment 4a. Only `RESISC45` dataset is used.
All experiment use a ResNet18 student model trained with a ResNet50 teacher model. The model resulting form the
following was used: [Experiment 4a, Run 35](results/Experiment_3a/Run_35/config.toml) as starting point.

**Results**
- [Results](results/Experiment_7a/experiment_summary.md)

---



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
| [5]  | Train with teacher KD only in the first stage to see if kd in second stage is necessary - ResNet10<br>- EuroSAT<br>**- Stage 1:** Train with teacher --> [Experiment 4, Run 6](results/Experiment_4/Run_6/config.toml)<br>**- Stage 2:** Compression with hyperparameter from [Experiment 7, Run 2](results/Experiment_7/Run_2/config.toml)<br>**- Stage 3:** No Teacher                               |
| [6]  | Train with teacher KD only in the first stage to see if kd in second stage is necessary - ResNet10<br>- EuroSAT<br>**- Stage 1:** Train with teacher --> [Experiment 4, Run 6](results/Experiment_4/Run_6/config.toml)<br>**- Stage 2:** Compression with hyperparameter from [Experiment 7, Run 11](results/Experiment_7/Run_11/config.toml)<br>**- Stage 3:** No Teacher                             |
| [7]  | Train with teacher KD only in the first stage to see if kd in second stage is necessary - ResNet18<br>- EuroSAT<br>**- Stage 1:** Train with teacher --> [Experiment 3, Run 3](results/Experiment_3/Run_3/config.toml)<br>**- Stage 2:** Compression with hyperparameter from [Experiment 6, Run 2](results/Experiment_6/Run_2/config.toml)<br>**- Stage 3:** No Teacher                               |
| [8]  | Train with teacher KD only in the first stage to see if kd in second stage is necessary - ResNet18<br>- EuroSAT<br>**- Stage 1:** Train with teacher --> [Experiment 3, Run 3](results/Experiment_3/Run_3/config.toml)<br>**- Stage 2:** Compression with hyperparameter from [Experiment 6, Run 11](results/Experiment_6/Run_11/config.toml)<br>**- Stage 3:** No Teacher                             |
| [9]  | Train with teacher KD only in the second stage to see if kd in first stage is necessary - ResNet10<br>- EuroSAT<br>**- Stage 1:** No teacher --> [Experiment 1, Run 4](results/Experiment_1/Run_4/config.toml)<br>**- Stage 2:** Compression with hyperparameter from [Experiment 7, Run 2](results/Experiment_7/Run_2/config.toml)<br>**- Stage 3:** Train with teacher --> use model after pruning   |
| [10] | Train with teacher KD only in the second stage to see if kd in first stage is necessary - ResNet10<br>- EuroSAT<br>**- Stage 1:** No teacher --> [Experiment 1, Run 4](results/Experiment_1/Run_4/config.toml)<br>**- Stage 2:** Compression with hyperparameter from [Experiment 7, Run 11](results/Experiment_7/Run_11/config.toml)<br>**- Stage 3:** Train with teacher --> use model after pruning |
| [11] | Train with teacher KD only in the second stage to see if kd in first stage is necessary - ResNet18<br>- EuroSAT<br>**- Stage 1:** No teacher --> [Experiment 1, Run 1](results/Experiment_1/Run_1/config.toml)<br>**- Stage 2:** Compression with hyperparameter from [Experiment 6, Run 2](results/Experiment_6/Run_2/config.toml)<br>**- Stage 3:** Train with teacher --> use model after pruning   |
| [12] | Train with teacher KD only in the second stage to see if kd in first stage is necessary - ResNet18<br>- EuroSAT<br>**- Stage 1:** No teacher --> [Experiment 1, Run 1](results/Experiment_1/Run_1/config.toml)<br>**- Stage 2:** Compression with hyperparameter from [Experiment 6, Run 11](results/Experiment_6/Run_11/config.toml)<br>**- Stage 3:** Train with teacher --> use model after pruning |


### ResNet10 on EuroSAT
| Model                                       | Parameter | Validation Accuracy (%) - Seed42 | Validation Accuracy (%) std | Run ID                                              | Notes                                                       |
|---------------------------------------------|-----------|----------------------------------|-----------------------------|-----------------------------------------------------|-------------------------------------------------------------|
| ResNet10 - Our Baseline                     | 4,921,197 | 98.40                            | 98.40                       | [Experiment 1, Run_4](results/Experiment_1/Run_4)   | Train from scratch without a teacher for 50 epochs          |
| ResNet10 + KD                               | 4,921,197 | 98.51                            | 98.40                       | [Experiment 4, Run_6](results/Experiment_4/Run_6)   | Train from scratch with a teacher for 50 epochs             |
| ResNet10 - No KD + Pruning (38.46%) - No KD | 3,020,831 | 98.04                            | 98.40                       | [Experiment 8, Run_1](results/Experiment_8/Run_1)   | Compress model ResNet10 - Baseline and fine-tune without KD |
| ResNet10 - No KD + Pruning (34.59%) + KD    | 3,210,783 | 98.19                            | 98.40                       | [Experiment 8, Run_9](results/Experiment_8/Run_9)   | Compress model ResNet10 - No KD and fine-tune with KD       |
| ResNet10 + KD + Pruning (38.46%) - No KD    | 3,020,831 | 98.20                            | 98.40                       | [Experiment 8, Run_5](results/Experiment_8/Run_5)   | Compress model ResNet10 - KD and fine-tune without KD       |
| ResNet10 + KD + Pruning (38.46%) + KD       | 3,020,831 | 98.31                            | 98.40                       | [Experiment 7, Run_2](results/Experiment_7/Run_2)   | Compress model ResNet10 - KD and fine-tune with KD          |
| ResNet10 - No KD + Pruning (65.75%) - No KD | 1'681'300 | 97.83                            | 98.40                       | [Experiment 8, Run_2](results/Experiment_8/Run_2)   | Compress model ResNet10 - Baseline and fine-tune without KD |
| ResNet10 - No KD + Pruning (65.75%) + KD    | 1'681'300 | 98.15                            | 98.40                       | [Experiment 8, Run_10](results/Experiment_8/Run_10) | Compress model ResNet10 - No KD and fine-tune with KD       |
| ResNet10 + KD + Pruning (65.75%) - No KD    | 1'681'300 | 97.89                            | 98.40                       | [Experiment 8, Run_6](results/Experiment_8/Run_6)   | Compress model ResNet10 - KD and fine-tune without KD       |
| ResNet10 + KD + Pruning (63.12%) + KD       | 1,810,324 | 98.17                            | 98.40                       | [Experiment 7, Run_11](results/Experiment_7/Run_11) | Compress model ResNet10 - KD and fine-tune with KD          |

### ResNet18 on EuroSAT
| Model                                       | Parameter | Validation Accuracy (%) - Seed42 | Validation Accuracy (%) std | Run ID                                              | Notes                                                       |
|---------------------------------------------|-----------|----------------------------------|-----------------------------|-----------------------------------------------------|-------------------------------------------------------------|
| ResNet18 - Our Baseline                     | 4,921,197 | 98.27                            | 98.40                       | [Experiment 1, Run_1](results/Experiment_1/Run_1)   | Train from scratch without a teacher for 50 epochs          |
| ResNet18 + KD                               | 4,921,197 | 98.74                            | 98.40                       | [Experiment 3, Run_3](results/Experiment_3/Run_3)   | Train from scratch with a teacher for 50 epochs             |
| ResNet18 - No KD + Pruning (46.03%) - No KD | 6,032,870 | 98.28                            | 98.40                       | [Experiment 8, Run_3](results/Experiment_8/Run_3)   | Compress model ResNet18 - Baseline and fine-tune without KD |
| ResNet18 - No KD + Pruning (46.03%) + KD    | 6,032,870 | 98.54                            | 98.40                       | [Experiment 8, Run_11](results/Experiment_8/Run_11) | Compress model ResNet18 - No KD and fine-tune with KD       |
| ResNet18 + KD + Pruning (46.03%) - No KD    | 6,032,870 | 98.37                            | 98.40                       | [Experiment 8, Run_7](results/Experiment_8/Run_7)   | Compress model ResNet18 - KD and fine-tune without KD       |
| ResNet18 + KD + Pruning (46.03%) + KD       | 6,032,870 | 98.56                            | 98.40                       | [Experiment 6, Run_2](results/Experiment_6/Run_2)   | Compress model ResNet18 - KD and fine-tune with KD          |
| ResNet18 - No KD + Pruning (75.00%) - No KD | 2,703,657 | 97.69                            | 98.40                       | [Experiment 8, Run_4](results/Experiment_8/Run_4)   | Compress model ResNet18 - Baseline and fine-tune without KD |
| ResNet18 - No KD + Pruning (75.00%) + KD    | 2,794,281 | 98.00                            | 98.40                       | [Experiment 8, Run_12](results/Experiment_8/Run_12) | Compress model ResNet18 - No KD and fine-tune with KD       |
| ResNet18 + KD + Pruning (75.00%) - No KD    | 2,794,281 | 98.19                            | 98.40                       | [Experiment 8, Run_8](results/Experiment_8/Run_8)   | Compress model ResNet18 - KD and fine-tune without KD       |
| ResNet18 + KD + Pruning (75.00%) + KD       | 2,794,281 | 98.67                            | 98.40                       | [Experiment 6, Run_11](results/Experiment_6/Run_11) | Compress model ResNet18 - KD and fine-tune with KD          |

---

## Experiment 8a: Ablation Experiments - RESISC45, ResNet10, ResNet18

**Goal:** The goal of this experiment is to perform ablation studies to better understand the impact of the proposed
distillation aware pruning loop.
All ablation studies were done on the Resisc45 dataset using ResNet10 and ResNet18 as student models and ResNet50 as a
teacher.


| Run  | Description                                                                                                                                                                                                                                                                                                                                                                     |
|------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [1]  | Train without any teacher KD in the complete loop to get a baseline - ResNet10<br>- Resisc45<br>**- Stage 1:** No teacher --> [Experiment 1a, Run 2](results/Experiment_1a/Run_2/config.toml)<br>**- Stage 2:** Compression with hyperparameter from [Experiment 7a, Run 5](results/Experiment_7a/Run_5/config.toml)<br>**- Stage 3:** No Teacher                               |
| [2]  | Train without any teacher KD in the complete loop to get a baseline - ResNet10<br>- Resisc45<br>**- Stage 1:** No teacher --> [Experiment 1a, Run 2](results/Experiment_1a/Run_2/config.toml)<br>**- Stage 2:** Compression with hyperparameter from [Experiment 7a, Run 6](results/Experiment_7a/Run_6/config.toml)<br>**- Stage 3:** No Teacher                               |
| [3]  | Train without any teacher KD in the complete loop to get a baseline - ResNet18<br>- Resisc45<br>**- Stage 1:** No teacher --> [Experiment 1a, Run 1](results/Experiment_1a/Run_1/config.toml)<br>**- Stage 2:** Compression with hyperparameter from [Experiment 6a, Run 1](results/Experiment_6a/Run_1/config.toml)<br>**- Stage 3:** No Teacher                               |
| [4]  | Train without any teacher KD in the complete loop to get a baseline - ResNet18<br>- Resisc45<br>**- Stage 1:** No teacher --> [Experiment 1a, Run 1](results/Experiment_1a/Run_1/config.toml)<br>**- Stage 2:** Compression with hyperparameter from [Experiment 6a, Run 3](results/Experiment_6a/Run_3/config.toml)<br>**- Stage 3:** No Teacher                               |
| [5]  | Train with teacher KD only in the first stage to see if kd in second stage is necessary - ResNet10<br>- Resisc45<br>**- Stage 1:** Train with teacher --> [Experiment 4a, Run 35](results/Experiment_4a/Run_35/config.toml)<br>**- Stage 2:** Compression with hyperparameter from [Experiment 7a, Run 5](results/Experiment_7a/Run_5/config.toml)<br>**- Stage 3:** No Teacher |
| [6]  | Train with teacher KD only in the first stage to see if kd in second stage is necessary - ResNet10<br>- Resisc45<br>**- Stage 1:** Train with teacher --> [Experiment 4a, Run 35](results/Experiment_4a/Run_35/config.toml)<br>**- Stage 2:** Compression with hyperparameter from [Experiment 7a, Run 6](results/Experiment_7a/Run_6/config.toml)<br>**- Stage 3:** No Teacher |
| [7]  | Train with teacher KD only in the first stage to see if kd in second stage is necessary - ResNet18<br>- Resisc45<br>**- Stage 1:** Train with teacher --> [Experiment 3a, Run 23](results/Experiment_3a/Run_23/config.toml)<br>**- Stage 2:** Compression with hyperparameter from [Experiment 6a, Run 1](results/Experiment_6a/Run_1/config.toml)<br>**- Stage 3:** No Teacher |
| [8]  | Train with teacher KD only in the first stage to see if kd in second stage is necessary - ResNet18<br>- Resisc45<br>**- Stage 1:** Train with teacher --> [Experiment 3a, Run 23](results/Experiment_3a/Run_23/config.toml)<br>**- Stage 2:** Compression with hyperparameter from [Experiment 6a, Run 3](results/Experiment_6a/Run_3/config.toml)<br>**- Stage 3:** No Teacher |
| [9]  | Train with teacher KD only in the second stage to see if kd in first stage is necessary - ResNet10<br>- Resisc45<br>**- Stage 1:** No teacher<br>**- Stage 2:** Compression with hyperparameter from [Experiment 7a, Run 5](results/Experiment_7a/Run_5/config.toml)<br>**- Stage 3:** Train with teacher --> use model after pruning                                           |
| [10] | Train with teacher KD only in the second stage to see if kd in first stage is necessary - ResNet10<br>- Resisc45<br>**- Stage 1:** No teacher<br>**- Stage 2:** Compression with hyperparameter from [Experiment 7a, Run 6](results/Experiment_7a/Run_6/config.toml)<br>**- Stage 3:** Train with teacher --> use model after pruning                                           |
| [11] | Train with teacher KD only in the second stage to see if kd in first stage is necessary - ResNet18<br>- Resisc45<br>**- Stage 1:** No teacher<br>**- Stage 2:** Compression with hyperparameter from [Experiment 6a, Run 1](results/Experiment_6a/Run_1/config.toml)<br>**- Stage 3:** Train with teacher --> use model after pruning                                           |
| [12] | Train with teacher KD only in the second stage to see if kd in first stage is necessary - ResNet18<br>- Resisc45<br>**- Stage 1:** No teacher<br>**- Stage 2:** Compression with hyperparameter from [Experiment 6a, Run 3](results/Experiment_6a/Run_3/config.toml)<br>**- Stage 3:** Train with teacher --> use model after pruning                                           |

### ResNet10 on RESISC45
| Model                                                | Parameter | Validation Accuracy (%) - Seed42 | Validation Accuracy (%) std | Run ID                                                                            | Notes                                                       |
|------------------------------------------------------|-----------|----------------------------------|-----------------------------|-----------------------------------------------------------------------------------|-------------------------------------------------------------|
| ResNet10 - Our Baseline                              | 4,921,197 | 88.55                            | 89.88                       | [Experiment 1a, Run_2](../experiments/results/Experiment_1a/Run_2)                | Train from scratch without a teacher for 50 epochs          |
| ResNet10 + KD                                        | 4,921,197 | 89.44                            | 89.88                       | [Experiment 4a, Run_35](../experiments/results/Experiment_4a/Run_35)              | Train from scratch with a teacher for 50 epochs             |
| ResNet10 - No KD + Pruning (34.57%) - No KD          | 3,219,996 | 88.38                            | 89.88                       | [Experiment 8a, Run_1](../experiments/results/Experiment_8a/Run_1)                | Compress model ResNet10 - Baseline and fine-tune without KD |
| ResNet10 - No KD + Pruning (34.57%) + KD             | 3,219,996 | 89.59                            | 89.88                       | [Experiment 8a, Run_9](../experiments/results/Experiment_8a/Run_9)                | Compress model ResNet10 - No KD and fine-tune with KD       |
| ResNet10 + KD + Pruning (34.57%) - No KD             | 3,219,996 | 88.43                            | 89.88                       | [Experiment 8a, Run_5](../experiments/results/Experiment_8a/Run_5)                | Compress model ResNet10 - KD and fine-tune without KD       |
| ResNet10 + KD + Pruning (34.57%) + KD                | 3,219,996 | 90.37                            | 89.88                       | [Experiment 7a, Run_5](../experiments/results/Experiment_7a/Run_5)                | Compress model ResNet10 - KD and fine-tune with KD          |
| ResNet10 - No KD + Pruning (63.09%) - No KD          | 1,816,587 | 87.49                            | 89.88                       | [Experiment 8a, Run_2](../experiments/results/Experiment_8a/Run_2)                | Compress model ResNet10 - Baseline and fine-tune without KD |
| ResNet10 - No KD + Pruning (63.09%) + KD             | 1,816,587 | 88.17                            | 89.88                       | [Experiment 8a, Run_10](../experiments/results/Experiment_8a/Run_10)              | Compress model ResNet10 - No KD and fine-tune with KD       |
| ResNet10 + KD + Pruning (63.09%) - No KD             | 1,816,587 | 87.37                            | 89.88                       | [Experiment 8a, Run_6](../experiments/results/Experiment_8a/Run_6)                | Compress model ResNet10 - KD and fine-tune without KD       |
| ResNet10 + KD + Pruning (63.09%) + KD                | 1,816,587 | 89.1                             | 89.88                       | [Experiment 7a, Run_6](../experiments/results/Experiment_7a/Run_6)                | Compress model ResNet10 - KD and fine-tune with KD          |
| ResNet10 + 6 iterations of KD aware compression loop | 1,258,861 | **89.19**                        | 89.88                       | [Experiment 10a, Run_1, Iteration 4](../experiments/results/Experiment_10a/Run_1) | Distillation-Aware compression loop with 2 iterations       |
| ResNet10 + 8 iterations of KD aware compression loop | 796,575   | **88.14**                        | 89.88                       | [Experiment 10a, Run_1, Iteration 4](../experiments/results/Experiment_10a/Run_1) | Distillation-Aware compression loop with 4 iterations       |

### ResNet18 on RESISC45
| Model                                                | Parameter  | Validation Accuracy (%) - Seed42 | Validation Accuracy (%) std | Run ID                                                                          | Notes                                                       |
|------------------------------------------------------|------------|----------------------------------|-----------------------------|---------------------------------------------------------------------------------|-------------------------------------------------------------|
| ResNet18 - Our Baseline                              | 11,191,917 | 89.88                            | 89.88                       | [Experiment 1a, Run_1](../experiments/results/Experiment_1a/Run_1)              | Train from scratch without a teacher for 50 epochs          |
| ResNet18 + KD                                        | 11,191,917 | 91.76                            | 89.88                       | [Experiment 3a, Run_23](../experiments/results/Experiment_3a/Run_23)            | Train from scratch with a teacher for 50 epochs             |
| ResNet18 - No KD + Pruning (46.03%) - No KD          | 6,040,725  | 90.98                            | 89.88                       | [Experiment 8a, Run_3](../experiments/results/Experiment_8a/Run_3)              | Compress model ResNet18 - Baseline and fine-tune without KD |
| ResNet18 - No KD + Pruning (46.03%) + KD             | 6,040,725  | 92.14                            | 89.88                       | [Experiment 8a, Run_11](../experiments/results/Experiment_8a/Run_11)            | Compress model ResNet18 - No KD and fine-tune with KD       |
| ResNet18 + KD + Pruning (46.03%) - No KD             | 6,040,725  | 92.06                            | 89.88                       | [Experiment 8a, Run_7](../experiments/results/Experiment_8a/Run_7)              | Compress model ResNet18 - KD and fine-tune without KD       |
| ResNet18 + KD + Pruning (46.03%) + KD                | 6,040,725  | 92.19                            | 89.88                       | [Experiment 6a, Run_1](../experiments/results/Experiment_6a/Run_1)              | Compress model ResNet18 - KD and fine-tune with KD          |
| ResNet18 - No KD + Pruning (74.99%) - No KD          | 2,798,648  | 87.78                            | 89.88                       | [Experiment 8a, Run_4](../experiments/results/Experiment_8a/Run_4)              | Compress model ResNet18 - Baseline and fine-tune without KD |
| ResNet18 - No KD + Pruning (74.99%) + KD             | 2,798,648  | 90.05                            | 89.88                       | [Experiment 8a, Run_12](../experiments/results/Experiment_8a/Run_12)            | Compress model ResNet18 - No KD and fine-tune with KD       |
| ResNet18 + KD + Pruning (74.99%) - No KD             | 2,798,648  | 89.02                            | 89.88                       | [Experiment 8a, Run_8](../experiments/results/Experiment_8a/Run_8)              | Compress model ResNet18 - KD and fine-tune without KD       |
| ResNet18 + KD + Pruning (74.99%) + KD                | 2,798,648  | 91.21                            | 89.88                       | [Experiment 6a, Run_3](../experiments/results/Experiment_6a/Run_3)              | Compress model ResNet18 - KD and fine-tune with KD          |
| ResNet18 + 7 iterations of KD aware compression loop | 1,715,497  | **91.89**                        | 89.88                       | [Experiment 9a, Run_1, Iteration 6](../experiments/results/Experiment_9a/Run_1) | Distillation-Aware compression loop with 6 iterations       |
| ResNet18 + 8 iterations of KD aware compression loop | 923,889    | **90.62**                        | 89.88                       | [Experiment 9a, Run_1, Iteration 9](../experiments/results/Experiment_9a/Run_1) | Distillation-Aware compression loop with 9 iterations       |

---

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