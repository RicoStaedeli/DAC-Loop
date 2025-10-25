# Experiment Descriptions

In this document we track the different experiments.  
All configurations, results, and decisions are documented for reproducibility.

---

## General Guidelines

For each experiment and run, the following information should be recorded:

- **Date**
- **Author/Responsible**
- **Config file**
- **Model architecture**
- **Dataset (train/val/test)**
- **Preprocessing steps**
- **Hyperparameters**
- **Training duration**
- **Metrics (training/validation/test)**
- **Plots (learning curves, losses, accuracy, etc.)**
- **Observations & Issues**
- **Decisions & Next Steps**

---


## Experiment 1 - baselines

**Goal:** Train a large teacher model with ResNet18. The training will be done with different combination of spectral bands.
Also, every smaller model will be trained from scratch as a baseline.
<details>
  <summary>Experiment 1 – overview plot</summary>
  <img src="Experiment_1/overview_plot_Experiment_1.png" alt="overview">
</details>



### Runs

| Run | Date       | Author | Config                                                       | Model    | Dataset                  | Hardware   | Train Time         | Metrics                                                        | Plots                                            | Bands                                      | Notes/Issues                                      | Next Steps         |
|-----|------------|--------|--------------------------------------------------------------|----------|--------------------------|------------|--------------------|----------------------------------------------------------------|--------------------------------------------------|---------------------------------------------------|---------------------------------------------------|--------------------|
| [1] | 2025-10-15 | Rico   | [Config.toml](Experiment_1/Run_1/config.toml) | ResNet18 | Train/Val/Test split of blanchon/EuroSAT_MSI | DGX-7 Rico |  | [metrics.json](../experiments/Experiment_1/Run_1/metrics.json) | [plots](Experiment_1/Run_1/plots) | ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"] |  | |
| [2] | 2025-10-15 | Rico   | [Config.toml](Experiment_1/Run_2/config.toml) | ResNet18 | Train/Val/Test split of blanchon/EuroSAT_MSI | DGX-7 Rico |  | [metrics.json](../experiments/Experiment_1/Run_2/metrics.json) | [plots](Experiment_1/Run_1/plots) |["B02", "B03", "B04"]|  | |
| [3] | 2025-10-15 | Rico   | [Config.toml](Experiment_1/Run_3/config.toml) | ResNet18 | Train/Val/Test split of blanchon/EuroSAT_MSI | DGX-7 Rico |  | [metrics.json](../experiments/Experiment_1/Run_3/metrics.json) | [plots](Experiment_1/Run_1/plots) |["NDVI", "NDWI", "NDMI"]["B02", "B03", "B04", "B07", "B08"] |  | |
| [4] | 2025-10-15 | Rico   | [Config.toml](Experiment_1/Run_4/config.toml) | ResNet10 | Train/Val/Test split of blanchon/EuroSAT_MSI | DGX-7 Rico |  | [metrics.json](../experiments/Experiment_1/Run_4/metrics.json) | [plots](Experiment_1/Run_1/plots) | ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"] |  | |
| [5] | 2025-10-15 | Rico   | [Config.toml](Experiment_1/Run_5/config.toml) | ResNet10 | Train/Val/Test split of blanchon/EuroSAT_MSI | DGX-7 Rico |  | [metrics.json](../experiments/Experiment_1/Run_5/metrics.json) | [plots](Experiment_1/Run_1/plots) |["B02", "B03", "B04"]|  | |
| [6] | 2025-10-15 | Rico   | [Config.toml](Experiment_1/Run_6/config.toml) | ResNet10 | Train/Val/Test split of blanchon/EuroSAT_MSI | DGX-7 Rico |  | [metrics.json](../experiments/Experiment_1/Run_6/metrics.json) | [plots](Experiment_1/Run_1/plots) |["NDVI", "NDWI", "NDMI"]["B02", "B03", "B04", "B07", "B08"] |  | |
| [7] | 2025-10-15 | Rico   | [Config.toml](Experiment_1/Run_7/config.toml) | TinyCNN  | Train/Val/Test split of blanchon/EuroSAT_MSI | DGX-7 Rico |  | [metrics.json](../experiments/Experiment_1/Run_7/metrics.json) | [plots](Experiment_1/Run_1/plots) | ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"] |  | |
| [8] | 2025-10-15 | Rico   | [Config.toml](Experiment_1/Run_8/config.toml) | TinyCNN  | Train/Val/Test split of blanchon/EuroSAT_MSI | DGX-7 Rico |  | [metrics.json](../experiments/Experiment_1/Run_8/metrics.json) | [plots](Experiment_1/Run_1/plots) |["B02", "B03", "B04"]|  | |
| [9] | 2025-10-15 | Rico   | [Config.toml](Experiment_1/Run_9/config.toml) | TinyCNN  | Train/Val/Test split of blanchon/EuroSAT_MSI | DGX-7 Rico |  | [metrics.json](../experiments/Experiment_1/Run_9/metrics.json) | [plots](Experiment_1/Run_1/plots) |["NDVI", "NDWI", "NDMI"]["B02", "B03", "B04", "B07", "B08"] |  | |

## Experiment 2 - students

**Goal:** Train the smaller models (ResNet10 and TiniCNN) with teacher-student training. Use the best teacher model (best validation loss) ResNet18 as a guiding teacher.
<details>
<summary>Experiment 1– extra notes & insights</summary>
- space for extra comments here
</details>

### Runs

| Run | Date       | Author | Config                                                       | Model Student  | Model Teacher | Dataset                  | Hardware   | Train Time         | Metrics                                                        | Plots                                            | Bands                                      | Notes/Issues                                      | Next Steps         |
|-----|------------|--------|--------------------------------------------------------------|---------|---------------|--------------------------|------------|--------------------|----------------------------------------------------------------|--------------------------------------------------|---------------------------------------------------|---------------------------------------------------|--------------------|
| [1] | 2025-10-15 | Rico   | [Config.toml](Experiment_2/Run_1/config.toml) | ResNet10 | ResNet18      | Train/Val/Test split of blanchon/EuroSAT_MSI | DGX-7 Rico |  | [metrics.json](Experiment_2/Run_1/metrics.json) | [plots](Experiment_1/Run_1/plots) | ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"] |  | |
| [2] | 2025-10-15 | Rico   | [Config.toml](Experiment_2/Run_2/config.toml) | ResNet10 | ResNet18      |Train/Val/Test split of blanchon/EuroSAT_MSI | DGX-7 Rico |  | [metrics.json](Experiment_2/Run_2/metrics.json) | [plots](Experiment_1/Run_1/plots) |["B02", "B03", "B04"]|  | |
| [3] | 2025-10-15 | Rico   | [Config.toml](Experiment_2/Run_3/config.toml) | ResNet10 | ResNet18      |Train/Val/Test split of blanchon/EuroSAT_MSI | DGX-7 Rico |  | [metrics.json](Experiment_2/Run_3/metrics.json) | [plots](Experiment_1/Run_1/plots) |["NDVI", "NDWI", "NDMI"]["B02", "B03", "B04", "B07", "B08"] |  | |
| [4] | 2025-10-15 | Rico   | [Config.toml](Experiment_2/Run_4/config.toml) | TinyCNN | ResNet18      |Train/Val/Test split of blanchon/EuroSAT_MSI | DGX-7 Rico |  | [metrics.json](Experiment_2/Run_4/metrics.json) | [plots](Experiment_1/Run_1/plots) | ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"] |  | |
| [5] | 2025-10-15 | Rico   | [Config.toml](Experiment_2/Run_5/config.toml) | TinyCNN | ResNet18      |Train/Val/Test split of blanchon/EuroSAT_MSI | DGX-7 Rico |  | [metrics.json](Experiment_2/Run_5/metrics.json) | [plots](Experiment_1/Run_1/plots) |["B02", "B03", "B04"]|  | |
| [6] | 2025-10-15 | Rico   | [Config.toml](Experiment_2/Run_6/config.toml) | TinyCNN | ResNet18      |Train/Val/Test split of blanchon/EuroSAT_MSI | DGX-7 Rico |  | [metrics.json](Experiment_2/Run_6/metrics.json) | [plots](Experiment_1/Run_1/plots) |["NDVI", "NDWI", "NDMI"]["B02", "B03", "B04", "B07", "B08"] |  | |


## Experiment 3 - Hyperparameter tuning students

**Goal:** Train the smaller models (TinyCNN) with teacher-student training. Use the best teacher model (best validation loss) ResNet18 as a guiding teacher.
In this experiment we will do hyperparameter tuning for the student model training. Also we use different knowledge distillation methods such as CosineEmbeddingLoss, CrossEntropyLoss and KLDivergenceLoss.
<details>
  <summary>Experiment 3 – overview plot</summary>
  <img src="Experiment_3/overview_plot_Experiment_3.png" alt="overview">
</details>

### Runs

| Run  | Date       | Author | Config                                           | Model Student | Model Teacher | KD-Type | Dataset             | Hardware   | Train Time | Metrics                                             | Plots                                | Bands                                                        | Notes/Issues | Next Steps |
|------|------------|--------|--------------------------------------------------|---------------|---------------|---------|---------------------|------------|------------|----------------------------------------------------|--------------------------------------|--------------------------------------------------------------|--------------|------------|
| [1] | 2025-10-25 | Rico   | [Config.toml](Experiment_3/Run_1/config.toml)  | TinyCNN       | ResNet18      | Train/Val/Test split of blanchon/EuroSAT_MSI | CosineEmbeddingLoss | DGX-7 Rico |            | [metrics.json](Experiment_3/Run_1/metrics.jsonl) | [plots](Experiment_3/Run_1/plots)  | ["NDVI", "NDWI", "NDMI"]["B02", "B03", "B04", "B07", "B08"]  |              |            |
| [2] | 2025-10-25 | Rico   | [Config.toml](Experiment_3/Run_2/config.toml)  | TinyCNN       | ResNet18      | Train/Val/Test split of blanchon/EuroSAT_MSI | CosineEmbeddingLoss | DGX-7 Rico |            | [metrics.json](Experiment_3/Run_2/metrics.jsonl) | [plots](Experiment_3/Run_2/plots)  | ["NDVI", "NDWI", "NDMI"]["B02", "B03", "B04", "B07", "B08"]  |              |            |
| [3] | 2025-10-25 | Rico   | [Config.toml](Experiment_3/Run_3/config.toml)  | TinyCNN       | ResNet18      | Train/Val/Test split of blanchon/EuroSAT_MSI | CosineEmbeddingLoss | DGX-7 Rico |            | [metrics.json](Experiment_3/Run_3/metrics.jsonl) | [plots](Experiment_3/Run_3/plots)  | ["NDVI", "NDWI", "NDMI"]["B02", "B03", "B04", "B07", "B08"]  |              |            |
| [4] | 2025-10-25 | Rico   | [Config.toml](Experiment_3/Run_4/config.toml)  | TinyCNN       | ResNet18      | Train/Val/Test split of blanchon/EuroSAT_MSI | CosineEmbeddingLoss | DGX-7 Rico |            | [metrics.json](Experiment_3/Run_4/metrics.jsonl) | [plots](Experiment_3/Run_4/plots)  | ["NDVI", "NDWI", "NDMI"]["B02", "B03", "B04", "B07", "B08"]  |              |            |
| [5] | 2025-10-25 | Rico   | [Config.toml](Experiment_3/Run_5/config.toml)  | TinyCNN       | ResNet18      | Train/Val/Test split of blanchon/EuroSAT_MSI | CosineEmbeddingLoss | DGX-7 Rico |            | [metrics.json](Experiment_3/Run_5/metrics.jsonl) | [plots](Experiment_3/Run_5/plots)  | ["NDVI", "NDWI", "NDMI"]["B02", "B03", "B04", "B07", "B08"]  |              |            |
| [6] | 2025-10-25 | Rico   | [Config.toml](Experiment_3/Run_6/config.toml)  | TinyCNN       | ResNet18      | Train/Val/Test split of blanchon/EuroSAT_MSI | CosineEmbeddingLoss | DGX-7 Rico |            | [metrics.json](Experiment_3/Run_6/metrics.jsonl) | [plots](Experiment_3/Run_6/plots)  | ["NDVI", "NDWI", "NDMI"]["B02", "B03", "B04", "B07", "B08"]  |              |            |
| [7] | 2025-10-25 | Rico   | [Config.toml](Experiment_3/Run_7/config.toml)  | TinyCNN       | ResNet18      | Train/Val/Test split of blanchon/EuroSAT_MSI | CosineEmbeddingLoss | DGX-7 Rico |            | [metrics.json](Experiment_3/Run_7/metrics.jsonl) | [plots](Experiment_3/Run_7/plots)  | ["NDVI", "NDWI", "NDMI"]["B02", "B03", "B04", "B07", "B08"]  |              |            |
| [8] | 2025-10-25 | Rico   | [Config.toml](Experiment_3/Run_8/config.toml)  | TinyCNN       | ResNet18      | Train/Val/Test split of blanchon/EuroSAT_MSI | CosineEmbeddingLoss | DGX-7 Rico |            | [metrics.json](Experiment_3/Run_8/metrics.jsonl) | [plots](Experiment_3/Run_8/plots)  | ["NDVI", "NDWI", "NDMI"]["B02", "B03", "B04", "B07", "B08"]  |              |            |
| [9] | 2025-10-25 | Rico   | [Config.toml](Experiment_3/Run_9/config.toml)  | TinyCNN       | ResNet18      | Train/Val/Test split of blanchon/EuroSAT_MSI | CosineEmbeddingLoss | DGX-7 Rico |            | [metrics.json](Experiment_3/Run_9/metrics.jsonl) | [plots](Experiment_3/Run_9/plots)  | ["NDVI", "NDWI", "NDMI"]["B02", "B03", "B04", "B07", "B08"]  |              |            |
| [10] | 2025-10-25 | Rico   | [Config.toml](Experiment_3/Run_10/config.toml)  | TinyCNN       | ResNet18      | Train/Val/Test split of blanchon/EuroSAT_MSI | CosineEmbeddingLoss | DGX-7 Rico |            | [metrics.json](Experiment_3/Run_10/metrics.jsonl) | [plots](Experiment_3/Run_10/plots)  | ["NDVI", "NDWI", "NDMI"]["B02", "B03", "B04", "B07", "B08"]  |              |            |
| [11] | 2025-10-25 | Rico   | [Config.toml](Experiment_3/Run_11/config.toml)  | TinyCNN       | ResNet18      | Train/Val/Test split of blanchon/EuroSAT_MSI | CrossEntropyLoss    | DGX-7 Rico |            | [metrics.json](Experiment_3/Run_11/metrics.jsonl) | [plots](Experiment_3/Run_11/plots)  | ["NDVI", "NDWI", "NDMI"]["B02", "B03", "B04", "B07", "B08"]  |              |            |
| [12] | 2025-10-25 | Rico   | [Config.toml](Experiment_3/Run_12/config.toml)  | TinyCNN       | ResNet18      | Train/Val/Test split of blanchon/EuroSAT_MSI | CrossEntropyLoss    | DGX-7 Rico |            | [metrics.json](Experiment_3/Run_12/metrics.jsonl) | [plots](Experiment_3/Run_12/plots)  | ["NDVI", "NDWI", "NDMI"]["B02", "B03", "B04", "B07", "B08"]  |              |            |
| [13] | 2025-10-25 | Rico   | [Config.toml](Experiment_3/Run_13/config.toml)  | TinyCNN       | ResNet18      | Train/Val/Test split of blanchon/EuroSAT_MSI | CrossEntropyLoss    | DGX-7 Rico |            | [metrics.json](Experiment_3/Run_13/metrics.jsonl) | [plots](Experiment_3/Run_13/plots)  | ["NDVI", "NDWI", "NDMI"]["B02", "B03", "B04", "B07", "B08"]  |              |            |
| [14] | 2025-10-25 | Rico   | [Config.toml](Experiment_3/Run_14/config.toml)  | TinyCNN       | ResNet18      | Train/Val/Test split of blanchon/EuroSAT_MSI | CrossEntropyLoss    | DGX-7 Rico |            | [metrics.json](Experiment_3/Run_14/metrics.jsonl) | [plots](Experiment_3/Run_14/plots)  | ["NDVI", "NDWI", "NDMI"]["B02", "B03", "B04", "B07", "B08"]  |              |            |
| [15] | 2025-10-25 | Rico   | [Config.toml](Experiment_3/Run_15/config.toml)  | TinyCNN       | ResNet18      | Train/Val/Test split of blanchon/EuroSAT_MSI | CrossEntropyLoss    | DGX-7 Rico |            | [metrics.json](Experiment_3/Run_15/metrics.jsonl) | [plots](Experiment_3/Run_15/plots)  | ["NDVI", "NDWI", "NDMI"]["B02", "B03", "B04", "B07", "B08"]  |              |            |
| [16] | 2025-10-25 | Rico   | [Config.toml](Experiment_3/Run_16/config.toml)  | TinyCNN       | ResNet18      | Train/Val/Test split of blanchon/EuroSAT_MSI | CrossEntropyLoss    | DGX-7 Rico |            | [metrics.json](Experiment_3/Run_16/metrics.jsonl) | [plots](Experiment_3/Run_16/plots)  | ["NDVI", "NDWI", "NDMI"]["B02", "B03", "B04", "B07", "B08"]  |              |            |
| [17] | 2025-10-25 | Rico   | [Config.toml](Experiment_3/Run_17/config.toml)  | TinyCNN       | ResNet18      | Train/Val/Test split of blanchon/EuroSAT_MSI | CrossEntropyLoss    | DGX-7 Rico |            | [metrics.json](Experiment_3/Run_17/metrics.jsonl) | [plots](Experiment_3/Run_17/plots)  | ["NDVI", "NDWI", "NDMI"]["B02", "B03", "B04", "B07", "B08"]  |              |            |
| [18] | 2025-10-25 | Rico   | [Config.toml](Experiment_3/Run_18/config.toml)  | TinyCNN       | ResNet18      | Train/Val/Test split of blanchon/EuroSAT_MSI | CrossEntropyLoss    | DGX-7 Rico |            | [metrics.json](Experiment_3/Run_18/metrics.jsonl) | [plots](Experiment_3/Run_18/plots)  | ["NDVI", "NDWI", "NDMI"]["B02", "B03", "B04", "B07", "B08"]  |              |            |
| [19] | 2025-10-25 | Rico   | [Config.toml](Experiment_3/Run_19/config.toml)  | TinyCNN       | ResNet18      | Train/Val/Test split of blanchon/EuroSAT_MSI | CrossEntropyLoss    | DGX-7 Rico |            | [metrics.json](Experiment_3/Run_19/metrics.jsonl) | [plots](Experiment_3/Run_19/plots)  | ["NDVI", "NDWI", "NDMI"]["B02", "B03", "B04", "B07", "B08"]  |              |            |
| [20] | 2025-10-25 | Rico   | [Config.toml](Experiment_3/Run_20/config.toml)  | TinyCNN       | ResNet18      | Train/Val/Test split of blanchon/EuroSAT_MSI | CrossEntropyLoss    | DGX-7 Rico |            | [metrics.json](Experiment_3/Run_20/metrics.jsonl) | [plots](Experiment_3/Run_20/plots)  | ["NDVI", "NDWI", "NDMI"]["B02", "B03", "B04", "B07", "B08"]  |              |            |
| [21] | 2025-10-25 | Rico   | [Config.toml](Experiment_3/Run_21/config.toml)  | TinyCNN       | ResNet18      | Train/Val/Test split of blanchon/EuroSAT_MSI | CrossEntropyLoss    | DGX-7 Rico |            | [metrics.json](Experiment_3/Run_21/metrics.jsonl) | [plots](Experiment_3/Run_21/plots)  | ["NDVI", "NDWI", "NDMI"]["B02", "B03", "B04", "B07", "B08"]  |              |            |
| [22] | 2025-10-25 | Rico   | [Config.toml](Experiment_3/Run_22/config.toml)  | TinyCNN       | ResNet18      | Train/Val/Test split of blanchon/EuroSAT_MSI | CrossEntropyLoss    | DGX-7 Rico |            | [metrics.json](Experiment_3/Run_22/metrics.jsonl) | [plots](Experiment_3/Run_22/plots)  | ["NDVI", "NDWI", "NDMI"]["B02", "B03", "B04", "B07", "B08"]  |              |            |
| [23] | 2025-10-25 | Rico   | [Config.toml](Experiment_3/Run_23/config.toml)  | TinyCNN       | ResNet18      | Train/Val/Test split of blanchon/EuroSAT_MSI | CrossEntropyLoss    | DGX-7 Rico |            | [metrics.json](Experiment_3/Run_23/metrics.jsonl) | [plots](Experiment_3/Run_23/plots)  | ["NDVI", "NDWI", "NDMI"]["B02", "B03", "B04", "B07", "B08"]  |              |            |
| [24] | 2025-10-25 | Rico   | [Config.toml](Experiment_3/Run_24/config.toml)  | TinyCNN       | ResNet18      | Train/Val/Test split of blanchon/EuroSAT_MSI | KLDivLoss           | DGX-7 Rico |            | [metrics.json](Experiment_3/Run_24/metrics.jsonl) | [plots](Experiment_3/Run_24/plots)  | ["NDVI", "NDWI", "NDMI"]["B02", "B03", "B04", "B07", "B08"]  |              |            |

