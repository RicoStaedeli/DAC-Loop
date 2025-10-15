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
<summary>Experiment 1– extra notes & insights</summary>
- space for extra comments here
</details>

### Runs

| Run | Date       | Author | Config                                                       | Model    | Dataset                  | Hardware   | Train Time         | Metrics                                                        | Plots                                            | Bands                                      | Notes/Issues                                      | Next Steps         |
|-----|------------|--------|--------------------------------------------------------------|----------|--------------------------|------------|--------------------|----------------------------------------------------------------|--------------------------------------------------|---------------------------------------------------|---------------------------------------------------|--------------------|
| [1] | 2025-10-15 | Rico   | [Config.toml](../experiments/Experiment_1/Run_1/config.toml) | ResNet18 | Train/Val/Test split of blanchon/EuroSAT_MSI | DGX-7 Rico |  | [metrics.json](../experiments/Experiment_1/Run_1/metrics.json) | [plots](../experiments/Experiment_1/Run_1/plots) | ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"] |  | |
| [2] | 2025-10-15 | Rico   | [Config.toml](../experiments/Experiment_1/Run_2/config.toml) | ResNet18 | Train/Val/Test split of blanchon/EuroSAT_MSI | DGX-7 Rico |  | [metrics.json](../experiments/Experiment_1/Run_2/metrics.json) | [plots](../experiments/Experiment_1/Run_1/plots) |["B02", "B03", "B04"]|  | |
| [3] | 2025-10-15 | Rico   | [Config.toml](../experiments/Experiment_1/Run_3/config.toml) | ResNet18 | Train/Val/Test split of blanchon/EuroSAT_MSI | DGX-7 Rico |  | [metrics.json](../experiments/Experiment_1/Run_3/metrics.json) | [plots](../experiments/Experiment_1/Run_1/plots) |["NDVI", "NDWI", "NDMI"]["B02", "B03", "B04", "B07", "B08"] |  | |
| [4] | 2025-10-15 | Rico   | [Config.toml](../experiments/Experiment_1/Run_4/config.toml) | ResNet10 | Train/Val/Test split of blanchon/EuroSAT_MSI | DGX-7 Rico |  | [metrics.json](../experiments/Experiment_1/Run_4/metrics.json) | [plots](../experiments/Experiment_1/Run_1/plots) | ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"] |  | |
| [5] | 2025-10-15 | Rico   | [Config.toml](../experiments/Experiment_1/Run_5/config.toml) | ResNet10 | Train/Val/Test split of blanchon/EuroSAT_MSI | DGX-7 Rico |  | [metrics.json](../experiments/Experiment_1/Run_5/metrics.json) | [plots](../experiments/Experiment_1/Run_1/plots) |["B02", "B03", "B04"]|  | |
| [6] | 2025-10-15 | Rico   | [Config.toml](../experiments/Experiment_1/Run_6/config.toml) | ResNet10 | Train/Val/Test split of blanchon/EuroSAT_MSI | DGX-7 Rico |  | [metrics.json](../experiments/Experiment_1/Run_6/metrics.json) | [plots](../experiments/Experiment_1/Run_1/plots) |["NDVI", "NDWI", "NDMI"]["B02", "B03", "B04", "B07", "B08"] |  | |
| [7] | 2025-10-15 | Rico   | [Config.toml](../experiments/Experiment_1/Run_7/config.toml) | TinyCNN  | Train/Val/Test split of blanchon/EuroSAT_MSI | DGX-7 Rico |  | [metrics.json](../experiments/Experiment_1/Run_7/metrics.json) | [plots](../experiments/Experiment_1/Run_1/plots) | ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"] |  | |
| [8] | 2025-10-15 | Rico   | [Config.toml](../experiments/Experiment_1/Run_8/config.toml) | TinyCNN  | Train/Val/Test split of blanchon/EuroSAT_MSI | DGX-7 Rico |  | [metrics.json](../experiments/Experiment_1/Run_8/metrics.json) | [plots](../experiments/Experiment_1/Run_1/plots) |["B02", "B03", "B04"]|  | |
| [9] | 2025-10-15 | Rico   | [Config.toml](../experiments/Experiment_1/Run_9/config.toml) | TinyCNN  | Train/Val/Test split of blanchon/EuroSAT_MSI | DGX-7 Rico |  | [metrics.json](../experiments/Experiment_1/Run_9/metrics.json) | [plots](../experiments/Experiment_1/Run_1/plots) |["NDVI", "NDWI", "NDMI"]["B02", "B03", "B04", "B07", "B08"] |  | |

## Experiment 2 - students

**Goal:** Train the smaller models (ResNet10 and TiniCNN) with teacher-student training. Use the best teacher model (best validation loss) ResNet18 as a guiding teacher.
<details>
<summary>Experiment 1– extra notes & insights</summary>
- space for extra comments here
</details>

### Runs

| Run | Date       | Author | Config                                                       | Model Student  | Model Teacher | Dataset                  | Hardware   | Train Time         | Metrics                                                        | Plots                                            | Bands                                      | Notes/Issues                                      | Next Steps         |
|-----|------------|--------|--------------------------------------------------------------|---------|---------------|--------------------------|------------|--------------------|----------------------------------------------------------------|--------------------------------------------------|---------------------------------------------------|---------------------------------------------------|--------------------|
| [1] | 2025-10-15 | Rico   | [Config.toml](../experiments/Experiment_2/Run_1/config.toml) | ResNet10 | ResNet18      | Train/Val/Test split of blanchon/EuroSAT_MSI | DGX-7 Rico |  | [metrics.json](../experiments/Experiment_2/Run_1/metrics.json) | [plots](../experiments/Experiment_1/Run_1/plots) | ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"] |  | |
| [2] | 2025-10-15 | Rico   | [Config.toml](../experiments/Experiment_2/Run_2/config.toml) | ResNet10 | ResNet18      |Train/Val/Test split of blanchon/EuroSAT_MSI | DGX-7 Rico |  | [metrics.json](../experiments/Experiment_2/Run_2/metrics.json) | [plots](../experiments/Experiment_1/Run_1/plots) |["B02", "B03", "B04"]|  | |
| [3] | 2025-10-15 | Rico   | [Config.toml](../experiments/Experiment_2/Run_3/config.toml) | ResNet10 | ResNet18      |Train/Val/Test split of blanchon/EuroSAT_MSI | DGX-7 Rico |  | [metrics.json](../experiments/Experiment_2/Run_3/metrics.json) | [plots](../experiments/Experiment_1/Run_1/plots) |["NDVI", "NDWI", "NDMI"]["B02", "B03", "B04", "B07", "B08"] |  | |
| [4] | 2025-10-15 | Rico   | [Config.toml](../experiments/Experiment_2/Run_4/config.toml) | TinyCNN | ResNet18      |Train/Val/Test split of blanchon/EuroSAT_MSI | DGX-7 Rico |  | [metrics.json](../experiments/Experiment_2/Run_4/metrics.json) | [plots](../experiments/Experiment_1/Run_1/plots) | ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"] |  | |
| [5] | 2025-10-15 | Rico   | [Config.toml](../experiments/Experiment_2/Run_5/config.toml) | TinyCNN | ResNet18      |Train/Val/Test split of blanchon/EuroSAT_MSI | DGX-7 Rico |  | [metrics.json](../experiments/Experiment_2/Run_5/metrics.json) | [plots](../experiments/Experiment_1/Run_1/plots) |["B02", "B03", "B04"]|  | |
| [6] | 2025-10-15 | Rico   | [Config.toml](../experiments/Experiment_2/Run_6/config.toml) | TinyCNN | ResNet18      |Train/Val/Test split of blanchon/EuroSAT_MSI | DGX-7 Rico |  | [metrics.json](../experiments/Experiment_2/Run_6/metrics.json) | [plots](../experiments/Experiment_1/Run_1/plots) |["NDVI", "NDWI", "NDMI"]["B02", "B03", "B04", "B07", "B08"] |  | |

