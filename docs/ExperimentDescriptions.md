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

**Goal:** Train a large teacher model. The training will be done with different combination of spectral bands.
<details>
<summary>Experiment 1– extra notes & insights</summary>
- space for extra comments here
</details>

### Runs

| Run      | Date       | Author | Config                                            | Model                            | Dataset                  | Hardware   | Train Time         | Metrics                                                        | Plots                                            | Notes/Issues                                      | Next Steps         |
|----------|------------|--------|---------------------------------------------------|----------------------------------|--------------------------|------------|--------------------|----------------------------------------------------------------|--------------------------------------------------|---------------------------------------------------|--------------------|
| [1] | 2025-10-05 | Rico   | [Config.toml](../configs/Experiment_1/Experiment-1-Run-1.toml) | GPT2 (1 decoder, 1 head)         | Train/Val/Test split of blanchon/EuroSAT_MSI | DGX-7 Rico |  | [metrics.json](../experiments/Experiment_1/Run_1/metrics.json) | [plots](../experiments/Experiment_1/Run_1/plots) |  |  |

