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

**Goal:** Create a baseline on the **full ATN dataset** (tokens_min = 225 // tokens_max = 511) with minimal GPT-2 implementation (reduced MLP + including MLP) to observe the convergence point and compare future subset approaches against it.
<details>
<summary>Experiment 1– extra notes & insights</summary>
- space for extra comments here
</details>

### Runs

| Run       | Date       | Author | Config                                   | Model                            | Dataset                                                | Hardware     | Train Time         | Metrics                                                        | Plots                                            | Notes/Issues                                       | Next Steps         |
|-----------|------------|--------|------------------------------------------|----------------------------------|--------------------------------------------------------|-------------|--------------------|----------------------------------------------------------------|--------------------------------------------------|----------------------------------------------------|--------------------|
| [1] inval | 2025-09-26 | Lorena | [Config.toml](../configs/exp1_run1.toml) | GPT2 (1 decoder, 1 head)         | HF: /cleaned-atn<br/>(tokens_min = 225 // tokens_max = 511) | DGX-8 Lorena | 16,25 h (58515.3s) | [metrics.json](../Experiments/Experiment_1/Run_1/metrics.json) | [plots](../Experiments/Experiment_1/Run_1/plots) | will be redone inclduing **training loss** as well | TO DELETE, replaced by E1 R3 |

