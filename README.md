![KD aware compression loop](docs/plots/0_KD%20Aware%20Compression%20Loop.png)

# DAC-LOOP: DISTILLATION-AWARE COMPRESSION LOOP WITH TEACHER-STUDENT LEARNING FOR REMOTE SENSING CNN

## Project Overview

This is the official implementation of the paper: DAC-Lop: Distillation-Aware Compression Loop with Teacher-Student
Learning for Remote Sensing CNN.
The goal of this paper is to explore how much a knowledge-distilled student model can be further compressed
while preserving teacher-level accuracy on RS scene classification. In particular, our main contributions are:

- We introduce a Distillation-Aware Compression Loop tailored to CNN backbones and evaluated on EuroSAT and
  NWPU-RESISC45.
- Ablation studies demonstrate that iterative DAC-Loop cycles outperform standard pruning on every level of compressio.

Our proposed DAC-Loop approach achieves state-of-the-art results on the NWPU-RESISC45 dataset with a ResNet18 backbone,
compressing the model by 85.2% while still achieving 92.00% accuracy, outperforming previous methods.
We implemented a dynamic framework allowing to create scheduled compression loops. The loop follows these steps:

1. A teacher model is trained on the dataset.
2. A student model (smaller than teacher) is trained using knowledge distillation from the teacher.
3. The KD-trained student model is compressed via structured pruning.
4. The compressed model becomes the new student, and steps 2-4 are repeated for N iterations.

The framework allows to flexibly define different compression rates, methods and parameters during each iteration. This
enables a possibility to start in a first iteration e.g. with global unstructured pruning with a low compression rate
and end with structured pruning with a high compression rate in the last iteration.

## Project Structure

The directory structure of **DAC-Loop** is organized as follows:


---

### Module Overview

| Directory                       | Description                                                                                   |
|---------------------------------|-----------------------------------------------------------------------------------------------|
| **[experiments](experiments/)** | Contains the experiment configurations and the results                                        |
| **[framework](framework/)**     | This is the core of the DAC-Loop. It contains the complete logic to handle KD and Compression |
| **[notebooks](notebooks/)**     | Notebooks for data exploration                                                                |
| **[runner.py](runner.py)**      | The primary script to execute experiments                                                     |

## Getting Started

The project uses UV as the main dependency manager. To set up the environment, follow these steps:

1. **Clone the Repository**:

```bash
git https://github.com/RicoStaedeli/DAC-Loop.git
cd DAC-Loop
```

2. **Install Dependencies**:

```bash
pip install uv      # Install UV if not already installed
uv install
```

3. **Run Experiments**:
   The experiments are defined in the [configs](experiments/configurations) directory. Each Experiments consist of an
   `Experiment_Orchestrator_Config.toml`
   file and the corresponding `Run-xy.toml` files. The orchestrator config defines which runs to execute. The parameters
   of the specific run are in the run config files.
   An example experiment can be found in [Experiment_XY](experiments/configurations/Experiment_XY). The name
   of the experiment folder is
   passed as an argument to the `runner.py` script.

   To execute an experiment, use the following command:

```bash
uv run runner.py --gpu 0 --experiment Experiment_XY
```

## Evaluation

The following table summarizes the evaluation results of our DAC-Loop approach on the NWPU-RESISC45 dataset using a
ResNet18 backbone.
The table compares the base model, compressed models without knowledge distillation (KD), and models trained with KD.
Also included are the results of our DAC-Loop approach with multiple iterations (N) of compression and KD.

**ResNet-18 Ablation Study Results on RESISC45**

|             | KD-1 | KD-2 | N | Comp. | Params |        Val. Acc. |
|-------------|:----:|:----:|--:|------:|-------:|-----------------:|
| **Base**    |  –   |  –   | 1 |    0% | 11.2 M |     90.29 ± 0.28 |
|             |  –   |  –   | 1 |    0% | 2.0 M* |     83.50 ± 0.38 |
|             |  ✓   |  –   | 1 |    0% | 11.2 M |     92.85 ± 0.15 |
| **Comp.+**  |  –   |  –   | 1 | 47.5% |  5.8 M |     90.32 ± 0.14 |
|             |  ✓   |  –   | 1 | 47.5% |  5.8 M |     91.26 ± 0.07 |
|             |  –   |  ✓   | 1 | 47.5% |  5.8 M |     92.06 ± 0.11 |
|             |  ✓   |  ✓   | 1 | 47.5% |  5.8 M |     92.68 ± 0.24 |
| **Comp.++** |  –   |  –   | 1 | 75.8% |  2.7 M |     88.60 ± 0.15 |
|             |  ✓   |  –   | 1 | 75.8% |  2.7 M |     89.52 ± 0.33 |
|             |  –   |  ✓   | 1 | 75.8% |  2.7 M |     90.39 ± 0.12 |
|             |  ✓   |  ✓   | 1 | 75.8% |  2.7 M |     91.23 ± 0.16 |
| **Loop**    |  –   |  –   | 6 | 85.2% |  1.6 M | **92.00 ± 0.35** |
|             |  –   |  ✓   | 9 | 92.1% |  0.8 M |     90.71 ± 0.25 |

\* 2.0M model: ResNet10mini, our own small-model baseline.

![Final Plot](docs/plots/10_dac_compression_curves.png)

## Contact
Feel free to get in touch with us with any questions on the project. Reach out to [rico.staedeli@student.unisg.ch](mailto:rico.staedeli@student.unisg.ch).

## Citation
If you find this work useful for your research, please consider citing the following paper:

```
@article{staedeli2024dacloop,
  title={DAC-Loop: Distillation-Aware Compression Loop with Teacher-Student Learning for Remote Sensing CNN},
  author={Staedeli, Rico and Raichle, Lorena},
  journal={DeepLearning 2025},
  year={2025}
}
```