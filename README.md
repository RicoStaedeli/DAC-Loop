# DISTILLATION-AWARE PRUNING LOOP: ENHANCING CNN COMPRESSION WITH TEACHER-STUDENT LEARNING FOR REMOTE SENSING

## Project Overview

This project implements a distillation-aware pruning loop designed to enhance the compression of Convolutional Neural
Networks (CNNs) using teacher-student learning techniques. The primary goal is to reduce the model size while
maintaining high performance, making it suitable for deployment in resource-constrained environments, particularly in
remote sensing applications.

## Project Structure

The directory structure of **R-L-Net** is organized as follows:

```
R-L-Net/
├── configs/                            # Configuration files for the experiments
├── docs/                               # Documentation
├── experiments/                        # Experiment logs and outputs
├── framework/                          # Core application logic
│   ├── config_parser/                  # Configuration parsing utilities
│   ├── dataset/                        # Data loading and processing
│   ├── models/                         # Neural network architecture definitions
│   │   ├── ResNet10.py
│   │   ├── ResNet18.py
│   │   ├── ResNet50.py
│   │   └── TinyCNN.py
│   ├── pruning/                        # Pruning algorithms and logic
│   │   ├── services/
│   │   └── strategies/
│   │       ├── ports.py
│   │       └── pruning_orchestrator.py
│   ├── training/                       # Training loop handlers
│   │   ├── **strategies/**
│   │   │   ├── factories.py
│   │   │   ├── fitnets.py
│   │   │   ├── io_ckpt.py
│   │   │   └── metrics_writer.py
│   │   └── training_orchestrator.py
│   └── utils/                          # General utility functions
├── notebooks/                          # Jupyter notebooks for data analysis
├── .gitignore
├── pyproject.toml                      # Python project settings (Poetry/PEP 621)
└── runner.py                           # Main entry point for training and pruning
```

---

### Module Overview

| Directory              | Description                                                                        |
|------------------------|------------------------------------------------------------------------------------|
| **framework/models**   | Contains model architectures including various ResNet implementations and TinyCNN. |
| **framework/pruning**  | Handles model pruning strategies and orchestration logic.                          |
| **framework/training** | Manages training loops with KD methods                                             |
| **notebooks**          | Notebooks for data exploration                                                     |
| **runner.py**          | The primary script to execute training or pruning tasks.                           |

## Getting Started

The project uses UV as the main dependency manager. To set up the environment, follow these steps:

1. **Clone the Repository**:

```bash
git clone https://github.com/RicoStaedeli/R-L-Net.git
cd R-L-Net
```

2. **Install Dependencies**:

```bash
pip install uv
uv install
```

3. **Run Experiments**:
   The experiments are defined in the [configs](experiments/configurations) directory. Each Experiments consist of an
   `Experiment_Orchestrator_Config.toml`
   file and the corresponding `Run-xy.toml` files. The orchestrator config defines which runs to execute. The parameters
   of the specific run are in the run config files.
   An example experiment can be found in [Experiment_XY](experiments/configurations/Experiment_XY). The name of the experiment folder is
   passed as an argument to the `runner.py` script.
4. To execute an experiment, use the following command:

```bash
uv run runner.py --gpu 0 --experiment Experiment_XY
```
