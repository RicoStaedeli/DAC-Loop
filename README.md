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
