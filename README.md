# Isotopologue Extrapolation Neural Network Framework

A Physics-Informed Neural Network framework for predicting minor isotopologue energy levels using the "Isotopologue Extrapolation" method.

## Overview

This repository provides a configurable deep learning framework designed to predict energy levels for minor isotopologues (e.g., CO2, H2O) by extrapolating from the main isotopologue. It builds on the work of Polyansky _et al._ (2017) and McKemmish _et al._ (2024), utilizing the residual between the main isotopologue's experimental and calculated energy levels as a correction factor.

The model addresses the lack of experimental data for minor isotopologues by learning to transfer accuracy from the main species.

## Model Architecture

The framework supports multiple flexible architectures defined in `src/models.py`, selectable via configuration files:

1.  **Single Trunk MLP** (`CO2_Single_Trunk`, `H2O_Single_Trunk`):
    * A deep fully connected network (e.g., 1024 -> 512 -> ... -> 1) using GELU activations and Dropout for regularization.
    * Designed for learning global corrections across a dataset.

2.  **Hybrid Gated Networks** (`COCO2_Combined_Shared_Partial_Heads`, `H2O_Hybrid`):
    * **Shared Trunk**: A "brain" that learns common physics across all isotopologues/molecules.
    * **Specialist Heads**: Lightweight adapters fine-tuned for specific isotopes.
    * **Gating Mechanism**: A learnable sigmoid gate that dynamically blends the output of the shared trunk and the specialist heads (0–1 weighting), allowing the model to decide when to trust general physics versus specific isotope corrections.

## Key Features

* **Config-Driven Experiments**: All training parameters, model selections, and dataset paths are defined in `.yml` files (e.g., `configs/co2_basic.yml`).
* **Automated Data Handling**: Pipelines to automatically combine multiple datasets (e.g., CO and CO2) and generate features at runtime.
* **Advanced Architectures**: Includes support for embeddings, skip connections, and hybrid shared/specialized trunks.
* **Reproducibility**: Each run creates a unique, timestamped output directory containing logs, model checkpoints, plots, and a copy of the configuration used.

## Repository Structure

```text
.
├── configs/             # YAML configuration files for experiments
├── scripts/             # Data processing and analysis scripts
│   ├── combine_datasets.py
│   ├── preprocess_co2.py
│   ├── inference_analysis.py
│   └── postprocess_inference.py
├── src/                 # Source code package
│   ├── main.py          # Main entry point for training
│   ├── models.py        # PyTorch architectures (MLP, Hybrid, Gated)
│   ├── data_loader.py   # Data loading and scaling logic
│   ├── training.py      # Training loop and loss calculation
│   ├── inference.py     # Inference logic
│   └── plotting.py      # Visualization utilities
├── pyproject.toml       # Dependencies and project metadata
└── uv.lock              # Dependency lockfile
```

## Installation

This project is managed using uv for fast and reliable dependency management.

1. **Install uv** (if not already installed):
```bash
curl -LsSf [https://astral.sh/uv/install.sh](https://astral.sh/uv/install.sh) | sh 
```

2. **Clone the repository**:
```bash
git clone https://github.com/mbarnfield63/ML_Isotopologue_Extrapolation.git
cd ML_Isotopologue_Extrapolation
```

3. **Sync the environment**:
This command creates the virtual environment and installs all dependencies (PyTorch, pandas, scikit-learn, etc.) locked in ```uv.lock```

```bash
uv sync
```

4. **Activate the environment**:
- Linux/Mac: ```source .venv/bin/activate```
- Windows: ```.venv\Scripts\activate```

## Usage
1. **Data Preparation**:
Preprocess raw ExoMol states files into CSV format suitable for training. Scripts are provided for specific molecules.
```bash
uv run python -m scripts.preprocess_co2.py
```

2. **Experiments**:
    **a. Training**: Run an experiment using a configuration file. The `main.py` script handles data loading, model instatiation, and the training loop.
    ```bash
    uv run python -m src.main --config configs/co2_basic.yml
    ```

    **b. Visualisation** can be enabled in the configuration file to produce plots for loss curves, predicted vs true energy levels and residual histograms in the experiment's output directory.

    **c. Inference** can also be enabled in the configuration file, which retrains the model on the full training dataset before running inference.

3. **Inference Analysis**:
Generate plots and summary of the results.
```bash
uv run python -m scripts.inference_analysis
```
