# Molecular Energy Neural Network Framework

This repository provides a configurable framework for training neural networks to predict molecular energy levels. It is designed to be extensible, allowing for new molecules, datasets, and model architectures.

This work is based on previous research predicting CO2 and CO minor isotopologue energies using MARVEL, TROVE, and calculated values. (See [CO2 Neural Network Repository](https://github.com/mbarnfield63/CO2_NN_IE))

## Features

- **Config-Driven**: All experiments are defined in `.yml` files. No code changes needed to change models, learning rates, or datasets.
- **Automated Data Handling**:
    * Automatically combines multiple datasets (e.g., CO and CO2) at runtime.
    * Features are added by default; only non-feature columns need to be specified.
- **Experiment Modes**: Run a single train/val/test split, K-Fold Cross-Validation, or multiple runs with different random seeds, all from the config file.
- **Reproducible Outputs**: Each run creates a unique, timestamped output directory containing all plots, logs, model files, and a copy of the config.