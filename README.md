# Isotopologue Extrapolation Neural Network Framework

This repository provides a configurable framework for training neural networks to predict minor isotopologue molecular energy levels. It is designed to be extensible, allowing for new molecules, datasets, and model architectures.

This work builds on the work of Polyansky _et al._ (2017) and McKemmish _et al._ (2024) formalising the ExoMol method of "Isotopologue Extrapolation"; utilising the residual of the main isotopologue's experimental and calculated energy levels as a correction factor for the calculated energy levels of the minor isotopologues where experimental data was not present.

## Features

- **Config-Driven**: All experiments are defined in `.yml` files. No code changes needed to change models, learning rates, or datasets.
- **Automated Data Handling**:
    * Automatically combines multiple datasets (e.g., CO and CO2) at runtime.
    * Features are added by default; only non-feature columns need to be specified.
- **Experiment Modes**: Run a single train/val/test split, Stratified K-Fold Cross-Validation, or multiple runs with different random seeds, all from the config file.
- **Reproducible Outputs**: Each run creates a unique, timestamped output directory containing all plots, logs, model files, and a copy of the config.
