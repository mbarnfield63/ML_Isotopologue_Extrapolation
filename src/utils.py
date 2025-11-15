import numpy as np
import os
import random
import time
import torch
import yaml


def load_config(config_path: str) -> dict:
    """
    Loads a YAML config file.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def create_output_dir(config: dict) -> str:
    """
    Creates a unique, timestamped output directory for the experiment.

    Example: outputs/co2_basic_20251114_163045/
    """
    base_output_dir = config.get("output_dir", "outputs")
    run_name = config.get("run_name", "experiment")
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    output_dir = os.path.join(base_output_dir, f"{run_name}_{timestamp}")

    # Create all necessary subdirectories
    os.makedirs(os.path.join(output_dir, "CSVs"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "Plots"), exist_ok=True)

    return output_dir


def setup_reproducibility(seed: int):
    """
    Sets random seeds for torch, numpy, and random for reproducible results.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Reproducibility seed set to {seed}.")
