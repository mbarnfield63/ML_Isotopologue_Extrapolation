import copy
import numpy as np
import torch
import torch.nn as nn
from torch import optim


def get_loss_function(config: dict, reduction="mean"):
    """
    Returns an instantiated loss function.
    Added 'reduction' param to support weighted training (requires reduction='none').
    """
    train_config = config.get("training", {})
    loss_name = train_config.get("loss_function", "SmoothL1Loss")

    # Map of names to classes
    LOSS_FUNCTIONS = {
        "SmoothL1Loss": nn.SmoothL1Loss,
        "MSELoss": nn.MSELoss,
        "L1Loss": nn.L1Loss,
        "HuberLoss": nn.HuberLoss,
    }

    if loss_name not in LOSS_FUNCTIONS:
        raise ValueError(
            f"Unknown loss function: {loss_name}. "
            f"Available options are: {list(LOSS_FUNCTIONS.keys())}"
        )

    # Instantiate the loss function
    loss_class = LOSS_FUNCTIONS[loss_name]

    # Handle Huber specific arg 'delta' if present in config, else default
    if loss_name == "HuberLoss" or loss_name == "SmoothL1Loss":
        return loss_class(reduction=reduction, beta=1.0)

    return loss_class(reduction=reduction)


# === Optimizer Factory ===
OPTIMIZERS = {
    "Adam": optim.Adam,
    "AdamW": optim.AdamW,
    "SGD": optim.SGD,
}


def get_optimizer(model: nn.Module, config: dict):
    """
    Reads the config and returns an instantiated optimizer.
    Defaults to Adam if not specified.
    """
    train_config = config.get("training", {})
    optim_name = train_config.get("optimizer", "Adam")
    learning_rate = float(train_config.get("learning_rate", 1e-3))
    weight_decay = float(train_config.get("weight_decay", 0.0))

    if optim_name not in OPTIMIZERS:
        raise ValueError(
            f"Unknown optimizer: {optim_name}. " f"Available: {list(OPTIMIZERS.keys())}"
        )

    optim_class = OPTIMIZERS[optim_name]
    return optim_class(model.parameters(), lr=learning_rate, weight_decay=weight_decay)


# === EarlyStopping Class ===
class EarlyStopping:
    """
    Handles early stopping logic.
    Monitors validation loss and stops training if it doesn't improve.
    """

    def __init__(self, patience: int = 10, delta: float = 0.0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
        """
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf

    def __call__(self, val_loss, model):
        """
        Call this method at the end of each validation epoch.
        Returns:
            bool: True if training should stop, False otherwise.
            model_state: The state dict of the best model found so far.
        """
        score = -val_loss  # We want to maximize the negative loss

        if self.best_score is None:
            self.best_score = score
            self.val_loss_min = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
        elif score < self.best_score + self.delta:
            # Loss did not improve (or not by enough)
            self.counter += 1
            print(f"  EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
            best_model_state = None  # Don't return a new best state
        else:
            # Loss improved
            self.best_score = score
            self.val_loss_min = val_loss
            self.counter = 0
            best_model_state = copy.deepcopy(model.state_dict())

        return self.early_stop, best_model_state


# === Core Training & Evaluation Utilities ===
def train(
    model: nn.Module,
    dataloader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    criterion: nn.Module,
    class_weights: torch.Tensor = None,
):
    """
    Training loop with optional class-based weighting.
    """
    model.train()
    total_loss = 0.0

    for batch in dataloader:
        # 1. Unpack Batch
        if len(batch) >= 4:
            X, y = batch[0].to(device), batch[1].to(device)
            # We need iso_idx (index 3) to look up weights
            iso_idx = batch[3].to(device)

            # Handle model inputs
            if len(batch) >= 4:
                mol_idx = batch[2].to(device)
                outputs = model(X, mol_idx, iso_idx)
            else:
                outputs = model(X)
        else:
            # Fallback for simple datasets
            X, y = batch[0].to(device), batch[1].to(device)
            outputs = model(X)
            iso_idx = None

        optimizer.zero_grad()

        # 2. Calculate Loss
        if class_weights is not None and iso_idx is not None:
            # Look up the weight for each sample in the batch
            # class_weights shape: [Num_Isos] -> batch_weights shape: [Batch_Size]
            batch_weights = class_weights[iso_idx]

            # Compute raw unreduced loss (vector of errors)
            raw_loss = criterion(outputs, y)

            # Apply weights manually and mean
            loss = (raw_loss * batch_weights).mean()
        else:
            # Standard unweighted loss
            loss = criterion(outputs, y)
            if loss.ndim > 0:  # If criterion was 'none' but no weights provided
                loss = loss.mean()

        loss.backward()
        optimizer.step()
        total_loss += float(loss.item())

    return total_loss / max(1, len(dataloader))


def evaluate(
    model: nn.Module,
    dataloader,
    device: torch.device,
    criterion: nn.Module,
):
    """
    Evaluation loop.
    Returns: average loss, RMSE, and MAE over the dataloader.
    """
    model.eval()
    total_loss = 0.0
    preds, trues = [], []
    with torch.no_grad():
        for batch in dataloader:
            if len(batch) == 4:
                X, y, mol_idx, iso_idx = batch
                X, y = X.to(device), y.to(device)
                mol_idx, iso_idx = mol_idx.to(device), iso_idx.to(device)
                outputs = model(X, mol_idx, iso_idx)
            else:
                X, y = batch[0].to(device), batch[1].to(device)
                outputs = model(X)

            loss = criterion(outputs, y)
            total_loss += float(loss.item())
            preds.append(outputs.view(-1).cpu().numpy())
            trues.append(y.view(-1).cpu().numpy())

    avg_loss = total_loss / max(1, len(dataloader))
    if not preds:
        return avg_loss, float("nan"), float("nan")

    y_pred = np.concatenate(preds)
    y_true = np.concatenate(trues)
    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    mae = float(np.mean(np.abs(y_pred - y_true)))
    return avg_loss, rmse, mae


def get_predictions(
    model: nn.Module,
    dataloader,
    device: torch.device,
):
    """
    Collects predictions and true values from a dataloader.
    Returns: y_true (N,), y_pred (N,)
    """
    model.eval()
    y_true_list, y_pred_list = [], []
    with torch.no_grad():
        for batch in dataloader:
            if len(batch) == 4:
                X, y, mol_idx, iso_idx = batch
                X, y = X.to(device), y.to(device)
                mol_idx, iso_idx = mol_idx.to(device), iso_idx.to(device)
                outputs = model(X, mol_idx, iso_idx)
            else:
                X, y = batch[0].to(device), batch[1].to(device)
                outputs = model(X)

            y_pred_list.append(outputs.view(-1).cpu().numpy())
            y_true_list.append(y.view(-1).cpu().numpy())

    y_true = np.concatenate(y_true_list) if y_true_list else np.array([])
    y_pred = np.concatenate(y_pred_list) if y_pred_list else np.array([])
    return y_true, y_pred
