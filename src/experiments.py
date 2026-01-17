import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler


# === Imports from other src modules ===
try:
    from .data_loader import MoleculeDataset, calculate_class_weights
    from .models import get_model
    from .training import (
        get_loss_function,
        get_optimizer,
        EarlyStopping,
        train,
        evaluate,
        get_predictions,
    )
except ImportError:
    print("Running as script, using standard imports.")
    from data_loader import MoleculeDataset, calculate_class_weights
    from models import get_model
    from training import (
        get_loss_function,
        get_optimizer,
        EarlyStopping,
        train,
        evaluate,
        get_predictions,
    )


def train_final_model(
    config: dict,
    full_train_df: pd.DataFrame,
    feature_cols: list,
    target_col: str,
    device: torch.device,
    epochs: int,
):
    """
    Trains a final model on the entire available dataset (Train + Val).
    """
    print(f"Training Final Model for fixed {epochs} epochs (derived from CV average)")

    # 1. Setup Data
    mol_col = config["data"].get("molecule_idx_col")
    iso_col = config["data"].get("iso_idx_col")

    scaler = StandardScaler()
    scaled_cols = config["data"].get("scaled_cols", []) or []
    valid_scaled_cols = [col for col in scaled_cols if col in feature_cols]

    if valid_scaled_cols:
        full_train_df = full_train_df.copy()  # Avoid SettingWithCopy
        full_train_df[valid_scaled_cols] = full_train_df[valid_scaled_cols].astype(
            float
        )
        full_train_df.loc[:, valid_scaled_cols] = scaler.fit_transform(
            full_train_df[valid_scaled_cols]
        )
        print(f"Final model: Scaled {len(valid_scaled_cols)} features.")

    train_ds = MoleculeDataset(
        full_train_df, feature_cols, target_col, mol_col, iso_col
    )

    sampler = None
    if config["data"].get("use_weighted_sampler", False):
        try:
            from .data_loader import get_weighted_sampler

            sampler = get_weighted_sampler(full_train_df, target_col)
            print("Using WeightedSampler for final refit.")
        except ImportError:
            print(
                "Warning: Could not import get_weighted_sampler. using random shuffle."
            )

    batch_size = config["training"].get("batch_size", 128)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=(sampler is None), sampler=sampler
    )

    # 2. Setup Model & Optimizer
    input_dim = len(feature_cols)
    model = get_model(config, input_dim).to(device)
    optimizer = get_optimizer(model, config)

    # === Loss Setup (Split Train vs Eval not needed here, but weighting is) ===
    # reduction='none' because manual weights might be applied in train()
    train_criterion = get_loss_function(config, reduction="none")

    loss_weights_tensor = None
    weight_config = config.get("weighting", {})
    if weight_config.get("enabled", False):
        iso_col_name = config["data"].get("iso_col", "iso")
        class_weights_dict = calculate_class_weights(
            full_train_df, iso_col_name, weight_config
        )

        iso_idx_col = config["data"].get("iso_idx_col", "iso_idx_encoded")
        if iso_idx_col in full_train_df.columns:
            idx_map = (
                full_train_df[[iso_idx_col, iso_col_name]]
                .drop_duplicates()
                .set_index(iso_idx_col)[iso_col_name]
            )
            max_idx = full_train_df[iso_idx_col].max()
            loss_weights_tensor = torch.ones(
                max_idx + 1, dtype=torch.float32, device=device
            )

            for idx, iso_label in idx_map.items():
                if iso_label in class_weights_dict:
                    loss_weights_tensor[idx] = class_weights_dict[iso_label]
            print("  Final Refit: Loss weights applied.")

    # Init Bias
    if config["training"].get("init_bias_to_mean", False):
        try:
            mean_target = float(
                np.mean(full_train_df[target_col].values.astype(np.float32))
            )
            model.init_output_bias(mean_target)
        except Exception:
            pass

    # 3. Training Loop
    model.train()
    for epoch in range(epochs):
        avg_loss = train(
            model,
            train_loader,
            optimizer,
            device,
            train_criterion,
            class_weights=loss_weights_tensor,
        )

        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            print(f"  Refit Epoch {epoch+1:3d}/{epochs} | Train Loss: {avg_loss:.6f}")

    print("Refit complete.")
    return model, scaler


def run_single_train_test(
    config: dict,
    model: nn.Module,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list,
    target_col: str,
    device: torch.device,
    train_sampler=None,
):
    """
    Runs a single train/validation/test experiment.

    Returns:
        tuple: (
            model (trained model),
            train_losses (list),
            val_losses (list),
            test_results (dict),
            test_preds_df (pd.DataFrame)
        )
    """
    train_config = config["training"]
    data_config = config["data"]
    batch_size = train_config.get("batch_size", 128)
    epochs = train_config.get("epochs", 100)

    # === Dataloaders ===
    mol_col = data_config.get("molecule_idx_col")
    iso_col = data_config.get("iso_idx_col")

    train_ds = MoleculeDataset(train_df, feature_cols, target_col, mol_col, iso_col)
    val_ds = MoleculeDataset(val_df, feature_cols, target_col, mol_col, iso_col)
    test_ds = MoleculeDataset(test_df, feature_cols, target_col, mol_col, iso_col)

    # Get WeightedSampler if specified in config
    sampler = train_sampler
    if sampler is None and data_config.get("use_weighted_sampler", False):
        from .data_loader import get_weighted_sampler

        sampler = get_weighted_sampler(train_df, target_col)
        print("Using WeightedSampler for training.")

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=(sampler is None), sampler=sampler
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    # 1. For Training: 'none' reduction to allow manual weighting in train()
    train_criterion = get_loss_function(config, reduction="none")
    # 2. For Evaluation: 'mean' reduction so evaluate() receives a scalar
    eval_criterion = get_loss_function(config, reduction="mean")

    optimizer = get_optimizer(model, config)
    print(f"Using Optimizer: {train_config.get('optimizer', 'Adam')}")
    print(f"Using Loss Function: {train_config.get('loss_function', 'SmoothL1Loss')}")

    # === Calculate Loss Weights ===
    loss_weights_tensor = None
    weight_config = config.get("weighting", {})
    if weight_config.get("enabled", False):
        iso_col_name = config["data"].get("iso_col", "iso")
        class_weights_dict = calculate_class_weights(
            train_df, iso_col_name, weight_config
        )

        iso_idx_col = config["data"].get("iso_idx_col", "iso_idx_encoded")
        if iso_idx_col in train_df.columns:
            idx_map = (
                train_df[[iso_idx_col, iso_col_name]]
                .drop_duplicates()
                .set_index(iso_idx_col)[iso_col_name]
            )
            max_idx = train_df[iso_idx_col].max()
            loss_weights_tensor = torch.ones(
                max_idx + 1, dtype=torch.float32, device=device
            )

            for idx, iso_label in idx_map.items():
                if iso_label in class_weights_dict:
                    loss_weights_tensor[idx] = class_weights_dict[iso_label]
            print("Loss weights calculated and applied.")

    # Scheduler & Early Stopping
    scheduler_config = train_config.get("scheduler", {})
    scheduler = None
    if scheduler_config.get("enabled", False):
        print(f"Using ReduceLROnPlateau Scheduler: {scheduler_config}")
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode=scheduler_config.get("mode", "min"),
            factor=scheduler_config.get("factor", 0.1),
            patience=scheduler_config.get("patience", 10),
            min_lr=float(scheduler_config.get("min_lr", 1e-6)),
        )

    early_stopping_config = train_config.get("early_stopping", {})
    early_stopper = None
    if early_stopping_config.get("enabled", False):
        patience = early_stopping_config.get("patience", 10)
        delta = float(early_stopping_config.get("delta", 1e-4))
        early_stopper = EarlyStopping(patience=patience, delta=delta)
        print(f"Early Stopping enabled: Patience={patience}, Delta={delta}")
    else:
        print("Early Stopping disabled.")

    # Bias initialization
    if train_config.get("init_bias_to_mean", False):
        try:
            train_targets = train_df[target_col].values.astype(np.float32)
            mean_target = float(np.mean(train_targets))
            model.init_output_bias(mean_target)
            print(f"Initialized final bias to train target mean = {mean_target:.6f}")
        except Exception as e:
            print(f"Warning: Could not initialize output bias. Error: {e}")

    print("\n" + "=" * 60)
    print(f"STARTING: Single Run ({config['run_name']})")
    print("=" * 60)

    train_losses, val_losses = [], []
    best_model_state = copy.deepcopy(model.state_dict())
    best_val_loss = np.inf
    epoch_stopped = epochs

    for epoch in range(epochs):
        # Use train_criterion (none) + weights
        train_loss = train(
            model,
            train_loader,
            optimizer,
            device,
            train_criterion,
            class_weights=loss_weights_tensor,
        )

        # Use eval_criterion (mean) -> returns scalar
        val_loss, val_rmse, val_mae = evaluate(
            model, val_loader, device, eval_criterion
        )

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if scheduler:
            scheduler.step(val_loss)

        # Log progress
        if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == epochs - 1:
            print(
                f"Epoch {epoch+1:3d}/{epochs} | "
                f"Train Loss: {train_loss:.6f} | "
                f"Val Loss: {val_loss:.6f} | "
                f"Val RMSE: {val_rmse:.6f} | "
                f"Val MAE: {val_mae:.6f}"
            )

        # Check for early stopping
        if early_stopper:
            stop_training, new_best_state = early_stopper(val_loss, model)
            if new_best_state:
                best_model_state = new_best_state
                best_val_loss = early_stopper.val_loss_min
            if stop_training:
                print(f"Early stopping triggered at epoch {epoch+1}")
                epoch_stopped = epoch + 1
                break
        else:
            # If no early stopping, save model on best val loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = copy.deepcopy(model.state_dict())

    if epoch_stopped == epochs:
        print("Training completed (all epochs).")

    # Load best model state
    model.load_state_dict(best_model_state)
    stopped_at_epoch = epoch_stopped - (early_stopper.counter if early_stopper else 0)
    print(
        f"\nLoaded best model state (from Epoch {stopped_at_epoch}, Val Loss: {best_val_loss:.6f}) for final testing."
    )

    # Final Test Evaluation
    print("\n" + "=" * 60)
    print("FINAL TEST PERFORMANCE:")
    test_loss, test_rmse, test_mae = evaluate(
        model, test_loader, device, eval_criterion
    )
    print(
        f"  Test Loss: {test_loss:.6f}\n  Test RMSE: {test_rmse:.6f}\n  Test MAE: {test_mae:.6f}"
    )
    print("=" * 60)

    test_results = {
        "test_loss": test_loss,
        "test_rmse": test_rmse,
        "test_mae": test_mae,
    }

    # Get predictions on test set for analysis
    y_true, y_pred = get_predictions(model, test_loader, device)

    # Create final predictions DataFrame, merging back all original columns
    pred_df = test_df.copy()
    pred_df["y_true"] = y_true
    pred_df["y_pred"] = y_pred

    return model, train_losses, val_losses, test_results, pred_df


def run_cross_validation(
    config: dict,
    full_df: pd.DataFrame,
    feature_cols: list,
    target_col: str,
    device: torch.device,
):
    """
    Runs a K-Fold cross-validation experiment on the given dataframe.

    This function uses a StratifiedKFold. This addresses the core project
    goal of ensuring that rare isotopologues (which are imbalanced classes)
    are fairly represented in every validation fold. This is more robust
    than simple KFold or random splitting.
    """
    exp_config = config["experiment"]
    train_config = config["training"]
    data_config = config["data"]
    k_folds = exp_config.get("k_folds", 5)
    seed = exp_config.get("seed", 42)
    batch_size = train_config.get("batch_size", 128)
    epochs = train_config.get("epochs", 50)

    fold_results, all_fold_preds = [], []

    # 1. For Training: 'none' reduction to allow manual weighting in train()
    train_criterion = get_loss_function(config, reduction="none")
    # 2. For Evaluation: 'mean' reduction so evaluate() receives a scalar
    eval_criterion = get_loss_function(config, reduction="mean")

    print("\n" + "=" * 60)
    print(f"STARTING: {k_folds}-Fold Cross-Validation ({config['run_name']})")
    print(f"Using Optimizer: {train_config.get('optimizer', 'Adam')}")
    print(f"Using Loss Function: {train_config.get('loss_function', 'SmoothL1Loss')}")

    # Stratification logic
    stratify_on_col = "iso"
    if stratify_on_col not in full_df.columns:
        print(f"WARNING: Stratification column '{stratify_on_col}' not found.")
        print(
            "... Falling back to standard KFold (results may be biased if data is imbalanced)."
        )
        kf = KFold(n_splits=k_folds, shuffle=True, random_state=seed)
        split_iterator = kf.split(full_df)
    else:
        print(f"Using StratifiedKFold, splitting on '{stratify_on_col}' column.")
        kf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=seed)
        # We pass the 'iso' column to the split method
        split_iterator = kf.split(full_df, full_df[stratify_on_col])

    # Index cols for dataloader
    mol_col = data_config.get("molecule_idx_col")
    iso_col = data_config.get("iso_idx_col")

    for fold, (train_idx, val_idx) in enumerate(split_iterator):
        print(f"\n=== Fold {fold+1}/{k_folds} ===")
        train_df_fold = full_df.iloc[train_idx].copy()
        val_df_fold = full_df.iloc[val_idx].copy()

        # Keep unscaled copy for reporting
        val_df_fold_unscaled = val_df_fold.copy()

        # Scaler must be fit inside the loop
        scaler = StandardScaler()
        scaled_cols = config["data"].get("scaled_cols", []) or []
        valid_scaled_cols = [col for col in scaled_cols if col in feature_cols]
        if valid_scaled_cols:
            # Cast all to float before scaling
            train_df_fold[valid_scaled_cols] = train_df_fold[valid_scaled_cols].astype(
                float
            )
            val_df_fold[valid_scaled_cols] = val_df_fold[valid_scaled_cols].astype(
                float
            )

            train_df_fold.loc[:, valid_scaled_cols] = scaler.fit_transform(
                train_df_fold[valid_scaled_cols]
            )
            val_df_fold.loc[:, valid_scaled_cols] = scaler.transform(
                val_df_fold[valid_scaled_cols]
            )

        # === Calculate Loss Weights for this Fold ===
        loss_weights_tensor = None
        weight_config = config.get("weighting", {})
        if weight_config.get("enabled", False):
            iso_col_name = config["data"].get("iso_col", "iso")
            class_weights_dict = calculate_class_weights(
                train_df_fold, iso_col_name, weight_config
            )

            iso_idx_col = config["data"].get("iso_idx_col", "iso_idx_encoded")
            if iso_idx_col in train_df_fold.columns:
                idx_map = (
                    train_df_fold[[iso_idx_col, iso_col_name]]
                    .drop_duplicates()
                    .set_index(iso_idx_col)[iso_col_name]
                )
                max_idx = train_df_fold[iso_idx_col].max()
                loss_weights_tensor = torch.ones(
                    max_idx + 1, dtype=torch.float32, device=device
                )

                for idx, iso_label in idx_map.items():
                    if iso_label in class_weights_dict:
                        loss_weights_tensor[idx] = class_weights_dict[iso_label]
                print(f"  Fold {fold+1}: Loss weights calculated.")

        train_ds = MoleculeDataset(
            train_df_fold, feature_cols, target_col, mol_col, iso_col
        )
        val_ds = MoleculeDataset(
            val_df_fold, feature_cols, target_col, mol_col, iso_col
        )

        # Get WeightedSampler if specified in config
        sampler = None
        if data_config.get("use_weighted_sampler", False):
            from .data_loader import get_weighted_sampler

            sampler = get_weighted_sampler(train_df_fold, target_col)
            print(f"  Fold {fold+1}: Using WeightedSampler for training.")

        pin_memory = device.type == "cuda"
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=(sampler is None),
            sampler=sampler,
            pin_memory=pin_memory,
        )
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False, pin_memory=pin_memory
        )

        # Get a fresh model and optimizer for each fold
        model = get_model(config, input_dim=len(feature_cols)).to(device)
        optimizer = get_optimizer(model, config)

        scheduler_config = train_config.get("scheduler", {})
        scheduler = None
        if scheduler_config.get("enabled", False):
            # Only print for first fold
            if fold == 0:
                print(f"Using ReduceLROnPlateau Scheduler: {scheduler_config}")
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=scheduler_config.get("factor", 0.1),
                patience=scheduler_config.get("patience", 10),
                min_lr=float(scheduler_config.get("min_lr", 1e-6)),
            )

        # Initialize Early Stopping for the fold
        early_stopping_config = train_config.get("early_stopping", {})
        early_stopper = None
        if early_stopping_config.get("enabled", False):
            patience = early_stopping_config.get("patience", 10)
            delta = float(early_stopping_config.get("delta", 1e-4))
            early_stopper = EarlyStopping(patience=patience, delta=delta)
            print(f"  Fold {fold+1}: Early Stopping enabled.")

        # Bias initialization
        if train_config.get("init_bias_to_mean", False):
            try:
                mean_target = float(
                    np.mean(train_df_fold[target_col].values.astype(np.float32))
                )
                model.init_output_bias(mean_target)
            except Exception as e:
                print(f"Warning: Could not initialize output bias. Error: {e}")

        best_model_state = copy.deepcopy(model.state_dict())
        best_val_loss = np.inf
        epoch_stopped = epochs

        for epoch in range(epochs):
            # TRAIN: Pass 'train_criterion' (none) + weights
            train_loss = train(
                model,
                train_loader,
                optimizer,
                device,
                train_criterion,
                class_weights=loss_weights_tensor,
            )

            # EVAL: Pass 'eval_criterion' (mean)
            val_loss, val_rmse, val_mae = evaluate(
                model, val_loader, device, eval_criterion
            )

            if scheduler:
                scheduler.step(val_loss)

            # Log ~5 times per fold
            if (epoch + 1) % (epochs // 5 or 1) == 0 or epoch == epochs - 1:
                print(
                    f"  Epoch {epoch+1:3d}/{epochs} | "
                    f"Train Loss: {train_loss:.6f} | "
                    f"Val Loss: {val_loss:.6f} | "
                    f"Val MAE: {val_mae:.6f}"
                )

            # Check for early stopping
            if early_stopper:
                stop_training, new_best_state = early_stopper(val_loss, model)
                if new_best_state:
                    best_model_state = new_best_state
                    best_val_loss = early_stopper.val_loss_min
                if stop_training:
                    print(
                        f"  Fold {fold+1}: Early stopping triggered at epoch {epoch+1}"
                    )
                    epoch_stopped = epoch + 1
                    break
            else:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = copy.deepcopy(model.state_dict())

        actual_stopped_epoch = epoch_stopped
        if early_stopper and early_stopper.early_stop:
            actual_stopped_epoch = epoch_stopped - early_stopper.counter

        # Final evaluation for this fold
        model.load_state_dict(best_model_state)
        val_loss, val_rmse, val_mae = evaluate(
            model, val_loader, device, eval_criterion
        )
        print(
            f"Fold {fold+1} Final | Val RMSE: {val_rmse:.6f} | Val MAE: {val_mae:.6f}"
        )

        fold_results.append(
            {
                "fold": fold + 1,
                "val_loss": val_loss,
                "val_rmse": val_rmse,
                "val_mae": val_mae,
                "stopped_epoch": actual_stopped_epoch,
            }
        )

        # Get predictions for this fold
        y_true, y_pred = get_predictions(model, val_loader, device)

        # Store predictions, merging back all original columns
        fold_pred_df = val_df_fold_unscaled.copy()
        fold_pred_df["fold"] = fold + 1
        fold_pred_df["y_true"] = y_true
        fold_pred_df["y_pred"] = y_pred
        all_fold_preds.append(fold_pred_df)

    # CV Summary
    cv_results_df = pd.DataFrame(fold_results)
    print("\n" + "=" * 60)
    print("CROSS-VALIDATION SUMMARY:")
    print(cv_results_df.to_string(index=False))
    print("-" * 60)
    print(
        f"Avg RMSE: {cv_results_df['val_rmse'].mean():.6f} ± {cv_results_df['val_rmse'].std():.6f}"
    )
    print(
        f"Avg MAE:  {cv_results_df['val_mae'].mean():.6f} ± {cv_results_df['val_mae'].std():.6f}"
    )
    print("=" * 60)

    cv_preds_df = pd.concat(all_fold_preds, ignore_index=True)
    return cv_results_df, cv_preds_df


# === Main Experiment Dispatcher ===
def run_experiment(
    config: dict,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list,
    target_col: str,
    device: torch.device,
    train_sampler=None,
):
    """
    Main dispatcher function.
    Reads the config and calls the appropriate experiment runner.
    """
    exp_type = config.get("experiment", {}).get("type", "single_run")
    input_dim = len(feature_cols)

    # Dictionary to hold all final results for main.py
    results = {
        "model": None,
        "train_losses": [],
        "val_losses": [],
        "test_results": {},
        "test_predictions_df": pd.DataFrame(),
        "cv_results_df": pd.DataFrame(),
        "cv_predictions_df": pd.DataFrame(),
    }

    if exp_type == "single_run":
        # --- Run a single Train/Val/Test experiment ---
        model = get_model(config, input_dim).to(device)

        (model, train_losses, val_losses, test_results, test_preds_df) = (
            run_single_train_test(
                config,
                model,
                train_df,
                val_df,
                test_df,
                feature_cols,
                target_col,
                device,
                train_sampler=train_sampler,
            )
        )
        results.update(
            {
                "model": model,
                "train_losses": train_losses,
                "val_losses": val_losses,
                "test_results": test_results,
                "test_predictions_df": test_preds_df,
            }
        )

    elif exp_type == "cv":
        # --- Run a K-Fold Cross-Validation experiment ---
        full_cv_df = pd.concat([train_df, val_df], ignore_index=True)
        print(f"Running CV on combined train+val splits (n={len(full_cv_df)}).")

        cv_results_df, cv_preds_df = run_cross_validation(
            config,
            full_cv_df,
            feature_cols,
            target_col,
            device,
        )

        # The main results from CV are the summary and the out-of-fold predictions
        results.update(
            {
                "cv_results_df": cv_results_df,
                "cv_predictions_df": cv_preds_df,
                # Use cv_preds for analysis, as it's the main output
                "test_predictions_df": cv_preds_df,
            }
        )

    elif exp_type == "multi_seed":
        # --- Run multiple 'single_run' experiments with different seeds ---
        print(f"Running Multi-Seed experiment... (Not yet fully implemented)")
        pass

    else:
        raise ValueError(
            f"Unknown experiment type: {exp_type}. "
            "Must be 'single_run', 'cv', or 'multi_seed'."
        )

    return results
