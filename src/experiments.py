import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler


# === Imports from other src modules ===
try:
    from .data_loader import MoleculeDataset
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
    from data_loader import MoleculeDataset
    from models import get_model
    from training import (
        get_loss_function,
        get_optimizer,
        EarlyStopping,
        train,
        evaluate,
        get_predictions,
    )


# === Experiment Runners ===
def run_single_train_test(
    config: dict,
    model: nn.Module,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list,
    target_col: str,
    device: torch.device,
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
    batch_size = train_config.get("batch_size", 128)
    epochs = train_config.get("epochs", 100)

    # Dataloaders
    train_ds = MoleculeDataset(train_df, feature_cols, target_col)
    val_ds = MoleculeDataset(val_df, feature_cols, target_col)
    test_ds = MoleculeDataset(test_df, feature_cols, target_col)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    # Get criterion and optimizer from factories
    criterion = get_loss_function(config)
    optimizer = get_optimizer(model, config)
    print(f"Using Optimizer: {train_config.get('optimizer', 'Adam')}")
    print(f"Using Loss Function: {train_config.get('loss_function', 'SmoothL1Loss')}")

    # Initialize Early Stopping
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
    epoch_stopped = epochs  # Track which epoch we actually stop on

    for epoch in range(epochs):
        train_loss = train(model, train_loader, optimizer, device, criterion)
        val_loss, val_rmse, val_mae = evaluate(model, val_loader, device, criterion)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

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
    test_loss, test_rmse, test_mae = evaluate(model, test_loader, device, criterion)
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
    k_folds = exp_config.get("k_folds", 5)
    seed = exp_config.get("seed", 42)
    batch_size = train_config.get("batch_size", 128)
    epochs = train_config.get("epochs", 50)  # CV folds often use fewer epochs

    fold_results, all_fold_preds = [], []

    # Get non-feature columns to merge back for analysis
    non_feature_cols = [col for col in full_df.columns if col not in feature_cols]

    # Get loss function once
    criterion = get_loss_function(config)

    print("\n" + "=" * 60)
    print(f"STARTING: {k_folds}-Fold Cross-Validation ({config['run_name']})")
    print(f"Using Optimizer: {train_config.get('optimizer', 'Adam')}")
    print(f"Using Loss Function: {train_config.get('loss_function', 'SmoothL1Loss')}")

    # === CHANGED: Use StratifiedKFold ===
    # This is the key change to address the rare isotopologue problem.
    # We will stratify on the 'iso' column. This guarantees that each
    # fold has the same *proportion* of each isotopologue as the full dataset.

    # We assume the 'iso' column is used for stratification.
    # This column must be in the dataframe (i.e., in 'not_feature_cols').
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

    print("=" * 60)

    # === CHANGED ===
    # We now loop over the 'split_iterator' which is either KFold or StratifiedKFold
    for fold, (train_idx, val_idx) in enumerate(split_iterator):
        print(f"\n=== Fold {fold+1}/{k_folds} ===")
        train_df_fold = full_df.iloc[train_idx].copy()
        val_df_fold = full_df.iloc[val_idx].copy()

        # Scaler must be fit *inside* the loop
        scaler = StandardScaler()
        scaled_cols = config["data"].get("scaled_cols", []) or []
        valid_scaled_cols = [col for col in scaled_cols if col in feature_cols]
        if valid_scaled_cols:
            train_df_fold.loc[:, valid_scaled_cols] = scaler.fit_transform(
                train_df_fold[valid_scaled_cols]
            )
            val_df_fold.loc[:, valid_scaled_cols] = scaler.transform(
                val_df_fold[valid_scaled_cols]
            )

        # Dataloaders for this fold
        train_ds = MoleculeDataset(train_df_fold, feature_cols, target_col)
        val_ds = MoleculeDataset(val_df_fold, feature_cols, target_col)
        pin_memory = device.type == "cuda"
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, pin_memory=pin_memory
        )
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False, pin_memory=pin_memory
        )

        # Get a fresh model and optimizer for each fold
        model = get_model(config, input_dim=len(feature_cols)).to(device)
        optimizer = get_optimizer(model, config)

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
            train_loss = train(model, train_loader, optimizer, device, criterion)
            val_loss, val_rmse, val_mae = evaluate(model, val_loader, device, criterion)

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

        if epoch_stopped == epochs:
            print(f"  Fold {fold+1}: Training completed (all epochs).")

        # Load best model state for this fold
        model.load_state_dict(best_model_state)

        # Final evaluation for this fold on its validation set
        val_loss, val_rmse, val_mae = evaluate(model, val_loader, device, criterion)
        print(
            f"Fold {fold+1} Final | Val RMSE: {val_rmse:.6f} | Val MAE: {val_mae:.6f}"
        )
        fold_results.append(
            {
                "fold": fold + 1,
                "val_loss": val_loss,
                "val_rmse": val_rmse,
                "val_mae": val_mae,
                "stopped_epoch": epoch_stopped,
            }
        )

        # Get predictions for this fold
        y_true, y_pred = get_predictions(model, val_loader, device)

        # Store predictions, merging back non-feature columns
        fold_pred_df = val_df_fold[non_feature_cols].copy()
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

        # We perform CV on the combined train + val sets.
        full_cv_df = pd.concat([train_df, val_df], ignore_index=True)
        print(f"Running CV on combined train+val splits (n={len(full_cv_df)}).")
        if not test_df.empty:
            print(
                f"Held-out test set (n={len(test_df)}) is not used by this version of CV."
            )

        cv_results_df, cv_preds_df = run_cross_validation(
            config, full_cv_df, feature_cols, target_col, device
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
        # Note: CV does not produce one single 'model' or 'test_results'
        # The 'cv_results_df' (avg scores) is the key performance metric.

    elif exp_type == "multi_seed":
        # --- Run multiple 'single_run' experiments with different seeds ---
        print(f"Running Multi-Seed experiment... (Not yet fully implemented)")
        # This is where your original idea of "running an 80:20 split 5 times"
        # would be implemented. It's now called 'multi_seed' to be clear.

        # TODO: This would involve:
        # 1. Reading config['experiment']['seeds'] (e.g., [42, 43, 44])
        # 2. Looping over each seed.
        # 3. In the loop, re-split the data *from the beginning*
        #    using the new seed. This means `load_data` would need to
        #    accept a seed override.
        # 4. Get a new model, and call run_single_train_test.
        # 5. Aggregate all 'test_results' and 'test_predictions_df'
        #    into lists in the 'results' dictionary.
        pass

    else:
        raise ValueError(
            f"Unknown experiment type: {exp_type}. "
            "Must be 'single_run', 'cv', or 'multi_seed'."
        )

    return results
