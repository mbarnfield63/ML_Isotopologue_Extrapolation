import numpy as np
import os
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.preprocessing import StandardScaler

from .plotting import plot_feature_importance


def run_post_processing(
    pred_df: pd.DataFrame, pp_config: dict, scaler: StandardScaler, full_config: dict
) -> pd.DataFrame:
    """
    Applies workflow-specific post-processing to the prediction dataframe.
    This function is "opt-in" via the config `analysis.post_processing.enabled`.

      a. Un-scales data
      b. Calculates corrected energies
      c. Computes detailed errors.

    Args:
        pred_df (pd.DataFrame): The dataframe with 'y_true' and 'y_pred'.
        pp_config (dict): The `analysis.post_processing` section of the config.
        scaler (StandardScaler): The scaler object fit on the training data.
        full_config (dict): The complete config object (to get `data.scaled_cols`).

    Returns:
        pd.DataFrame: The `pred_df` with new columns added.
    """

    # === 1. Get required column names from config ===
    true_energy_col = pp_config.get("true_energy_col")
    original_energy_col = pp_config.get("original_energy_col")
    nn_correction_col = pp_config.get("nn_correction_col", "NN_correction")

    # === 2. Validate that required columns exist ===
    required_cols = [true_energy_col, original_energy_col]
    missing_cols = [col for col in required_cols if col not in pred_df.columns]

    if missing_cols:
        print(
            f"  WARNING: Skipping post-processing. Missing required columns: {missing_cols}"
        )
        print("           (Check `analysis.post_processing` in config)")
        return pred_df

    # === 3. Un-scale data ===
    scaled_cols = full_config.get("data", {}).get("scaled_cols", []) or []
    valid_scaled_cols = [col for col in scaled_cols if col in pred_df.columns]

    # Check using hasattr to avoid crash if scaler not fitted
    if valid_scaled_cols and scaler is not None and hasattr(scaler, "mean_"):
        print(f"  Un-scaling {len(valid_scaled_cols)} columns for post-processing...")

        # Create a copy to avoid SettingWithCopyWarning
        pred_df_unscaled = pred_df.copy()
        pred_df_unscaled.loc[:, valid_scaled_cols] = scaler.inverse_transform(
            pred_df_unscaled[valid_scaled_cols]
        )
    else:
        print(
            "  No columns to un-scale (or scaler not fitted). Assuming data is already unscaled."
        )
        pred_df_unscaled = pred_df.copy()

    # === 4. Apply NN correction ===

    # y_pred is the NN's prediction of the *target*
    # which is the correction to apply
    pred_df_unscaled[nn_correction_col] = pred_df_unscaled["y_pred"]

    # E_IE_corrected = E_IE_original + (NN correction)
    pred_df_unscaled["E_IE_corrected"] = (
        pred_df_unscaled[original_energy_col] + pred_df_unscaled[nn_correction_col]
    )

    # === 5. Calculate errors against the true value ===
    # Original_error = True_Energy - Original_Calculated
    pred_df_unscaled["Original_error"] = (
        pred_df_unscaled[true_energy_col] - pred_df_unscaled[original_energy_col]
    )

    # Corrected_error = True_Energy - Corrected_Calculated
    pred_df_unscaled["Corrected_error"] = (
        pred_df_unscaled[true_energy_col] - pred_df_unscaled["E_IE_corrected"]
    )

    # Absolute errors
    pred_df_unscaled["Original_abs_error"] = np.abs(pred_df_unscaled["Original_error"])
    pred_df_unscaled["Corrected_abs_error"] = np.abs(
        pred_df_unscaled["Corrected_error"]
    )

    # Improvement metrics
    pred_df_unscaled["Error_reduction"] = (
        pred_df_unscaled["Original_abs_error"] - pred_df_unscaled["Corrected_abs_error"]
    )

    pred_df_unscaled["Error_reduction_pct"] = 100 * (
        pred_df_unscaled["Error_reduction"] / pred_df_unscaled["Original_abs_error"]
    )
    pred_df_unscaled["Error_reduction_pct"] = pred_df_unscaled[
        "Error_reduction_pct"
    ].replace([np.inf, -np.inf, np.nan], 0)

    print("  Post-processing complete. Added corrected energy and error columns.")
    return pred_df_unscaled


def analyze_grouped_errors(
    pred_df: pd.DataFrame, iso_config: dict, output_dir: str
) -> pd.DataFrame:
    """
    Groups errors by a specified column (usually by iso) and saves a report.

    It robustly reports on *either* the simple baseline error (y_true vs y_pred)
    or the detailed "IE" errors if post-processing was run.
    """

    group_by_col = iso_config.get("group_by_col", "iso")
    if group_by_col not in pred_df.columns:
        print(
            f"  WARNING: Skipping grouped analysis. Grouping column '{group_by_col}' not in dataframe."
        )
        return pd.DataFrame()

    # Check if post-processing was run by looking for its output columns
    has_pp_cols = (
        "Original_abs_error" in pred_df.columns
        and "Corrected_abs_error" in pred_df.columns
    )

    # Check if simple error columns exist (should be added in main.py)
    has_simple_error_cols = (
        "Error" in pred_df.columns and "Abs_Error" in pred_df.columns
    )

    if has_pp_cols:
        print(
            f"  Analyzing detailed (post-processed) errors, grouped by '{group_by_col}'."
        )
    elif has_simple_error_cols:
        print(
            f"  Analyzing simple errors (y_true vs y_pred), grouped by '{group_by_col}'."
        )
    else:
        print(
            f"  WARNING: No error columns found ('Error' or 'Original_error'). Skipping grouped analysis."
        )
        return pd.DataFrame()

    results = {}

    groups_to_analyze = iso_config.get("isos_of_interest", [])
    if not groups_to_analyze:  # If list is empty, get all unique groups
        groups_to_analyze = sorted(pred_df[group_by_col].unique())

    print(f"  Found {len(groups_to_analyze)} groups to analyze...")

    for group in groups_to_analyze:
        mask = pred_df[group_by_col] == group
        if mask.sum() == 0:
            continue

        group_data = {"Group": group, "Count": mask.sum()}

        if has_pp_cols:
            # Report on detailed "IE" workflow errors
            mean_orig_mae = pred_df.loc[mask, "Original_abs_error"].mean()
            mean_corr_mae = pred_df.loc[mask, "Corrected_abs_error"].mean()
            mean_error_reduction = mean_orig_mae - mean_corr_mae

            # This calculation is now robust to divide-by-zero or outliers
            with np.errstate(divide="ignore", invalid="ignore"):
                mean_pct_reduction = 100 * (mean_error_reduction / mean_orig_mae)
                if not np.isfinite(mean_pct_reduction):
                    mean_pct_reduction = 0.0  # Set to 0 if mean_orig_mae was 0

            group_data.update(
                {
                    "Original MAE": mean_orig_mae,
                    "ML Corrected MAE": mean_corr_mae,
                    "Original RMSE": np.sqrt(
                        np.mean(pred_df.loc[mask, "Original_error"] ** 2)
                    ),
                    "ML Corrected RMSE": np.sqrt(
                        np.mean(pred_df.loc[mask, "Corrected_error"] ** 2)
                    ),
                    "Mean Error Reduction": mean_error_reduction,
                    "Mean Pct Reduction": mean_pct_reduction,
                }
            )
        elif has_simple_error_cols:
            # Report on simple, baseline errors
            group_data.update(
                {
                    "MAE": pred_df.loc[mask, "Abs_Error"].mean(),
                    "RMSE": np.sqrt(np.mean(pred_df.loc[mask, "Error"] ** 2)),
                    "Mean Error": pred_df.loc[mask, "Error"].mean(),
                }
            )

        results[group] = group_data

    if not results:
        print("  No data found for any specified groups.")
        return pd.DataFrame()

    df = pd.DataFrame.from_dict(results, orient="index").reset_index(drop=True)

    # --- Define column order for consistent CSVs ---
    if has_pp_cols:
        cols_order = [
            "Group",
            "Count",
            "Original MAE",
            "ML Corrected MAE",
            "Original RMSE",
            "ML Corrected RMSE",
            "Mean Error Reduction",
            "Mean Pct Reduction",
        ]
    elif has_simple_error_cols:
        cols_order = ["Group", "Count", "MAE", "RMSE", "Mean Error"]
    else:
        cols_order = ["Group", "Count"]  # Fallback

    # Filter for columns that actually exist
    final_cols = [col for col in cols_order if col in df.columns]
    df = df[final_cols]

    report_path = os.path.join(output_dir, "CSVs", "grouped_error_report.csv")
    df.to_csv(report_path, index=False)
    print(f"  Grouped error report saved to {report_path}")

    return df


def get_feature_importance(
    model: torch.nn.Module,
    dataloader,
    device: torch.device,
    feature_cols: list[str],
    criterion: torch.nn.Module = None,
    metric: str = "mae",
    output_dir: str = None,
    plot_fi: bool = True,
):
    """
    Permutation feature importance.

    Measures the increase in a metric (MAE, RMSE) when permuting a feature.
    """
    assert metric in {"rmse", "mae"}, "Metric must be 'rmse', or 'mae'"
    model.eval()

    X_list, y_list = [], []
    mol_idx_list, iso_idx_list = [], []
    has_indices = False

    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, (list, tuple)):
                if len(batch) >= 4:
                    # Assuming X, y, mol_idx, iso_idx
                    X_batch, y_batch = batch[0], batch[1]
                    mol_idx_batch, iso_idx_batch = batch[2], batch[3]
                    mol_idx_list.append(mol_idx_batch)
                    iso_idx_list.append(iso_idx_batch)
                    has_indices = True
                elif len(batch) >= 2:
                    X_batch, y_batch = batch[0], batch[1]
            else:
                print("Warning: Unexpected batch structure")
                return pd.DataFrame()

            X_list.append(X_batch)
            y_list.append(y_batch)

    if not X_list:
        print("  WARNING: Dataloader empty, cannot compute feature importance.")
        return pd.DataFrame()

    # Concatenate all batches
    X_full = torch.cat(X_list, dim=0).to(device)
    y_full = torch.cat(y_list, dim=0).to(device)

    mol_idx_full = None
    iso_idx_full = None
    if has_indices:
        mol_idx_full = torch.cat(mol_idx_list, dim=0).to(device)
        iso_idx_full = torch.cat(iso_idx_list, dim=0).to(device)

        # === SAFETY CHECK: Validate Indices ===
        if (mol_idx_full < 0).any() or (iso_idx_full < 0).any():
            print("  ERROR: Negative indices detected! Cannot compute FI.")
            return pd.DataFrame()

        # Check upper bounds if possible (requires access to model internals)
        if hasattr(model, "mol_embed") and hasattr(model.mol_embed, "num_embeddings"):
            max_mol = model.mol_embed.num_embeddings
            if (mol_idx_full >= max_mol).any():
                print(
                    f"  ERROR: Molecule indices out of bounds (>= {max_mol})! Cannot compute FI."
                )
                return pd.DataFrame()

        if hasattr(model, "iso_embed") and hasattr(model.iso_embed, "num_embeddings"):
            max_iso = model.iso_embed.num_embeddings
            if (iso_idx_full >= max_iso).any():
                print(
                    f"  ERROR: Isotopologue indices out of bounds (>= {max_iso})! Cannot compute FI."
                )
                return pd.DataFrame()

    # Helper to call model with correct args
    def predict(input_x):
        if has_indices:
            return model(input_x, mol_idx_full, iso_idx_full)
        else:
            return model(input_x)

    with torch.no_grad():
        y_hat = predict(X_full)

    def compute_metric(y_true_t, y_pred_t):
        # For MAE/RMSE, use numpy for simplicity and robustness
        y_true_np = y_true_t.view(-1).cpu().numpy()
        y_pred_np = y_pred_t.view(-1).cpu().numpy()

        if metric == "rmse":
            return float(root_mean_squared_error(y_true_np, y_pred_np))
        else:  # mae
            return float(mean_absolute_error(y_true_np, y_pred_np))

    baseline_score = compute_metric(y_full, y_hat)
    print(f"  Feature importance baseline {metric.upper()}: {baseline_score:.6f}")

    g = torch.Generator(device=device)
    g.manual_seed(42)  # Using a fixed seed for permutations

    importance = {}

    with torch.no_grad():
        for j, feat in enumerate(feature_cols):
            X_perm = X_full.clone()
            idx = torch.randperm(X_perm.size(0), generator=g, device=device)
            X_perm[:, j] = X_perm[idx, j]

            y_perm = predict(X_perm)
            perm_score = compute_metric(y_full, y_perm)

            importance[feat] = perm_score - baseline_score

    df_imp = pd.DataFrame(
        {
            "feature": feature_cols,
            "importance": [importance[feat] for feat in feature_cols],
        }
    ).sort_values("importance", ascending=False)

    if output_dir and plot_fi:
        try:
            # Pass output_dir to the plotting function
            plot_feature_importance(df_imp, output_dir)
        except Exception as e:
            print(f"  WARNING: Failed to plot feature importance. Error: {e}")

    return df_imp
