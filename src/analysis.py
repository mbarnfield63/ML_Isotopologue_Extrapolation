import numpy as np
import os
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.preprocessing import StandardScaler

# TODO from .plotting import plot_feature_importance


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
    # We must un-scale *before* calculations
    scaled_cols = full_config.get("data", {}).get("scaled_cols", []) or []
    valid_scaled_cols = [col for col in scaled_cols if col in pred_df.columns]

    if valid_scaled_cols and scaler is not None and scaler.n_features_in_ > 0:
        print(f"  Un-scaling {len(valid_scaled_cols)} columns for post-processing...")

        # Create a copy to avoid SettingWithCopyWarning
        pred_df_unscaled = pred_df.copy()
        pred_df_unscaled.loc[:, valid_scaled_cols] = scaler.inverse_transform(
            pred_df_unscaled[valid_scaled_cols]
        )
    else:
        print("  No columns to un-scale or scaler not provided.")
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

    if has_pp_cols:
        print(
            f"  Analyzing detailed (post-processed) errors, grouped by '{group_by_col}'."
        )
    else:
        print(
            f"  Analyzing simple errors (y_true vs y_pred), grouped by '{group_by_col}'."
        )

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
            group_data.update(
                {
                    "Original MAE": pred_df.loc[mask, "Original_abs_error"].mean(),
                    "ML Corrected MAE": pred_df.loc[mask, "Corrected_abs_error"].mean(),
                    "Original RMSE": np.sqrt(
                        np.mean(pred_df.loc[mask, "Original_error"] ** 2)
                    ),
                    "ML Corrected RMSE": np.sqrt(
                        np.mean(pred_df.loc[mask, "Corrected_error"] ** 2)
                    ),
                    "Mean Error Reduction": pred_df.loc[mask, "Error_reduction"].mean(),
                    "Mean Pct Reduction": pred_df.loc[
                        mask, "Error_reduction_pct"
                    ].mean(),
                }
            )
        else:
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

    df = pd.DataFrame.from_dict(results, orient="index")

    report_path = os.path.join(output_dir, "CSVs", "grouped_error_report.csv")
    df.to_csv(report_path, index=False)
    print(f"  Grouped error report saved to {report_path}")

    return df


def get_feature_importance(
    model: torch.nn.Module,
    dataloader,
    device: torch.device,
    feature_cols: list[str],
    criterion: torch.nn.Module,
    metric: str = "mae",
):
    """
    Permutation feature importance.

    Measures the increase in a metric (MAE, RMSE) when permuting a feature.
    """
    assert metric in {"rmse", "mae"}, "Metric must be 'rmse', or 'mae'"
    model.eval()

    X_list, y_list = [], []
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_list.append(X_batch)
            y_list.append(y_batch)

    if not X_list:
        print("  WARNING: Dataloader empty, cannot compute feature importance.")
        return pd.DataFrame()

    X_full = torch.cat(X_list, dim=0).to(device)
    y_full = torch.cat(y_list, dim=0).to(device)

    with torch.no_grad():
        y_hat = model(X_full)

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

            y_perm = model(X_perm)
            perm_score = compute_metric(y_full, y_perm)

            importance[feat] = perm_score - baseline_score

    df_imp = pd.DataFrame(
        {
            "feature": feature_cols,
            "importance": [importance[feat] for feat in feature_cols],
        }
    ).sort_values("importance", ascending=False)

    # TODO: Call plot_feature_importance here
    # if output_dir:
    #     plot_feature_importance(df_imp, output_dir)

    return df_imp
