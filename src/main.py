import argparse
import numpy as np
import pandas as pd
import os
import shutil
import time
import torch

# === Imports from other src modules ===
from .data_loader import load_data, MoleculeDataset
from .experiments import run_experiment
from .analysis import (
    run_post_processing,
    analyze_grouped_errors,
    get_feature_importance,
)

from .plotting import plot_all_results
from .utils import load_config, create_output_dir, setup_reproducibility


def main(config_path: str):
    """
    Main entry point for the training and evaluation pipeline.
    """
    start_time = time.time()

    # === 1. Load Config and Setup ===
    print(f"Loading configuration from: {config_path}")
    config = load_config(config_path)
    analysis_config = config.get("analysis", {})  # Get analysis config section

    output_dir = create_output_dir(config)
    shutil.copy(config_path, os.path.join(output_dir, "run_config.yml"))

    seed = config.get("experiment", {}).get("seed", 42)
    setup_reproducibility(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # === 2. Load Data ===
    print("\n" + "=" * 60)
    print("STEP 1: LOADING DATA")
    print("=" * 60)
    (
        train_df,
        val_df,
        test_df,
        feature_cols,
        target_col,
        scaler,
        n_molecules,
        n_isos
    ) = load_data(config)
    print(f"Loaded data with {len(feature_cols)} features.")
    print(f"Found {n_molecules} molecules and {n_isos} isotopologues.")

    # === Update Config with Model Params ===
    if 'params' not in config['model']:
        config['model']['params'] = {}
        
    config['model']['params']['n_molecules'] = n_molecules
    config['model']['params']['n_isos'] = n_isos

    # === 3. Run Experiment ===
    print("\n" + "=" * 60)
    print("STEP 2: RUNNING EXPERIMENT")
    print("=" * 60)
    results = run_experiment(
        config, train_df, val_df, test_df, feature_cols, target_col, device
    )

    # === 4. Post-Processing & Analysis ===
    print("\n" + "=" * 60)
    print("STEP 3: POST-PROCESSING & ANALYSIS")
    print("=" * 60)

    pred_df = results["test_predictions_df"]
    iso_results_df = pd.DataFrame()
    overall_metrics = {}

    if pred_df.empty:
        print("No predictions generated. Skipping analysis.")
    else:
        # --- Add baseline error columns for ALL workflows ---
        try:
            pred_df["Error"] = pred_df["y_true"] - pred_df["y_pred"]
            pred_df["Abs_Error"] = np.abs(pred_df["Error"])
        except Exception as e:
            print(f"Warning: Could not compute baseline error columns. Error: {e}")

        # --- 4a. Run Optional Post-Processing ---
        pp_config = analysis_config.get("post_processing", {})
        if pp_config.get("enabled", False):
            print("\nRunning post-processing...")
            pred_df = run_post_processing(pred_df, pp_config, scaler, config)

            if "Original_abs_error" in pred_df.columns:
                try:
                    mean_orig_mae = pred_df["Original_abs_error"].mean()
                    mean_corr_mae = pred_df["Corrected_abs_error"].mean()
                    with np.errstate(divide='ignore', invalid='ignore'):
                        pct_imp = 100 * (mean_orig_mae - mean_corr_mae) / mean_orig_mae
                        overall_metrics["overall_pct_improvement"] = pct_imp if np.isfinite(pct_imp) else 0.0
                except Exception as e:
                    print(f"  Could not calculate overall improvement: {e}")
        else:
            print("Skipping post-processing (not enabled in config).")

        # --- 4b. Run Optional Grouped Error Analysis ---
        iso_config = analysis_config.get("isotopologue_analysis", {})
        if iso_config.get("enabled", False):
            print("\nRunning grouped error analysis...")
            iso_results_df = analyze_grouped_errors(pred_df, iso_config, output_dir)
            print("\nGrouped Error Analysis Summary:")
            if iso_results_df is not None:
                print(iso_results_df)
        else:
            print("Skipping grouped error analysis (not enabled in config).")

        # --- 4c. Run Optional Feature Importance ---
        fi_config = analysis_config.get("feature_importance", {})
        if fi_config.get("enabled", False) and results["model"] and not test_df.empty:

            print("\nCalculating feature importance...")
            mol_col = config['data'].get('molecule_idx_col')
            iso_col = config['data'].get('iso_idx_col')
            
            test_ds = MoleculeDataset(test_df, feature_cols, target_col, mol_col, iso_col)
            test_loader = torch.utils.data.DataLoader(
                test_ds,
                batch_size=config.get("training", {}).get("batch_size", 128),
                shuffle=False,
            )

            feature_importance_df = get_feature_importance(
                results["model"],
                test_loader,
                device,
                feature_cols,
                metric=fi_config.get("metric", "mae"),
                output_dir=output_dir,
                plot_fi=fi_config.get("plot", True)
            )
            fi_path = os.path.join(output_dir, "CSVs", "feature_importance.csv")
            feature_importance_df.to_csv(fi_path, index=False)
            print(f"Feature importance saved to {fi_path}")
        else:
            print("Skipping feature importance (not enabled in config).")

    # === 5. Plotting (To be added) ===
    print("\n" + "=" * 60)
    print("STEP 4: PLOTTING RESULTS")
    print("=" * 60)
    if plot_all_results is not None:
        plot_all_results(
            results=results,
            pred_df=pred_df,
            iso_results_df=iso_results_df,
            config=config,
            overall_metrics=overall_metrics,
            output_dir=output_dir,
        )
    else:
        print("Skipping plotting (src.plotting module not found).")

    # === 6. Save Final Outputs ===
    print("\n" + "=" * 60)
    print("STEP 5: SAVING OUTPUTS")
    print("=" * 60)

    if not pred_df.empty:
        pred_path = os.path.join(output_dir, "CSVs", "final_predictions.csv")
        pred_df.to_csv(pred_path, index=False)
        print(f"Final predictions saved to {pred_path}")

    if results["model"]:
        model_path = os.path.join(output_dir, "model.pt")
        torch.save(results["model"].state_dict(), model_path)
        print(f"Model saved to {model_path}")

    if not results["cv_results_df"].empty:
        cv_summary_path = os.path.join(output_dir, "CSVs", "cv_summary_results.csv")
        results["cv_results_df"].to_csv(cv_summary_path, index=False)
        print(f"CV summary saved to {cv_summary_path}")

    end_time = time.time()
    print("\n" + "=" * 60)
    print(f"Workflow finished in {end_time - start_time:.2f} seconds.")
    print(f"All outputs saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run a molecular energy ML experiment."
    )
    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="Path to the experiment configuration YAML file.",
    )
    args = parser.parse_args()

    main(args.config_path)
