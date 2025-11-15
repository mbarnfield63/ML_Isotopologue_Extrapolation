import argparse
import numpy as np
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
from .training import get_loss_function
from .utils import load_config, create_output_dir, setup_reproducibility

# TODO from .plotting import plot_all_results


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
    (train_df, val_df, test_df, feature_cols, target_col, scaler) = load_data(config)
    print(f"Loaded data with {len(feature_cols)} features.")

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
            print("Running post-processing (as per config)...")
            pred_df = run_post_processing(pred_df, pp_config, scaler, config)
        else:
            print("Skipping post-processing (not enabled in config).")

        # --- 4b. Run Optional Grouped Error Analysis ---
        iso_config = analysis_config.get("isotopologue_analysis", {})
        if iso_config.get("enabled", False):
            print("Running grouped error analysis (as per config)...")
            iso_results_df = analyze_grouped_errors(pred_df, iso_config, output_dir)
            print("Grouped Error Analysis Summary:")
            if iso_results_df is not None:
                print(iso_results_df)
        else:
            print("Skipping grouped error analysis (not enabled in config).")

        # --- 4c. Run Optional Feature Importance ---
        fi_config = analysis_config.get("feature_importance", {})
        if fi_config.get("enabled", False) and results["model"] and not test_df.empty:

            print("Calculating feature importance...")
            test_ds = MoleculeDataset(test_df, feature_cols, target_col)
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
                criterion=get_loss_function(config),
                metric=fi_config.get("metric", "mae"),
            )
            fi_path = os.path.join(output_dir, "CSVs", "feature_importance.csv")
            feature_importance_df.to_csv(fi_path, index=False)
            print(f"Feature importance saved to {fi_path}")
        else:
            print("Skipping feature importance.")

    # === 5. Plotting (To be added) ===
    print("\n" + "=" * 60)
    print("STEP 4: PLOTTING RESULTS")
    print("=" * 60)
    # TODO: plot_all_results(results, pred_df, iso_results_df, output_dir)
    print("(Plotting functions to be ported to src/plotting.py)")

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
