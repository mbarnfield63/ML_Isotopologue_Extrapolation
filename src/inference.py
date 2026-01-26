import os
import pandas as pd
import numpy as np
import torch
import traceback

# === Imports from other src modules ===
from .data_loader import MoleculeDataset


def run_inference_pipeline(
    config: dict,
    model: torch.nn.Module,
    scaler,
    feature_cols: list,
    target_col: str,
    device: torch.device,
    base_output_dir: str,
    mol_map: dict,
    iso_map: dict,
):
    """
    Orchestrates the full inference pipeline:
    1. Loads external data from CSV.
    2. Validates that required feature columns exist.
    3. Scales the data using the training scaler.
    4. Runs the model in evaluation mode.
    5. Saves predictions alongside original data to a new CSV.
    """

    # 1. Check Configuration
    inference_config = config.get("inference", {})
    if not inference_config.get("enabled", False):
        return

    inf_data_path = inference_config.get("data_path")
    if not inf_data_path or not os.path.exists(inf_data_path):
        print(f"Error: Inference data not found at: {inf_data_path}")
        print("Please check the 'inference.data_path' in your config file.")
        return

    print(f"Loading inference data from: {inf_data_path}")

    try:
        # 2. Load Data
        inf_df = pd.read_csv(inf_data_path)

        # 3. Validate Features
        # Ensure all columns the model was trained on are present in the new file
        missing_cols = [c for c in feature_cols if c not in inf_df.columns]
        if missing_cols:
            raise ValueError(
                f"Inference file is missing {len(missing_cols)} required feature columns: {missing_cols}"
            )

        # 4. Apply Scaling
        # Leave original DataFrame unscaled/readable
        model_input_df = inf_df.copy()

        if scaler:
            print("Applying feature scaling (using scaler fitted on training data)...")

            # Use the features the scaler saw during fit()
            if hasattr(scaler, "feature_names_in_"):
                cols_to_scale = scaler.feature_names_in_
                print(
                    f"  Scaling features based on fitted names: {list(cols_to_scale)}"
                )
                model_input_df[cols_to_scale] = scaler.transform(
                    model_input_df[cols_to_scale]
                )
            else:
                # Fallback: Use config defined columns if scaler doesn't store names
                scaled_cols = config["data"].get("scaled_cols", [])
                if scaled_cols:
                    print(f"  Scaling features based on config: {scaled_cols}")
                    model_input_df[scaled_cols] = scaler.transform(
                        model_input_df[scaled_cols]
                    )
                else:
                    print(
                        "  Warning: Scaler provided but no columns identified for scaling."
                    )

        # Apply Molecule/Iso Mappings
        print("Mapping molecule and isotopologue indices...")

        # Get raw column names from config (e.g., "molecule", "iso")
        raw_mol_col = config["data"].get("molecule_col", "molecule")
        raw_iso_col = config["data"].get("iso_col", "iso")

        # Get target encoded column names (e.g., "molecule_idx_encoded")
        target_mol_idx = config["data"].get("molecule_idx_col", "molecule_idx_encoded")
        target_iso_idx = config["data"].get("iso_idx_col", "iso_idx_encoded")

        # Map Molecules
        if mol_map and raw_mol_col in model_input_df.columns:
            # map the strings to integers using the dictionary
            model_input_df[target_mol_idx] = (
                model_input_df[raw_mol_col].astype(str).map(mol_map)
            )

            # Check for unknown molecules (NaNs after mapping)
            if model_input_df[target_mol_idx].isnull().any():
                print(
                    f"Warning: Found molecules in inference data not present in training! Filling with 0."
                )
                model_input_df[target_mol_idx] = (
                    model_input_df[target_mol_idx].fillna(0).astype(int)
                )
            else:
                model_input_df[target_mol_idx] = model_input_df[target_mol_idx].astype(
                    int
                )

        elif target_mol_idx not in model_input_df.columns:
            # Fallback if no map provided
            model_input_df[target_mol_idx] = 0

        # Map Isotopologues
        if iso_map and raw_iso_col in model_input_df.columns:
            model_input_df[target_iso_idx] = (
                model_input_df[raw_iso_col].astype(str).map(iso_map)
            )

            if model_input_df[target_iso_idx].isnull().any():
                print(
                    f"Warning: Found isotopologues in inference data not present in training! Filling with 0."
                )
                model_input_df[target_iso_idx] = (
                    model_input_df[target_iso_idx].fillna(0).astype(int)
                )
            else:
                model_input_df[target_iso_idx] = model_input_df[target_iso_idx].astype(
                    int
                )

        elif target_iso_idx not in model_input_df.columns:
            model_input_df[target_iso_idx] = 0

        # 5. Handle Target Column
        # MoleculeDataset expects a target column to exist. If missing, fill with dummy values (0.0).
        if target_col not in model_input_df.columns:
            model_input_df[target_col] = 0.0

        # 6. Prepare Dataset & Loader
        mol_idx_col = config["data"].get("molecule_idx_col")
        iso_idx_col = config["data"].get("iso_idx_col")

        # Create Dataset
        inf_ds = MoleculeDataset(
            model_input_df, feature_cols, target_col, mol_idx_col, iso_idx_col
        )

        # Create Loader
        batch_size = config.get("training", {}).get("batch_size", 128)
        inf_loader = torch.utils.data.DataLoader(
            inf_ds, batch_size=batch_size, shuffle=False
        )

        # 7. Run Prediction Loop
        model.eval()
        inference_preds = []
        print(f"Running inference on {len(inf_df)} samples...")

        with torch.no_grad():
            for batch in inf_loader:
                features = batch[0].to(device)

                # Check if model requires molecule/isotope indices (e.g. for embeddings)
                if len(batch) >= 4:
                    mol_idxs = batch[2].to(device)
                    iso_idxs = batch[3].to(device)
                    preds = model(features, mol_idxs, iso_idxs)
                else:
                    preds = model(features)

                inference_preds.append(preds.cpu().numpy())

        # 8. Process & Save Results
        if inference_preds:
            # Flatten list of arrays into a single 1D array
            inference_preds = np.concatenate(inference_preds).flatten()

            # Attach predictions to the ORIGINAL dataframe
            inf_df["predicted_IE_correction"] = inference_preds

            # Determine Output Directory
            inf_out_dir = inference_config.get("output_dir")
            if not inf_out_dir:
                # Default: Create an 'Inference' folder inside the main output directory
                inf_out_dir = os.path.join(base_output_dir, "Inference")

            if not os.path.exists(inf_out_dir):
                os.makedirs(inf_out_dir)

            # Generate Output Filename
            out_filename = "inference_results.csv"
            out_path = os.path.join(inf_out_dir, out_filename)

            inf_df.to_csv(out_path, index=False)
            print(f"Success! Inference predictions saved to: {out_path}")

            if config.get("inference", {}).get("analysis", False):
                print("Running inference analysis...")
                from plotting import (
                    plot_inference_results_individual,
                    plot_inference_results_all,
                )
                from analysis import summarize_inference_results

                plot_inference_results_individual(inf_df, inf_out_dir)
                plot_inference_results_all(inf_df, inf_out_dir)

                summarize_inference_results(inf_df, inf_out_dir)

        else:
            print("Warning: No predictions were generated.")

    except Exception as e:
        print("\n" + "!" * 60)
        print(f"CRITICAL ERROR DURING INFERENCE: {e}")
        print("!" * 60)
        traceback.print_exc()
        print("Skipping inference due to error.")
