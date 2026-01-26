import os
import numpy as np
import pandas as pd

INFERENCE_DIR = "./outputs/co_co2_cv_20260122_1607/"
OUTPUT_DIR = "./outputs/energy_levels/co/"
OUTPUT_COLS = ["v", "J"]  # not including energy level

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Import predictions
preds = pd.read_csv(os.path.join(INFERENCE_DIR, "Inference/inference_results.csv"))

# Create E_ML
preds["E_ML"] = preds["E_IE"] - preds["predicted_IE_correction"]

# Filter predictions for output columns only in order
preds_filtered = preds[OUTPUT_COLS + ["E_ML", "iso"]]

for iso in preds_filtered["iso"].unique():
    preds_iso = preds_filtered[preds_filtered["iso"] == iso]

    # Order by v, then J
    preds_iso = preds_iso.sort_values(by=["v", "J"]).reset_index(drop=True)

    # Make output cols integers
    for col in OUTPUT_COLS:
        preds_iso[col] = preds_iso[col].astype(int)

    # Drop iso column for output
    preds_iso = preds_iso.drop(columns=["iso"])

    output_path = os.path.join(OUTPUT_DIR, f"IE_ML_energy_levels_{iso}.csv")
    preds_iso.to_csv(output_path, index=False)
