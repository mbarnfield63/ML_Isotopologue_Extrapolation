import numpy as np
import pandas as pd
import os

DIR = "data/processed"
INPUTS = ["CO_minor_isos_ma.txt", "CO2_minor_isos_ma.txt"]
OUTPUT = "co_co2_training.csv"
E_CUTOFF = None  # cm^-1
REMOVE_MOLECULES = []  # e.g., ["CO2"]


def combine_datasets(input_files, output_file):
    """
    Combines multiple CSV datasets into a single dataset by concatenating them row-wise.

    Parameters:
    input_files (list of str): List of file paths to the input CSV files.
    output_file (str): File path for the output combined CSV file.
    """
    all_dfs = []
    for file in input_files:
        if os.path.exists(file):
            df = pd.read_csv(file)

            # --- Auto-infer molecule column ---
            if "molecule" not in df.columns:
                # infer molecule name from the first input filename
                filename = os.path.basename(file)
                inferred_mol = filename.split("_")[0]

                df["molecule"] = inferred_mol
                print(
                    f"  > Column 'molecule' not found. Inferred '{inferred_mol}' from filename."
                )

            all_dfs.append(df)
        else:
            print(f"Warning: {file} does not exist and will be skipped.")
    if not all_dfs:
        print("No valid input files found. Creating empty output.")
        pd.DataFrame().to_csv(output_file, index=False)
        return

    all_cols_list = sorted({col for df in all_dfs for col in df.columns})

    if len(all_dfs) > 1:
        print("Combining datasets...")
        processed_dfs = []
        for df in all_dfs:
            # Re-index to include all columns, fill missing with 0.0
            df = df.reindex(columns=all_cols_list, fill_value=0.0)
            processed_dfs.append(df)

        combined_df = pd.concat(processed_dfs, ignore_index=True)
        print(
            f"Combined dataset created with {len(combined_df)} rows, {len(all_cols_list)} columns."
        )
    else:
        # Single dataset: ensure it has all columns (fill missing with 0.0)
        combined_df = all_dfs[0].reindex(columns=all_cols_list, fill_value=0.0)
        print(f"Loaded single dataset with {len(combined_df)} rows.")

    # Remove lines where E_Ma_iso > 40000
    if E_CUTOFF is not None and "E_Ma_parent" in combined_df.columns:
        initial_count = len(combined_df)
        combined_df = combined_df[combined_df["E_Ma_parent"] <= E_CUTOFF]
        filtered_count = len(combined_df)
        print(
            f"Filtered out {initial_count - filtered_count} rows where E_Ma_parent > {E_CUTOFF}."
        )

    # Remove specified molecules
    if REMOVE_MOLECULES and "molecule" in combined_df.columns:
        initial_count = len(combined_df)
        combined_df = combined_df[~combined_df["molecule"].isin(REMOVE_MOLECULES)]
        filtered_count = len(combined_df)
        print(
            f"Filtered out {initial_count - filtered_count} rows for molecules: {REMOVE_MOLECULES}."
        )
        print(f"Remaining molecules: {combined_df['molecule'].unique().tolist()}")

    combined_df.to_csv(output_file, index=False)
    print(f"Combined dataset saved to {output_file}")


if __name__ == "__main__":
    input_paths = [os.path.join(DIR, fname) for fname in INPUTS]
    output_path = os.path.join(DIR, OUTPUT)
    combine_datasets(input_paths, output_path)
