from concurrent.futures import ProcessPoolExecutor
import functools
import glob
import numpy as np
import os
import pandas as pd
import sys


# Import mass database
try:
    import isotopologue_masses as iso_db
except ImportError:
    print(
        "Error: isotopologue_masses.py not found. Please ensure it is in the same directory."
    )
    sys.exit(1)

# --- Configuration ---
PARENT_ISO = 626
INPUT_DIR = "data/co2"
OUTPUT_DIR = "data/processed"
NUM_WORKERS = 6
INF_MAX = 15000  # cm^-1 (max inference energy)


# --- Feature Engineering Functions ---
def calc_mu1(row):
    """Symmetric stretch: average of reduced masses for each side."""
    m_o1, m_c, m_o2 = row["mass_o_1"], row["mass_c"], row["mass_o_2"]
    side1 = (m_o1 * m_c) / (m_o1 + m_c)
    side2 = (m_o2 * m_c) / (m_o2 + m_c)
    return (side1 + side2) / 2


def calc_mu2(row):
    """Bend: reduced mass for just oxygens."""
    m_o1, m_o2 = row["mass_o_1"], row["mass_o_2"]
    return (m_o1 * m_o2) / (m_o1 + m_o2)


def calc_mu3(row):
    """Asymmetric stretch: combined mu for both sides."""
    m_o1, m_c, m_o2 = row["mass_o_1"], row["mass_c"], row["mass_o_2"]
    return ((m_o1 + m_o2) * m_c) / (m_o1 + m_o2 + m_c)


def calc_mu_all(row):
    """Combined reduced mass for all three atoms."""
    m_o1, m_c, m_o2 = row["mass_o_1"], row["mass_c"], row["mass_o_2"]
    return (m_o1 * m_c * m_o2) / (m_o1 + m_c + m_o2)


def prepare_atomic_features(df, parent_mus):
    """
    Adds mass-based features, ratios to parent, and one-hot encodings.
    """
    print("   Computing atomic mass features...")

    # 1. Map Masses using isotopologue_masses.py
    # Returns [mass_o1, mass_c, mass_o2] for CO2
    try:
        mass_data = df["iso"].apply(lambda x: iso_db.get_mass_list(x))
    except ValueError as e:
        print(f"Error mapping masses: {e}")
        sys.exit(1)

    df[["mass_o_1", "mass_c", "mass_o_2"]] = pd.DataFrame(
        mass_data.tolist(), index=df.index
    )

    # 2. Calculate Reduced Masses
    df["mu1"] = df.apply(calc_mu1, axis=1)
    df["mu2"] = df.apply(calc_mu2, axis=1)
    df["mu3"] = df.apply(calc_mu3, axis=1)
    df["mu_all"] = df.apply(calc_mu_all, axis=1)

    # 3. Calculate Ratios (using parent mu values passed in)
    df["mu1_ratio"] = df["mu1"] / parent_mus["mu1"]
    df["mu2_ratio"] = df["mu2"] / parent_mus["mu2"]
    df["mu3_ratio"] = df["mu3"] / parent_mus["mu3"]
    df["mu_all_ratio"] = df["mu_all"] / parent_mus["mu_all"]

    # 4. One-Hot Encoding for Masses
    mass_cols = ["mass_c", "mass_o_1", "mass_o_2"]
    for col in mass_cols:
        if col in df.columns:
            one_hot = pd.get_dummies(df[col], prefix=col)
            df = pd.concat([df, one_hot], axis=1)
            df.drop(columns=[col], inplace=True)

    # 5. One-Hot Encoding for Symmetries
    if "e_f" in df.columns:
        # Ensure boolean/int representation
        ef_one_hot = pd.get_dummies(df["e_f"]).astype(int)
        df = pd.concat([df, ef_one_hot], axis=1)
        df.drop(columns=["e_f"], inplace=True)

    if "tot_sym" in df.columns:
        sym_one_hot = pd.get_dummies(df["tot_sym"], prefix="Sym")
        df = pd.concat([df, sym_one_hot], axis=1)
        df.drop(columns=["tot_sym"], inplace=True)

    return df


def generate_match_key(df):
    """
    Generates a tuple key for matching quantum states.
    Using Trove columns + J + e_f + l2 from Herzberg.
    """
    # Matches "Trove" definition in analyze_assignments.py
    key_cols = ["J", "e_f", "hzb_l2", "Trove_v1", "Trove_v2", "Trove_v3"]
    # Create tuple key
    return df[key_cols].apply(tuple, axis=1)


# --- Parsing Functions ---
def load_file_pair(iso, input_dir):
    """
    Loads _ma.csv (matched/MARVEL) and _ca.csv (calc only) for a given iso.
    """
    ma_path = os.path.join(input_dir, f"CO2_{iso}_ma.csv")
    ca_path = os.path.join(input_dir, f"CO2_{iso}_ca.csv")

    matched = pd.DataFrame()
    ca_only = pd.DataFrame()

    # Load Matched (Training Candidates)
    if os.path.exists(ma_path):
        matched = pd.read_csv(ma_path, sep=",")
        matched["iso"] = iso
        # Rename E to E_Ma for consistency with CO script
        if "E" in matched.columns:
            matched.rename(columns={"E": "E_Ma"}, inplace=True)

        # Safety: Remove duplicate columns
        matched = matched.loc[:, ~matched.columns.duplicated()]

    # Load Ca-Only (Inference Candidates)
    if os.path.exists(ca_path):
        ca_only = pd.read_csv(ca_path, sep=",")
        ca_only["iso"] = iso

        # Rename E to E_Ca for inference data to match pipeline expectations
        if "E" in ca_only.columns:
            ca_only.rename(columns={"E": "E_Ca"}, inplace=True)

        # Ensure E_Ma is NaN for these
        ca_only["E_Ma"] = np.nan

        # Safety: Remove duplicate columns
        ca_only = ca_only.loc[:, ~ca_only.columns.duplicated()]

    return matched, ca_only


# Parallelized functions
def process_iso_files(iso, input_dir):
    """
    Worker function for Step 1.
    Loads files and generates keys for a single isotopologue.
    """
    print(f"   Processing ISO {iso}...")
    matched, ca_only = load_file_pair(iso, input_dir)

    # Generate Match Keys immediately so we don't have to do it later
    if not matched.empty:
        matched["match_key"] = generate_match_key(matched)
    if not ca_only.empty:
        ca_only["match_key"] = generate_match_key(ca_only)

    return iso, matched, ca_only


def process_minor_iso_features(iso, matched_df, ca_only_df, lookup_ma, lookup_ca):
    """
    Worker function for Step 3.
    Calculates features for a minor isotopologue using parent lookups.
    """
    train_result = None
    inf_result = None

    # --- A. Process Training Data ---
    if not matched_df.empty:
        df_train = matched_df.copy()
        # Map Parent Info
        df_train["E_Ma_parent"] = df_train["match_key"].map(lookup_ma)
        df_train["E_Ca_parent"] = df_train["match_key"].map(lookup_ca)

        # Rename and Clean
        df_train.rename(columns={"E_Ca": "E_Ca_iso", "E_Ma": "E_Ma_iso"}, inplace=True)
        df_train.dropna(subset=["E_Ma_parent", "E_Ca_parent"], inplace=True)

        # Calculate IE
        df_train["E_IE"] = (
            df_train["E_Ca_iso"] + df_train["E_Ma_parent"] - df_train["E_Ca_parent"]
        )
        df_train["Error_IE"] = df_train["E_Ma_iso"] - df_train["E_IE"]

        train_result = df_train

    # --- B. Process Inference Data ---
    if not ca_only_df.empty:
        df_inf = ca_only_df.copy()
        # Map Parent Info
        df_inf["E_Ma_parent"] = df_inf["match_key"].map(lookup_ma)
        df_inf["E_Ca_parent"] = df_inf["match_key"].map(lookup_ca)

        # Rename and Clean
        df_inf.rename(columns={"E_Ca": "E_Ca_iso", "E_Ma": "E_Ma_iso"}, inplace=True)
        df_inf.dropna(subset=["E_Ma_parent", "E_Ca_parent"], inplace=True)

        # Calculate IE
        df_inf["E_IE"] = (
            df_inf["E_Ca_iso"] + df_inf["E_Ma_parent"] - df_inf["E_Ca_parent"]
        )

        inf_result = df_inf

    return train_result, inf_result


# --- Main Pipeline ---
if __name__ == "__main__":

    # Ensure input directories exist
    if not os.path.exists(INPUT_DIR):
        print(f"Error: Data directory not found at {INPUT_DIR}")
        print("Please ensure the script is run from the project root.")
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Identify Isotopologues
    all_files = glob.glob(os.path.join(INPUT_DIR, "CO2_*_*.csv"))

    # Extract iso numbers safely
    isos = set()
    for f in all_files:
        try:
            filename = os.path.basename(f)
            # Filter for expected file naming convention
            if filename.startswith("CO2_") and (
                "_ma.csv" in filename or "_ca.csv" in filename
            ):
                parts = filename.split("_")
                # Format is usually CO2_{iso}_{type}.csv, so index 1 is iso
                iso_str = parts[1]
                isos.add(int(iso_str))
        except (IndexError, ValueError):
            continue

    isos = sorted(list(isos))
    processed_data = {}  # iso -> {'matched': df, 'ca_only': df}

    print(f"1. Parsing Files from {INPUT_DIR} using {NUM_WORKERS} workers...")

    processed_data = {}

    # Use ProcessPoolExecutor to run load_and_key in parallel
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        # Submit all tasks
        futures = [executor.submit(process_iso_files, iso, INPUT_DIR) for iso in isos]

        # Collect results as they complete
        for future in futures:
            iso, matched, ca_only = future.result()
            processed_data[iso] = {"matched": matched, "ca_only": ca_only}

    # 2. Process Parent Isotopologue
    if PARENT_ISO not in processed_data:
        print(f"Error: Parent isotopologue {PARENT_ISO} missing.")
        sys.exit(1)

    print(f"2. Processing Parent Isotopologue ({PARENT_ISO})...")

    # Use the 'matched' parent file to create the lookup
    parent_matched = processed_data[PARENT_ISO]["matched"]

    if parent_matched.empty:
        print("Error: Parent matched data is empty. Cannot build lookups.")
        sys.exit(1)

    # Create Lookups
    lookup_ma = parent_matched.set_index("match_key")["E_Ma"].to_dict()
    lookup_ca = parent_matched.set_index("match_key")["E_Ca"].to_dict()

    # Calculate Parent Reduced Masses for Ratio features
    # Retrieve mass list from external DB [O1, C, O2]
    p_masses = iso_db.get_mass_list(PARENT_ISO)
    dummy_row = {
        "mass_o_1": p_masses[0],
        "mass_c": p_masses[1],
        "mass_o_2": p_masses[2],
    }

    parent_mus = {
        "mu1": calc_mu1(dummy_row),
        "mu2": calc_mu2(dummy_row),
        "mu3": calc_mu3(dummy_row),
        "mu_all": calc_mu_all(dummy_row),
    }

    # 3. Construct Final Datasets
    final_training_rows = []
    final_inference_rows = []

    print("3. Generating Features and Final Tables for Minor Isotopologues...")
    print(
        f"   Processing {len(processed_data) - 1} minor isotopologues using {NUM_WORKERS} workers..."
    )

    # Prepare list of isos to process (excluding parent)
    minor_isos = [iso for iso in processed_data.keys() if iso != PARENT_ISO]

    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        # Need to pass the lookup dicts to every worker.
        # Note: If lookups are massive, this might use a lot of RAM.
        futures = []
        for iso in minor_isos:
            data = processed_data[iso]
            futures.append(
                executor.submit(
                    process_minor_iso_features,
                    iso,
                    data["matched"],
                    data["ca_only"],
                    lookup_ma,
                    lookup_ca,
                )
            )

        for future in futures:
            train_df, inf_df = future.result()
            if train_df is not None:
                final_training_rows.append(train_df)
            if inf_df is not None:
                final_inference_rows.append(inf_df)

    # 4. Concatenate and Cleanup
    # -- Process Training Set --
    if final_training_rows:
        df_final_train = pd.concat(final_training_rows, ignore_index=True)
        df_final_train = prepare_atomic_features(df_final_train, parent_mus)

        df_final_train["molecule"] = "CO2"

        # Cleanup columns
        drops = ["match_key", "ID", "unc", "??", "Source"]
        df_final_train.drop(
            columns=[c for c in drops if c in df_final_train.columns], inplace=True
        )
        # Rename Symmetries
        df_final_train.rename(
            columns={'Sym_A"': "Sym_Adp", "Sym_A'": "Sym_Ap"}, inplace=True
        )

        train_path = os.path.join(OUTPUT_DIR, "CO2_minor_isos_training.csv")
        df_final_train.to_csv(train_path, index=False)
        print(f"Saved Training Data: {train_path} ({len(df_final_train)} rows)")
    else:
        print("Warning: No valid training data generated.")

    # -- Process Inference Set --
    if final_inference_rows:
        df_final_inf = pd.concat(final_inference_rows, ignore_index=True)
        df_final_inf = prepare_atomic_features(df_final_inf, parent_mus)

        df_final_inf["molecule"] = "CO2"

        # Filter by max inference energy
        df_final_inf = df_final_inf[df_final_inf["E_Ma_parent"] <= INF_MAX]

        # Cleanup columns
        drops = ["match_key", "ID", "unc", "??", "Source"]
        df_final_inf.drop(
            columns=[c for c in drops if c in df_final_inf.columns], inplace=True
        )
        # Rename Symmetries
        df_final_inf.rename(
            columns={'Sym_A"': "Sym_Adp", "Sym_A'": "Sym_Ap"}, inplace=True
        )

        inf_path = os.path.join(OUTPUT_DIR, "CO2_minor_isos_inference.csv")
        df_final_inf.to_csv(inf_path, index=False)
        print(f"Saved Inference Data: {inf_path} ({len(df_final_inf)} rows)")
    else:
        print("Warning: No valid inference data generated.")
