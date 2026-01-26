import os
import sys
import glob
import pandas as pd
import numpy as np

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
    CO2 uses AFGL columns + J + e_f.
    """
    # Identify AFGL columns dynamically
    afgl_cols = [col for col in df.columns if col.startswith("AFGL")]
    key_cols = ["J"] + afgl_cols

    # Check for symmetry columns
    if "e_f" in df.columns:
        key_cols.append("e_f")

    # Create tuple key
    return df[key_cols].apply(tuple, axis=1)


# --- Parsing Functions ---
def load_file_pair(iso, input_dir):
    """
    Loads _ma.txt (matched/MARVEL) and _ca.txt (calc only) for a given iso.
    """
    ma_path = os.path.join(input_dir, f"CO2_{iso}_ma.txt")
    ca_path = os.path.join(input_dir, f"CO2_{iso}_ca.txt")

    matched = pd.DataFrame()
    ca_only = pd.DataFrame()

    # Load Matched (Training Candidates)
    if os.path.exists(ma_path):
        matched = pd.read_csv(ma_path, sep="\t")
        matched["iso"] = iso
        # Rename E to E_Ma for consistency with CO script
        if "E" in matched.columns:
            matched.rename(columns={"E": "E_Ma"}, inplace=True)

    # Load Ca-Only (Inference Candidates)
    if os.path.exists(ca_path):
        ca_only = pd.read_csv(ca_path, sep="\t")
        ca_only["iso"] = iso
        # Ensure E_Ma is NaN for these
        ca_only["E_Ma"] = np.nan

    return matched, ca_only


# --- Main Pipeline ---
if __name__ == "__main__":

    # Ensure input directories exist
    if not os.path.exists(INPUT_DIR):
        print(f"Error: Data directory not found at {INPUT_DIR}")
        print("Please ensure the script is run from the project root.")
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Identify Isotopologues
    all_files = glob.glob(os.path.join(INPUT_DIR, "CO2_*_*.txt"))

    # Extract iso numbers safely
    isos = set()
    for f in all_files:
        try:
            filename = os.path.basename(f)
            # Filter for expected file naming convention
            if filename.startswith("CO2_") and (
                "_ma.txt" in filename or "_ca.txt" in filename
            ):
                parts = filename.split("_")
                # Format is usually CO2_{iso}_{type}.txt, so index 1 is iso
                iso_str = parts[1]
                isos.add(int(iso_str))
        except (IndexError, ValueError):
            continue

    isos = sorted(list(isos))
    processed_data = {}  # iso -> {'matched': df, 'ca_only': df}

    print(f"1. Parsing Files from {INPUT_DIR}...")

    for iso in isos:
        print(f"   Processing Isotopologue {iso}...")
        matched, ca_only = load_file_pair(iso, INPUT_DIR)

        # Generate Match Keys
        if not matched.empty:
            matched["match_key"] = generate_match_key(matched)
        if not ca_only.empty:
            ca_only["match_key"] = generate_match_key(ca_only)

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

    for iso, data in processed_data.items():
        if iso == PARENT_ISO:
            continue

        # --- A. Training Data (Minor Isos with MARVEL) ---
        df_train = data["matched"].copy()

        if not df_train.empty:
            # Map Parent Info
            df_train["E_Ma_main"] = df_train["match_key"].map(lookup_ma)
            df_train["E_Ca_main"] = df_train["match_key"].map(lookup_ca)

            # Rename
            df_train.rename(
                columns={"E_Ca": "E_Ca_iso", "E_Ma": "E_Ma_iso"}, inplace=True
            )

            # Filter: Must have parent info to calculate IE
            df_train.dropna(subset=["E_Ma_main", "E_Ca_main"], inplace=True)
            df_train["E_IE"] = (
                df_train["E_Ca_iso"] + df_train["E_Ma_main"] - df_train["E_Ca_main"]
            )
            df_train["Error_IE"] = df_train["E_Ma_iso"] - df_train["E_IE"]

            final_training_rows.append(df_train)

        # --- B. Inference Data (Calc Only) ---
        df_inf = data["ca_only"].copy()

        if not df_inf.empty:
            # Map Parent Info
            df_inf["E_Ma_main"] = df_inf["match_key"].map(lookup_ma)
            df_inf["E_Ca_main"] = df_inf["match_key"].map(lookup_ca)

            # Filter: Must have parent info to predict accurately
            df_inf.dropna(subset=["E_Ma_main", "E_Ca_main"], inplace=True)

            # Rename
            df_inf.rename(
                columns={"E_Ca": "E_Ca_iso", "E_Ma": "E_Ma_iso"}, inplace=True
            )

            df_inf["E_IE"] = (
                df_inf["E_Ca_iso"] + df_inf["E_Ma_main"] - df_inf["E_Ca_main"]
            )

            final_inference_rows.append(df_inf)

    # 4. Concatenate and Cleanup
    # -- Process Training Set --
    if final_training_rows:
        df_final_train = pd.concat(final_training_rows, ignore_index=True)
        df_final_train = prepare_atomic_features(df_final_train, parent_mus)

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
