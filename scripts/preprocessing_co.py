import os
import sys
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
PARENT_ISO = 26  # C12-O16
MATCH_KEYS = ["v", "J"]

# Paths
DUO_DIR = "data/co/calc"
MARVEL_DIR = "data/co/marvel"
OUTPUT_DIR = "data/processed"


# --- Feature Engineering Functions ---
def calculate_reduced_mass(row):
    """
    Calculates reduced mass for diatomic molecules (m1-m2).
    """
    m1 = row["mass_c"]
    m2 = row["mass_o"]

    mu = (m1 * m2) / (m1 + m2)
    return mu


def prepare_atomic_features(df, parent_mu):
    """
    Adds mass-based features and one-hot encodings.
    """
    print("   Computing atomic mass features...")

    # 1. Map Masses using isotopologue_masses.py
    # Returns [mass_c, mass_o]
    mass_data = df["iso"].apply(lambda x: iso_db.get_mass_list(x))
    df[["mass_c", "mass_o"]] = pd.DataFrame(mass_data.tolist(), index=df.index)

    # 2. Calculate Reduced Mass
    df["mu"] = df.apply(calculate_reduced_mass, axis=1)
    df["mu_ratio"] = df["mu"] / parent_mu

    # 3. One-Hot Encoding for Masses
    mass_cols = ["mass_c", "mass_o"]
    for col in mass_cols:
        if col in df.columns:
            one_hot = pd.get_dummies(df[col], prefix=col)
            df = pd.concat([df, one_hot], axis=1)
            df.drop(columns=[col], inplace=True)

    return df


# --- Feature Engineering Functions ---


def calculate_reduced_mass(row):
    """
    Calculates reduced mass for diatomic molecules (m1-m2).
    """
    m1 = row["mass_c"]
    m2 = row["mass_o"]

    mu = (m1 * m2) / (m1 + m2)
    return mu


def prepare_atomic_features(df, parent_mu):
    """
    Adds mass-based features and one-hot encodings.
    """
    print("   Computing atomic mass features...")

    # 1. Map Masses using isotopologue_masses.py
    # Returns [mass_c, mass_o]
    mass_data = df["iso"].apply(lambda x: iso_db.get_mass_list(x))
    df[["mass_c", "mass_o"]] = pd.DataFrame(mass_data.tolist(), index=df.index)

    # 2. Calculate Reduced Mass
    df["mu"] = df.apply(calculate_reduced_mass, axis=1)
    df["mu_ratio"] = df["mu"] / parent_mu

    # 3. One-Hot Encoding for Masses
    mass_cols = ["mass_c", "mass_o"]
    for col in mass_cols:
        if col in df.columns:
            one_hot = pd.get_dummies(df[col], prefix=col)
            df = pd.concat([df, one_hot], axis=1)
            df.drop(columns=[col], inplace=True)

    return df


# --- Parsing Functions (Restored from Original Workflow) ---


def parse_duo_output(filename):
    """
    Parses DUO output file to extract energy levels.
    Uses strict column indexing from original split_MaHi_Ca_CO.py.
    """
    with open(filename, "r") as f:
        lines = f.readlines()

    duo_data = []

    # Skip header lines and look for level data
    for line in lines:
        # Check if line contains level information with all required columns
        if line.strip() and not line.startswith("#") and not line.startswith("*"):
            try:
                # Split the line and convert to appropriate types
                parts = line.split()
                if len(parts) >= 11:  # Ensure line has all required columns
                    level = {
                        "J": int(
                            float(parts[0])
                        ),  # Rotational quantum number (column 1)
                        "i": int(parts[1]),  # State index (column 2)
                        "E_Ca": float(parts[2]),  # Energy level (column 3)
                        "State": int(parts[3]),  # Electronic state (column 4)
                        "v": int(parts[4]),  # Vibrational quantum number (column 5)
                        "lambda": int(parts[5]),  # Lambda (column 6)
                        "spin": float(parts[6]),  # Spin (column 7)
                        "sigma": float(parts[7]),  # Sigma (column 8)
                        "omega": float(parts[8]),  # Omega (column 9)
                        "parity": parts[9],  # Parity (column 10)
                        "label": parts[10],  # State label (column 11)
                    }
                    duo_data.append(level)
            except (ValueError, IndexError):
                continue

    return pd.DataFrame(duo_data)


def load_marvel_data(filename):
    """
    Parses MARVEL energy file using original logic.
    """
    try:
        # Read and process the MARVEL file using fixed columns
        marvel_data = pd.read_csv(filename, sep=r"\s+", header=None)

        # Original column definition from split_MaHi_Ca_CO.py
        marvel_data.columns = ["v", "J", "E", "Unc_E", "N"]

        # Rename E to E_Ma for internal consistency
        marvel_data.rename(columns={"E": "E_Ma"}, inplace=True)

        # Keep only necessary columns
        return marvel_data[["v", "J", "E_Ma"]]

    except Exception as e:
        print(f"Warning: Could not parse MARVEL file {filename}: {e}")
        return pd.DataFrame()


# --- Main Pipeline ---

if __name__ == "__main__":
    # Ensure input directories exist
    if not os.path.exists(DUO_DIR):
        print(f"Error: Data directory not found at {DUO_DIR}")
        print("Please ensure the script is run from the project root.")
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Identify files
    duo_files = sorted(
        [f for f in os.listdir(DUO_DIR) if f.endswith("_output_duo.out")]
    )

    # Storage
    processed_data = {}  # Key: iso, Value: {'matched': df, 'ca_only': df}

    print(f"1. Parsing and Merging Raw Files from {DUO_DIR}...")

    for d_file in duo_files:
        try:
            iso_str = d_file.split("_")[0][2:]
            iso = int(iso_str)
        except (IndexError, ValueError):
            print(f"Skipping file with unknown format: {d_file}")
            continue

        print(f"   Processing Isotopologue {iso}...")

        # Load DUO
        df_duo = parse_duo_output(os.path.join(DUO_DIR, d_file))
        if df_duo.empty:
            print(f"   Warning: No DUO data found for {iso} (or format mismatch).")
            continue

        # Create match key tuple for merging
        df_duo["match_key"] = df_duo[MATCH_KEYS].apply(tuple, axis=1)

        # Load MARVEL
        m_file = f"MARVEL_Energies_CO{iso}.txt"
        m_path = os.path.join(MARVEL_DIR, m_file)

        if os.path.exists(m_path):
            df_marvel = load_marvel_data(m_path)
            if not df_marvel.empty:
                df_marvel["match_key"] = df_marvel[MATCH_KEYS].apply(tuple, axis=1)

                # Merge DUO and MARVEL
                merged = pd.merge(
                    df_duo, df_marvel[["match_key", "E_Ma"]], on="match_key", how="left"
                )
            else:
                merged = df_duo.copy()
                merged["E_Ma"] = np.nan
        else:
            print(f"      No MARVEL file found. All rows treated as Ca-only.")
            merged = df_duo.copy()
            merged["E_Ma"] = np.nan

        merged["iso"] = iso

        # Split into Matched (Training) and Unmatched (Prediction/Ca-only)
        matched = merged.dropna(subset=["E_Ma"]).copy()
        ca_only = merged[merged["E_Ma"].isna()].copy()

        processed_data[iso] = {"matched": matched, "ca_only": ca_only}

    # 2. Process Parent Isotopologue
    if PARENT_ISO not in processed_data:
        print(f"Error: Parent isotopologue {PARENT_ISO} missing.")
        sys.exit(1)

    print(f"2. Processing Parent Isotopologue ({PARENT_ISO})...")
    parent_matched = processed_data[PARENT_ISO]["matched"]

    # Create Lookups for Parent Energies
    lookup_ma = parent_matched.set_index("match_key")["E_Ma"].to_dict()
    lookup_ca = parent_matched.set_index("match_key")["E_Ca"].to_dict()

    # Calculate Parent Reduced Mass for Ratios
    parent_masses = iso_db.get_mass_list(PARENT_ISO)
    mu_parent = (parent_masses[0] * parent_masses[1]) / (
        parent_masses[0] + parent_masses[1]
    )

    # 3. Construct Final Datasets (Minor Isotopologues Only)
    final_training_rows = []
    final_ca_only_rows = []

    print("3. Generating Features and Final Tables for Minor Isotopologues...")

    for iso, data in processed_data.items():
        # Parent isotopologue is excluded from output files
        if iso == PARENT_ISO:
            continue

        # --- A. Training Data (Minor Isos with MARVEL) ---
        df_train = data["matched"].copy()

        # Map Parent Info
        df_train["E_Ma_main"] = df_train["match_key"].map(lookup_ma)
        df_train["E_Ca_main"] = df_train["match_key"].map(lookup_ca)

        # Rename for consistency
        df_train.rename(columns={"E_Ca": "E_Ca_iso", "E_Ma": "E_Ma_iso"}, inplace=True)

        # Filter: Parent info is required to calculate IE
        df_train.dropna(subset=["E_Ma_main", "E_Ca_main"], inplace=True)

        if not df_train.empty:
            # Calculate Physics-Informed Errors
            df_train["E_IE"] = (
                df_train["E_Ca_iso"] + df_train["E_Ma_main"] - df_train["E_Ca_main"]
            )
            df_train["Error_IE"] = df_train["E_Ma_iso"] - df_train["E_IE"]

            final_training_rows.append(df_train)

        # --- B. Ca-Only Data (Prediction Set) ---
        df_ca = data["ca_only"].copy()

        # Attach Parent Info
        df_ca["E_Ma_main"] = df_ca["match_key"].map(lookup_ma)
        df_ca["E_Ca_main"] = df_ca["match_key"].map(lookup_ca)

        # Filter: Parent info is required for inference (prediction)
        # Parent lines missing MARVEL are excluded
        df_ca.dropna(subset=["E_Ma_main", "E_Ca_main"], inplace=True)

        if not df_ca.empty:
            df_ca.rename(columns={"E_Ca": "E_Ca_iso", "E_Ma": "E_Ma_iso"}, inplace=True)

            # Calculate Pseudo-IE (Prediction Baseline) where possible
            df_ca["E_IE"] = df_ca["E_Ca_iso"] + df_ca["E_Ma_main"] - df_ca["E_Ca_main"]

            final_ca_only_rows.append(df_ca)

    # 4. Concatenate and Feature Engineering

    # -- Process Training Set --
    if final_training_rows:
        df_final_train = pd.concat(final_training_rows, ignore_index=True)
        df_final_train = prepare_atomic_features(df_final_train, mu_parent)

        # Cleanup
        drops = [
            "match_key",
            "parity",
            "i",
            "State",
            "lambda",
            "spin",
            "sigma",
            "omega",
            "label",
        ]
        df_final_train.drop(
            columns=[c for c in drops if c in df_final_train.columns], inplace=True
        )

        train_path = os.path.join(OUTPUT_DIR, "CO_minor_isos_training.csv")
        df_final_train.to_csv(train_path, index=False)
        print(f"Saved Training Data: {train_path} ({len(df_final_train)} rows)")
    else:
        print("Warning: No valid training data generated.")

    # -- Process Ca-Only Set --
    if final_ca_only_rows:
        df_final_ca = pd.concat(final_ca_only_rows, ignore_index=True)
        df_final_ca = prepare_atomic_features(df_final_ca, mu_parent)

        # Cleanup
        drops = [
            "match_key",
            "parity",
            "i",
            "State",
            "lambda",
            "spin",
            "sigma",
            "omega",
            "label",
        ]
        df_final_ca.drop(
            columns=[c for c in drops if c in df_final_ca.columns], inplace=True
        )

        ca_path = os.path.join(OUTPUT_DIR, "CO_minor_isos_inference.csv")
        df_final_ca.to_csv(ca_path, index=False)
        print(f"Saved Ca-Only Data: {ca_path} ({len(df_final_ca)} rows)")
    else:
        print(
            "Warning: No Ca-only data found (after filtering for missing parent data)."
        )
