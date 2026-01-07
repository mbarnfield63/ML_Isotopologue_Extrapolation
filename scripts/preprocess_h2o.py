import io
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
PARENT_ISO = 161
H2O_ISOS_MAP = {
    "H216O": 161,
    "H217O": 171,
    "H218O": 181,
    "1H2-16O": 161,
    "1H2-17O": 171,
    "1H2-18O": 181,
}

# --- Column Definitions ---
MARVEL_COLS = ["v1", "v2", "v3", "J", "Ka", "Kc", "E", "unc", "LTC", "LSC", "parity"]
STATES_COLS = [
    "id",
    "E_ma",
    "g_tot",
    "J",
    "tau",
    "unc",
    "Ka",
    "Kc",
    "v1",
    "v2",
    "v3",
    "sym",
    "E_calc",
    "source",
]
MATCH_KEYS = ["v1", "v2", "v3", "J", "Ka", "Kc"]


# --- Feature Engineering Functions ---
def calculate_reduced_masses(row):
    """Calculates reduced masses for triatomic molecules (m1-m2-m3)."""
    # H2O structure: H(1)-O(c)-H(2)
    m1 = row["mass_h_1"]
    m2 = row["mass_o"]
    m3 = row["mass_h_2"]

    # 1. Symmetric-like (average of sides)
    mu_side1 = (m1 * m2) / (m1 + m2)
    mu_side2 = (m3 * m2) / (m3 + m2)
    mu1 = (mu_side1 + mu_side2) / 2

    # 2. Bend-like (interaction of outer atoms)
    mu2 = (m1 * m3) / (m1 + m3)
    # 3. Asymmetric-like (combined)
    mu3 = ((m1 + m2) * m2) / (m1 + m2 + m2)

    # 4. Total reduced mass
    mu_all = (m1 * m2 * m3) / (m1 + m2 + m3)

    return pd.Series([mu1, mu2, mu3, mu_all], index=["mu1", "mu2", "mu3", "mu_all"])


def prepare_atomic_features(df):
    """Adds mass-based features and one-hot encodings."""
    print("  Computing atomic mass features and one-hot encodings...")

    # 1. Map Masses
    # isotopologue_masses.py defines 161 as ['H1', 'O16', 'H1']
    # We map these to mass_h_1, mass_o, mass_h_2
    mass_data = df["iso_code"].apply(lambda x: iso_db.get_mass_list(x))
    df[["mass_h_1", "mass_o", "mass_h_2"]] = pd.DataFrame(
        mass_data.tolist(), index=df.index
    )

    # 2. Calculate Reduced Masses (Continuous Features)
    mu_cols = df.apply(calculate_reduced_masses, axis=1)
    df = pd.concat([df, mu_cols], axis=1)

    # 3. Calculate Ratios relative to Parent
    parent_masses = iso_db.get_mass_list(PARENT_ISO)
    dummy_parent = pd.Series(
        {
            "mass_h_1": parent_masses[0],
            "mass_o": parent_masses[1],
            "mass_h_2": parent_masses[2],
        }
    )
    mu_parent = calculate_reduced_masses(dummy_parent)

    df["mu1_ratio"] = df["mu1"] / mu_parent["mu1"]
    df["mu2_ratio"] = df["mu2"] / mu_parent["mu2"]
    df["mu3_ratio"] = df["mu3"] / mu_parent["mu3"]
    df["mu_all_ratio"] = df["mu_all"] / mu_parent["mu_all"]

    # 4. One-Hot Encoding for Masses
    mass_cols = ["mass_h_1", "mass_o", "mass_h_2"]
    for col in mass_cols:
        one_hot = pd.get_dummies(df[col], prefix=col)
        df = pd.concat([df, one_hot], axis=1)
        df.drop(columns=[col], inplace=True)  # raw numerical mass columns

    # 5. One-Hot Encoding for Symmetries
    # Expecting 'sym' column with values like A1, B2, etc.
    if "sym" in df.columns:
        sym_one_hot = pd.get_dummies(df["sym"], prefix="Sym")
        df = pd.concat([df, sym_one_hot], axis=1)
        df.drop(columns=["sym"], inplace=True)

    return df


# --- Parsing Functions ---
def assign_symmetry(row):
    """Derives C2v symmetry labels (A1, A2, B1, B2) from Ka/Kc."""
    ka, kc = row["Ka"], row["Kc"]
    ka_even = ka % 2 == 0
    kc_even = kc % 2 == 0

    if ka_even and kc_even:
        return "A1"
    elif ka_even and not kc_even:
        return "B1"
    elif not ka_even and not kc_even:
        return "B2"
    elif not ka_even and kc_even:
        return "A2"
    return "U"


def load_data_marvel(file_path):
    cleaned_lines = []
    separator_count = 0
    with open(file_path, "r") as f:
        for line in f:
            if line.startswith("="):
                separator_count += 1
                if separator_count >= 3:
                    break
                continue
            if line.startswith("Component") or "assignment" in line:
                continue
            cleaned_lines.append(line)

    data_block = "".join(cleaned_lines)
    if not data_block.strip():
        return pd.DataFrame(columns=MARVEL_COLS)

    return pd.read_csv(io.StringIO(data_block), names=MARVEL_COLS, sep="\s+")


def preprocess_marvel(file_path):
    df = load_data_marvel(file_path)
    file_name = os.path.basename(file_path)

    try:
        iso_key = file_name.split("energy_levels_")[1].split(".")[0]
        if iso_key in H2O_ISOS_MAP:
            df["iso_code"] = H2O_ISOS_MAP[iso_key]
        else:
            return None
    except IndexError:
        return None

    # Determine Symmetry
    df["sym"] = df.apply(assign_symmetry, axis=1)
    if "parity" in df.columns:
        df.drop(columns=["parity"], inplace=True)

    # Quantum Numbers & Energy
    num_cols = ["v1", "v2", "v3", "J", "Ka", "Kc"]
    df[num_cols] = df[num_cols].astype(int)
    df["E"] = df["E"].astype(float)

    # Clean invalid zeros
    cond = (df["E"] == 0.0) & (df[num_cols].ne(0).any(axis=1))
    df = df.loc[~cond].copy()

    # Rename Energy to Target Name
    df.rename(columns={"E": "E_Ma_iso"}, inplace=True)

    # Create matching key
    df["match_key"] = df[MATCH_KEYS].apply(tuple, axis=1)
    return df


def preprocess_states(file_path):
    file_name = os.path.basename(file_path)
    try:
        source = file_name.split("__")[1].split(".states")[0]
    except IndexError:
        source = "Unknown"

    df = pd.read_csv(file_path, sep="\s+", header=None)

    if source == "POKAZATEL":
        df.columns = STATES_COLS
        df.drop(columns=["id", "E_ma", "g_tot", "tau", "sym"], inplace=True)
    else:
        df.columns = [c for c in STATES_COLS if c not in ("tau", "source")]
        df.drop(columns=["id", "E_ma", "g_tot"], inplace=True)

    iso_key = file_name.split(".states")[0].split("__")[0]
    if iso_key in H2O_ISOS_MAP:
        df["iso_code"] = H2O_ISOS_MAP[iso_key]
    else:
        return None

    df.replace(-2, np.nan, inplace=True)
    df.dropna(axis=0, how="any", inplace=True)

    # Rename Energy
    df.rename(columns={"E_calc": "E_Ca_iso"}, inplace=True)

    # Create matching key
    df["match_key"] = df[MATCH_KEYS].apply(tuple, axis=1)
    return df


# --- Main Pipeline ---

if __name__ == "__main__":
    input_dir = "data/h2o/raw"
    output_file = "data/processed/H2O_minor_isos_processed.csv"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    print("1. Parsing raw files...")
    files = [os.path.join(input_dir, f) for f in os.listdir(input_dir)]

    data_store = {}

    for f in files:
        if f.endswith(".txt"):
            df = preprocess_marvel(f)
            kind = "marvel"
        elif f.endswith(".states"):
            df = preprocess_states(f)
            kind = "calc"
        else:
            continue

        if df is not None and not df.empty:
            iso = df["iso_code"].iloc[0]
            if iso not in data_store:
                data_store[iso] = {}
            data_store[iso][kind] = df

    # Check for Parent Iso
    if PARENT_ISO not in data_store:
        print(f"Error: Parent isotopologue {PARENT_ISO} missing.")
        sys.exit(1)

    print(f"2. Processing Parent Isotopologue ({PARENT_ISO})...")

    parent_marvel = data_store[PARENT_ISO].get("marvel")
    parent_calc = data_store[PARENT_ISO].get("calc")

    if parent_calc is None or parent_marvel is None:
        print("Error: Parent isotopologue needs both MARVEL and CALC files.")
        sys.exit(1)

    # Merge Parent Data
    parent_merged = pd.merge(
        parent_marvel[["match_key", "E_Ma_iso"]].rename(
            columns={"E_Ma_iso": "E_Ma_parent"}
        ),
        parent_calc[["match_key", "E_Ca_iso"]].rename(
            columns={"E_Ca_iso": "E_Ca_parent"}
        ),
        on="match_key",
        how="inner",
    )

    # Create Lookups
    # Map Minor keys to Parent Energies (Ma and Ca)
    lookup_ma = parent_merged.set_index("match_key")["E_Ma_parent"].to_dict()
    lookup_ca = parent_merged.set_index("match_key")["E_Ca_parent"].to_dict()

    print(f"   Established {len(lookup_ma)} parent reference points.")

    final_dfs = []

    for iso, subsets in data_store.items():
        if iso == PARENT_ISO:
            continue
        if "calc" not in subsets:
            continue

        print(f"3. Processing Minor Isotopologue {iso}...")
        df_minor = subsets["calc"].copy()

        # 1. Attach Parent Info (E_Ma_parent, E_Ca_parent)
        df_minor["E_Ma_parent"] = df_minor["match_key"].map(lookup_ma)
        df_minor["E_Ca_parent"] = df_minor["match_key"].map(lookup_ca)

        # 2. Attach target (E_Ma_iso) if Marvel exists
        if "marvel" in subsets:
            marvel_dict = subsets["marvel"].set_index("match_key")["E_Ma_iso"].to_dict()
            df_minor["E_Ma_iso"] = df_minor["match_key"].map(marvel_dict)
        else:
            print(f"   Skipping {iso} - No MARVEL data for target labels.")
            continue

        # 3. Filter out invalid rows
        # Must have: E_Ca_iso (implicit), E_Ma_iso (Target), E_Ma_parent, E_Ca_parent
        df_minor.dropna(subset=["E_Ma_iso", "E_Ma_parent", "E_Ca_parent"], inplace=True)

        if df_minor.empty:
            continue

        # 4. Calculate IE
        # IE = Minor_Calc + (Parent_Marvel - Parent_Calc)
        df_minor["E_IE"] = df_minor["E_Ca_iso"] + (
            df_minor["E_Ma_parent"] - df_minor["E_Ca_parent"]
        )

        # 5. Calculate Error (Target - IE)
        df_minor["Error_IE"] = df_minor["E_Ma_iso"] - df_minor["E_IE"]

        final_dfs.append(df_minor)

    if final_dfs:
        combined_df = pd.concat(final_dfs, ignore_index=True)

        # Generate Mass/Sym Features
        combined_df = prepare_atomic_features(combined_df)

        # Final Cleanup
        cols_to_drop = ["match_key", "id", "source", "unc", "LTC", "LSC", "tau"]
        combined_df.drop(
            columns=[c for c in cols_to_drop if c in combined_df.columns], inplace=True
        )

        # Ensure 'iso' column exists if needed
        if "iso_code" in combined_df.columns:
            combined_df.rename(columns={"iso_code": "iso"}, inplace=True)

        combined_df.to_csv(output_file, index=False)
        print(f"Done. Saved {len(combined_df)} records to {output_file}")
    else:
        print("No valid minor isotopologue data generated.")
