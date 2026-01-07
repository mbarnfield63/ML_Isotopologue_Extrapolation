# Preprocess minor H2O isos
import io
import numpy as np
import os
import pandas as pd

h2o_isos = {
    "H216O": 161,
    "H217O": 171,
    "H218O": 181,
    "1H2-16O": 161,
    "1H2-17O": 171,
    "1H2-18O": 181,
}
marvel_column_names = [
    "v1",
    "v2",
    "v3",
    "J",
    "Ka",
    "Kc",
    "E",
    "unc",
    "LTC",
    "LSC",
    "parity",
]
states_column_names = [
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


def load_data_marvel(file_path):
    cleaned_lines = []
    separator_count = 0

    with open(file_path, "r") as f:
        for line in f:
            # 1. Check for separator lines
            if line.startswith("="):
                separator_count += 1
                # Hit the 3rd separator, Component 1 is finished. Stop reading.
                if separator_count >= 3:
                    break
                # Skip the separator line itself
                continue

            # 2. Skip metadata headers
            if line.startswith("Component") or "assignment" in line:
                continue

            # 3. Keep actual data lines
            cleaned_lines.append(line)

    data_block = "".join(cleaned_lines)
    if not data_block.strip():
        return pd.DataFrame(columns=marvel_column_names)

    # Load from the memory buffer
    df = pd.read_csv(io.StringIO(data_block), names=marvel_column_names, sep="\s+")

    return df


def get_files_in_directory(dir_path):
    files = []
    for file_name in os.listdir(dir_path):
        if file_name.endswith(".txt"):
            files.append(os.path.join(dir_path, file_name))
    return files


def preprocess_marvel(file_path):
    df = load_data_marvel(file_path)
    file_name = os.path.basename(file_path)
    iso_key = file_name.split("energy_levels_")[1].split(".")[0]
    if iso_key in h2o_isos:
        df["iso_code"] = h2o_isos[iso_key]
    else:
        assert False, f"Unknown isotope in file name: {file_name}"

    # Clean dataframe
    if "parity" in df.columns:
        df = df.drop(columns=["parity"])
    df["LTC"] = df["LTC"].astype(int)
    df["LSC"] = df["LSC"].astype(int)
    df["sym"] = df.apply(assign_symmetry, axis=1)

    # Drop rows where E == 0.0 and if any of v1, v2, v3, J, Ka, Kc are non-zero
    df["E"] = df["E"].astype(float)
    cols = ["v1", "v2", "v3", "J", "Ka", "Kc"]
    df[cols] = df[cols].astype(int)
    cond = (df["E"] == 0.0) & (df[cols].ne(0).any(axis=1))
    if cond.any():
        df = df.loc[~cond].reset_index(drop=True)

    return df


def assign_symmetry(row):
    """
    Derives C2v symmetry labels (A1, A2, B1, B2) for Water
    based on the parity of Ka and Kc.
    """
    ka = row["Ka"]
    kc = row["Kc"]

    # Check if Even (e) or Odd (o)
    ka_even = ka % 2 == 0
    kc_even = kc % 2 == 0

    if ka_even and kc_even:  # ee
        return "A1"
    elif ka_even and not kc_even:  # eo
        return "B1"
    elif not ka_even and not kc_even:  # oo
        return "B2"
    elif not ka_even and kc_even:  # oe
        return "A2"
    else:
        return "U"


def preprocess_states(file_path):
    df = pd.read_csv(file_path, sep="\s+", header=None)

    file_name = os.path.basename(file_path)
    source = file_name.split("__")[1].split(".states")[0]
    if source == "POKAZATEL":
        df.columns = states_column_names
        df.drop(columns=["id", "E_ma", "g_tot", "tau", "sym"], inplace=True)
    else:
        # Use a list comprehension to exclude 'tau' and 'source' from the column names
        df.columns = [c for c in states_column_names if c not in ("tau", "source")]
        df.drop(columns=["id", "E_ma", "g_tot"], inplace=True)

    iso_key = file_name.split(".states")[0].split("__")[0]
    if iso_key in h2o_isos:
        df["iso_code"] = h2o_isos[iso_key]
    else:
        assert False, f"Unknown isotope in file name: {file_name}"

    # Clean dataframe
    # Replace -2 with NaN and drop any rows containing NaN
    df.replace(-2, np.nan, inplace=True)
    df.dropna(axis=0, how="any", inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df


if __name__ == "__main__":
    input_dir = "data/h2o/raw"
    output_dir_marvel = "data/h2o/marvel"
    os.makedirs(output_dir_marvel, exist_ok=True)
    output_dir_calc = "data/h2o/calc"
    os.makedirs(output_dir_calc, exist_ok=True)

    files = get_files_in_directory(input_dir)
    for file_name in os.listdir(input_dir):
        file_path = os.path.join(input_dir, file_name)
        if file_name.endswith(".txt"):
            df = preprocess_marvel(file_path)
            iso = df["iso_code"][0]
            out_name = "H2O_" + iso.astype(str) + "_MARVEL.csv"
            output_path = os.path.join(output_dir_marvel, out_name)
        elif file_name.endswith(".states"):
            df = preprocess_states(file_path)
            iso = df["iso_code"][0]
            out_name = "H2O_" + iso.astype(str) + "_CALC.csv"
            output_path = os.path.join(output_dir_calc, out_name)
        else:
            print(f"Skipping unrecognized file: {file_name}")
            continue

        df.to_csv(output_path, index=False)
        print(f"Processed {file_name}, saved to {output_path}")
