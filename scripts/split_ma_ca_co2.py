import pandas as pd
import glob
import sys
import os
import time
import concurrent.futures
import functools

# --- Configuration ---
raw_dir = "C:\\Code\\Work\\raw_data_store\\CO2"
out_dir = "data/co2"

iso_formats = {
    "12C-16O2": 626,
    "16O-12C-17O": 627,
    "16O-12C-18O": 628,
    "12C-17O2": 727,
    "17O-12C-18O": 728,
    "12C-18O2": 828,
    "13C-16O2": 636,
    "16O-13C-17O": 637,
    "16O-13C-18O": 638,
    "13C-17O2": 737,
    "17O-13C-18O": 738,
    "13C-18O2": 838,
}

column_names = [
    "ID",
    "E",
    "gtot",
    "J",
    "unc",
    "??",
    "tot_sym",
    "e_f",
    "hzb_v1",
    "hzb_v2",
    "hzb_l2",
    "hzb_v3",
    "Trove_coeff",
    "AFGL_m1",
    "AFGL_m2",
    "AFGL_l2",
    "AFGL_m3",
    "AFGL_r",
    "Trove_v1",
    "Trove_v2",
    "Trove_v3",
    "Source",
    "E_Ca",
]


def split_raw_file(file_path, output_dir):
    """
    Splits the raw file into two separate files in the output directory:
    - CO2_[iso]_ma.txt = Marvel data
    - CO2_[iso]_ca.txt = Calculated data
    """
    try:
        # Extract filename from path (e.g., "path/to/12C-16O2__...cut" -> "12C-16O2")
        base_name = os.path.basename(file_path)
        iso_str = base_name.split("__")[0]

        iso = iso_formats.get(iso_str)
        if iso is None:
            print(f"Skipping unknown isotope format: {iso_str}")
            return

        print(f"Processing {iso} ({iso_str})...")

        # Read the file
        df = pd.read_csv(
            file_path, header=None, skiprows=1, sep=r"\s+", names=column_names
        )

        # Clean up whitespace
        for col in df.select_dtypes(include=["object"]).columns:
            df[col] = df[col].str.strip()

        # Split based on Source
        marvel_hitran = df[df["Source"].isin(["Ma", "MA", "Hi", "HI"])]
        calculated = df[df["Source"].isin(["Ca", "CA", "Eh", "EH"])]

        # Verification
        if len(marvel_hitran) + len(calculated) != len(df):
            print(
                f"Warning: Rows lost during split for {iso} (Check 'Source' column values)"
            )

        # Define Output Paths
        # Note: Using sep='\t' to match the preprocessing script expectations
        ma_path = os.path.join(output_dir, f"CO2_{iso}_ma.csv")
        ca_path = os.path.join(output_dir, f"CO2_{iso}_ca.csv")

        marvel_hitran.to_csv(ma_path, index=False, sep=",")
        calculated.to_csv(ca_path, index=False, sep=",")

        print(f"Saved: CO2_{iso} to {output_dir}")

    except Exception as e:
        print(f"Error processing {file_path}: {e}")


if __name__ == "__main__":
    time_start = time.time()

    # Validation
    if not os.path.exists(raw_dir):
        print(f"Error: Input directory '{raw_dir}' does not exist.")
        sys.exit(1)

    # Create output directory if it doesn't exist
    os.makedirs(out_dir, exist_ok=True)

    # Get files
    files = glob.glob(os.path.join(raw_dir, "*.states.cut"))

    if not files:
        print(f"No .states.cut files found in {raw_dir}")
        sys.exit(0)

    print(f"Found {len(files)} files. Starting processing...")

    # Partial function to pass output_dir to map
    process_func = functools.partial(split_raw_file, output_dir=out_dir)

    # Parallel Execution
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # We wrap the map in list() to force execution and catch exceptions if any
        list(executor.map(process_func, files))

    print(f"All files processed in {time.time() - time_start:.2f} seconds.")
