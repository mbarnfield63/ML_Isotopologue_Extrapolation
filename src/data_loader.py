import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class MoleculeDataset(Dataset):
    """
    Generic Dataset for molecular energy regression.
    This is adapted from the CO2Dataset in the original repository.

    """

    def __init__(self, df, feature_cols, target_col):
        """
        Args:
            df (pd.DataFrame): DataFrame containing the data.
            feature_cols (list[str]): List of feature column names.
            target_col (str): The name of the target column.
        """
        self.X = df[feature_cols].values.astype(np.float32)
        self.y = df[target_col].values.astype(np.float32).reshape(-1, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.X[idx]),
            torch.from_numpy(self.y[idx]),
        )


def load_data(config):
    """
    Loads, combines, splits, and scales data based on the config file.

    This function implements the new logic for:
    1. Auto-combining multiple CSVs listed in `data.paths`.
    2. Dynamically determining features using `data.not_feature_cols`.

    Args:
        config (dict): The loaded YAML config file.

    Returns:
        tuple: (
            train_df (pd.DataFrame),
            val_df (pd.DataFrame),
            test_df (pd.DataFrame),
            feature_cols (list[str]),
            target_col (str),
            scaler (StandardScaler)
        )
    """
    data_config = config['data']
    exp_config = config['experiment']

    # === 1. Load and Combine Data ===
    paths = data_config['paths']
    if not isinstance(paths, list):
        paths = [paths]  # Ensure it's a list

    all_dfs = []
    all_cols = set()

    print("Loading data from paths:")
    for path in paths:
        if not path:  # Skip empty path entries
            continue
        print(f"- {path}")
        try:
            df = pd.read_csv(path)
            all_dfs.append(df)
            all_cols.update(df.columns)
        except FileNotFoundError:
            print(f"  WARNING: File not found, skipping: {path}")

    if not all_dfs:
        raise FileNotFoundError(
            "No valid data files were found. Check `data.paths` in your config.")

    all_cols_list = sorted(list(all_cols))
    combined_df = None

    if len(all_dfs) > 1:
        print("Multiple data paths found. Combining datasets...")
        processed_dfs = []
        for df in all_dfs:
            # Re-index to include all columns, fill missing with 0.0
            df = df.reindex(columns=all_cols_list, fill_value=0.0)
            processed_dfs.append(df)

        combined_df = pd.concat(processed_dfs, ignore_index=True)
        print(
            f"Combined dataset created with {len(combined_df)} rows and {len(all_cols_list)} columns.")
    else:
        combined_df = all_dfs[0]
        print(f"Loaded single dataset with {len(combined_df)} rows.")

    # === 2. Pre-processing ===
    # Drop specified isotopologues to filter out
    filter_out_isos = data_config.get('filter_out_isos', []) or []
    if filter_out_isos:
        initial_count = len(combined_df)
        combined_df = combined_df[~combined_df['iso'].isin(filter_out_isos)]
        dropped_count = initial_count - len(combined_df)
        print(f"    Dropped {dropped_count} rows based on filtered out isotopologues: {filter_out_isos}")
        print(f"    New dataset size: {len(combined_df)} rows.")

    # Drop rows with NaN in any critical columns (target or non-features)
    target_col = data_config['target_col']
    # Ensure not_feature_cols is a list, even if empty or None
    not_feature_cols_config = data_config.get('not_feature_cols', []) or []

    critical_cols = [target_col] + not_feature_cols_config
    # Filter out any None or empty strings from critical_cols
    critical_cols = [
        col for col in critical_cols if col and col in combined_df.columns]

    initial_count = len(combined_df)
    if critical_cols:
        combined_df = combined_df.dropna(subset=critical_cols)
        dropped_rows = initial_count - len(combined_df)
        if dropped_rows > 0:
            print(
                f"Dropped {dropped_rows} rows containing NaN values in critical columns.")

    # === 3. Determine Feature Columns ===
    # This section implements the `not_feature_cols` logic
    all_cols_in_df = set(combined_df.columns)
    not_feature_cols_set = set(not_feature_cols_config)
    not_feature_cols_set.add(target_col)

    # Determine feature columns dynamically
    feature_cols = [
        col for col in all_cols_in_df if col not in not_feature_cols_set]
    feature_cols.sort()  # For consistency

    print(f"Determined {len(feature_cols)} feature columns.")
    if len(feature_cols) < 5:  # Just a friendly warning
        print(f"  Features: {feature_cols}")

    # === 4. Train/Val/Test Split ===
    # Get split sizes from config with defaults
    train_size = exp_config.get('train_size', 0.7)
    val_size = exp_config.get('val_size', 0.1)
    test_size = exp_config.get('test_size', 0.2)
    random_state = exp_config.get('seed', 42)

    # Ensure sizes sum to 1
    if not np.isclose(train_size + val_size + test_size, 1.0):
        raise ValueError(
            f"ERROR: Split sizes (train={train_size}, val={val_size}, test={test_size}) do not sum to 1. Please ensure they add up to 1."
        )

    # First split: Train vs. (Val + Test)
    temp_size = val_size + test_size
    if temp_size > 0:
        train_df, temp_df = train_test_split(
            combined_df, test_size=temp_size, random_state=random_state, shuffle=True
        )
    else:
        train_df = combined_df.copy()
        temp_df = pd.DataFrame(columns=combined_df.columns)

    # Second split: Val vs. Test
    if val_size > 0 and test_size > 0:
        val_ratio = val_size / temp_size
        val_df, test_df = train_test_split(
            temp_df, test_size=(
                1.0 - val_ratio), random_state=random_state + 1, shuffle=True
        )
    elif val_size > 0:
        val_df = temp_df.copy()
        test_df = pd.DataFrame(columns=combined_df.columns)
    elif test_size > 0:
        test_df = temp_df.copy()
        val_df = pd.DataFrame(columns=combined_df.columns)
    else:
        val_df = pd.DataFrame(columns=combined_df.columns)
        test_df = pd.DataFrame(columns=combined_df.columns)

    print(
        f"Data split: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    # === 5. Scaling ===
    scaler = StandardScaler()
    scaled_cols = data_config.get('scaled_cols', []) or []

    # Find intersection of specified scaled_cols and the actual feature_cols
    valid_scaled_cols = [
        col for col in scaled_cols if col in feature_cols and col in train_df.columns]

    if len(valid_scaled_cols) > 0:
        print(
            f"Fitting scaler on {len(valid_scaled_cols)} columns from training data.")

        # Ensure columns are float before scaling
        train_df[valid_scaled_cols] = train_df[valid_scaled_cols].astype(np.float32)
        if not val_df.empty:
            val_df[valid_scaled_cols] = val_df[valid_scaled_cols].astype(np.float32)
        if not test_df.empty:
            test_df[valid_scaled_cols] = test_df[valid_scaled_cols].astype(np.float32)

        # Fit on train only
        scaler.fit(train_df[valid_scaled_cols])

        # Apply to all splits
        train_df.loc[:, valid_scaled_cols] = scaler.transform(
            train_df[valid_scaled_cols])
        if not val_df.empty:
            val_df.loc[:, valid_scaled_cols] = scaler.transform(
                val_df[valid_scaled_cols])
        if not test_df.empty:
            test_df.loc[:, valid_scaled_cols] = scaler.transform(
                test_df[valid_scaled_cols])
                
    else:
        print("No valid columns specified for scaling, or scaled_cols is empty.")

    return train_df, val_df, test_df, feature_cols, target_col, scaler
