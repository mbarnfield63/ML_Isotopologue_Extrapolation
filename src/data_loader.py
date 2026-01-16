import pandas as pd
import numpy as np
import torch
import os  # Added to handle filename parsing
from torch.utils.data import Dataset, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder


class MoleculeDataset(Dataset):
    """
    Generic Dataset for molecular energy regression.
    Updated to support molecule and isotopologue indices for embedding models.
    """

    def __init__(
        self, df, feature_cols, target_col, mol_idx_col=None, iso_idx_col=None
    ):
        """
        Args:
            df (pd.DataFrame): DataFrame containing the data.
            feature_cols (list[str]): List of feature column names.
            target_col (str): The name of the target column.
            mol_idx_col (str, optional): Name of the encoded molecule index column.
            iso_idx_col (str, optional): Name of the encoded isotopologue index column.
        """
        self.X = df[feature_cols].values.astype(np.float32)
        self.y = df[target_col].values.astype(np.float32).reshape(-1, 1)

        # Store indices if provided
        self.mol_idx = None
        self.iso_idx = None

        if mol_idx_col and mol_idx_col in df.columns:
            self.mol_idx = torch.tensor(df[mol_idx_col].values.astype(np.int64))

        if iso_idx_col and iso_idx_col in df.columns:
            self.iso_idx = torch.tensor(df[iso_idx_col].values.astype(np.int64))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Basic return: X, y
        items = [torch.from_numpy(self.X[idx]), torch.from_numpy(self.y[idx])]

        # Add indices if they exist (for combined regressors)
        if self.mol_idx is not None:
            items.append(self.mol_idx[idx])
        else:
            items.append(torch.tensor(-1))  # Dummy value

        if self.iso_idx is not None:
            items.append(self.iso_idx[idx])
        else:
            items.append(torch.tensor(-1))  # Dummy value

        return tuple(items)


def calculate_class_weights(df, iso_col, weight_config):
    """
    Calculates weights for each class based on inverse frequency and manual overrides.
    Returns a dictionary: {iso_label: weight_value}
    """
    if not iso_col or iso_col not in df.columns:
        return {}

    # 1. Calculate base weights (inverse frequency)
    iso_counts = df[iso_col].value_counts()
    total_samples = len(df)
    n_classes = len(iso_counts)

    # Base weight = Total / (Count * N_Classes)
    class_weights = {
        iso: total_samples / (count * n_classes) for iso, count in iso_counts.items()
    }

    # 2. Apply manual overrides from config
    manual_weights = weight_config.get("manual_weights", {})
    first_val = df[iso_col].iloc[0]

    for iso_key, multiplier in manual_weights.items():
        # Match key type to dataframe column type
        try:
            if isinstance(first_val, (int, np.integer)):
                iso_key = int(iso_key)
            elif isinstance(first_val, str):
                iso_key = str(iso_key)
        except:
            pass

        if iso_key in class_weights:
            class_weights[iso_key] *= multiplier

    return class_weights


def get_weighted_sampler(df, iso_col, weight_config):
    """
    Creates a WeightedRandomSampler using calculate_class_weights.
    """
    # Reuse the logic above!
    class_weights = calculate_class_weights(df, iso_col, weight_config)

    if not class_weights:
        return None

    # Assign a weight to every sample
    sample_weights = df[iso_col].map(class_weights).fillna(0).values
    sample_weights = torch.from_numpy(sample_weights).double()

    # Create sampler
    sampler = WeightedRandomSampler(
        sample_weights, len(sample_weights), replacement=True
    )

    return sampler


def load_data(config):
    """
    Loads, combines, splits, and scales data based on the config file.
    Now handles molecule/iso encoding.

    Returns:
        tuple: (
            train_df, val_df, test_df,
            feature_cols, target_col, scaler,
            n_molecules, n_isos
        )
    """
    data_config = config["data"]
    exp_config = config["experiment"]

    # === 1. Load and Combine Data ===
    training_data = data_config["training"]
    if not isinstance(training_data, list):
        training_data = [training_data]

    print("Loading data:")
    for path in training_data:
        if not path:
            continue  # Skip empty path entries
        print(f"- {path}")
        try:
            df = pd.read_csv(path)

        except FileNotFoundError:
            print(f"  WARNING: File not found, skipping: {path}")

    # === 2. Pre-processing & Filtering ===
    target_col = data_config["target_col"]
    not_feature_cols_config = data_config.get("not_feature_cols", []) or []

    # Filter by isotopologues if specified
    filter_out_isos = data_config.get("filter_out_isos", []) or []
    if filter_out_isos:
        initial_count = len(df)
        df = df[~df["iso"].isin(filter_out_isos)]
        dropped = initial_count - len(df)
        if dropped > 0:
            print(f"Dropped {dropped} rows where iso in {filter_out_isos}")

    # Drop NaNs
    critical_cols = [target_col] + not_feature_cols_config
    critical_cols = [col for col in critical_cols if col and col in df.columns]

    if critical_cols:
        initial_count = len(df)
        df = df.dropna(subset=critical_cols)
        if len(df) < initial_count:
            print(f"Dropped {initial_count - len(df)} rows with NaNs.")

    # === 3. Determine Feature Columns ===
    # This section implements the `not_feature_cols` logic
    all_cols_in_df = set(df.columns)
    not_feature_cols_set = set(not_feature_cols_config)
    not_feature_cols_set.add(target_col)

    feature_cols = [col for col in all_cols_in_df if col not in not_feature_cols_set]
    feature_cols.sort()
    print(f"Determined {len(feature_cols)} feature columns.")

    # === 4. Encode Molecule/Iso Indices ===
    # Required for embedding layers in combination regressors
    mol_col = data_config.get(
        "molecule_col", "molecule"
    )  # Default column name 'molecule'
    iso_col = data_config.get("iso_col", "iso")  # Default column name 'iso'

    n_molecules = 0
    n_isos = 0

    # Handle Molecule Index
    if mol_col in df.columns:
        # Even if numeric, encoding required to ensure 0 to N-1 contiguous indices for Embeddings
        print(f"Encoding molecule column '{mol_col}'...")
        df["molecule_idx_encoded"] = LabelEncoder().fit_transform(
            df[mol_col].astype(str)
        )
        data_config["molecule_idx_col"] = "molecule_idx_encoded"
        n_molecules = df["molecule_idx_encoded"].nunique()
        print(f"Found {n_molecules} unique molecules.")
    else:
        # Fallback: if no molecule col, assume 1 molecule (idx 0)
        df["molecule_idx_encoded"] = 0
        data_config["molecule_idx_col"] = "molecule_idx_encoded"
        n_molecules = 1

    # Handle Iso Index
    if iso_col in df.columns:
        print(f"Encoding isotopologue column '{iso_col}' to 0..N indices...")
        df["iso_idx_encoded"] = LabelEncoder().fit_transform(df[iso_col].astype(str))
        data_config["iso_idx_col"] = "iso_idx_encoded"
        n_isos = df["iso_idx_encoded"].nunique()
        print(f"Found {n_isos} unique isotopologues.")
    else:
        df["iso_idx_encoded"] = 0
        data_config["iso_idx_col"] = "iso_idx_encoded"
        n_isos = 1

    # === 5. Train/Val/Test Split ===
    train_size = exp_config.get("train_size", 0.7)
    val_size = exp_config.get("val_size", 0.1)
    test_size = exp_config.get("test_size", 0.2)
    random_state = exp_config.get("seed", 42)

    # Ensure sizes sum to 1
    if not np.isclose(train_size + val_size + test_size, 1.0):
        raise ValueError(
            f"ERROR: Split sizes (train={train_size}, val={val_size}, test={test_size}) do not sum to 1. Please ensure they add up to 1."
        )

    # First split: Train vs. (Val + Test)
    temp_size = val_size + test_size
    if temp_size > 0:
        train_df, temp_df = train_test_split(
            df, test_size=temp_size, random_state=random_state, shuffle=True
        )
    else:
        train_df = df.copy()
        temp_df = pd.DataFrame(columns=df.columns)

    # Second split: Val vs. Test
    if val_size > 0 and test_size > 0:
        val_ratio = val_size / temp_size
        val_df, test_df = train_test_split(
            temp_df,
            test_size=(1.0 - val_ratio),
            random_state=random_state + 1,
            shuffle=True,
        )
    elif val_size > 0:
        val_df = temp_df.copy()
        test_df = pd.DataFrame(columns=df.columns)
    elif test_size > 0:
        test_df = temp_df.copy()
        val_df = pd.DataFrame(columns=df.columns)
    else:
        val_df = pd.DataFrame(columns=df.columns)
        test_df = pd.DataFrame(columns=df.columns)

    print(f"Data split: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    # === 6. Scaling ===
    scaler = StandardScaler()
    scaled_cols = data_config.get("scaled_cols", []) or []
    valid_scaled_cols = [
        col for col in scaled_cols if col in feature_cols and col in train_df.columns
    ]

    # Only scale globally if NOT running CV
    # Scaling is handled per-fold in that case
    is_cv = exp_config.get("type") == "cv"

    if valid_scaled_cols and not is_cv:
        print(f"Fitting scaler on {len(valid_scaled_cols)} columns...")
        # Ensure floats
        train_df[valid_scaled_cols] = train_df[valid_scaled_cols].astype(np.float32)
        if not val_df.empty:
            val_df[valid_scaled_cols] = val_df[valid_scaled_cols].astype(np.float32)
        if not test_df.empty:
            test_df[valid_scaled_cols] = test_df[valid_scaled_cols].astype(np.float32)

        scaler.fit(train_df[valid_scaled_cols])
        train_df.loc[:, valid_scaled_cols] = scaler.transform(
            train_df[valid_scaled_cols]
        )
        if not val_df.empty:
            val_df.loc[:, valid_scaled_cols] = scaler.transform(
                val_df[valid_scaled_cols]
            )
        if not test_df.empty:
            test_df.loc[:, valid_scaled_cols] = scaler.transform(
                test_df[valid_scaled_cols]
            )

    elif is_cv:
        print(
            "Experiment type is 'cv'. Skipping global scaling (will be handled per-fold)."
        )

    # === 7. Weighted Sampler ===
    # Only weight the training set
    weight_config = config.get("weighting", {})
    train_sampler = None

    if weight_config.get("enabled", False):
        print("Creating weighted sampler for training data...")
        # Use the original iso column (not the encoded index) for easier config matching
        train_sampler = get_weighted_sampler(train_df, iso_col, weight_config)
        if train_sampler:
            print(
                f"  Sampler created. Manual weights: {weight_config.get('manual_weights', 'None')}"
            )

    return (
        train_df,
        val_df,
        test_df,
        feature_cols,
        target_col,
        scaler,
        n_molecules,
        n_isos,
        train_sampler,
    )
