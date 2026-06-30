import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.preprocessing import RobustScaler
from torch.utils.data import DataLoader, Dataset

# Data loading and column renaming
CSV_PATH = Path("data/coal_gas_20250826.csv")


def load_data_csv(csv_path=CSV_PATH, nrows=None) -> pd.DataFrame:
    df = pd.read_csv(csv_path, encoding="utf-8-sig", nrows=nrows)
    rename_map = {
        "moisture(%)": "moisture_wt",
        "ash(%)": "ash_wt",
        "volatile(%)": "volatile_wt",
        "temperature(℃)": "temperature_C",
        "pressure(MPa)": "pressure_MPa",
        "adsorption capacity(m3/t)": "CH4_adsorption_m3t",
    }
    df = df.rename(columns=rename_map)
    return df


# Basic validation
def validate_data(df: pd.DataFrame) -> pd.DataFrame:
    # Check for missing values
    required_columns = [
        "moisture_wt",
        "ash_wt",
        "volatile_wt",
        "temperature_C",
        "pressure_MPa",
        "CH4_adsorption_m3t",
    ]
    missing_columns = []

    for col in required_columns:
        if col not in df.columns:
            missing_columns.append(col)

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    if df[required_columns].isna().any().any():
        raise ValueError("Dataset contains missing values.")

    # pressure should be positive for Sips model
    if not (df["pressure_MPa"] > 0).all():
        raise ValueError("Sips model requires pressure_MPa > 0 for all samples.")

    min_pressure = float(df["pressure_MPa"].min())
    print(f"Minimum pressure: {min_pressure:.7f} MPa")

    if min_pressure < 0.01:
        warnings.warn(
            f"Minimum pressure is {min_pressure:.7f} MPa, below 0.01 MPa.",
            RuntimeWarning,
            stacklevel=2,
        )

    # we use log transform for adsorption capacity, so it must be positive.
    if not (df["CH4_adsorption_m3t"] > 0).all():
        raise ValueError("Adsorption values must be positive for log transform.")

    # Adding a new column for temperature in Kelvin, as it needed for physical formulas.
    df = df.copy()
    df["temperature_K"] = df["temperature_C"] + 273.15

    return df


# Infer experiment IDs from repeated coal-condition curves
def infer_experiment_id(df: pd.DataFrame) -> pd.DataFrame:
    group_columns = [
        "moisture_wt",
        "ash_wt",
        "volatile_wt",
        "temperature_C",
    ]

    df = df.copy()

    df["experiment_id"] = df.groupby(group_columns, sort=False).ngroup().add(1)

    df["experiment_id"] = df["experiment_id"].map(lambda i: f"exp_{i:03d}")

    return df


# 12 Feature engineering; input variables for the model.
def engineer_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    df = df.copy()

    df["Tr"] = df["temperature_K"] / 190.6
    df["Pr"] = df["pressure_MPa"] / 4.60
    df["FC"] = 100 - df["moisture_wt"] - df["ash_wt"] - df["volatile_wt"]
    df["OM"] = (df["volatile_wt"] + df["FC"]) / 100
    df["beta"] = 1 / (8.314 * df["temperature_K"])
    df["PT"] = df["pressure_MPa"] * df["temperature_K"]
    df["MV"] = df["moisture_wt"] * df["volatile_wt"]

    feature_names = [
        "moisture_wt",
        "ash_wt",
        "volatile_wt",
        "temperature_K",
        "pressure_MPa",
        "Tr",
        "Pr",
        "FC",
        "OM",
        "beta",
        "PT",
        "MV",
    ]

    return df, feature_names


def select_stratified_test_experiments(
    group_summary: pd.DataFrame,
    joint_strata: pd.Series,
    test_size: float,
    random_state: int,
) -> set:
    n_groups = len(group_summary)
    n_test_groups = int(np.ceil(test_size * n_groups))
    stratum_counts = joint_strata.value_counts(sort=False)

    if (
        (stratum_counts < 2).any()
        or n_test_groups < len(stratum_counts)
        or n_groups - n_test_groups < len(stratum_counts)
    ):
        raise ValueError("Joint strata are too sparse for a leakage-free split.")

    desired_counts = stratum_counts * test_size
    test_counts = np.floor(desired_counts).astype(int)
    test_counts = test_counts.clip(lower=1, upper=stratum_counts - 1)

    while test_counts.sum() < n_test_groups:
        candidates = test_counts[test_counts < stratum_counts - 1].index
        residuals = desired_counts[candidates] - test_counts[candidates]
        test_counts[residuals.idxmax()] += 1

    while test_counts.sum() > n_test_groups:
        candidates = test_counts[test_counts > 1].index
        excess = test_counts[candidates] - desired_counts[candidates]
        test_counts[excess.idxmax()] -= 1

    rng = np.random.default_rng(random_state)
    test_experiment_ids = set()

    for stratum, n_selected in test_counts.items():
        stratum_ids = group_summary.index[joint_strata == stratum].to_numpy(copy=True)
        rng.shuffle(stratum_ids)
        test_experiment_ids.update(stratum_ids[:n_selected])

    return test_experiment_ids


# Experiment-level train/test split with joint temperature-pressure stratification
def make_main_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    n_strata_bins: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    group_summary = df.groupby("experiment_id", sort=False).agg(
        temperature_K=("temperature_K", "median"),
        max_pressure_MPa=("pressure_MPa", "max"),
    )

    T1_experiment_ids = None
    last_error = None

    for n_bins in range(n_strata_bins, 1, -1):
        temperature_bin = pd.qcut(
            group_summary["temperature_K"],
            q=n_bins,
            labels=False,
            duplicates="drop",
        )
        pressure_bin = pd.qcut(
            group_summary["max_pressure_MPa"],
            q=n_bins,
            labels=False,
            duplicates="drop",
        )
        joint_strata = temperature_bin.astype(str) + "_" + pressure_bin.astype(str)

        try:
            T1_experiment_ids = select_stratified_test_experiments(
                group_summary,
                joint_strata,
                test_size,
                random_state,
            )
            break
        except ValueError as error:
            last_error = error

    if T1_experiment_ids is None:
        raise ValueError(
            "Unable to create a stratified experiment-level main split."
        ) from last_error

    train_experiment_ids = set(group_summary.index) - T1_experiment_ids

    train_df = (
        df[df["experiment_id"].isin(train_experiment_ids)].copy().reset_index(drop=True)
    )
    T1_df = (
        df[df["experiment_id"].isin(T1_experiment_ids)].copy().reset_index(drop=True)
    )

    train_experiments = set(train_df["experiment_id"])
    T1_experiments = set(T1_df["experiment_id"])

    overlap = train_experiments.intersection(T1_experiments)

    if overlap:
        raise ValueError(f"Experiment leakage detected: {overlap}")
    return train_df, T1_df


def make_t3_split(
    T1_df: pd.DataFrame,
    pressure_threshold: float = 8.0,
) -> pd.DataFrame:

    T3_df = (
        T1_df[T1_df["pressure_MPa"] > pressure_threshold].copy().reset_index(drop=True)
    )

    if T3_df.empty:
        raise ValueError(f"T3 has no samples above {pressure_threshold} MPa.")

    return T3_df


# Experiment-level train/validation split
def make_train_val_split(
    train_df: pd.DataFrame,
    val_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=val_size,
        random_state=random_state,
    )

    train_idx, val_idx = next(
        splitter.split(train_df, groups=train_df["experiment_id"])
    )

    final_train_df = train_df.iloc[train_idx].reset_index(drop=True)
    val_df = train_df.iloc[val_idx].reset_index(drop=True)

    train_experiments = set(final_train_df["experiment_id"])
    val_experiments = set(val_df["experiment_id"])

    overlap = train_experiments.intersection(val_experiments)

    if overlap:
        raise ValueError(f"Train/validation leakage detected: {overlap}")

    return final_train_df, val_df


# 5-fold GroupKFold cross-validation
def make_cv_folds(
    train_val_df: pd.DataFrame,
    n_splits: int = 5,
) -> list[dict]:

    splitter = GroupKFold(n_splits=n_splits)
    fold_indices = []

    groups = train_val_df["experiment_id"]

    for fold_number, (train_idx, val_idx) in enumerate(
        splitter.split(train_val_df, groups=groups),
        start=1,
    ):
        fold_train_df = train_val_df.iloc[train_idx]
        fold_val_df = train_val_df.iloc[val_idx]

        train_experiments = set(fold_train_df["experiment_id"])
        val_experiments = set(fold_val_df["experiment_id"])

        overlap = train_experiments.intersection(val_experiments)

        if overlap:
            raise ValueError(f"Experiment leakage in fold {fold_number}: {overlap}")

        fold_indices.append(
            {
                "fold": fold_number,
                "train_idx": train_idx,
                "val_idx": val_idx,
            }
        )

    return fold_indices


# Scaling and target transform
def scale_features_and_target(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    T1_df: pd.DataFrame,
    T3_df: pd.DataFrame,
    feature_names: list[str],
):
    scaler = RobustScaler()

    X_train = scaler.fit_transform(train_df[feature_names])
    X_val = scaler.transform(val_df[feature_names])
    X_T1 = scaler.transform(T1_df[feature_names])
    X_T3 = scaler.transform(T3_df[feature_names])

    y_train = np.log(train_df["CH4_adsorption_m3t"].to_numpy()).reshape(-1, 1)
    y_val = np.log(val_df["CH4_adsorption_m3t"].to_numpy()).reshape(-1, 1)
    y_T1 = np.log(T1_df["CH4_adsorption_m3t"].to_numpy()).reshape(-1, 1)
    y_T3 = np.log(T3_df["CH4_adsorption_m3t"].to_numpy()).reshape(-1, 1)

    raw_pressure_train = train_df["pressure_MPa"].to_numpy().reshape(-1, 1)
    raw_pressure_val = val_df["pressure_MPa"].to_numpy().reshape(-1, 1)
    raw_pressure_T1 = T1_df["pressure_MPa"].to_numpy().reshape(-1, 1)
    raw_pressure_T3 = T3_df["pressure_MPa"].to_numpy().reshape(-1, 1)

    experiment_id_train = train_df["experiment_id"].str[4:].astype(np.int64).to_numpy()
    experiment_id_val = val_df["experiment_id"].str[4:].astype(np.int64).to_numpy()
    experiment_id_T1 = T1_df["experiment_id"].str[4:].astype(np.int64).to_numpy()
    experiment_id_T3 = T3_df["experiment_id"].str[4:].astype(np.int64).to_numpy()

    return {
        "scaler": scaler,
        "X_train": X_train,
        "X_val": X_val,
        "X_T1": X_T1,
        "X_T3": X_T3,
        "y_train": y_train,
        "y_val": y_val,
        "y_T1": y_T1,
        "y_T3": y_T3,
        "raw_pressure_train": raw_pressure_train,
        "raw_pressure_val": raw_pressure_val,
        "raw_pressure_T1": raw_pressure_T1,
        "raw_pressure_T3": raw_pressure_T3,
        "experiment_id_train": experiment_id_train,
        "experiment_id_val": experiment_id_val,
        "experiment_id_T1": experiment_id_T1,
        "experiment_id_T3": experiment_id_T3,
    }


# final verification function for all splits.
def verify_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    T1_df: pd.DataFrame,
    T3_df: pd.DataFrame,
    pressure_threshold: float = 8.0,
):
    train_experiments = set(train_df["experiment_id"])
    val_experiments = set(val_df["experiment_id"])
    T1_experiments = set(T1_df["experiment_id"])
    T3_experiments = set(T3_df["experiment_id"])

    if train_experiments.intersection(val_experiments):
        raise ValueError("Train/validation leakage detected.")
    if train_experiments.intersection(T1_experiments):
        raise ValueError("Train/T1 leakage detected.")
    if val_experiments.intersection(T1_experiments):
        raise ValueError("Validation/T1 leakage detected.")

    if not T3_experiments.issubset(T1_experiments):
        raise ValueError("T3 experiments must belong to T1.")

    T1_rows = pd.MultiIndex.from_frame(T1_df)
    T3_rows = pd.MultiIndex.from_frame(T3_df)
    if not T3_rows.isin(T1_rows).all():
        raise ValueError("T3 must contain only rows from T1.")

    splits = {
        "train": train_df,
        "validation": val_df,
        "T1": T1_df,
        "T3": T3_df,
    }

    for split_name, split_df in splits.items():
        if not (split_df["pressure_MPa"] > 0).all():
            raise ValueError(f"{split_name} contains non-positive pressure.")

    if not (T3_df["pressure_MPa"] > pressure_threshold).all():
        raise ValueError(f"T3 contains pressure at or below {pressure_threshold} MPa.")


# PyTorch dataset
class AdsorptionDataset(Dataset):
    def __init__(self, X, y, raw_pressure, experiment_id):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.raw_pressure = torch.tensor(raw_pressure, dtype=torch.float32)
        self.experiment_id = torch.tensor(experiment_id, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return {
            "x": self.X[idx],
            "y": self.y[idx],
            "raw_pressure": self.raw_pressure[idx],
            "experiment_id": self.experiment_id[idx],
        }


# PyTorch DataLoaders
def make_dataloaders(arrays: dict, batch_size: int = 64):
    train_dataset = AdsorptionDataset(
        arrays["X_train"],
        arrays["y_train"],
        arrays["raw_pressure_train"],
        arrays["experiment_id_train"],
    )

    val_dataset = AdsorptionDataset(
        arrays["X_val"],
        arrays["y_val"],
        arrays["raw_pressure_val"],
        arrays["experiment_id_val"],
    )

    T1_dataset = AdsorptionDataset(
        arrays["X_T1"],
        arrays["y_T1"],
        arrays["raw_pressure_T1"],
        arrays["experiment_id_T1"],
    )

    T3_dataset = AdsorptionDataset(
        arrays["X_T3"],
        arrays["y_T3"],
        arrays["raw_pressure_T3"],
        arrays["experiment_id_T3"],
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    T1_loader = DataLoader(
        T1_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    T3_loader = DataLoader(
        T3_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    return train_loader, val_loader, T1_loader, T3_loader


def make_cv_fold_data(
    train_val_df: pd.DataFrame,
    T1_df: pd.DataFrame,
    T3_df: pd.DataFrame,
    feature_names: list[str],
    fold_indices: list[dict],
    batch_size: int = 64,
) -> list[dict]:
    cv_folds = []

    for fold in fold_indices:
        fold_train_df = (
            train_val_df.iloc[fold["train_idx"]].copy().reset_index(drop=True)
        )
        fold_val_df = train_val_df.iloc[fold["val_idx"]].copy().reset_index(drop=True)

        fold_arrays = scale_features_and_target(
            fold_train_df,
            fold_val_df,
            T1_df,
            T3_df,
            feature_names,
        )
        (
            fold_train_loader,
            fold_val_loader,
            fold_T1_loader,
            fold_T3_loader,
        ) = make_dataloaders(
            fold_arrays,
            batch_size=batch_size,
        )

        cv_folds.append(
            {
                "fold": fold["fold"],
                "train_idx": fold["train_idx"],
                "val_idx": fold["val_idx"],
                "train_loader": fold_train_loader,
                "val_loader": fold_val_loader,
                "T1_loader": fold_T1_loader,
                "T3_loader": fold_T3_loader,
                "scaler": fold_arrays["scaler"],
                "arrays": fold_arrays,
                "splits": {
                    "train": fold_train_df,
                    "validation": fold_val_df,
                },
            }
        )

    return cv_folds


def build_data_pipeline(
    csv_path: Path = CSV_PATH,
    nrows=None,
    batch_size: int = 64,
    test_size: float = 0.2,
    val_size: float = 0.2,
    pressure_threshold: float = 8.0,
    n_splits: int = 5,
    n_strata_bins: int = 3,
    random_state: int = 42,
) -> dict:
    df = load_data_csv(csv_path=csv_path, nrows=nrows)
    df = validate_data(df)
    df = infer_experiment_id(df)
    df, feature_names = engineer_features(df)
    train_val_df, T1_df = make_main_split(
        df,
        test_size=test_size,
        random_state=random_state,
        n_strata_bins=n_strata_bins,
    )
    T3_df = make_t3_split(
        T1_df,
        pressure_threshold=pressure_threshold,
    )
    train_df, val_df = make_train_val_split(
        train_val_df,
        val_size=val_size,
        random_state=random_state,
    )
    fold_indices = make_cv_folds(
        train_val_df,
        n_splits=n_splits,
    )
    verify_splits(
        train_df,
        val_df,
        T1_df,
        T3_df,
        pressure_threshold=pressure_threshold,
    )

    arrays = scale_features_and_target(
        train_df,
        val_df,
        T1_df,
        T3_df,
        feature_names,
    )
    train_loader, val_loader, T1_loader, T3_loader = make_dataloaders(
        arrays,
        batch_size=batch_size,
    )
    cv_folds = make_cv_fold_data(
        train_val_df,
        T1_df,
        T3_df,
        feature_names,
        fold_indices,
        batch_size=batch_size,
    )

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "T1_loader": T1_loader,
        "T3_loader": T3_loader,
        "scaler": arrays["scaler"],
        "feature_names": feature_names,
        "fold_indices": fold_indices,
        "cv_folds": cv_folds,
        "arrays": arrays,
        "splits": {
            "full": df,
            "train_val": train_val_df,
            "train": train_df,
            "validation": val_df,
            "T1": T1_df,
            "T3": T3_df,
        },
        "raw_pressure": {
            "train": train_loader.dataset.raw_pressure,
            "validation": val_loader.dataset.raw_pressure,
            "T1": T1_loader.dataset.raw_pressure,
            "T3": T3_loader.dataset.raw_pressure,
        },
        "experiment_id": {
            "train": train_loader.dataset.experiment_id,
            "validation": val_loader.dataset.experiment_id,
            "T1": T1_loader.dataset.experiment_id,
            "T3": T3_loader.dataset.experiment_id,
        },
    }


if __name__ == "__main__":
    pipeline = build_data_pipeline()
    splits = pipeline["splits"]
    arrays = pipeline["arrays"]

    df = splits["full"]
    train_df = splits["train"]
    val_df = splits["validation"]
    T1_df = splits["T1"]
    T3_df = splits["T3"]

    print("Full data:", df.shape)
    print("Train data:", train_df.shape)
    print("Validation data:", val_df.shape)
    print("T1 data:", T1_df.shape)
    print("T3 data:", T3_df.shape)
    print()

    print("Experiment overlap:")
    print(set(train_df["experiment_id"]).intersection(set(T1_df["experiment_id"])))
    print(set(val_df["experiment_id"]).intersection(set(train_df["experiment_id"])))
    print()

    print("Total experiments:", df["experiment_id"].nunique())
    print("Train experiments:", train_df["experiment_id"].nunique())
    print("Validation experiments:", val_df["experiment_id"].nunique())
    print("T1 experiments:", T1_df["experiment_id"].nunique())
    print("T3 experiments:", T3_df["experiment_id"].nunique())
    print()

    for fold in pipeline["fold_indices"]:
        print(
            f"Fold {fold['fold']}: "
            f"{len(fold['train_idx'])} training samples, "
            f"{len(fold['val_idx'])} validation samples"
        )

    print("X_train:", arrays["X_train"].shape)
    print("y_train:", arrays["y_train"].shape)
    print("raw_pressure_train:", arrays["raw_pressure_train"].shape)

    batch = next(iter(pipeline["train_loader"]))
    print("Batch x:", batch["x"].shape)
    print("Batch y:", batch["y"].shape)
    print("Batch raw pressure:", batch["raw_pressure"].shape)
    print("Batch experiment ID:", batch["experiment_id"].shape)
