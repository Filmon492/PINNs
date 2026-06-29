import numpy as np 
from pathlib import Path
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import RobustScaler
import torch
from torch.utils.data import Dataset, DataLoader

# Data loading and column renaming
CSV_PATH = Path("data/coal_gas_20250826.csv")

def load_data_csv(csv_path= CSV_PATH, nrows= None)-> pd.DataFrame:
    df = pd.read_csv(csv_path, encoding="utf-8-sig", nrows=nrows)
    rename_map = {
        "moisture(%)": "moisture_wt",
        "ash(%)": "ash_wt",
        "volatile(%)": "volatile_wt",
        "temperature(℃)": "temperature_C",
        "pressure(MPa)": "pressure_MPa",
        "adsorption capacity(m3/t)": "CH4_adsorption_m3t",
    }
    df= df.rename(columns= rename_map)
    return df


# Basic validation
def validate_data(df: pd.DataFrame)-> pd.DataFrame:
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

    df["experiment_id"] = (
        df.groupby(group_columns, sort=False).ngroup().add(1))

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

# Experiment-level train/test split
def make_main_split(df: pd.DataFrame, test_size=0.2, random_state=42) -> tuple[pd.DataFrame, pd.DataFrame]:
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    
    train_idx, T1_idx = next(splitter.split(df, groups=df["experiment_id"]))
    
    train_df = df.iloc[train_idx].reset_index(drop=True) # contains the rows used for training the model.
    T1_df = df.iloc[T1_idx].reset_index(drop=True) # contains the rows used for T1 interpolation testing.

    train_experiments = set(train_df["experiment_id"]) # contains the unique experiment IDs in the training set.
    T1_experiments = set(T1_df["experiment_id"]) # contains the unique experiment IDs in the T1 set.

    overlap = train_experiments.intersection(T1_experiments)

    if overlap:
        raise ValueError(f"Experiment leakage detected: {overlap}")
    return train_df, T1_df

def make_t3_split(
    T1_df: pd.DataFrame,
    pressure_threshold: float = 8.0,
) -> pd.DataFrame:

    T3_df = T1_df[
        T1_df["pressure_MPa"] > pressure_threshold
    ].copy().reset_index(drop=True)

    if T3_df.empty:
        raise ValueError(
            f"T3 has no samples above {pressure_threshold} MPa."
        )

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
    }

# PyTorch dataset
class AdsorptionDataset(Dataset):
    def __init__(self, X, y, raw_pressure):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.raw_pressure = torch.tensor(raw_pressure, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return {
            "x": self.X[idx],
            "y": self.y[idx],
            "raw_pressure": self.raw_pressure[idx],
        }

    # PyTorch DataLoaders
def make_dataloaders(arrays: dict, batch_size: int = 64):
    train_dataset = AdsorptionDataset(
        arrays["X_train"],
        arrays["y_train"],
        arrays["raw_pressure_train"],
    )

    val_dataset = AdsorptionDataset(
        arrays["X_val"],
        arrays["y_val"],
        arrays["raw_pressure_val"],
    )

    T1_dataset = AdsorptionDataset(
        arrays["X_T1"],
        arrays["y_T1"],
        arrays["raw_pressure_T1"],
    )

    T3_dataset = AdsorptionDataset(
        arrays["X_T3"],
        arrays["y_T3"],
        arrays["raw_pressure_T3"],
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

if __name__== "__main__":
    df = load_data_csv()
    df = validate_data(df)
    df = infer_experiment_id(df)
    df, feature_names = engineer_features(df)
    train_val_df, T1_df = make_main_split(df)
    T3_df = make_t3_split(T1_df)
    train_df, val_df = make_train_val_split(train_val_df)

   
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

    arrays = scale_features_and_target(
        train_df,
        val_df,
        T1_df,
        T3_df,
        feature_names,
    )
    print("X_train:", arrays["X_train"].shape)
    print("y_train:", arrays["y_train"].shape)
    print("raw_pressure_train:", arrays["raw_pressure_train"].shape)

    train_loader, val_loader, T1_loader, T3_loader = make_dataloaders(arrays)

    batch = next(iter(train_loader))
    print("Batch x:", batch["x"].shape)
    print("Batch y:", batch["y"].shape)
    print("Batch raw pressure:", batch["raw_pressure"].shape)
