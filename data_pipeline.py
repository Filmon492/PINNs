import numpy as np 
from pathlib import Path
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

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
    
    train_idx, test_idx = next(splitter.split(df, groups=df["experiment_id"]))
    
    train_df = df.iloc[train_idx].reset_index(drop=True) # contains the rows used for training the model.
    test_df = df.iloc[test_idx].reset_index(drop=True) # contains the rows used for testing the model.

    train_experiments = set(train_df["experiment_id"]) # contains the unique experiment IDs in the training set.
    test_experiments = set(test_df["experiment_id"]) # contains the unique experiment IDs in the test set.

    overlap = train_experiments.intersection(test_experiments)

    if overlap:
        raise ValueError(f"Experiment leakage detected: {overlap}")
    return train_df, test_df

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

if __name__== "__main__":
    df = load_data_csv()
    df = validate_data(df)
    df = infer_experiment_id(df)
    df, feature_names = engineer_features(df)
    train_val_df, test_df = make_main_split(df)
    train_df, val_df = make_train_val_split(train_val_df)

    print("Full data:", df.shape)
    print("Train data:", train_df.shape)
    print("Validation data:", val_df.shape)
    print("Test data:", test_df.shape)
    print()

    print("Experiment overlap:")
    print(set(train_df["experiment_id"]).intersection(set(test_df["experiment_id"])))
    print(set(val_df["experiment_id"]).intersection(set(train_df["experiment_id"])))
    print()

    print("Total experiments:", df["experiment_id"].nunique())
    print("Train experiments:", train_df["experiment_id"].nunique())
    print("Validation experiments:", val_df["experiment_id"].nunique())
    print("Test experiments:", test_df["experiment_id"].nunique())