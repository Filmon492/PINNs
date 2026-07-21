from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.optimize import curve_fit
from scipy.stats import f_oneway, pearsonr, spearmanr
from statsmodels.stats.multicomp import pairwise_tukeyhsd

from ablation_analysis import VARIANT_SPECS, load_variant_model
from data_pipeline import build_data_pipeline
from result_paths import ensure_step_results_dir
from sips_gauge_variant import compute_parameter_shap_summary

ANALYSIS_SPLITS = ("train", "validation", "T1", "T2", "T3")
UNIQUE_ANALYSIS_SPLITS = ("train", "validation", "T1", "T2")
SPLIT_SUFFIXES = {
    "train": "train",
    "validation": "val",
    "val": "val",
    "T1": "T1",
    "T2": "T2",
    "T3": "T3",
}
V5_PARAMETER_COLUMNS = ("q_max", "K", "n")
V5_SHAP_PARAMETER_ORDER = {
    name: index for index, name in enumerate(V5_PARAMETER_COLUMNS)
}
DEFAULT_LITERATURE_DH_RANGE_KJMOL = (-60.0, -5.0)
BUSCH_GENSTERBLUM_REFERENCE = (
    "Busch and Gensterblum (2011), International Journal of Coal Geology 87, 49-71"
)


# Select a variant definition from the shared ablation registry.
def get_variant_spec(variant_id: str):
    for spec in VARIANT_SPECS:
        if spec.variant_id == variant_id:
            return spec

    raise KeyError(f"Unknown variant_id: {variant_id}")


# Load a trained blueprint variant from the checkpoints directory.
def load_trained_variant(
    variant_id: str,
    checkpoint_dir=Path("checkpoints"),
    device: str = "cpu",
):
    spec = get_variant_spec(variant_id)

    return load_variant_model(spec, Path(checkpoint_dir), device=device)


# Convert one named split into model tensors and matching metadata.
def get_split_inputs(
    pipeline: dict,
    split_name: str,
    device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor, pd.DataFrame]:
    if split_name not in SPLIT_SUFFIXES:
        raise KeyError(f"Unsupported split: {split_name}")

    suffix = SPLIT_SUFFIXES[split_name]
    split_key = "validation" if split_name == "val" else split_name
    split_df = pipeline["splits"][split_key].reset_index(drop=True)
    arrays = pipeline["arrays"]

    x = torch.tensor(
        arrays[f"X_{suffix}"],
        dtype=torch.float32,
        device=device,
    )
    raw_pressure = torch.tensor(
        arrays[f"raw_pressure_{suffix}"],
        dtype=torch.float32,
        device=device,
    )

    if len(split_df) != x.shape[0]:
        raise ValueError(
            f"Split dataframe and tensor length mismatch for {split_name}."
        )

    return x, raw_pressure, split_df


# Build the common metadata columns used in Step 11 output tables.
def make_metadata_frame(split_df: pd.DataFrame, split_name: str) -> pd.DataFrame:
    metadata_columns = [
        "experiment_id",
        "coal_rank_inferred",
        "moisture_wt",
        "ash_wt",
        "volatile_wt",
        "temperature_K",
        "pressure_MPa",
        "CH4_adsorption_m3t",
    ]
    missing_columns = [col for col in metadata_columns if col not in split_df.columns]

    if missing_columns:
        raise ValueError(f"Missing metadata columns: {missing_columns}")

    metadata = split_df.loc[:, metadata_columns].copy()
    metadata.insert(0, "split", split_name)
    metadata = metadata.rename(columns={"coal_rank_inferred": "coal_rank"})

    return metadata


# Collect V5 Sips gauge parameters for each sample in the chosen splits.
def collect_v5_parameter_table(
    model,
    pipeline: dict,
    split_names: tuple[str, ...] = ANALYSIS_SPLITS,
    device: str = "cpu",
) -> pd.DataFrame:
    rows = []
    model = model.to(device)
    was_training = model.training
    model.eval()

    try:
        with torch.inference_mode():
            for split_name in split_names:
                x, raw_pressure, split_df = get_split_inputs(
                    pipeline,
                    split_name,
                    device=device,
                )
                output = model(x, raw_pressure)
                metadata = make_metadata_frame(split_df, split_name)

                for column in ("q_hat", *V5_PARAMETER_COLUMNS):
                    if column not in output:
                        raise KeyError(f"V5 output is missing {column}.")

                    metadata[column] = output[column].detach().cpu().numpy().reshape(-1)

                rows.append(metadata)
    finally:
        model.train(was_training)

    return pd.concat(rows, ignore_index=True)


# Use one row per experiment when analyzing composition-level Sips parameters.
def make_experiment_level_v5_table(
    parameter_table: pd.DataFrame,
    split_names: tuple[str, ...] = UNIQUE_ANALYSIS_SPLITS,
) -> pd.DataFrame:
    filtered = parameter_table[parameter_table["split"].isin(split_names)].copy()

    return filtered.drop_duplicates(["experiment_id"]).reset_index(drop=True)


# Summarize V5 Sips parameters by split and coal rank.
def summarize_v5_parameters_by_rank(parameter_table: pd.DataFrame) -> pd.DataFrame:
    return (
        parameter_table.groupby(["split", "coal_rank"], sort=True)
        .agg(
            n_experiments=("experiment_id", "nunique"),
            n_samples=("q_max", "size"),
            q_max_mean=("q_max", "mean"),
            q_max_std=("q_max", "std"),
            q_max_min=("q_max", "min"),
            q_max_max=("q_max", "max"),
            K_mean=("K", "mean"),
            K_std=("K", "std"),
            K_min=("K", "min"),
            K_max=("K", "max"),
            n_mean=("n", "mean"),
            n_std=("n", "std"),
            n_min=("n", "min"),
            n_max=("n", "max"),
        )
        .reset_index()
    )


# Check the expected negative relation between volatile matter and q_max.
def compute_qmax_volatile_correlation(
    experiment_level_table: pd.DataFrame,
    required_max_correlation: float = -0.3,
    strict: bool = False,
) -> pd.DataFrame:
    q_max = experiment_level_table["q_max"].to_numpy()
    volatile = experiment_level_table["volatile_wt"].to_numpy()

    if len(experiment_level_table) < 3:
        raise ValueError("At least three experiments are needed for Pearson r.")

    if np.allclose(q_max, q_max[0]) or np.allclose(volatile, volatile[0]):
        correlation = float("nan")
        p_value = float("nan")
    else:
        correlation, p_value = pearsonr(q_max, volatile)
        correlation = float(correlation)
        p_value = float(p_value)

    passed = bool(np.isfinite(correlation) and correlation < required_max_correlation)

    if strict and not passed:
        raise ValueError(
            "Pearson corr(q_max, volatile_wt) failed: "
            f"{correlation:.3f} is not below {required_max_correlation:.3f}."
        )

    return pd.DataFrame(
        [
            {
                "n_experiments": int(len(experiment_level_table)),
                "pearson_corr_qmax_volatile": correlation,
                "p_value": p_value,
                "required_corr_less_than": float(required_max_correlation),
                "passes_blueprint_rule": passed,
            }
        ]
    )


# Evaluate the classical Langmuir isotherm.
def langmuir_isotherm(
    pressure_MPa: np.ndarray,
    q_L: float,
    b: float,
) -> np.ndarray:
    return q_L * b * pressure_MPa / (1.0 + b * pressure_MPa)


# Fit one Langmuir curve for each inferred experiment.
def fit_langmuir_by_experiment(
    full_df: pd.DataFrame,
    min_points: int = 3,
) -> pd.DataFrame:
    rows = []

    for experiment_id, group in full_df.groupby("experiment_id", sort=True):
        pressure = group["pressure_MPa"].to_numpy(dtype=float)
        adsorption = group["CH4_adsorption_m3t"].to_numpy(dtype=float)
        coal_rank = group["coal_rank_inferred"].iloc[0]
        row = {
            "experiment_id": experiment_id,
            "coal_rank": coal_rank,
            "n_points": int(len(group)),
            "q_L": float("nan"),
            "b": float("nan"),
            "RMSE": float("nan"),
            "R2": float("nan"),
            "fit_success": False,
            "fit_message": "",
        }

        if len(group) < min_points:
            row["fit_message"] = f"Need at least {min_points} pressure points."
            rows.append(row)
            continue

        if np.allclose(pressure, pressure[0]):
            row["fit_message"] = "Pressure values are not variable."
            rows.append(row)
            continue

        initial_q_L = max(float(np.max(adsorption)) * 1.2, 1e-6)
        initial_b = 1.0 / max(float(np.median(pressure)), 1e-6)

        try:
            params, _ = curve_fit(
                langmuir_isotherm,
                pressure,
                adsorption,
                p0=(initial_q_L, initial_b),
                bounds=([1e-8, 1e-8], [np.inf, np.inf]),
                maxfev=20_000,
            )
            q_L, b = params
            prediction = langmuir_isotherm(pressure, q_L, b)
            residuals = adsorption - prediction
            ss_res = float(np.sum(residuals**2))
            ss_tot = float(np.sum((adsorption - np.mean(adsorption)) ** 2))
            row.update(
                {
                    "q_L": float(q_L),
                    "b": float(b),
                    "RMSE": float(np.sqrt(np.mean(residuals**2))),
                    "R2": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
                    "fit_success": True,
                    "fit_message": "ok",
                }
            )
        except (RuntimeError, ValueError) as error:
            row["fit_message"] = str(error)

        rows.append(row)

    return pd.DataFrame(rows)


# Summarize classical Langmuir fits by inferred coal rank.
def summarize_langmuir_by_rank(langmuir_table: pd.DataFrame) -> pd.DataFrame:
    return (
        langmuir_table.groupby("coal_rank", sort=True)
        .agg(
            n_experiments=("experiment_id", "size"),
            n_success=("fit_success", "sum"),
            q_L_mean=("q_L", "mean"),
            q_L_std=("q_L", "std"),
            b_mean=("b", "mean"),
            b_std=("b", "std"),
            RMSE_mean=("RMSE", "mean"),
            R2_mean=("R2", "mean"),
        )
        .reset_index()
    )


# Compare V5 experiment-level Sips parameters with classical Langmuir fits.
def compare_v5_to_langmuir(
    v5_experiment_table: pd.DataFrame,
    langmuir_table: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    comparison = v5_experiment_table.merge(
        langmuir_table,
        on=["experiment_id", "coal_rank"],
        how="left",
        suffixes=("_v5", "_langmuir"),
    )
    successful = comparison["fit_success"].fillna(False)

    comparison["qmax_minus_q_L"] = comparison["q_max"] - comparison["q_L"]
    comparison["K_minus_b"] = comparison["K"] - comparison["b"]
    comparison["qmax_over_q_L"] = comparison["q_max"] / comparison["q_L"]
    comparison["K_over_b"] = comparison["K"] / comparison["b"]

    summary = {
        "n_experiments": int(len(comparison)),
        "n_successful_langmuir_fits": int(successful.sum()),
        "qmax_minus_q_L_mean": float(
            comparison.loc[successful, "qmax_minus_q_L"].mean()
        ),
        "K_minus_b_mean": float(comparison.loc[successful, "K_minus_b"].mean()),
        "qmax_over_q_L_mean": float(comparison.loc[successful, "qmax_over_q_L"].mean()),
        "K_over_b_mean": float(comparison.loc[successful, "K_over_b"].mean()),
    }

    for v5_column, langmuir_column, label in (
        ("q_max", "q_L", "qmax_vs_langmuir_q_L"),
        ("K", "b", "K_vs_langmuir_b"),
    ):
        paired = comparison.loc[successful, [v5_column, langmuir_column]].dropna()

        if len(paired) < 3:
            correlation = float("nan")
            p_value = float("nan")
        else:
            correlation, p_value = pearsonr(paired[v5_column], paired[langmuir_column])
            correlation = float(correlation)
            p_value = float(p_value)

        summary[f"pearson_corr_{label}"] = correlation
        summary[f"pearson_p_value_{label}"] = p_value

    return comparison, pd.DataFrame([summary])


# Collect V6 Arrhenius enthalpy predictions for each sample.
def collect_v6_enthalpy_table(
    model,
    pipeline: dict,
    split_names: tuple[str, ...] = ANALYSIS_SPLITS,
    device: str = "cpu",
) -> pd.DataFrame:
    rows = []
    model = model.to(device)
    was_training = model.training
    model.eval()

    try:
        with torch.inference_mode():
            for split_name in split_names:
                x, raw_pressure, split_df = get_split_inputs(
                    pipeline,
                    split_name,
                    device=device,
                )
                output = model(x, raw_pressure)
                metadata = make_metadata_frame(split_df, split_name)

                for column in ("q_hat", "q_max", "K", "n", "delta_H_kJmol"):
                    if column not in output:
                        raise KeyError(f"V6 output is missing {column}.")

                    metadata[column] = output[column].detach().cpu().numpy().reshape(-1)

                rows.append(metadata)
    finally:
        model.train(was_training)

    return pd.concat(rows, ignore_index=True)


# Summarize V6 enthalpy values by split and coal rank.
def summarize_v6_enthalpy_by_rank(enthalpy_table: pd.DataFrame) -> pd.DataFrame:
    return (
        enthalpy_table.groupby(["split", "coal_rank"], sort=True)
        .agg(
            n_experiments=("experiment_id", "nunique"),
            n_samples=("delta_H_kJmol", "size"),
            delta_H_kJmol_mean=("delta_H_kJmol", "mean"),
            delta_H_kJmol_std=("delta_H_kJmol", "std"),
            delta_H_kJmol_min=("delta_H_kJmol", "min"),
            delta_H_kJmol_max=("delta_H_kJmol", "max"),
        )
        .reset_index()
    )


# Compare V6 enthalpy estimates with a literature-compatible benchmark range.
def compare_v6_enthalpy_to_literature(
    enthalpy_table: pd.DataFrame,
    literature_range_kJmol: tuple[float, float] = DEFAULT_LITERATURE_DH_RANGE_KJMOL,
) -> pd.DataFrame:
    low, high = literature_range_kJmol
    rows = []

    for (split_name, coal_rank), group in enthalpy_table.groupby(
        ["split", "coal_rank"],
        sort=True,
    ):
        delta_H = group["delta_H_kJmol"]
        in_range = delta_H.between(low, high)
        mean_delta_H = float(delta_H.mean())

        rows.append(
            {
                "split": split_name,
                "coal_rank": coal_rank,
                "n_samples": int(len(group)),
                "delta_H_kJmol_mean": mean_delta_H,
                "delta_H_kJmol_std": float(delta_H.std()),
                "literature_reference": BUSCH_GENSTERBLUM_REFERENCE,
                "literature_delta_H_min_kJmol": float(low),
                "literature_delta_H_max_kJmol": float(high),
                "mean_in_literature_range": bool(low <= mean_delta_H <= high),
                "fraction_samples_in_literature_range": float(in_range.mean()),
                "abs_delta_H_kJmol_mean": float(delta_H.abs().mean()),
            }
        )

    return pd.DataFrame(rows)


# Run one-way ANOVA for V6 enthalpy differences across coal ranks.
def compute_enthalpy_anova_by_split(
    enthalpy_table: pd.DataFrame,
    split_names: tuple[str, ...] = ANALYSIS_SPLITS,
) -> pd.DataFrame:
    rows = []

    for split_name in split_names:
        split_table = enthalpy_table[enthalpy_table["split"] == split_name]
        grouped_values = [
            group["delta_H_kJmol"].to_numpy()
            for _, group in split_table.groupby("coal_rank")
            if len(group) >= 2
        ]

        if len(grouped_values) < 2:
            f_statistic = float("nan")
            p_value = float("nan")
        else:
            f_statistic, p_value = f_oneway(*grouped_values)
            f_statistic = float(f_statistic)
            p_value = float(p_value)

        rows.append(
            {
                "split": split_name,
                "n_ranks": int(split_table["coal_rank"].nunique()),
                "n_samples": int(len(split_table)),
                "F_statistic": f_statistic,
                "p_value": p_value,
            }
        )

    return pd.DataFrame(rows)


# Run Tukey HSD pairwise rank comparisons for V6 enthalpy.
def compute_enthalpy_tukey_by_split(
    enthalpy_table: pd.DataFrame,
    split_names: tuple[str, ...] = ANALYSIS_SPLITS,
    alpha: float = 0.05,
) -> pd.DataFrame:
    rows = []

    for split_name in split_names:
        split_table = enthalpy_table[enthalpy_table["split"] == split_name]

        if split_table["coal_rank"].nunique() < 2:
            continue

        tukey = pairwise_tukeyhsd(
            endog=split_table["delta_H_kJmol"],
            groups=split_table["coal_rank"],
            alpha=alpha,
        )
        summary_rows = tukey.summary().data
        headers = summary_rows[0]

        for values in summary_rows[1:]:
            row = dict(zip(headers, values))
            rows.append(
                {
                    "split": split_name,
                    "alpha": float(alpha),
                    "group1": row["group1"],
                    "group2": row["group2"],
                    "mean_difference": float(row["meandiff"]),
                    "p_value_adjusted": float(row["p-adj"]),
                    "lower_ci": float(row["lower"]),
                    "upper_ci": float(row["upper"]),
                    "reject_equal_means": bool(row["reject"]),
                }
            )

    return pd.DataFrame(rows)


# Load Matplotlib/Seaborn in file-saving mode.
def load_plotting_backend():
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    import seaborn as sns

    return plt, sns


# Save one violin plot for a V5 Sips parameter across coal ranks.
def save_v5_parameter_violin_plot(
    parameter_table: pd.DataFrame,
    parameter_name: str,
    output_path: Path,
) -> Path:
    if parameter_name not in V5_PARAMETER_COLUMNS:
        raise KeyError(f"Unsupported V5 parameter: {parameter_name}")

    plt, sns = load_plotting_backend()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    sns.violinplot(
        data=parameter_table,
        x="coal_rank",
        y=parameter_name,
        inner="quartile",
        cut=0,
        color="#6baed6",
        ax=ax,
    )
    sns.stripplot(
        data=parameter_table,
        x="coal_rank",
        y=parameter_name,
        color="#222222",
        size=2.5,
        alpha=0.45,
        ax=ax,
    )
    ax.set_xlabel("Coal rank")
    ax.set_ylabel(parameter_name)
    ax.set_title(f"Step 11: V5 {parameter_name} by Coal Rank")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    return output_path


# Save a violin plot for V6 Arrhenius enthalpy across coal ranks.
def save_v6_enthalpy_violin_plot(
    enthalpy_table: pd.DataFrame,
    output_path: Path,
) -> Path:
    plt, sns = load_plotting_backend()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    sns.violinplot(
        data=enthalpy_table,
        x="coal_rank",
        y="delta_H_kJmol",
        inner="quartile",
        cut=0,
        color="#74c476",
        ax=ax,
    )
    sns.stripplot(
        data=enthalpy_table,
        x="coal_rank",
        y="delta_H_kJmol",
        color="#222222",
        size=2.5,
        alpha=0.45,
        ax=ax,
    )
    ax.set_xlabel("Coal rank")
    ax.set_ylabel("delta_H (kJ/mol)")
    ax.set_title("Step 11: V6 Arrhenius Enthalpy by Coal Rank")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    return output_path


# Randomly sample rows without changing the original array.
def sample_rows(array: np.ndarray, max_rows: int, random_state: int) -> np.ndarray:
    if len(array) <= max_rows:
        return array

    rng = np.random.default_rng(random_state)
    indices = rng.choice(len(array), size=max_rows, replace=False)

    return array[indices]


# Add ranks and normalized importance to V5 parameter-head SHAP values.
def rank_v5_parameter_shap(shap_table: pd.DataFrame) -> pd.DataFrame:
    required_columns = ["parameter", "feature", "mean_abs_shap"]
    missing_columns = [col for col in required_columns if col not in shap_table.columns]

    if missing_columns:
        raise ValueError(f"Missing SHAP columns: {missing_columns}")

    ranked = shap_table.copy()
    ranked["parameter_order"] = ranked["parameter"].map(V5_SHAP_PARAMETER_ORDER)
    ranked = ranked.sort_values(
        ["parameter_order", "mean_abs_shap"],
        ascending=[True, False],
    )
    ranked["rank"] = ranked.groupby("parameter").cumcount().add(1)
    parameter_totals = ranked.groupby("parameter")["mean_abs_shap"].transform("sum")
    ranked["normalized_importance"] = ranked["mean_abs_shap"] / parameter_totals

    return ranked.drop(columns=["parameter_order"]).reset_index(drop=True)


# Save V5 SHAP results as one combined table, three head-specific tables, and a plot.
def save_v5_parameter_shap_outputs(
    shap_table: pd.DataFrame,
    results_dir=Path("results"),
) -> dict[str, Path]:
    results_dir = ensure_step_results_dir(results_dir, step=11)
    ranked_shap = rank_v5_parameter_shap(shap_table)

    paths = {
        "v5_parameter_shap": results_dir / "step11_shap_v5_parameter_importance.csv",
        "v5_q_max_shap": results_dir / "step11_shap_v5_qmax_importance.csv",
        "v5_K_shap": results_dir / "step11_shap_v5_K_importance.csv",
        "v5_n_shap": results_dir / "step11_shap_v5_n_importance.csv",
        "v5_parameter_shap_plot": results_dir
        / "step11_shap_v5_parameter_importance.png",
    }

    ranked_shap.to_csv(paths["v5_parameter_shap"], index=False)

    for parameter_name in V5_PARAMETER_COLUMNS:
        parameter_table = ranked_shap[ranked_shap["parameter"] == parameter_name]
        parameter_table.to_csv(paths[f"v5_{parameter_name}_shap"], index=False)

    save_v5_parameter_shap_plot(ranked_shap, paths["v5_parameter_shap_plot"])

    return paths


# Save a three-panel bar plot for q_max, K, and n SHAP feature importance.
def save_v5_parameter_shap_plot(
    shap_table: pd.DataFrame,
    output_path: Path,
) -> Path:
    plt, _ = load_plotting_backend()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6), sharex=False)

    for ax, parameter_name in zip(axes, V5_PARAMETER_COLUMNS):
        parameter_table = shap_table[shap_table["parameter"] == parameter_name]
        parameter_table = parameter_table.sort_values("mean_abs_shap", ascending=True)

        ax.barh(
            parameter_table["feature"],
            parameter_table["mean_abs_shap"],
            color="#4c78a8",
        )
        ax.set_title(parameter_name)
        ax.set_xlabel("mean |SHAP|")
        ax.grid(True, axis="x", alpha=0.25)

    axes[0].set_ylabel("Feature")
    fig.suptitle("Step 11: V5 SHAP Importance by Sips Parameter Head")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    return output_path


# Recover raw pressure from the scaled pressure feature used by SHAP.
def raw_pressure_from_scaled_features(x_np: np.ndarray, pipeline: dict) -> np.ndarray:
    feature_names = pipeline["feature_names"]
    pressure_idx = feature_names.index("pressure_MPa")
    scaler = pipeline["scaler"]
    raw_pressure = x_np[:, pressure_idx] * scaler.scale_[pressure_idx]
    raw_pressure = raw_pressure + scaler.center_[pressure_idx]

    return np.maximum(raw_pressure, 1e-8).reshape(-1, 1)


# Compute SHAP feature importance for a model's adsorption prediction q_hat.
def compute_global_prediction_shap_summary(
    model,
    pipeline: dict,
    variant_id: str,
    split_name: str = "validation",
    max_background: int = 25,
    max_explain: int = 25,
    nsamples: int = 100,
    device: str = "cpu",
    random_state: int = 42,
) -> pd.DataFrame:
    import warnings

    import shap
    from sklearn.exceptions import ConvergenceWarning

    suffix = SPLIT_SUFFIXES[split_name]
    arrays = pipeline["arrays"]
    background = sample_rows(arrays["X_train"], max_background, random_state)
    explain = sample_rows(arrays[f"X_{suffix}"], max_explain, random_state + 1)

    model = model.to(device)
    was_training = model.training
    model.eval()

    def predict_q_hat(x_np):
        x_tensor = torch.tensor(x_np, dtype=torch.float32, device=device)
        raw_pressure = torch.tensor(
            raw_pressure_from_scaled_features(x_np, pipeline),
            dtype=torch.float32,
            device=device,
        )

        with torch.inference_mode():
            output = model(x_tensor, raw_pressure)

        return output["q_hat"].detach().cpu().numpy().reshape(-1)

    try:
        explainer = shap.KernelExplainer(predict_q_hat, background)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            try:
                shap_values = explainer.shap_values(
                    explain,
                    nsamples=nsamples,
                    silent=True,
                )
            except TypeError:
                shap_values = explainer.shap_values(explain, nsamples=nsamples)
    finally:
        model.train(was_training)

    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    mean_abs_values = np.abs(np.asarray(shap_values)).mean(axis=0)

    return pd.DataFrame(
        {
            "variant_id": variant_id,
            "feature": pipeline["feature_names"],
            "mean_abs_shap": mean_abs_values,
        }
    )


# Add global feature ranks and normalized SHAP importance.
def rank_global_shap(shap_table: pd.DataFrame) -> pd.DataFrame:
    required_columns = ["variant_id", "feature", "mean_abs_shap"]
    missing_columns = [col for col in required_columns if col not in shap_table.columns]

    if missing_columns:
        raise ValueError(f"Missing global SHAP columns: {missing_columns}")

    ranked = shap_table.copy()
    ranked = ranked.sort_values(
        ["variant_id", "mean_abs_shap"],
        ascending=[True, False],
    )
    ranked["rank"] = ranked.groupby("variant_id").cumcount().add(1)
    variant_totals = ranked.groupby("variant_id")["mean_abs_shap"].transform("sum")
    ranked["normalized_importance"] = ranked["mean_abs_shap"] / variant_totals

    return ranked.reset_index(drop=True)


# Aggregate V5 parameter-head SHAP into one physical-parameter importance ranking.
def aggregate_v5_parameter_shap(ranked_v5_shap: pd.DataFrame) -> pd.DataFrame:
    aggregated = (
        ranked_v5_shap.groupby("feature", sort=False)
        .agg(
            mean_abs_shap=("mean_abs_shap", "mean"),
            normalized_importance=("normalized_importance", "mean"),
        )
        .reset_index()
        .sort_values("mean_abs_shap", ascending=False)
    )
    aggregated.insert(0, "variant_id", "V5_parameter_heads")
    aggregated["rank"] = np.arange(1, len(aggregated) + 1)

    return aggregated.reset_index(drop=True)


# Compare V1 and V5 prediction SHAP rankings feature by feature.
def compare_v1_v5_global_shap(
    v1_shap: pd.DataFrame,
    v5_shap: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    v1_ranked = rank_global_shap(v1_shap)
    v5_ranked = rank_global_shap(v5_shap)
    merged = v1_ranked.merge(
        v5_ranked,
        on="feature",
        suffixes=("_V1", "_V5"),
    )
    merged["rank_difference_V5_minus_V1"] = merged["rank_V5"] - merged["rank_V1"]
    merged["importance_difference_V5_minus_V1"] = (
        merged["normalized_importance_V5"] - merged["normalized_importance_V1"]
    )

    if len(merged) < 3:
        rank_correlation = float("nan")
        rank_p_value = float("nan")
        importance_correlation = float("nan")
        importance_p_value = float("nan")
    else:
        rank_correlation, rank_p_value = spearmanr(
            merged["rank_V1"],
            merged["rank_V5"],
        )
        importance_correlation, importance_p_value = spearmanr(
            merged["mean_abs_shap_V1"],
            merged["mean_abs_shap_V5"],
        )

    summary = pd.DataFrame(
        [
            {
                "n_shared_features": int(len(merged)),
                "spearman_rank_corr": float(rank_correlation),
                "spearman_rank_p_value": float(rank_p_value),
                "spearman_importance_corr": float(importance_correlation),
                "spearman_importance_p_value": float(importance_p_value),
            }
        ]
    )

    return merged.sort_values("rank_V1").reset_index(drop=True), summary


# Save a side-by-side plot comparing V1 and V5 global SHAP importance.
def save_global_shap_comparison_plot(
    comparison_table: pd.DataFrame,
    output_path: Path,
) -> Path:
    plt, _ = load_plotting_backend()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plot_table = comparison_table.sort_values(
        "normalized_importance_V1",
        ascending=True,
    )
    y_positions = np.arange(len(plot_table))
    bar_height = 0.38

    fig, ax = plt.subplots(figsize=(8.5, 5.8))
    ax.barh(
        y_positions - bar_height / 2,
        plot_table["normalized_importance_V1"],
        height=bar_height,
        label="V1 q_hat",
        color="#4c78a8",
    )
    ax.barh(
        y_positions + bar_height / 2,
        plot_table["normalized_importance_V5"],
        height=bar_height,
        label="V5 q_hat",
        color="#f58518",
    )
    ax.set_yticks(y_positions)
    ax.set_yticklabels(plot_table["feature"])
    ax.set_xlabel("Normalized mean |SHAP|")
    ax.set_title("Step 11: Global SHAP Comparison")
    ax.grid(True, axis="x", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    return output_path


# Save all global SHAP comparison outputs for V1 and V5.
def save_global_shap_outputs(
    v1_shap: pd.DataFrame,
    v5_shap: pd.DataFrame,
    ranked_v5_parameter_shap: pd.DataFrame,
    results_dir=Path("results"),
) -> tuple[dict[str, Path], dict[str, pd.DataFrame]]:
    results_dir = ensure_step_results_dir(results_dir, step=11)
    v1_ranked = rank_global_shap(v1_shap)
    v5_ranked = rank_global_shap(v5_shap)
    comparison, correlation = compare_v1_v5_global_shap(v1_shap, v5_shap)
    v5_parameter_aggregate = aggregate_v5_parameter_shap(ranked_v5_parameter_shap)

    paths = {
        "v1_global_shap": results_dir / "step11_shap_v1_global_importance.csv",
        "v5_global_shap": results_dir / "step11_shap_v5_global_importance.csv",
        "v5_parameter_aggregate_shap": results_dir
        / "step11_shap_v5_parameter_aggregate_importance.csv",
        "v1_v5_global_shap": results_dir / "step11_shap_v1_vs_v5_global_importance.csv",
        "v1_v5_shap_rank_correlation": results_dir
        / "step11_shap_v1_vs_v5_rank_correlation.csv",
        "v1_v5_global_shap_plot": results_dir
        / "step11_shap_v1_vs_v5_global_importance.png",
    }

    v1_ranked.to_csv(paths["v1_global_shap"], index=False)
    v5_ranked.to_csv(paths["v5_global_shap"], index=False)
    v5_parameter_aggregate.to_csv(paths["v5_parameter_aggregate_shap"], index=False)
    comparison.to_csv(paths["v1_v5_global_shap"], index=False)
    correlation.to_csv(paths["v1_v5_shap_rank_correlation"], index=False)
    save_global_shap_comparison_plot(comparison, paths["v1_v5_global_shap_plot"])

    tables = {
        "v1_global_shap": v1_ranked,
        "v5_global_shap": v5_ranked,
        "v5_parameter_aggregate_shap": v5_parameter_aggregate,
        "v1_v5_global_shap": comparison,
        "v1_v5_shap_rank_correlation": correlation,
    }

    return paths, tables


# Run the first Step 11 interpretability pass and save tables/plots.
def run_and_save_step11_interpretability(
    results_dir=Path("results"),
    checkpoint_dir=Path("checkpoints"),
    batch_size: int = 64,
    device: str = "cpu",
    strict_qmax_correlation: bool = True,
    run_v5_shap: bool = True,
    run_global_shap: bool = True,
    shap_max_background: int = 25,
    shap_max_explain: int = 25,
    shap_nsamples: int = 100,
    literature_delta_H_range_kJmol: tuple[float, float] = (
        DEFAULT_LITERATURE_DH_RANGE_KJMOL
    ),
    random_state: int = 42,
    verbose: bool = True,
) -> dict[str, object]:
    results_dir = ensure_step_results_dir(results_dir, step=11)
    pipeline = build_data_pipeline(batch_size=batch_size)

    v5_model = load_trained_variant("V5", checkpoint_dir=checkpoint_dir, device=device)
    v6_model = load_trained_variant("V6", checkpoint_dir=checkpoint_dir, device=device)
    v1_model = None

    if run_global_shap:
        v1_model = load_trained_variant(
            "V1", checkpoint_dir=checkpoint_dir, device=device
        )

    v5_sample_table = collect_v5_parameter_table(
        v5_model,
        pipeline,
        device=device,
    )
    v5_experiment_table = make_experiment_level_v5_table(v5_sample_table)
    v5_summary = summarize_v5_parameters_by_rank(v5_experiment_table)
    qmax_correlation = compute_qmax_volatile_correlation(
        v5_experiment_table,
        strict=strict_qmax_correlation,
    )

    v6_sample_table = collect_v6_enthalpy_table(
        v6_model,
        pipeline,
        device=device,
    )
    v6_summary = summarize_v6_enthalpy_by_rank(v6_sample_table)
    v6_literature_check = compare_v6_enthalpy_to_literature(
        v6_sample_table,
        literature_range_kJmol=literature_delta_H_range_kJmol,
    )
    v6_anova = compute_enthalpy_anova_by_split(v6_sample_table)
    v6_tukey = compute_enthalpy_tukey_by_split(v6_sample_table)
    langmuir_table = fit_langmuir_by_experiment(pipeline["splits"]["full"])
    langmuir_summary = summarize_langmuir_by_rank(langmuir_table)
    v5_langmuir_comparison, v5_langmuir_summary = compare_v5_to_langmuir(
        v5_experiment_table,
        langmuir_table,
    )

    paths = {
        "v5_sample_parameters": results_dir / "step11_v5_sips_parameters_samples.csv",
        "v5_experiment_parameters": results_dir
        / "step11_v5_sips_parameters_experiments.csv",
        "v5_parameter_summary": results_dir / "step11_v5_sips_params_by_rank.csv",
        "qmax_volatile_correlation": results_dir
        / "step11_qmax_volatile_correlation.csv",
        "v6_sample_enthalpy": results_dir / "step11_v6_enthalpy_samples.csv",
        "v6_enthalpy_summary": results_dir / "step11_v6_enthalpy_by_rank.csv",
        "v6_enthalpy_literature_check": results_dir
        / "step11_v6_enthalpy_literature_check.csv",
        "v6_enthalpy_anova": results_dir / "step11_v6_enthalpy_anova.csv",
        "v6_enthalpy_tukey": results_dir / "step11_v6_enthalpy_tukey.csv",
        "langmuir_fits": results_dir / "step11_langmuir_fits_by_experiment.csv",
        "langmuir_summary": results_dir / "step11_langmuir_fits_by_rank.csv",
        "v5_vs_langmuir": results_dir / "step11_v5_vs_langmuir_parameters.csv",
        "v5_vs_langmuir_summary": results_dir / "step11_v5_vs_langmuir_summary.csv",
        "v5_qmax_plot": results_dir / "step11_v5_qmax_by_rank.png",
        "v5_K_plot": results_dir / "step11_v5_K_by_rank.png",
        "v5_n_plot": results_dir / "step11_v5_n_by_rank.png",
        "v6_enthalpy_plot": results_dir / "step11_v6_enthalpy_by_rank.png",
    }

    v5_sample_table.to_csv(paths["v5_sample_parameters"], index=False)
    v5_experiment_table.to_csv(paths["v5_experiment_parameters"], index=False)
    v5_summary.to_csv(paths["v5_parameter_summary"], index=False)
    qmax_correlation.to_csv(paths["qmax_volatile_correlation"], index=False)
    v6_sample_table.to_csv(paths["v6_sample_enthalpy"], index=False)
    v6_summary.to_csv(paths["v6_enthalpy_summary"], index=False)
    v6_literature_check.to_csv(paths["v6_enthalpy_literature_check"], index=False)
    v6_anova.to_csv(paths["v6_enthalpy_anova"], index=False)
    v6_tukey.to_csv(paths["v6_enthalpy_tukey"], index=False)
    langmuir_table.to_csv(paths["langmuir_fits"], index=False)
    langmuir_summary.to_csv(paths["langmuir_summary"], index=False)
    v5_langmuir_comparison.to_csv(paths["v5_vs_langmuir"], index=False)
    v5_langmuir_summary.to_csv(paths["v5_vs_langmuir_summary"], index=False)

    save_v5_parameter_violin_plot(
        v5_experiment_table,
        "q_max",
        paths["v5_qmax_plot"],
    )
    save_v5_parameter_violin_plot(
        v5_experiment_table,
        "K",
        paths["v5_K_plot"],
    )
    save_v5_parameter_violin_plot(
        v5_experiment_table,
        "n",
        paths["v5_n_plot"],
    )
    save_v6_enthalpy_violin_plot(v6_sample_table, paths["v6_enthalpy_plot"])

    v5_shap = None
    v5_ranked_shap = None
    global_shap_tables = {}
    if run_v5_shap:
        v5_shap = compute_parameter_shap_summary(
            v5_model,
            pipeline,
            split_name="validation",
            max_background=shap_max_background,
            max_explain=shap_max_explain,
            nsamples=shap_nsamples,
            device=device,
            random_state=random_state,
        )
        paths.update(save_v5_parameter_shap_outputs(v5_shap, results_dir=results_dir))
        v5_ranked_shap = rank_v5_parameter_shap(v5_shap)

    if run_global_shap:
        if v1_model is None:
            raise ValueError("V1 model must be loaded for global SHAP comparison.")

        v1_global_shap = compute_global_prediction_shap_summary(
            v1_model,
            pipeline,
            variant_id="V1",
            split_name="validation",
            max_background=shap_max_background,
            max_explain=shap_max_explain,
            nsamples=shap_nsamples,
            device=device,
            random_state=random_state + 10,
        )
        v5_global_shap = compute_global_prediction_shap_summary(
            v5_model,
            pipeline,
            variant_id="V5",
            split_name="validation",
            max_background=shap_max_background,
            max_explain=shap_max_explain,
            nsamples=shap_nsamples,
            device=device,
            random_state=random_state + 20,
        )

        if v5_ranked_shap is None:
            v5_shap = compute_parameter_shap_summary(
                v5_model,
                pipeline,
                split_name="validation",
                max_background=shap_max_background,
                max_explain=shap_max_explain,
                nsamples=shap_nsamples,
                device=device,
                random_state=random_state,
            )
            v5_ranked_shap = rank_v5_parameter_shap(v5_shap)

        global_paths, global_shap_tables = save_global_shap_outputs(
            v1_global_shap,
            v5_global_shap,
            v5_ranked_shap,
            results_dir=results_dir,
        )
        paths.update(global_paths)

    if verbose:
        for path in paths.values():
            print(f"Saved Step 11 output: {path}")

    return {
        "v5_sample_parameters": v5_sample_table,
        "v5_experiment_parameters": v5_experiment_table,
        "v5_parameter_summary": v5_summary,
        "qmax_volatile_correlation": qmax_correlation,
        "v6_sample_enthalpy": v6_sample_table,
        "v6_enthalpy_summary": v6_summary,
        "v6_enthalpy_literature_check": v6_literature_check,
        "v6_enthalpy_anova": v6_anova,
        "v6_enthalpy_tukey": v6_tukey,
        "langmuir_fits": langmuir_table,
        "langmuir_summary": langmuir_summary,
        "v5_vs_langmuir": v5_langmuir_comparison,
        "v5_vs_langmuir_summary": v5_langmuir_summary,
        "v5_parameter_shap": v5_shap,
        **global_shap_tables,
        "paths": paths,
    }


# Quick local check for the first Step 11 outputs.
def smoke_test_step11_interpretability() -> None:
    result = run_and_save_step11_interpretability(
        results_dir=Path("/private/tmp/uioenergy_step11_smoke"),
        shap_max_background=5,
        shap_max_explain=5,
        shap_nsamples=20,
        verbose=False,
    )
    print("V5 experiment rows:", len(result["v5_experiment_parameters"]))
    print("V5 rank summary rows:", len(result["v5_parameter_summary"]))
    print("V6 enthalpy summary rows:", len(result["v6_enthalpy_summary"]))
    print("V5 SHAP rows:", len(result["v5_parameter_shap"]))
    print("V1 global SHAP rows:", len(result["v1_global_shap"]))
    print("Langmuir fit rows:", len(result["langmuir_fits"]))
    print(result["qmax_volatile_correlation"].to_string(index=False))
    print(result["v1_v5_shap_rank_correlation"].to_string(index=False))


if __name__ == "__main__":
    smoke_test_step11_interpretability()
