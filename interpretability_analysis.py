from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import f_oneway, pearsonr
from statsmodels.stats.multicomp import pairwise_tukeyhsd

from ablation_analysis import VARIANT_SPECS, load_variant_model
from data_pipeline import build_data_pipeline

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


# Run the first Step 11 interpretability pass and save tables/plots.
def run_and_save_step11_interpretability(
    results_dir=Path("results"),
    checkpoint_dir=Path("checkpoints"),
    batch_size: int = 64,
    device: str = "cpu",
    strict_qmax_correlation: bool = False,
    verbose: bool = True,
) -> dict[str, object]:
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    pipeline = build_data_pipeline(batch_size=batch_size)

    v5_model = load_trained_variant("V5", checkpoint_dir=checkpoint_dir, device=device)
    v6_model = load_trained_variant("V6", checkpoint_dir=checkpoint_dir, device=device)

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
    v6_anova = compute_enthalpy_anova_by_split(v6_sample_table)
    v6_tukey = compute_enthalpy_tukey_by_split(v6_sample_table)

    paths = {
        "v5_sample_parameters": results_dir / "step11_v5_sips_parameters_samples.csv",
        "v5_experiment_parameters": results_dir
        / "step11_v5_sips_parameters_experiments.csv",
        "v5_parameter_summary": results_dir / "step11_v5_sips_params_by_rank.csv",
        "qmax_volatile_correlation": results_dir
        / "step11_qmax_volatile_correlation.csv",
        "v6_sample_enthalpy": results_dir / "step11_v6_enthalpy_samples.csv",
        "v6_enthalpy_summary": results_dir / "step11_v6_enthalpy_by_rank.csv",
        "v6_enthalpy_anova": results_dir / "step11_v6_enthalpy_anova.csv",
        "v6_enthalpy_tukey": results_dir / "step11_v6_enthalpy_tukey.csv",
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
    v6_anova.to_csv(paths["v6_enthalpy_anova"], index=False)
    v6_tukey.to_csv(paths["v6_enthalpy_tukey"], index=False)

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
        "v6_enthalpy_anova": v6_anova,
        "v6_enthalpy_tukey": v6_tukey,
        "paths": paths,
    }


# Quick local check for the first Step 11 outputs.
def smoke_test_step11_interpretability() -> None:
    result = run_and_save_step11_interpretability(
        results_dir=Path("/private/tmp/uioenergy_step11_smoke"),
        verbose=False,
    )
    print("V5 experiment rows:", len(result["v5_experiment_parameters"]))
    print("V5 rank summary rows:", len(result["v5_parameter_summary"]))
    print("V6 enthalpy summary rows:", len(result["v6_enthalpy_summary"]))
    print(result["qmax_volatile_correlation"].to_string(index=False))


if __name__ == "__main__":
    smoke_test_step11_interpretability()
