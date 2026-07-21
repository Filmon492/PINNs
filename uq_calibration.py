from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from scipy.stats import norm, spearmanr

from ablation_analysis import TIER_NAMES, VARIANT_SPECS, load_variant_model
from data_pipeline import build_data_pipeline
from unified_eval_harness import collect_mc_dropout_predictions, collect_predictions

CONFIDENCE_LEVELS = (0.68, 0.90, 0.95, 0.99)
DEFAULT_TAU_GRID = np.geomspace(0.1, 10.0, 81)
EVALUATION_SPLITS = ("validation", *TIER_NAMES)
LOADER_KEYS = {
    "validation": "val_loader",
    "T1": "T1_loader",
    "T2": "T2_loader",
    "T3": "T3_loader",
}
CONSTRAINT_HARDNESS_SCORE = {
    "V1": 0,
    "V3": 1,
    "V4": 1,
    "V6": 1,
    "V7": 2,
    "V5": 3,
    "V2": 4,
    "V8": 4,
}
VARIANT_ORDER = {spec.variant_id: index for index, spec in enumerate(VARIANT_SPECS)}
SPLIT_ORDER = {split_name: index for index, split_name in enumerate(EVALUATION_SPLITS)}


# Convert MC Dropout outputs into mean prediction and uncertainty components.
def compute_uncertainty_components(
    q_hat_mc: torch.Tensor,
    sigma_ale_mc: Optional[torch.Tensor] = None,
) -> dict[str, np.ndarray]:
    mc_values = q_hat_mc.detach().cpu().numpy()

    if mc_values.ndim < 2:
        raise ValueError("q_hat_mc must contain MC passes and samples.")

    mc_values = mc_values.reshape(mc_values.shape[0], -1)

    if mc_values.shape[0] < 2:
        raise ValueError("At least two MC passes are required.")

    mean_prediction = mc_values.mean(axis=0)
    sigma_epi = mc_values.std(axis=0, ddof=1)

    if sigma_ale_mc is None:
        sigma_ale = np.zeros_like(sigma_epi)
    else:
        ale_values = sigma_ale_mc.detach().cpu().numpy()
        ale_values = ale_values.reshape(ale_values.shape[0], -1)

        if ale_values.shape != mc_values.shape:
            raise ValueError("sigma_ale_mc must match q_hat_mc shape.")

        sigma_ale = ale_values.mean(axis=0)

    sigma_total = np.sqrt(sigma_epi**2 + sigma_ale**2)
    sigma_total = np.maximum(sigma_total, 1e-8)
    epistemic_fraction = sigma_epi**2 / np.maximum(sigma_total**2, 1e-12)

    return {
        "mean_prediction": mean_prediction,
        "sigma_epi": sigma_epi,
        "sigma_ale": sigma_ale,
        "sigma_total": sigma_total,
        "epistemic_fraction": epistemic_fraction,
    }


# Compute calibration and proper scoring metrics after scaling sigma by tau.
def compute_calibrated_uq_metrics(
    q_true: torch.Tensor,
    q_hat_mc: torch.Tensor,
    sigma_ale_mc: Optional[torch.Tensor] = None,
    tau: float = 1.0,
) -> dict[str, float]:
    if tau <= 0:
        raise ValueError("tau must be positive.")

    true_values = q_true.detach().cpu().numpy().reshape(-1)
    components = compute_uncertainty_components(q_hat_mc, sigma_ale_mc)
    mean_prediction = components["mean_prediction"]
    sigma_total = np.maximum(tau * components["sigma_total"], 1e-8)
    absolute_errors = np.abs(true_values - mean_prediction)

    if true_values.shape != mean_prediction.shape:
        raise ValueError("q_true and q_hat_mc must contain the same samples.")

    coverage_errors = []
    metrics = {
        "tau": float(tau),
        "sigma_epi": float(np.mean(components["sigma_epi"])),
        "sigma_ale": float(np.mean(components["sigma_ale"])),
        "sigma_total_uncalibrated": float(np.mean(components["sigma_total"])),
        "sigma_total_calibrated": float(np.mean(sigma_total)),
        "epistemic_fraction_pct": float(
            np.mean(components["epistemic_fraction"]) * 100.0
        ),
    }

    for confidence_level in CONFIDENCE_LEVELS:
        z_value = norm.ppf((1 + confidence_level) / 2)
        covered = absolute_errors <= z_value * sigma_total
        coverage = float(np.mean(covered))
        metrics[f"coverage_{int(confidence_level * 100)}"] = coverage
        coverage_errors.append(abs(coverage - confidence_level))

    metrics["ECE"] = float(np.mean(coverage_errors))

    if (
        true_values.size < 2
        or np.allclose(absolute_errors, absolute_errors[0])
        or np.allclose(sigma_total, sigma_total[0])
    ):
        metrics["rho_s"] = float("nan")
    else:
        metrics["rho_s"] = float(spearmanr(absolute_errors, sigma_total).statistic)

    variance = sigma_total**2
    standardized_error = (true_values - mean_prediction) / sigma_total
    nll_values = 0.5 * np.log(2 * np.pi * variance)
    nll_values += 0.5 * standardized_error**2
    crps_values = sigma_total * (
        standardized_error * (2 * norm.cdf(standardized_error) - 1)
        + 2 * norm.pdf(standardized_error)
        - 1 / np.sqrt(np.pi)
    )

    metrics["NLL"] = float(np.mean(nll_values))
    metrics["CRPS"] = float(np.mean(crps_values))
    metrics["sharpness_95"] = float(np.mean(2 * norm.ppf(0.975) * sigma_total))

    return metrics


# Select the scalar uncertainty correction that minimizes validation ECE.
def fit_tau_by_ece(
    q_true: torch.Tensor,
    q_hat_mc: torch.Tensor,
    sigma_ale_mc: Optional[torch.Tensor] = None,
    tau_grid=DEFAULT_TAU_GRID,
) -> dict[str, float]:
    if len(tau_grid) == 0:
        raise ValueError("tau_grid must contain at least one candidate.")

    candidates = []

    for tau in tau_grid:
        metrics = compute_calibrated_uq_metrics(
            q_true,
            q_hat_mc,
            sigma_ale_mc=sigma_ale_mc,
            tau=float(tau),
        )
        candidates.append(
            {
                "tau": float(tau),
                "ECE": metrics["ECE"],
                "NLL": metrics["NLL"],
                "CRPS": metrics["CRPS"],
            }
        )

    return min(candidates, key=lambda row: (row["ECE"], row["NLL"]))


# Run MC Dropout and calibrated UQ evaluation for one data split.
def evaluate_split_uq(
    model,
    loader,
    n_mc: int,
    tau: float,
    device: str = "cpu",
) -> tuple[dict[str, float], dict[str, float]]:
    predictions = collect_predictions(
        model,
        loader,
        device=device,
    )
    mc_predictions = collect_mc_dropout_predictions(
        model,
        loader,
        device=device,
        n_mc=n_mc,
    )
    uncalibrated_metrics = compute_calibrated_uq_metrics(
        predictions["q_true"],
        mc_predictions["q_hat_mc"],
        mc_predictions.get("sigma_ale_mc"),
        tau=1.0,
    )
    calibrated_metrics = compute_calibrated_uq_metrics(
        predictions["q_true"],
        mc_predictions["q_hat_mc"],
        mc_predictions.get("sigma_ale_mc"),
        tau=tau,
    )

    return uncalibrated_metrics, calibrated_metrics


# Fit tau on validation data for one variant.
def fit_variant_tau(
    model,
    pipeline: dict,
    n_mc: int,
    tau_grid=DEFAULT_TAU_GRID,
    device: str = "cpu",
) -> dict[str, float]:
    validation_loader = pipeline["val_loader"]
    validation_predictions = collect_predictions(
        model,
        validation_loader,
        device=device,
    )
    validation_mc = collect_mc_dropout_predictions(
        model,
        validation_loader,
        device=device,
        n_mc=n_mc,
    )

    tau_result = fit_tau_by_ece(
        validation_predictions["q_true"],
        validation_mc["q_hat_mc"],
        validation_mc.get("sigma_ale_mc"),
        tau_grid=tau_grid,
    )
    uncalibrated_metrics = compute_calibrated_uq_metrics(
        validation_predictions["q_true"],
        validation_mc["q_hat_mc"],
        validation_mc.get("sigma_ale_mc"),
        tau=1.0,
    )

    return {
        "tau": tau_result["tau"],
        "validation_ECE_uncalibrated": uncalibrated_metrics["ECE"],
        "validation_ECE_calibrated": tau_result["ECE"],
        "validation_NLL_calibrated": tau_result["NLL"],
        "validation_CRPS_calibrated": tau_result["CRPS"],
    }


# Evaluate calibrated UQ for one trained variant on validation/T1/T2/T3.
def evaluate_variant_uq(
    spec,
    model,
    pipeline: dict,
    n_mc: int = 100,
    tau_grid=DEFAULT_TAU_GRID,
    device: str = "cpu",
) -> tuple[list[dict[str, float]], dict[str, float]]:
    tau_info = fit_variant_tau(
        model,
        pipeline,
        n_mc=n_mc,
        tau_grid=tau_grid,
        device=device,
    )
    tau = tau_info["tau"]
    rows = []

    for split_name in EVALUATION_SPLITS:
        loader_key = LOADER_KEYS[split_name]
        uncalibrated_metrics, calibrated_metrics = evaluate_split_uq(
            model,
            pipeline[loader_key],
            n_mc=n_mc,
            tau=tau,
            device=device,
        )
        row = {
            "variant_id": spec.variant_id,
            "variant_label": spec.label,
            "split": split_name,
            "constraint_hardness_score": float(
                CONSTRAINT_HARDNESS_SCORE[spec.variant_id]
            ),
            "n_mc": float(n_mc),
            "tau": tau,
            "ECE_uncalibrated": uncalibrated_metrics["ECE"],
            "ECE_calibrated": calibrated_metrics["ECE"],
            "NLL_uncalibrated": uncalibrated_metrics["NLL"],
            "NLL_calibrated": calibrated_metrics["NLL"],
            "CRPS_uncalibrated": uncalibrated_metrics["CRPS"],
            "CRPS_calibrated": calibrated_metrics["CRPS"],
            "rho_s_uncalibrated": uncalibrated_metrics["rho_s"],
            "rho_s_calibrated": calibrated_metrics["rho_s"],
        }

        for metric_name, metric_value in calibrated_metrics.items():
            if metric_name == "tau":
                continue
            row[metric_name] = metric_value

        rows.append(row)

    tau_row = {
        "variant_id": spec.variant_id,
        "variant_label": spec.label,
        "constraint_hardness_score": float(CONSTRAINT_HARDNESS_SCORE[spec.variant_id]),
        "n_mc": float(n_mc),
        **tau_info,
    }

    return rows, tau_row


# Evaluate calibrated UQ for all blueprint variants.
def evaluate_all_variants_uq(
    pipeline: dict,
    checkpoint_dir=Path("checkpoints"),
    n_mc: int = 100,
    tau_grid=DEFAULT_TAU_GRID,
    device: str = "cpu",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    checkpoint_dir = Path(checkpoint_dir)
    metric_rows = []
    tau_rows = []
    cached_all_hard_result = None

    for spec in VARIANT_SPECS:
        if spec.variant_id == "V8" and cached_all_hard_result is not None:
            cached_metrics, cached_tau = cached_all_hard_result
            metric_rows.extend(
                {
                    **row,
                    "variant_id": "V8",
                    "variant_label": spec.label,
                    "constraint_hardness_score": float(CONSTRAINT_HARDNESS_SCORE["V8"]),
                }
                for row in cached_metrics
            )
            tau_rows.append(
                {
                    **cached_tau,
                    "variant_id": "V8",
                    "variant_label": spec.label,
                    "constraint_hardness_score": float(CONSTRAINT_HARDNESS_SCORE["V8"]),
                }
            )
            continue

        model = load_variant_model(spec, checkpoint_dir, device=device)
        rows, tau_row = evaluate_variant_uq(
            spec,
            model,
            pipeline,
            n_mc=n_mc,
            tau_grid=tau_grid,
            device=device,
        )
        metric_rows.extend(rows)
        tau_rows.append(tau_row)

        if spec.variant_id == "V2":
            cached_all_hard_result = (rows, tau_row)

    metrics_table = pd.DataFrame(metric_rows)
    tau_table = pd.DataFrame(tau_rows)

    return metrics_table, tau_table


# Check that a metrics table has the columns needed by a reporting helper.
def require_columns(table: pd.DataFrame, required_columns: list[str]) -> None:
    missing_columns = [col for col in required_columns if col not in table.columns]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")


# Sort Step 10 tables in the blueprint variant and split order.
def sort_step10_table(table: pd.DataFrame) -> pd.DataFrame:
    sorted_table = table.copy()
    sorted_table["_variant_order"] = sorted_table["variant_id"].map(VARIANT_ORDER)

    sort_columns = ["_variant_order"]
    if "split" in sorted_table.columns:
        sorted_table["_split_order"] = sorted_table["split"].map(SPLIT_ORDER)
        sort_columns.append("_split_order")

    sorted_table = sorted_table.sort_values(sort_columns)
    ordering_columns = [
        col for col in ("_variant_order", "_split_order") if col in sorted_table
    ]
    return sorted_table.drop(columns=ordering_columns).reset_index(drop=True)


# Load Matplotlib in script-safe mode for saving figures without a GUI window.
def load_plotting_backend():
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt


# Extract the epistemic uncertainty fraction values needed for the Step 10 plot.
def make_epistemic_fraction_table(metrics_table: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "variant_id",
        "variant_label",
        "split",
        "constraint_hardness_score",
        "n_mc",
        "tau",
        "sigma_epi",
        "sigma_ale",
        "sigma_total_uncalibrated",
        "sigma_total_calibrated",
        "epistemic_fraction_pct",
    ]
    require_columns(metrics_table, columns)

    return sort_step10_table(metrics_table.loc[:, columns])


# Summarize how much calibration error grows from interpolation to extrapolation.
def make_ece_shift_ratio_table(metrics_table: pd.DataFrame) -> pd.DataFrame:
    required_columns = [
        "variant_id",
        "variant_label",
        "split",
        "constraint_hardness_score",
        "tau",
        "ECE_uncalibrated",
        "ECE_calibrated",
    ]
    require_columns(metrics_table, required_columns)

    rows = []
    for variant_id in VARIANT_ORDER:
        variant_table = metrics_table[metrics_table["variant_id"] == variant_id]

        if variant_table.empty:
            continue

        split_rows = {row.split: row for row in variant_table.itertuples(index=False)}
        missing_splits = [
            split_name
            for split_name in ("validation", "T1", "T2", "T3")
            if split_name not in split_rows
        ]

        if missing_splits:
            raise ValueError(
                f"Variant {variant_id} is missing UQ splits: {missing_splits}"
            )

        t1_calibrated = float(split_rows["T1"].ECE_calibrated)
        t2_calibrated = float(split_rows["T2"].ECE_calibrated)
        t3_calibrated = float(split_rows["T3"].ECE_calibrated)
        t1_uncalibrated = float(split_rows["T1"].ECE_uncalibrated)
        t2_uncalibrated = float(split_rows["T2"].ECE_uncalibrated)
        t3_uncalibrated = float(split_rows["T3"].ECE_uncalibrated)

        rows.append(
            {
                "variant_id": variant_id,
                "variant_label": split_rows["T1"].variant_label,
                "constraint_hardness_score": float(
                    split_rows["T1"].constraint_hardness_score
                ),
                "tau": float(split_rows["T1"].tau),
                "ECE_validation_calibrated": float(
                    split_rows["validation"].ECE_calibrated
                ),
                "ECE_T1_calibrated": t1_calibrated,
                "ECE_T2_calibrated": t2_calibrated,
                "ECE_T3_calibrated": t3_calibrated,
                "ECE_T2_over_T1_calibrated": t2_calibrated / max(t1_calibrated, 1e-12),
                "ECE_T3_over_T1_calibrated": t3_calibrated / max(t1_calibrated, 1e-12),
                "ECE_T2_minus_T1_calibrated": t2_calibrated - t1_calibrated,
                "ECE_T3_minus_T1_calibrated": t3_calibrated - t1_calibrated,
                "ECE_validation_uncalibrated": float(
                    split_rows["validation"].ECE_uncalibrated
                ),
                "ECE_T1_uncalibrated": t1_uncalibrated,
                "ECE_T2_uncalibrated": t2_uncalibrated,
                "ECE_T3_uncalibrated": t3_uncalibrated,
                "ECE_T2_over_T1_uncalibrated": t2_uncalibrated
                / max(t1_uncalibrated, 1e-12),
                "ECE_T3_over_T1_uncalibrated": t3_uncalibrated
                / max(t1_uncalibrated, 1e-12),
            }
        )

    return sort_step10_table(pd.DataFrame(rows))


# Plot average epistemic fraction against hard-constraint score.
def save_epistemic_vs_hardness_plot(
    epistemic_table: pd.DataFrame,
    output_path: Path,
    split_names: tuple[str, ...] = TIER_NAMES,
) -> Path:
    plt = load_plotting_backend()

    require_columns(
        epistemic_table,
        [
            "variant_id",
            "constraint_hardness_score",
            "split",
            "epistemic_fraction_pct",
        ],
    )
    plot_table = epistemic_table[epistemic_table["split"].isin(split_names)]
    plot_table = (
        plot_table.groupby(
            ["variant_id", "constraint_hardness_score"],
            as_index=False,
        )
        .agg(
            epistemic_fraction_pct_mean=("epistemic_fraction_pct", "mean"),
            epistemic_fraction_pct_std=("epistemic_fraction_pct", "std"),
        )
        .sort_values(
            by="variant_id",
            key=lambda values: values.map(VARIANT_ORDER),
        )
    )

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.errorbar(
        plot_table["constraint_hardness_score"],
        plot_table["epistemic_fraction_pct_mean"],
        yerr=plot_table["epistemic_fraction_pct_std"].fillna(0.0),
        fmt="o",
        color="#1f77b4",
        ecolor="#7aa6d8",
        capsize=4,
    )

    for row in plot_table.itertuples(index=False):
        ax.annotate(
            row.variant_id,
            (
                row.constraint_hardness_score,
                row.epistemic_fraction_pct_mean,
            ),
            xytext=(5, 4),
            textcoords="offset points",
            fontsize=9,
        )

    ax.set_xlabel("Constraint hardness score")
    ax.set_ylabel("Epistemic fraction (%)")
    ax.set_title("Step 10: Epistemic Uncertainty vs Constraint Hardness")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    return output_path


# Plot calibrated interpolation ECE against coal-rank extrapolation ECE.
def save_ece_t1_t2_plot(ece_shift_table: pd.DataFrame, output_path: Path) -> Path:
    plt = load_plotting_backend()

    require_columns(
        ece_shift_table,
        [
            "variant_id",
            "constraint_hardness_score",
            "ECE_T1_calibrated",
            "ECE_T2_calibrated",
        ],
    )

    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    scatter = ax.scatter(
        ece_shift_table["ECE_T1_calibrated"],
        ece_shift_table["ECE_T2_calibrated"],
        c=ece_shift_table["constraint_hardness_score"],
        cmap="viridis",
        s=70,
        edgecolor="black",
        linewidth=0.5,
    )

    max_ece = max(
        float(ece_shift_table["ECE_T1_calibrated"].max()),
        float(ece_shift_table["ECE_T2_calibrated"].max()),
    )
    ax.plot([0, max_ece], [0, max_ece], linestyle="--", color="#777777")

    for row in ece_shift_table.itertuples(index=False):
        ax.annotate(
            row.variant_id,
            (row.ECE_T1_calibrated, row.ECE_T2_calibrated),
            xytext=(5, 4),
            textcoords="offset points",
            fontsize=9,
        )

    ax.set_xlabel("T1 calibrated ECE")
    ax.set_ylabel("T2 calibrated ECE")
    ax.set_title("Step 10: Calibration Shift Under Rank Extrapolation")
    ax.grid(True, alpha=0.25)
    fig.colorbar(scatter, ax=ax, label="Constraint hardness score")
    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    return output_path


# Save the Step 10 summary tables and plots expected by the blueprint.
def save_step10_summary_outputs(
    metrics_table: pd.DataFrame,
    results_dir=Path("results"),
) -> dict[str, Path]:
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    epistemic_table = make_epistemic_fraction_table(metrics_table)
    ece_shift_table = make_ece_shift_ratio_table(metrics_table)

    epistemic_path = results_dir / "step10_epistemic_fraction.csv"
    ece_shift_path = results_dir / "step10_ece_shift_ratio.csv"
    epistemic_plot_path = results_dir / "step10_epistemic_vs_hardness.png"
    ece_plot_path = results_dir / "step10_ece_t1_t2.png"

    epistemic_table.to_csv(epistemic_path, index=False)
    ece_shift_table.to_csv(ece_shift_path, index=False)
    save_epistemic_vs_hardness_plot(epistemic_table, epistemic_plot_path)
    save_ece_t1_t2_plot(ece_shift_table, ece_plot_path)

    return {
        "epistemic_fraction": epistemic_path,
        "ece_shift_ratio": ece_shift_path,
        "epistemic_vs_hardness_plot": epistemic_plot_path,
        "ece_t1_t2_plot": ece_plot_path,
    }


# Build Step 10 summary outputs from an existing metrics CSV.
def run_and_save_step10_summaries(
    metrics_path=Path("results/step10_uq_calibration_metrics.csv"),
    results_dir=Path("results"),
    verbose: bool = True,
) -> dict[str, object]:
    metrics_path = Path(metrics_path)

    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing Step 10 metrics table: {metrics_path}")

    metrics_table = pd.read_csv(metrics_path)
    summary_paths = save_step10_summary_outputs(metrics_table, results_dir=results_dir)

    if verbose:
        for path in summary_paths.values():
            print(f"Saved Step 10 summary output: {path}")

    return {
        "metrics": metrics_table,
        "paths": summary_paths,
    }


# Save all-variant calibrated UQ metrics.
def run_and_save_uq_calibration(
    results_dir=Path("results"),
    checkpoint_dir=Path("checkpoints"),
    batch_size: int = 64,
    n_mc: int = 100,
    tau_grid=DEFAULT_TAU_GRID,
    device: str = "cpu",
    verbose: bool = True,
) -> dict[str, object]:
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    pipeline = build_data_pipeline(batch_size=batch_size)
    metrics_table, tau_table = evaluate_all_variants_uq(
        pipeline,
        checkpoint_dir=checkpoint_dir,
        n_mc=n_mc,
        tau_grid=tau_grid,
        device=device,
    )

    metrics_path = results_dir / "step10_uq_calibration_metrics.csv"
    tau_path = results_dir / "step10_tau_by_variant.csv"
    metrics_table.to_csv(metrics_path, index=False)
    tau_table.to_csv(tau_path, index=False)
    summary_paths = save_step10_summary_outputs(
        metrics_table,
        results_dir=results_dir,
    )

    if verbose:
        print(f"Saved Step 10 UQ metrics: {metrics_path}")
        print(f"Saved Step 10 tau table: {tau_path}")
        for path in summary_paths.values():
            print(f"Saved Step 10 summary output: {path}")

    return {
        "metrics": metrics_table,
        "tau": tau_table,
        "paths": {
            "metrics": metrics_path,
            "tau": tau_path,
            **summary_paths,
        },
    }


# Quick first-step check on one variant and one validation loader.
def smoke_test_uq_calibration() -> None:
    pipeline = build_data_pipeline(batch_size=64)
    spec = VARIANT_SPECS[0]
    model = load_variant_model(spec, Path("checkpoints"))
    validation_predictions = collect_predictions(
        model,
        pipeline["val_loader"],
    )
    validation_mc = collect_mc_dropout_predictions(
        model,
        pipeline["val_loader"],
        n_mc=5,
    )
    tau_result = fit_tau_by_ece(
        validation_predictions["q_true"],
        validation_mc["q_hat_mc"],
        validation_mc.get("sigma_ale_mc"),
    )
    calibrated_metrics = compute_calibrated_uq_metrics(
        validation_predictions["q_true"],
        validation_mc["q_hat_mc"],
        validation_mc.get("sigma_ale_mc"),
        tau=tau_result["tau"],
    )

    print("variant:", spec.variant_id)
    print("tier:", "validation")
    print("tau:", tau_result["tau"])
    print("ECE:", calibrated_metrics["ECE"])
    print("epistemic_fraction_pct:", calibrated_metrics["epistemic_fraction_pct"])
    print("available_tiers:", TIER_NAMES)

    metrics_table, tau_table = evaluate_all_variants_uq(
        pipeline,
        n_mc=3,
    )
    print("all_variant_metric_rows:", len(metrics_table))
    print("all_variant_tau_rows:", len(tau_table))
    summary_paths = save_step10_summary_outputs(
        metrics_table,
        results_dir=Path("/private/tmp/uioenergy_step10_smoke"),
    )
    print("summary_outputs:", len(summary_paths))


if __name__ == "__main__":
    smoke_test_uq_calibration()
