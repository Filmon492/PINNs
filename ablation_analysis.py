from dataclasses import dataclass, fields
from itertools import combinations
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch

from all_hard_variant import AllHardConfig, AllHardSipsArrheniusPINN
from arrhenius_variant import ArrheniusConfig, ArrheniusPINN
from data_pipeline import build_data_pipeline
from hard_bounds_variant import HardBoundsPINN
from hard_monotonicity_variant import HardMonotonicityPINN
from joint_variants import HardBoundsMonotonicityPINN, JointVariantConfig
from sips_gauge_variant import SipsGaugeConfig, SipsGaugePINN
from soft_baseline import SoftBaselinePINN, V1ModelConfig
from unified_eval_harness import (
    anova_2x2,
    bonferroni_correct,
    collect_predictions,
    compare_variants,
    compute_named_accuracy_metric,
    mcnemar_test,
)


@dataclass(frozen=True)
class VariantSpec:
    variant_id: str
    label: str
    eval_metrics_file: str
    training_history_file: str
    checkpoint_file: str
    model_class: type[torch.nn.Module]
    config_class: type


VARIANT_SPECS = (
    VariantSpec(
        "V1",
        "All-soft",
        "soft_baseline_eval_metrics.csv",
        "soft_baseline_training_history.csv",
        "soft_baseline_v1.pt",
        SoftBaselinePINN,
        V1ModelConfig,
    ),
    VariantSpec(
        "V2",
        "All-hard",
        "all_hard_eval_metrics.csv",
        "all_hard_training_history.csv",
        "all_hard_v2_v8.pt",
        AllHardSipsArrheniusPINN,
        AllHardConfig,
    ),
    VariantSpec(
        "V3",
        "Hard bounds",
        "hard_bounds_eval_metrics.csv",
        "hard_bounds_training_history.csv",
        "hard_bounds_v3.pt",
        HardBoundsPINN,
        V1ModelConfig,
    ),
    VariantSpec(
        "V4",
        "Hard monotonicity",
        "hard_monotonicity_eval_metrics.csv",
        "hard_monotonicity_training_history.csv",
        "hard_monotonicity_v4.pt",
        HardMonotonicityPINN,
        V1ModelConfig,
    ),
    VariantSpec(
        "V5",
        "Hard Sips",
        "sips_gauge_eval_metrics.csv",
        "sips_gauge_training_history.csv",
        "sips_gauge_v5.pt",
        SipsGaugePINN,
        SipsGaugeConfig,
    ),
    VariantSpec(
        "V6",
        "Hard van't Hoff",
        "arrhenius_eval_metrics.csv",
        "arrhenius_training_history.csv",
        "arrhenius_v6.pt",
        ArrheniusPINN,
        ArrheniusConfig,
    ),
    VariantSpec(
        "V7",
        "Hard bounds + monotonicity",
        "joint_v7_eval_metrics.csv",
        "joint_v7_training_history.csv",
        "joint_v7.pt",
        HardBoundsMonotonicityPINN,
        JointVariantConfig,
    ),
    VariantSpec(
        "V8",
        "Hard Sips + van't Hoff",
        "all_hard_eval_metrics.csv",
        "all_hard_training_history.csv",
        "all_hard_v2_v8.pt",
        AllHardSipsArrheniusPINN,
        AllHardConfig,
    ),
)

TIER_NAMES = ("T1", "T2", "T3")
MAIN_EFFECT_METRICS = (
    "R2",
    "RMSE",
    "MAE",
    "MaxAE",
    "neg_rate",
    "mono_rate",
    "sat_rate",
    "ECE",
    "rho_s",
    "NLL",
    "eval_seconds",
    "total_parameters",
    "trainable_parameters",
)


# Build a dataclass config from the fields stored in a checkpoint.
def make_config(config_class: type, config_values: dict) -> object:
    valid_names = {field.name for field in fields(config_class)}
    filtered_values = {
        key: value for key, value in config_values.items() if key in valid_names
    }

    return config_class(**filtered_values)


# Load a trained model checkpoint for one blueprint variant.
def load_variant_model(
    spec: VariantSpec,
    checkpoint_dir: Path,
    device: str = "cpu",
) -> torch.nn.Module:
    checkpoint_path = checkpoint_dir / spec.checkpoint_file

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = make_config(spec.config_class, checkpoint.get("model_config", {}))
    model = spec.model_class(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    return model


# Collect prediction dictionaries for every variant and evaluation tier.
def collect_variant_predictions(
    pipeline: dict,
    checkpoint_dir: Path,
    device: str = "cpu",
) -> dict[str, dict[str, dict[str, torch.Tensor]]]:
    predictions: dict[str, dict[str, dict[str, torch.Tensor]]] = {}
    cached_all_hard_predictions = None

    for spec in VARIANT_SPECS:
        if spec.variant_id == "V8" and cached_all_hard_predictions is not None:
            predictions[spec.variant_id] = cached_all_hard_predictions
            continue

        model = load_variant_model(spec, checkpoint_dir, device=device)
        variant_predictions = {}

        for tier_name in TIER_NAMES:
            loader = pipeline[f"{tier_name}_loader"]
            variant_predictions[tier_name] = collect_predictions(
                model,
                loader,
                device=device,
            )

        predictions[spec.variant_id] = variant_predictions

        if spec.variant_id == "V2":
            cached_all_hard_predictions = variant_predictions

    return predictions


# Read training history metadata that belongs in the main-effects table.
def read_training_summary(
    spec: VariantSpec,
    results_dir: Path,
) -> dict[str, float]:
    history_path = results_dir / spec.training_history_file

    if not history_path.exists():
        return {
            "train_epochs": float("nan"),
            "best_epoch": float("nan"),
            "best_val_loss": float("nan"),
        }

    history = pd.read_csv(history_path)

    if history.empty:
        return {
            "train_epochs": 0.0,
            "best_epoch": float("nan"),
            "best_val_loss": float("nan"),
        }

    last_row = history.iloc[-1]

    return {
        "train_epochs": float(len(history)),
        "best_epoch": float(last_row.get("best_epoch", float("nan"))),
        "best_val_loss": float(last_row.get("best_val_loss", float("nan"))),
    }


# Combine all flattened evaluation CSVs into one long table.
def load_main_effect_rows(results_dir: Path) -> pd.DataFrame:
    tables = []

    for spec in VARIANT_SPECS:
        metrics_path = results_dir / spec.eval_metrics_file

        if not metrics_path.exists():
            raise FileNotFoundError(f"Missing evaluation metrics: {metrics_path}")

        table = pd.read_csv(metrics_path)
        table.insert(0, "variant_id", spec.variant_id)
        table.insert(1, "variant_label", spec.label)
        tables.append(table)

    long_table = pd.concat(tables, ignore_index=True)

    return long_table


# Create the compact Step 9 main-effects table.
def make_main_effects_table(
    long_table: pd.DataFrame,
    results_dir: Path,
) -> pd.DataFrame:
    rows = []
    spec_by_id = {spec.variant_id: spec for spec in VARIANT_SPECS}

    for variant_id in [spec.variant_id for spec in VARIANT_SPECS]:
        spec = spec_by_id[variant_id]
        training_summary = read_training_summary(spec, results_dir)

        for tier_name in TIER_NAMES:
            subset = long_table[
                (long_table["variant_id"] == variant_id)
                & (long_table["tier"] == tier_name)
            ]
            row = {
                "variant_id": variant_id,
                "variant_label": spec.label,
                "tier": tier_name,
                **training_summary,
                "train_time_seconds": float("nan"),
                "train_time_note": "not_recorded_in_existing_training_history",
            }

            for metric_name in MAIN_EFFECT_METRICS:
                metric_rows = subset[subset["metric"] == metric_name]

                if metric_rows.empty:
                    row[metric_name] = float("nan")
                    row[f"{metric_name}_ci_lower"] = float("nan")
                    row[f"{metric_name}_ci_upper"] = float("nan")
                    continue

                metric_row = metric_rows.iloc[0]
                row[metric_name] = float(metric_row["value"])
                row[f"{metric_name}_ci_lower"] = float(metric_row["ci_lower"])
                row[f"{metric_name}_ci_upper"] = float(metric_row["ci_upper"])

            total_parameters = row.get("total_parameters", float("nan"))
            row["parameter_memory_mb_float32"] = (
                total_parameters * 4.0 / (1024.0**2)
                if np.isfinite(total_parameters)
                else float("nan")
            )
            rows.append(row)

    return pd.DataFrame(rows)


# Extract one metric value from the compact main-effects table.
def get_metric_value(
    main_effects_table: pd.DataFrame,
    variant_id: str,
    tier_name: str,
    metric_name: str,
) -> float:
    rows = main_effects_table[
        (main_effects_table["variant_id"] == variant_id)
        & (main_effects_table["tier"] == tier_name)
    ]

    if rows.empty:
        raise KeyError(f"Missing {variant_id}/{tier_name}/{metric_name}.")

    return float(rows.iloc[0][metric_name])


# Compute explicit interaction deltas for the two 2x2 designs.
def compute_interaction_deltas(
    main_effects_table: pd.DataFrame,
    metrics: tuple[str, ...] = ("R2", "RMSE", "MAE", "MaxAE"),
) -> pd.DataFrame:
    families = (
        {
            "family": "Bounds_x_Monotonicity",
            "factor_a": "Bounds",
            "factor_b": "Monotonicity",
            "base": "V1",
            "a_only": "V3",
            "b_only": "V4",
            "joint": "V7",
        },
        {
            "family": "Sips_x_vH",
            "factor_a": "Sips",
            "factor_b": "van_t_Hoff",
            "base": "V1",
            "a_only": "V5",
            "b_only": "V6",
            "joint": "V8",
        },
    )
    lower_is_better = {"RMSE", "MAE", "MaxAE"}
    rows = []

    for tier_name in TIER_NAMES:
        for family in families:
            for metric_name in metrics:
                base = get_metric_value(
                    main_effects_table,
                    family["base"],
                    tier_name,
                    metric_name,
                )
                a_only = get_metric_value(
                    main_effects_table,
                    family["a_only"],
                    tier_name,
                    metric_name,
                )
                b_only = get_metric_value(
                    main_effects_table,
                    family["b_only"],
                    tier_name,
                    metric_name,
                )
                joint = get_metric_value(
                    main_effects_table,
                    family["joint"],
                    tier_name,
                    metric_name,
                )
                raw_delta = joint - a_only - b_only + base
                advantage_delta = (
                    -raw_delta if metric_name in lower_is_better else raw_delta
                )

                if advantage_delta > 0:
                    interaction_type = "superadditive"
                elif advantage_delta < 0:
                    interaction_type = "antagonistic"
                else:
                    interaction_type = "additive"

                rows.append(
                    {
                        "family": family["family"],
                        "tier": tier_name,
                        "metric": metric_name,
                        "base_variant": family["base"],
                        "factor_a_variant": family["a_only"],
                        "factor_b_variant": family["b_only"],
                        "joint_variant": family["joint"],
                        "base_value": base,
                        "factor_a_value": a_only,
                        "factor_b_value": b_only,
                        "joint_value": joint,
                        "raw_delta": raw_delta,
                        "advantage_delta": advantage_delta,
                        "interaction_type": interaction_type,
                    }
                )

    return pd.DataFrame(rows)


# Compute sample-level bounds violations for McNemar's test.
def sample_bounds_violations(predictions: dict[str, torch.Tensor]) -> np.ndarray:
    q_hat = predictions["q_hat"].reshape(-1)
    violations = q_hat < 0

    if "q_max" in predictions:
        q_max = predictions["q_max"].reshape(-1)
        violations = violations | (q_hat > q_max)

    return violations.detach().cpu().numpy().astype(bool)


# Compute adjacent pressure monotonicity violations for McNemar's test.
def monotonicity_edge_violations(
    predictions: dict[str, torch.Tensor],
    tolerance: float = 1e-8,
) -> np.ndarray:
    q_hat = predictions["q_hat"].detach().cpu().reshape(-1)
    raw_pressure = predictions["raw_pressure"].detach().cpu().reshape(-1)
    experiment_id = predictions["experiment_id"].detach().cpu().reshape(-1)
    violations = []

    for experiment in torch.unique(experiment_id):
        experiment_mask = experiment_id == experiment

        if experiment_mask.sum().item() < 2:
            continue

        pressure_values = raw_pressure[experiment_mask]
        prediction_values = q_hat[experiment_mask]
        order = torch.argsort(pressure_values)
        pressure_sorted = pressure_values[order]
        prediction_sorted = prediction_values[order]
        pressure_increases = pressure_sorted[1:] > pressure_sorted[:-1]
        prediction_decreases = (
            prediction_sorted[1:] < prediction_sorted[:-1] - tolerance
        )
        edge_violations = prediction_decreases[pressure_increases]
        violations.extend(edge_violations.numpy().astype(bool).tolist())

    return np.asarray(violations, dtype=bool)


# Run all pairwise RMSE comparisons and McNemar violation tests.
def compute_pairwise_tests(
    predictions_by_variant: dict[str, dict[str, dict[str, torch.Tensor]]],
    n_bootstrap: int = 100,
    alpha: float = 0.05,
    random_state: int = 42,
) -> pd.DataFrame:
    rows = []
    variant_ids = [spec.variant_id for spec in VARIANT_SPECS]
    n_comparisons = len(list(combinations(variant_ids, 2)))

    for tier_name in TIER_NAMES:
        tier_rows = []

        for pair_index, (variant_a, variant_b) in enumerate(
            combinations(variant_ids, 2)
        ):
            comparison = compare_variants(
                {"predictions": predictions_by_variant[variant_a][tier_name]},
                {"predictions": predictions_by_variant[variant_b][tier_name]},
                metric_name="RMSE",
                n_bootstrap=n_bootstrap,
                alpha=alpha,
                random_state=random_state + pair_index,
            )

            if abs(comparison["difference"]) < 1e-12:
                comparison["p_value"] = 1.0
                comparison["cohens_d"] = 0.0

            bounds_test = mcnemar_test(
                sample_bounds_violations(predictions_by_variant[variant_a][tier_name]),
                sample_bounds_violations(predictions_by_variant[variant_b][tier_name]),
            )
            mono_a = monotonicity_edge_violations(
                predictions_by_variant[variant_a][tier_name]
            )
            mono_b = monotonicity_edge_violations(
                predictions_by_variant[variant_b][tier_name]
            )

            if mono_a.size and mono_b.size:
                mono_test = mcnemar_test(mono_a, mono_b)
            else:
                mono_test = {
                    "p_value": float("nan"),
                    "discordant": float("nan"),
                }

            tier_rows.append(
                {
                    "tier": tier_name,
                    "variant_a": variant_a,
                    "variant_b": variant_b,
                    "RMSE_a": comparison["variant_a"],
                    "RMSE_b": comparison["variant_b"],
                    "RMSE_diff_a_minus_b": comparison["difference"],
                    "RMSE_diff_ci_lower": comparison["ci_lower"],
                    "RMSE_diff_ci_upper": comparison["ci_upper"],
                    "RMSE_p_value": comparison["p_value"],
                    "RMSE_cohens_d": comparison["cohens_d"],
                    "n_experiments": comparison["n_experiments"],
                    "bounds_mcnemar_p_value": bounds_test["p_value"],
                    "bounds_mcnemar_discordant": bounds_test["discordant"],
                    "mono_mcnemar_p_value": mono_test["p_value"],
                    "mono_mcnemar_discordant": mono_test["discordant"],
                }
            )

        correction = bonferroni_correct(
            [row["RMSE_p_value"] for row in tier_rows],
            alpha=alpha,
            n_comparisons=n_comparisons,
        )

        for row, adjusted_p, significant in zip(
            tier_rows,
            correction["adjusted_p_values"],
            correction["significant"],
        ):
            row["bonferroni_alpha"] = correction["corrected_alpha"]
            row["RMSE_p_value_bonferroni"] = adjusted_p
            row["RMSE_significant_bonferroni"] = bool(significant)

        rows.extend(tier_rows)

    return pd.DataFrame(rows)


# Save adjusted-p-value significance matrices for each tier.
def save_significance_matrices(
    pairwise_table: pd.DataFrame,
    output_dir: Path,
) -> dict[str, Path]:
    output_paths = {}
    variant_ids = [spec.variant_id for spec in VARIANT_SPECS]

    for tier_name in TIER_NAMES:
        tier_rows = pairwise_table[pairwise_table["tier"] == tier_name]
        matrix = pd.DataFrame(
            np.nan,
            index=variant_ids,
            columns=variant_ids,
        )

        for _, row in tier_rows.iterrows():
            variant_a = row["variant_a"]
            variant_b = row["variant_b"]
            adjusted_p = row["RMSE_p_value_bonferroni"]
            matrix.loc[variant_a, variant_b] = adjusted_p
            matrix.loc[variant_b, variant_a] = adjusted_p

        matrix_path = output_dir / f"step9_pairwise_significance_matrix_{tier_name}.csv"
        matrix.to_csv(matrix_path)
        output_paths[f"significance_matrix_{tier_name}"] = matrix_path

        try:
            import matplotlib.pyplot as plt

            plot_values = -np.log10(matrix.astype(float).to_numpy())
            finite_values = plot_values[np.isfinite(plot_values)]
            vmax = max(1.0, float(finite_values.max())) if finite_values.size else 1.0

            fig, ax = plt.subplots(figsize=(7, 6))
            image = ax.imshow(plot_values, vmin=0.0, vmax=vmax, cmap="viridis")
            ax.set_xticks(range(len(variant_ids)), labels=variant_ids, rotation=45)
            ax.set_yticks(range(len(variant_ids)), labels=variant_ids)
            ax.set_title(f"Step 9 RMSE Pairwise Significance: {tier_name}")
            fig.colorbar(image, ax=ax, label="-log10(Bonferroni p)")
            fig.tight_layout()
            heatmap_path = (
                output_dir / f"step9_pairwise_significance_heatmap_{tier_name}.png"
            )
            fig.savefig(heatmap_path, dpi=200)
            plt.close(fig)
            output_paths[f"significance_heatmap_{tier_name}"] = heatmap_path
        except ImportError:
            continue

    return output_paths


# Bootstrap replicated 2x2 ANOVA records from matched predictions.
def make_bootstrap_anova_records(
    predictions_by_variant: dict[str, dict[str, dict[str, torch.Tensor]]],
    family: dict[str, str],
    tier_name: str,
    metric_name: str,
    n_bootstrap: int,
    random_state: int,
) -> list[dict]:
    base_predictions = predictions_by_variant[family["base"]][tier_name]
    experiment_ids = base_predictions["experiment_id"].detach().cpu().numpy()
    unique_experiments = np.unique(experiment_ids)
    rng = np.random.default_rng(random_state)
    records = []

    cells = (
        (family["base"], "S", "S"),
        (family["a_only"], "H", "S"),
        (family["b_only"], "S", "H"),
        (family["joint"], "H", "H"),
    )

    for bootstrap_index in range(n_bootstrap):
        sampled_experiments = rng.choice(
            unique_experiments,
            size=len(unique_experiments),
            replace=True,
        )
        sampled_indices = np.concatenate(
            [
                np.flatnonzero(experiment_ids == experiment_id)
                for experiment_id in sampled_experiments
            ]
        )
        indices = torch.tensor(sampled_indices, dtype=torch.long)

        for variant_id, factor_a, factor_b in cells:
            metric_value = compute_named_accuracy_metric(
                predictions_by_variant[variant_id][tier_name],
                metric_name,
                indices,
            )

            if not np.isfinite(metric_value):
                continue

            records.append(
                {
                    "bootstrap_index": bootstrap_index,
                    "variant_id": variant_id,
                    family["factor_a"]: factor_a,
                    family["factor_b"]: factor_b,
                    metric_name: float(metric_value),
                }
            )

    return records


# Run bootstrap-based 2x2 ANOVA interaction checks.
def compute_anova_interactions(
    predictions_by_variant: dict[str, dict[str, dict[str, torch.Tensor]]],
    n_bootstrap: int = 100,
    random_state: int = 42,
) -> pd.DataFrame:
    families = (
        {
            "family": "Bounds_x_Monotonicity",
            "factor_a": "Bounds",
            "factor_b": "Monotonicity",
            "base": "V1",
            "a_only": "V3",
            "b_only": "V4",
            "joint": "V7",
        },
        {
            "family": "Sips_x_vH",
            "factor_a": "Sips",
            "factor_b": "van_t_Hoff",
            "base": "V1",
            "a_only": "V5",
            "b_only": "V6",
            "joint": "V8",
        },
    )
    rows = []

    for family_index, family in enumerate(families):
        for tier_index, tier_name in enumerate(TIER_NAMES):
            for metric_index, metric_name in enumerate(("R2", "RMSE")):
                records = make_bootstrap_anova_records(
                    predictions_by_variant,
                    family,
                    tier_name,
                    metric_name,
                    n_bootstrap=n_bootstrap,
                    random_state=random_state
                    + 1000 * family_index
                    + 100 * tier_index
                    + metric_index,
                )
                anova_results = anova_2x2(
                    records,
                    factor1=family["factor_a"],
                    factor2=family["factor_b"],
                    metric_name=metric_name,
                )

                for term_name, term_metrics in anova_results.items():
                    rows.append(
                        {
                            "family": family["family"],
                            "tier": tier_name,
                            "metric": metric_name,
                            "term": term_name,
                            **term_metrics,
                        }
                    )

    return pd.DataFrame(rows)


# Compute R2 retention ratios and pair them with physical violation rates.
def compute_extrapolation_retention(
    main_effects_table: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for spec in VARIANT_SPECS:
        r2_t1 = get_metric_value(main_effects_table, spec.variant_id, "T1", "R2")

        for extrap_tier in ("T2", "T3"):
            r2_extrap = get_metric_value(
                main_effects_table,
                spec.variant_id,
                extrap_tier,
                "R2",
            )
            violation_rates = [
                get_metric_value(
                    main_effects_table,
                    spec.variant_id,
                    extrap_tier,
                    metric_name,
                )
                for metric_name in ("neg_rate", "mono_rate", "sat_rate")
            ]
            finite_violation_rates = [
                value for value in violation_rates if np.isfinite(value)
            ]
            max_violation_rate = (
                max(finite_violation_rates) if finite_violation_rates else float("nan")
            )

            rows.append(
                {
                    "variant_id": spec.variant_id,
                    "variant_label": spec.label,
                    "tier": extrap_tier,
                    "R2_T1": r2_t1,
                    "R2_extrapolation": r2_extrap,
                    "R2_retention_ratio": (
                        r2_extrap / r2_t1
                        if np.isfinite(r2_t1) and abs(r2_t1) > 1e-12
                        else float("nan")
                    ),
                    "max_violation_rate": max_violation_rate,
                }
            )

    retention_table = pd.DataFrame(rows)

    if len(retention_table) >= 2:
        retention_table["spearman_r_retention_vs_violation"] = retention_table[
            "R2_retention_ratio"
        ].corr(
            retention_table["max_violation_rate"],
            method="spearman",
        )
    else:
        retention_table["spearman_r_retention_vs_violation"] = float("nan")

    return retention_table


# Save the retention-ratio plot requested by Step 9.
def save_retention_plot(
    retention_table: pd.DataFrame,
    output_path: Path,
) -> Optional[Path]:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    marker_by_tier = {"T2": "o", "T3": "s"}
    fig, ax = plt.subplots(figsize=(7, 5))

    for tier_name, tier_rows in retention_table.groupby("tier"):
        ax.scatter(
            tier_rows["max_violation_rate"],
            tier_rows["R2_retention_ratio"],
            label=tier_name,
            marker=marker_by_tier.get(tier_name, "o"),
            s=60,
        )

        for _, row in tier_rows.iterrows():
            ax.annotate(
                row["variant_id"],
                (row["max_violation_rate"], row["R2_retention_ratio"]),
                fontsize=8,
                xytext=(4, 4),
                textcoords="offset points",
            )

    ax.set_xlabel("Maximum physical violation rate")
    ax.set_ylabel("R2 retention ratio")
    ax.set_title("Step 9 Extrapolation Retention vs Violation Rate")
    ax.legend(title="Tier")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    return output_path


# Run and save the complete Step 9 ablation analysis.
def run_and_save_step9_analysis(
    results_dir=Path("results"),
    checkpoint_dir=Path("checkpoints"),
    batch_size: int = 64,
    device: str = "cpu",
    pairwise_n_bootstrap: int = 100,
    anova_n_bootstrap: int = 100,
    random_state: int = 42,
    verbose: bool = True,
) -> dict[str, object]:
    results_dir = Path(results_dir)
    checkpoint_dir = Path(checkpoint_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    long_table = load_main_effect_rows(results_dir)
    main_effects_table = make_main_effects_table(long_table, results_dir)

    main_effects_long_path = results_dir / "step9_main_effects_long.csv"
    main_effects_table_path = results_dir / "step9_main_effects_table.csv"
    long_table.to_csv(main_effects_long_path, index=False)
    main_effects_table.to_csv(main_effects_table_path, index=False)

    pipeline = build_data_pipeline(batch_size=batch_size)
    predictions_by_variant = collect_variant_predictions(
        pipeline,
        checkpoint_dir=checkpoint_dir,
        device=device,
    )

    pairwise_table = compute_pairwise_tests(
        predictions_by_variant,
        n_bootstrap=pairwise_n_bootstrap,
        random_state=random_state,
    )
    pairwise_path = results_dir / "step9_pairwise_tests.csv"
    pairwise_table.to_csv(pairwise_path, index=False)
    significance_paths = save_significance_matrices(pairwise_table, results_dir)

    interaction_deltas = compute_interaction_deltas(main_effects_table)
    interaction_deltas_path = results_dir / "step9_interaction_deltas.csv"
    interaction_deltas.to_csv(interaction_deltas_path, index=False)

    anova_table = compute_anova_interactions(
        predictions_by_variant,
        n_bootstrap=anova_n_bootstrap,
        random_state=random_state,
    )
    anova_path = results_dir / "step9_anova_interactions.csv"
    anova_table.to_csv(anova_path, index=False)

    retention_table = compute_extrapolation_retention(main_effects_table)
    retention_path = results_dir / "step9_extrapolation_retention.csv"
    retention_table.to_csv(retention_path, index=False)
    retention_plot_path = save_retention_plot(
        retention_table,
        results_dir / "step9_retention_vs_violation.png",
    )

    paths = {
        "main_effects_long": main_effects_long_path,
        "main_effects_table": main_effects_table_path,
        "pairwise_tests": pairwise_path,
        "interaction_deltas": interaction_deltas_path,
        "anova_interactions": anova_path,
        "extrapolation_retention": retention_path,
    }
    paths.update(significance_paths)

    if retention_plot_path is not None:
        paths["retention_plot"] = retention_plot_path

    if verbose:
        for name, path in paths.items():
            print(f"Saved {name}: {path}")

    return {
        "main_effects_long": long_table,
        "main_effects_table": main_effects_table,
        "pairwise_tests": pairwise_table,
        "interaction_deltas": interaction_deltas,
        "anova_interactions": anova_table,
        "extrapolation_retention": retention_table,
        "paths": paths,
    }


if __name__ == "__main__":
    run_and_save_step9_analysis()
