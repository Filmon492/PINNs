import time
from pathlib import Path
from typing import Optional, TypedDict

import numpy as np
import pandas as pd
import torch
from scipy.stats import binomtest, norm, spearmanr, ttest_1samp
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

from data_pipeline import build_data_pipeline


def main():
    data = build_data_pipeline()
    print("T1 samples:", len(data["T1_loader"].dataset))
    print("T3 samples:", len(data["T3_loader"].dataset))

class ModelOutput(TypedDict, total=False):
    q_hat: torch.Tensor
    q_max: torch.Tensor
    sigma_ale: torch.Tensor

# validate model output while running
def validate_model_output(output: ModelOutput,batch_size: int,) -> None:
    if "q_hat" not in output:
        raise KeyError("Model output must contain 'q_hat'.")

    if not isinstance(output["q_hat"], torch.Tensor):
        raise TypeError("q_hat must be a PyTorch tensor.")

    if output["q_hat"].numel() != batch_size:
        raise ValueError("q_hat must contain one prediction per sample.")

    if "q_max" in output:
        if not isinstance(output["q_max"], torch.Tensor):
            raise TypeError("q_max must be a PyTorch tensor.")

        if output["q_max"].numel() != batch_size:
            raise ValueError("q_max must contain one value per sample.")

    if "sigma_ale" in output:
        if not isinstance(output["sigma_ale"], torch.Tensor):
            raise TypeError("sigma_ale must be a PyTorch tensor.")

        if output["sigma_ale"].numel() != batch_size:
            raise ValueError("sigma_ale must contain one value per sample.")

        if (output["sigma_ale"] < 0).any().item():
            raise ValueError("sigma_ale must be non-negative.")

# Collect predictions and matching observations from a DataLoader.
def collect_predictions(
    model,
    loader,
    device: str = "cpu",
) -> dict[str, torch.Tensor]:
    device = torch.device(device)
    model = model.to(device)
    model.eval()

    predictions = []
    observations = []
    raw_pressures = []
    experiment_ids = []
    maximum_capacities = []

    with torch.inference_mode():
        for batch in loader:
            x = batch["x"].to(device)
            raw_pressure = batch["raw_pressure"].to(device)
            y_log = batch["y"].to(device)

            output = model(x, raw_pressure)
            validate_model_output(output, batch_size=len(x))

            q_hat = output["q_hat"].reshape(-1, 1)
            q_true = torch.exp(y_log).reshape(-1, 1)

            predictions.append(q_hat.cpu())
            observations.append(q_true.cpu())
            raw_pressures.append(raw_pressure.cpu())
            experiment_ids.append(batch["experiment_id"].cpu())

            if "q_max" in output:
                q_max = output["q_max"].reshape(-1, 1)
                maximum_capacities.append(q_max.cpu())

    if not predictions:
        raise ValueError("Cannot collect predictions from an empty DataLoader.")

    collected = {
        "q_hat": torch.cat(predictions),
        "q_true": torch.cat(observations),
        "raw_pressure": torch.cat(raw_pressures),
        "experiment_id": torch.cat(experiment_ids),
    }

    if maximum_capacities:
        if len(maximum_capacities) != len(predictions):
            raise ValueError("Model returned q_max for only some batches.")

        collected["q_max"] = torch.cat(maximum_capacities)

    return collected

# Activate dropout modules while keeping the rest of the model in eval mode.
def enable_dropout_layers(model) -> None:
    dropout_modules = (
        torch.nn.Dropout,
        torch.nn.Dropout1d,
        torch.nn.Dropout2d,
        torch.nn.Dropout3d,
        torch.nn.AlphaDropout,
        torch.nn.FeatureAlphaDropout,
    )

    model.eval()

    for module in model.modules():
        if isinstance(module, dropout_modules):
            module.train()

# Collect repeated stochastic predictions for MC Dropout uncertainty.
def collect_mc_dropout_predictions(
    model,
    loader,
    device: str = "cpu",
    n_mc: int = 100,
) -> dict[str, torch.Tensor]:
    if n_mc < 2:
        raise ValueError("MC Dropout requires at least two forward passes.")

    device = torch.device(device)
    model = model.to(device)
    was_training = model.training

    q_hat_passes = []
    sigma_ale_passes = []

    try:
        with torch.inference_mode():
            for _ in range(n_mc):
                enable_dropout_layers(model)

                pass_predictions = []
                pass_aleatoric = []

                for batch in loader:
                    x = batch["x"].to(device)
                    raw_pressure = batch["raw_pressure"].to(device)

                    output = model(x, raw_pressure)
                    validate_model_output(output, batch_size=len(x))

                    pass_predictions.append(
                        output["q_hat"].reshape(-1, 1).cpu()
                    )

                    if "sigma_ale" in output:
                        pass_aleatoric.append(
                            output["sigma_ale"].reshape(-1, 1).cpu()
                        )

                q_hat_passes.append(torch.cat(pass_predictions))

                if pass_aleatoric:
                    sigma_ale_passes.append(torch.cat(pass_aleatoric))
    finally:
        model.train(was_training)

    if not q_hat_passes:
        raise ValueError("Cannot collect MC predictions from an empty loader.")

    mc_predictions = {
        "q_hat_mc": torch.stack(q_hat_passes),
    }

    if sigma_ale_passes:
        if len(sigma_ale_passes) != len(q_hat_passes):
            raise ValueError("Model returned sigma_ale for only some MC passes.")

        mc_predictions["sigma_ale_mc"] = torch.stack(sigma_ale_passes)

    return mc_predictions

# Compute regression metrics on the original adsorption scale.
def compute_accuracy_metrics(
    q_true: torch.Tensor,
    q_hat: torch.Tensor,
) -> dict[str, float]:
    true_values = q_true.detach().cpu().numpy().reshape(-1)
    predictions = q_hat.detach().cpu().numpy().reshape(-1)

    if true_values.shape != predictions.shape:
        raise ValueError("q_true and q_hat must have the same shape.")

    if true_values.size == 0:
        raise ValueError("Cannot calculate metrics from empty tensors.")

    absolute_errors = np.abs(true_values - predictions)
    relative_errors = absolute_errors / np.maximum(
        np.abs(true_values),
        1e-8,
    )

    if len(true_values) < 2 or np.allclose(true_values, true_values[0]):
        r2 = float("nan")
    else:
        r2 = float(r2_score(true_values, predictions))

    return {
        "R2": r2,
        "RMSE": float(
            np.sqrt(mean_squared_error(true_values, predictions))
        ),
        "MAE": float(
            mean_absolute_error(true_values, predictions)
        ),
        "MaxAE": float(absolute_errors.max()),
        "median_relative_error_pct": float(
            np.median(relative_errors) * 100
        ),
        "mean_relative_error_pct": float(
            np.mean(relative_errors) * 100
        ),
    }

# Compute calibration and uncertainty metrics from MC Dropout predictions.
def compute_uq_metrics(
    q_true: torch.Tensor,
    q_hat_mc: torch.Tensor,
    sigma_ale_mc: Optional[torch.Tensor] = None,
) -> dict[str, float]:
    true_values = q_true.detach().cpu().numpy().reshape(-1)
    mc_values = q_hat_mc.detach().cpu().numpy()

    if mc_values.ndim < 2:
        raise ValueError("q_hat_mc must contain MC passes and samples.")

    mc_values = mc_values.reshape(mc_values.shape[0], -1)

    if mc_values.shape[0] < 2:
        raise ValueError("UQ metrics require at least two MC passes.")

    if mc_values.shape[1] != true_values.size:
        raise ValueError("q_hat_mc and q_true must contain the same samples.")

    mean_prediction = mc_values.mean(axis=0)
    sigma_epi_values = mc_values.std(axis=0, ddof=1)

    if sigma_ale_mc is None:
        sigma_ale_values = np.zeros_like(sigma_epi_values)
    else:
        ale_values = sigma_ale_mc.detach().cpu().numpy()
        ale_values = ale_values.reshape(ale_values.shape[0], -1)

        if ale_values.shape != mc_values.shape:
            raise ValueError("sigma_ale_mc must match q_hat_mc shape.")

        sigma_ale_values = ale_values.mean(axis=0)

    sigma_total_values = np.sqrt(
        sigma_epi_values**2 + sigma_ale_values**2
    )
    sigma_total_values = np.maximum(sigma_total_values, 1e-8)
    absolute_errors = np.abs(true_values - mean_prediction)

    confidence_levels = (0.68, 0.90, 0.95, 0.99)
    coverage_errors = []
    metrics = {
        "sigma_epi": float(np.mean(sigma_epi_values)),
        "sigma_ale": float(np.mean(sigma_ale_values)),
        "sigma_total": float(np.mean(sigma_total_values)),
    }

    for level in confidence_levels:
        z_value = norm.ppf((1 + level) / 2)
        covered = absolute_errors <= z_value * sigma_total_values
        coverage = float(np.mean(covered))
        metrics[f"coverage_{int(level * 100)}"] = coverage
        coverage_errors.append(abs(coverage - level))

    metrics["ECE"] = float(np.mean(coverage_errors))

    if (
        true_values.size < 2
        or np.allclose(absolute_errors, absolute_errors[0])
        or np.allclose(sigma_total_values, sigma_total_values[0])
    ):
        metrics["rho_s"] = float("nan")
    else:
        metrics["rho_s"] = float(
            spearmanr(absolute_errors, sigma_total_values).statistic
        )

    variance = sigma_total_values**2
    standardized_error = (true_values - mean_prediction) / sigma_total_values
    nll_values = 0.5 * np.log(2 * np.pi * variance)
    nll_values += 0.5 * standardized_error**2
    crps_values = sigma_total_values * (
        standardized_error * (2 * norm.cdf(standardized_error) - 1)
        + 2 * norm.pdf(standardized_error)
        - 1 / np.sqrt(np.pi)
    )

    metrics["NLL"] = float(np.mean(nll_values))
    metrics["CRPS"] = float(np.mean(crps_values))
    metrics["sharpness_95"] = float(
        np.mean(2 * norm.ppf(0.975) * sigma_total_values)
    )

    return metrics

# Synchronize accelerator work before reading wall-clock time.
def synchronize_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()

# Compute practical runtime and model-size metrics for one evaluation.
def compute_computational_metrics(
    model,
    n_samples: int,
    elapsed_seconds: float,
) -> dict[str, float]:
    if n_samples < 0:
        raise ValueError("n_samples cannot be negative.")

    if elapsed_seconds < 0:
        raise ValueError("elapsed_seconds cannot be negative.")

    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    if elapsed_seconds == 0:
        samples_per_second = float("inf")
    else:
        samples_per_second = n_samples / elapsed_seconds

    return {
        "n_samples": float(n_samples),
        "eval_seconds": float(elapsed_seconds),
        "samples_per_second": float(samples_per_second),
        "total_parameters": float(total_parameters),
        "trainable_parameters": float(trainable_parameters),
    }

# Raise an error when a hard physical constraint is violated.
def assert_hard_constraints(
    physical_metrics: dict[str, float],
    hard_constraints: tuple[str, ...],
    tolerance: float = 0.0,
) -> None:
    metric_names = {
        "nonnegative": "neg_rate",
        "saturation": "sat_rate",
        "monotonicity": "mono_rate",
    }

    for constraint_name in hard_constraints:
        if constraint_name not in metric_names:
            valid_names = ", ".join(metric_names)
            raise KeyError(
                f"Unknown hard constraint '{constraint_name}'. "
                f"Valid constraints are: {valid_names}."
            )

        metric_name = metric_names[constraint_name]
        metric_value = physical_metrics.get(metric_name, float("nan"))

        if not np.isfinite(metric_value):
            raise ValueError(
                f"Cannot verify hard constraint '{constraint_name}' "
                f"because {metric_name} is not available."
            )

        if metric_value > tolerance:
            raise ValueError(
                f"Hard constraint '{constraint_name}' was violated: "
                f"{metric_name} = {metric_value}."
            )

# Evaluate accuracy and uncertainty for one data tier.
def evaluate_accuracy(
    model,
    loader,
    device: str = "cpu",
    n_bootstrap: int = 100,
    n_mc: int = 100,
    hard_constraints: tuple[str, ...] = (),
    alpha: float = 0.05,
    random_state: int = 42,
) -> dict:
    evaluation_device = torch.device(device)
    synchronize_device(evaluation_device)
    start_time = time.perf_counter()

    collected = collect_predictions(
        model,
        loader,
        device=device,
    )

    synchronize_device(evaluation_device)
    elapsed_seconds = time.perf_counter() - start_time

    metrics = compute_accuracy_metrics(
        collected["q_true"],
        collected["q_hat"],
    )

    confidence_intervals = bootstrap_accuracy_metrics(
        collected,
        n_bootstrap=n_bootstrap,
        alpha=alpha,
        random_state=random_state,
    )

    physical_metrics = compute_physical_violation_metrics(collected)
    assert_hard_constraints(physical_metrics, hard_constraints)

    physical_confidence_intervals = bootstrap_physical_violation_metrics(
        collected,
        n_bootstrap=n_bootstrap,
        alpha=alpha,
        random_state=random_state,
    )

    computational_metrics = compute_computational_metrics(
        model,
        n_samples=collected["q_hat"].numel(),
        elapsed_seconds=elapsed_seconds,
    )

    if n_mc >= 2:
        mc_predictions = collect_mc_dropout_predictions(
            model,
            loader,
            device=device,
            n_mc=n_mc,
        )
        uq_metrics = compute_uq_metrics(
            collected["q_true"],
            mc_predictions["q_hat_mc"],
            mc_predictions.get("sigma_ale_mc"),
        )
    else:
        uq_metrics = {}

    return {
        "metrics": metrics,
        "confidence_intervals": confidence_intervals,
        "physical_metrics": physical_metrics,
        "physical_confidence_intervals": physical_confidence_intervals,
        "uq_metrics": uq_metrics,
        "computational_metrics": computational_metrics,
        "predictions": collected,
    }

# Evaluate one model on multiple data tiers using the same harness.
def evaluate_model(
    model,
    pipeline: dict,
    tier_names: tuple[str, ...] = ("T1", "T3"),
    device: str = "cpu",
    n_bootstrap: int = 100,
    n_mc: int = 100,
    hard_constraints: tuple[str, ...] = (),
    alpha: float = 0.05,
    random_state: int = 42,
) -> dict[str, dict]:
    loader_keys = {
        "train": "train_loader",
        "validation": "val_loader",
        "val": "val_loader",
        "T1": "T1_loader",
        "T3": "T3_loader",
    }

    results = {}

    for tier_name in tier_names:
        loader_key = loader_keys.get(tier_name, f"{tier_name}_loader")

        if loader_key not in pipeline:
            raise KeyError(f"Pipeline does not contain {loader_key}.")

        results[tier_name] = evaluate_accuracy(
            model,
            pipeline[loader_key],
            device=device,
            n_bootstrap=n_bootstrap,
            n_mc=n_mc,
            hard_constraints=hard_constraints,
            alpha=alpha,
            random_state=random_state,
        )

    return results

# Convert nested evaluation results into one tabular metrics DataFrame.
def flatten_evaluation_results(
    results: dict[str, dict],
    model_name: str = "model",
) -> pd.DataFrame:
    rows = []

    for tier_name, tier_results in results.items():
        accuracy_intervals = tier_results.get("confidence_intervals", {})
        physical_intervals = tier_results.get("physical_confidence_intervals", {})

        for metric_name, interval in accuracy_intervals.items():
            rows.append(
                {
                    "model": model_name,
                    "tier": tier_name,
                    "metric_group": "accuracy",
                    "metric": metric_name,
                    "value": interval["value"],
                    "ci_lower": interval["ci_lower"],
                    "ci_upper": interval["ci_upper"],
                }
            )

        for metric_name, value in tier_results.get("metrics", {}).items():
            if metric_name in accuracy_intervals:
                continue

            rows.append(
                {
                    "model": model_name,
                    "tier": tier_name,
                    "metric_group": "accuracy",
                    "metric": metric_name,
                    "value": value,
                    "ci_lower": float("nan"),
                    "ci_upper": float("nan"),
                }
            )

        for metric_name, interval in physical_intervals.items():
            rows.append(
                {
                    "model": model_name,
                    "tier": tier_name,
                    "metric_group": "physical",
                    "metric": metric_name,
                    "value": interval["value"],
                    "ci_lower": interval["ci_lower"],
                    "ci_upper": interval["ci_upper"],
                }
            )

        for metric_name, value in tier_results.get("physical_metrics", {}).items():
            if metric_name in physical_intervals:
                continue

            rows.append(
                {
                    "model": model_name,
                    "tier": tier_name,
                    "metric_group": "physical",
                    "metric": metric_name,
                    "value": value,
                    "ci_lower": float("nan"),
                    "ci_upper": float("nan"),
                }
            )

        for metric_name, value in tier_results.get("uq_metrics", {}).items():
            rows.append(
                {
                    "model": model_name,
                    "tier": tier_name,
                    "metric_group": "uq",
                    "metric": metric_name,
                    "value": value,
                    "ci_lower": float("nan"),
                    "ci_upper": float("nan"),
                }
            )

        for metric_name, value in tier_results.get(
            "computational_metrics",
            {},
        ).items():
            rows.append(
                {
                    "model": model_name,
                    "tier": tier_name,
                    "metric_group": "computational",
                    "metric": metric_name,
                    "value": value,
                    "ci_lower": float("nan"),
                    "ci_upper": float("nan"),
                }
            )

    columns = [
        "model",
        "tier",
        "metric_group",
        "metric",
        "value",
        "ci_lower",
        "ci_upper",
    ]
    return pd.DataFrame(rows, columns=columns)

# Save evaluation metrics to CSV for reproducible reporting.
def save_results_csv(
    results: dict[str, dict],
    output_path: Path,
    model_name: str = "model",
) -> pd.DataFrame:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results_table = flatten_evaluation_results(
        results,
        model_name=model_name,
    )
    results_table.to_csv(output_path, index=False)

    return results_table

# Estimate metric confidence intervals by resampling complete experiments.
def bootstrap_accuracy_metrics(
    predictions: dict[str, torch.Tensor],
    n_bootstrap: int = 100,
    alpha: float = 0.05,
    random_state: int = 42,
) -> dict[str, dict[str, float]]:
    q_true = predictions["q_true"]
    q_hat = predictions["q_hat"]
    experiment_ids = predictions["experiment_id"].numpy()

    unique_experiments = np.unique(experiment_ids)

    if len(unique_experiments) < 2:
        raise ValueError("Bootstrap requires at least two experiments.")

    rng = np.random.default_rng(random_state)
    point_metrics = compute_accuracy_metrics(q_true, q_hat)

    bootstrap_values = {
        metric_name: [] for metric_name in point_metrics
    }

    for _ in range(n_bootstrap):
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

        sample_metrics = compute_accuracy_metrics(
            q_true[indices],
            q_hat[indices],
        )

        for metric_name, value in sample_metrics.items():
            if np.isfinite(value):
                bootstrap_values[metric_name].append(value)

    results = {}

    for metric_name, point_value in point_metrics.items():
        values = np.asarray(bootstrap_values[metric_name])

        results[metric_name] = {
            "value": point_value,
            "ci_lower": float(np.quantile(values, alpha / 2)),
            "ci_upper": float(np.quantile(values, 1 - alpha / 2)),
        }

    return results

# Check that two evaluated variants were run on the same samples.
def validate_paired_variant_results(
    variant_a: dict,
    variant_b: dict,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    if "predictions" not in variant_a or "predictions" not in variant_b:
        raise KeyError("Both variants must contain collected predictions.")

    predictions_a = variant_a["predictions"]
    predictions_b = variant_b["predictions"]

    required_keys = ("q_true", "q_hat", "experiment_id")

    for key in required_keys:
        if key not in predictions_a or key not in predictions_b:
            raise KeyError(f"Both variants must contain {key}.")

    if predictions_a["q_true"].numel() != predictions_b["q_true"].numel():
        raise ValueError("Variants must contain the same number of samples.")

    if not torch.allclose(predictions_a["q_true"], predictions_b["q_true"]):
        raise ValueError("Variants must be evaluated on the same targets.")

    if not torch.equal(
        predictions_a["experiment_id"].reshape(-1),
        predictions_b["experiment_id"].reshape(-1),
    ):
        raise ValueError("Variants must use the same experiment ordering.")

    return predictions_a, predictions_b

# Compute one named accuracy metric from a prediction dictionary.
def compute_named_accuracy_metric(
    predictions: dict[str, torch.Tensor],
    metric_name: str,
    indices: torch.Tensor,
) -> float:
    metrics = compute_accuracy_metrics(
        predictions["q_true"][indices],
        predictions["q_hat"][indices],
    )

    if metric_name not in metrics:
        raise KeyError(f"Unknown accuracy metric: {metric_name}.")

    return metrics[metric_name]

# Compare two variants on one tier with paired experiment-level statistics.
def compare_variants(
    variant_a: dict,
    variant_b: dict,
    metric_name: str,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    random_state: int = 42,
) -> dict[str, float]:
    predictions_a, predictions_b = validate_paired_variant_results(
        variant_a,
        variant_b,
    )

    experiment_ids = predictions_a["experiment_id"].detach().cpu().numpy()
    unique_experiments = np.unique(experiment_ids)

    if len(unique_experiments) < 2:
        raise ValueError("Variant comparison requires at least two experiments.")

    all_indices = torch.arange(predictions_a["q_true"].numel())
    point_a = compute_named_accuracy_metric(
        predictions_a,
        metric_name,
        all_indices,
    )
    point_b = compute_named_accuracy_metric(
        predictions_b,
        metric_name,
        all_indices,
    )
    point_difference = point_a - point_b

    experiment_differences = []

    for experiment_id in unique_experiments:
        experiment_indices = torch.tensor(
            np.flatnonzero(experiment_ids == experiment_id),
            dtype=torch.long,
        )
        experiment_a = compute_named_accuracy_metric(
            predictions_a,
            metric_name,
            experiment_indices,
        )
        experiment_b = compute_named_accuracy_metric(
            predictions_b,
            metric_name,
            experiment_indices,
        )

        if np.isfinite(experiment_a) and np.isfinite(experiment_b):
            experiment_differences.append(experiment_a - experiment_b)

    experiment_differences = np.asarray(experiment_differences)

    if experiment_differences.size < 2:
        p_value = float("nan")
        cohens_d = float("nan")
    else:
        p_value = float(ttest_1samp(experiment_differences, 0.0).pvalue)
        difference_std = np.std(experiment_differences, ddof=1)

        if difference_std == 0:
            cohens_d = float("nan")
        else:
            cohens_d = float(np.mean(experiment_differences) / difference_std)

    rng = np.random.default_rng(random_state)
    bootstrap_differences = []

    for _ in range(n_bootstrap):
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

        sample_a = compute_named_accuracy_metric(
            predictions_a,
            metric_name,
            indices,
        )
        sample_b = compute_named_accuracy_metric(
            predictions_b,
            metric_name,
            indices,
        )

        if np.isfinite(sample_a) and np.isfinite(sample_b):
            bootstrap_differences.append(sample_a - sample_b)

    bootstrap_differences = np.asarray(bootstrap_differences)

    if bootstrap_differences.size == 0:
        ci_lower = float("nan")
        ci_upper = float("nan")
    else:
        ci_lower = float(np.quantile(bootstrap_differences, alpha / 2))
        ci_upper = float(np.quantile(bootstrap_differences, 1 - alpha / 2))

    return {
        "variant_a": float(point_a),
        "variant_b": float(point_b),
        "difference": float(point_difference),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "p_value": p_value,
        "cohens_d": cohens_d,
        "n_experiments": float(len(unique_experiments)),
    }

# Apply Bonferroni correction when many comparisons are tested together.
def bonferroni_correct(
    p_values: list[float],
    alpha: float = 0.05,
    n_comparisons: Optional[int] = None,
) -> dict[str, object]:
    if n_comparisons is None:
        n_comparisons = len(p_values)

    if n_comparisons <= 0:
        raise ValueError("n_comparisons must be positive.")

    corrected_alpha = alpha / n_comparisons
    adjusted_p_values = [
        min(float(p_value) * n_comparisons, 1.0)
        for p_value in p_values
    ]

    return {
        "corrected_alpha": float(corrected_alpha),
        "adjusted_p_values": adjusted_p_values,
        "significant": [
            float(p_value) <= corrected_alpha for p_value in p_values
        ],
    }

# Test two binary design factors and their interaction for one metric.
def anova_2x2(
    records: list[dict],
    factor1: str,
    factor2: str,
    metric_name: str,
) -> dict[str, dict[str, float]]:
    if not records:
        raise ValueError("ANOVA requires at least one record.")

    required_columns = {factor1, factor2, metric_name}
    missing_columns = [
        column
        for column in required_columns
        if any(column not in record for record in records)
    ]

    if missing_columns:
        raise KeyError(f"ANOVA records are missing columns: {missing_columns}.")

    df = pd.DataFrame(records)
    df = df[[factor1, factor2, metric_name]].dropna()
    df = df.rename(
        columns={
            factor1: "factor_a",
            factor2: "factor_b",
            metric_name: "metric_value",
        }
    )

    if df.empty:
        raise ValueError("ANOVA cannot run after dropping missing values.")

    factor_a_levels = df["factor_a"].nunique()
    factor_b_levels = df["factor_b"].nunique()

    if factor_a_levels != 2 or factor_b_levels != 2:
        raise ValueError("ANOVA 2x2 requires exactly two levels per factor.")

    cell_counts = df.groupby(["factor_a", "factor_b"]).size()

    if cell_counts.size != 4:
        raise ValueError("ANOVA 2x2 requires all four factor cells.")

    if len(df) <= 4:
        raise ValueError(
            "ANOVA 2x2 needs replicated observations, such as folds or "
            "bootstrap samples, not only one value per cell."
        )

    model = ols(
        "metric_value ~ C(factor_a) * C(factor_b)",
        data=df,
    ).fit()
    anova_table = anova_lm(model, typ=2)

    term_names = {
        factor1: "C(factor_a)",
        factor2: "C(factor_b)",
        "interaction": "C(factor_a):C(factor_b)",
        "residual": "Residual",
    }
    results = {}

    for output_name, table_name in term_names.items():
        if table_name not in anova_table.index:
            continue

        row = anova_table.loc[table_name]
        results[output_name] = {
            "sum_sq": float(row.get("sum_sq", float("nan"))),
            "df": float(row.get("df", float("nan"))),
            "F": float(row.get("F", float("nan"))),
            "p_value": float(row.get("PR(>F)", float("nan"))),
        }

    return results

# Compare paired binary violation outcomes between two variants.
def mcnemar_test(
    violations_a,
    violations_b,
) -> dict[str, float]:
    if isinstance(violations_a, torch.Tensor):
        violations_a = violations_a.detach().cpu().numpy()

    if isinstance(violations_b, torch.Tensor):
        violations_b = violations_b.detach().cpu().numpy()

    violations_a = np.asarray(violations_a, dtype=bool).reshape(-1)
    violations_b = np.asarray(violations_b, dtype=bool).reshape(-1)

    if violations_a.shape != violations_b.shape:
        raise ValueError("Violation arrays must have the same shape.")

    if violations_a.size == 0:
        raise ValueError("Cannot run McNemar test on empty arrays.")

    a_only = int(np.sum(violations_a & ~violations_b))
    b_only = int(np.sum(~violations_a & violations_b))
    both_violate = int(np.sum(violations_a & violations_b))
    neither_violate = int(np.sum(~violations_a & ~violations_b))
    discordant = a_only + b_only

    if discordant == 0:
        statistic = 0.0
        p_value = 1.0
    else:
        statistic = (max(abs(a_only - b_only) - 1, 0) ** 2) / discordant
        p_value = float(
            binomtest(
                min(a_only, b_only),
                n=discordant,
                p=0.5,
            ).pvalue
        )

    return {
        "a_only": float(a_only),
        "b_only": float(b_only),
        "both_violate": float(both_violate),
        "neither_violate": float(neither_violate),
        "discordant": float(discordant),
        "statistic": float(statistic),
        "p_value": p_value,
    }

# Compute the fraction of physically invalid negative predictions.
def compute_negative_violation_rate(
    q_hat: torch.Tensor,
) -> float:
    if q_hat.numel() == 0:
        raise ValueError("Cannot evaluate an empty prediction tensor.")

    negative_predictions = q_hat < 0

    return float(
        negative_predictions.float().mean().item()
    )
# Compute the fraction of predictions exceeding maximum capacity.
def compute_saturation_violation_rate(
    q_hat: torch.Tensor,
    q_max: torch.Tensor,
) -> float:
    if q_hat.numel() == 0:
        raise ValueError("Cannot evaluate empty prediction tensors.")

    if q_hat.numel() != q_max.numel():
        raise ValueError(
            "q_hat and q_max must contain the same number of values."
        )

    saturation_violations = (
        q_hat.reshape(-1) > q_max.reshape(-1)
    )

    return float(
        saturation_violations.float().mean().item()
    )

# Compute how often predictions decrease as pressure increases.
def compute_monotonicity_violation_rate(
    predictions: dict[str, torch.Tensor],
    tolerance: float = 1e-8,
) -> float:
    q_hat = predictions["q_hat"].detach().cpu().reshape(-1)
    raw_pressure = predictions["raw_pressure"].detach().cpu().reshape(-1)
    experiment_id = predictions["experiment_id"].detach().cpu().reshape(-1)

    if q_hat.numel() == 0:
        raise ValueError("Cannot evaluate an empty prediction tensor.")

    if not (
        q_hat.numel() == raw_pressure.numel() == experiment_id.numel()
    ):
        raise ValueError(
            "q_hat, raw_pressure, and experiment_id must have the same length."
        )

    violation_count = 0
    comparison_count = 0

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

        violations = pressure_increases & prediction_decreases
        violation_count += int(violations.sum().item())
        comparison_count += int(pressure_increases.sum().item())

    if comparison_count == 0:
        return float("nan")

    return violation_count / comparison_count

# Compute available physical violation rates from collected predictions.
def compute_physical_violation_metrics(
    predictions: dict[str, torch.Tensor],
) -> dict[str, float]:
    if "q_hat" not in predictions:
        raise KeyError("Predictions must contain q_hat.")

    if "raw_pressure" not in predictions:
        raise KeyError("Predictions must contain raw_pressure.")

    if "experiment_id" not in predictions:
        raise KeyError("Predictions must contain experiment_id.")

    metrics = {
        "neg_rate": compute_negative_violation_rate(
            predictions["q_hat"]
        ),
        "mono_rate": compute_monotonicity_violation_rate(
            predictions
        ),
    }

    if "q_max" in predictions:
        metrics["sat_rate"] = compute_saturation_violation_rate(
            predictions["q_hat"],
            predictions["q_max"],
        )
    else:
        metrics["sat_rate"] = float("nan")

    return metrics

# Estimate physical violation confidence intervals by resampling experiments.
def bootstrap_physical_violation_metrics(
    predictions: dict[str, torch.Tensor],
    n_bootstrap: int = 100,
    alpha: float = 0.05,
    random_state: int = 42,
) -> dict[str, dict[str, float]]:
    experiment_ids = predictions["experiment_id"].numpy()
    unique_experiments = np.unique(experiment_ids)

    if len(unique_experiments) < 2:
        raise ValueError("Bootstrap requires at least two experiments.")

    rng = np.random.default_rng(random_state)
    point_metrics = compute_physical_violation_metrics(predictions)

    bootstrap_values = {
        metric_name: [] for metric_name in point_metrics
    }

    for _ in range(n_bootstrap):
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

        sampled_predictions = {
            "q_hat": predictions["q_hat"][indices],
            "raw_pressure": predictions["raw_pressure"][indices],
            "experiment_id": predictions["experiment_id"][indices],
        }

        if "q_max" in predictions:
            sampled_predictions["q_max"] = predictions["q_max"][indices]

        sample_metrics = compute_physical_violation_metrics(
            sampled_predictions
        )

        for metric_name, value in sample_metrics.items():
            if np.isfinite(value):
                bootstrap_values[metric_name].append(value)

    results = {}

    for metric_name, point_value in point_metrics.items():
        values = np.asarray(bootstrap_values[metric_name])

        if values.size == 0:
            ci_lower = float("nan")
            ci_upper = float("nan")
        else:
            ci_lower = float(np.quantile(values, alpha / 2))
            ci_upper = float(np.quantile(values, 1 - alpha / 2))

        results[metric_name] = {
            "value": point_value,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
        }

    return results

if __name__ == "__main__":
   main()
