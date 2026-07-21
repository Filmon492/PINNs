from pathlib import Path
from typing import Optional

import numpy as np
import torch
from scipy.stats import norm, spearmanr

from ablation_analysis import TIER_NAMES, VARIANT_SPECS, load_variant_model
from data_pipeline import build_data_pipeline
from unified_eval_harness import collect_mc_dropout_predictions, collect_predictions

CONFIDENCE_LEVELS = (0.68, 0.90, 0.95, 0.99)
DEFAULT_TAU_GRID = np.geomspace(0.1, 10.0, 81)


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


if __name__ == "__main__":
    smoke_test_uq_calibration()
