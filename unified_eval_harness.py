from typing import TypedDict

import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from data_pipeline import build_data_pipeline


def main():
    data = build_data_pipeline()
    print("T1 samples:", len(data["T1_loader"].dataset))
    print("T3 samples:", len(data["T3_loader"].dataset))

class ModelOutput(TypedDict, total=False):
    q_hat: torch.Tensor
    q_max: torch.Tensor

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

# Evaluate accuracy and uncertainty for one data tier.
def evaluate_accuracy(
    model,
    loader,
    device: str = "cpu",
    n_bootstrap: int = 100,
    alpha: float = 0.05,
    random_state: int = 42,
) -> dict:
    collected = collect_predictions(
        model,
        loader,
        device=device,
    )

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

    return {
        "metrics": metrics,
        "confidence_intervals": confidence_intervals,
        "predictions": collected,
    }

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

# Compute available physical violation rates from collected predictions.
def compute_physical_violation_metrics(
    predictions: dict[str, torch.Tensor],
) -> dict[str, float]:
    if "q_hat" not in predictions:
        raise KeyError("Predictions must contain q_hat.")

    metrics = {
        "neg_rate": compute_negative_violation_rate(
            predictions["q_hat"]
        )
    }

    if "q_max" in predictions:
        metrics["sat_rate"] = compute_saturation_violation_rate(
            predictions["q_hat"],
            predictions["q_max"],
        )
    else:
        metrics["sat_rate"] = float("nan")

    return metrics

if __name__ == "__main__":
   main()
