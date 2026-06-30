import numpy as np
import pandas as pd
from typing import TypedDict
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from data_pipeline import build_data_pipeline


def main():
    data = build_data_pipeline()
    print("T1 samples:", len(data["T1_loader"].dataset))
    print("T3 samples:", len(data["T3_loader"].dataset))

class ModelOutput(TypedDict):
    q_hat: torch.Tensor

# validate model output while running
def validate_model_output(output: ModelOutput,batch_size: int,) -> None:
    if "q_hat" not in output:
        raise KeyError("Model output must contain 'q_hat'.")

    if not isinstance(output["q_hat"], torch.Tensor):
        raise TypeError("q_hat must be a PyTorch tensor.")

    if output["q_hat"].numel() != batch_size:
        raise ValueError("q_hat must contain one prediction per sample.")

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

    if not predictions:
        raise ValueError("Cannot collect predictions from an empty DataLoader.")

    return {
        "q_hat": torch.cat(predictions),
        "q_true": torch.cat(observations),
        "raw_pressure": torch.cat(raw_pressures),
        "experiment_id": torch.cat(experiment_ids),
    }

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

    return {
        "R2": float(r2_score(true_values, predictions)),
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

if __name__ == "__main__":
   main()