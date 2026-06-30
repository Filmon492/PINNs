import numpy as np
import pandas as pd
from typing import TypedDict
import torch

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
if __name__ == "__main__":
    main()