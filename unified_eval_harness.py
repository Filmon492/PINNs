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
def validate_model_output(
    output: ModelOutput,
    batch_size: int,
) -> None:
    if "q_hat" not in output:
        raise KeyError("Model output must contain 'q_hat'.")

    if not isinstance(output["q_hat"], torch.Tensor):
        raise TypeError("q_hat must be a PyTorch tensor.")

    if output["q_hat"].numel() != batch_size:
        raise ValueError(
            "q_hat must contain one prediction per sample."
        )

if __name__ == "__main__":
    main()