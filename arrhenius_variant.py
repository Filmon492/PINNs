from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F

from data_pipeline import build_data_pipeline
from soft_baseline import R_GAS, ResidualBlock, V1ModelConfig


@dataclass
class ArrheniusConfig(V1ModelConfig):
    temperature_center: float = 0.0
    temperature_scale: float = 1.0
    min_arrhenius_exponent: float = -50.0
    max_arrhenius_exponent: float = 50.0


# Arrhenius parameter head with K0 > 0 and exothermic delta_H < 0.
class ArrheniusHead(nn.Module):
    def __init__(self, input_dim: int, config: ArrheniusConfig):
        super().__init__()
        self.config = config
        self.head_K0 = nn.Linear(input_dim, 1)
        self.head_delta_H = nn.Linear(input_dim, 1)

    def forward(
        self,
        h: torch.Tensor,
        raw_temperature_K: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if raw_temperature_K.ndim == 1:
            raw_temperature_K = raw_temperature_K.reshape(-1, 1)

        if raw_temperature_K.shape[1] != 1:
            raise ValueError("raw_temperature_K must have shape (batch_size, 1).")

        if not (raw_temperature_K > 0).all():
            raise ValueError("Arrhenius relation requires positive temperature.")

        eps = self.config.softplus_epsilon
        K0 = F.softplus(self.head_K0(h)) + eps
        delta_H = -self.config.enthalpy_scale_j_per_mol
        delta_H = delta_H * (F.softplus(self.head_delta_H(h)) + eps)
        exponent = -delta_H / (R_GAS * raw_temperature_K)
        exponent = torch.clamp(
            exponent,
            min=self.config.min_arrhenius_exponent,
            max=self.config.max_arrhenius_exponent,
        )
        K = K0 * torch.exp(exponent)

        return {
            "K": K,
            "K0": K0,
            "delta_H": delta_H,
            "delta_H_kJmol": delta_H / 1000.0,
        }


# V6 hard-van't-Hoff PINN: K is computed from an Arrhenius form.
class ArrheniusPINN(nn.Module):
    def __init__(self, config: Optional[ArrheniusConfig] = None):
        super().__init__()
        self.config = config or ArrheniusConfig()

        blocks = []
        current_dim = self.config.input_dim

        for hidden_dim in self.config.hidden_dims:
            blocks.append(
                ResidualBlock(
                    input_dim=current_dim,
                    output_dim=hidden_dim,
                    dropout=self.config.dropout,
                )
            )
            current_dim = hidden_dim

        self.encoder = nn.Sequential(*blocks)
        self.mean_head = nn.Linear(current_dim, 1)
        self.logvar_head = nn.Linear(current_dim, 1)
        self.physics_head = nn.Linear(current_dim, 2)
        self.arrhenius_head = ArrheniusHead(current_dim, self.config)

    def raw_temperature_from_scaled_x(self, x: torch.Tensor) -> torch.Tensor:
        scaled_temperature = x[
            :,
            self.config.temperature_feature_idx : self.config.temperature_feature_idx
            + 1,
        ]

        return (
            scaled_temperature * self.config.temperature_scale
            + self.config.temperature_center
        )

    def forward(
        self,
        x: torch.Tensor,
        raw_pressure: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if x.shape[1] != self.config.input_dim:
            raise ValueError(f"Expected {self.config.input_dim} input features.")

        if not (raw_pressure > 0).all():
            raise ValueError("raw_pressure must be positive.")

        h = self.encoder(x)
        eps = self.config.softplus_epsilon
        q_hat = F.softplus(self.mean_head(h)) + eps
        log_sigma2 = torch.clamp(
            self.logvar_head(h),
            min=self.config.min_log_sigma2,
            max=self.config.max_log_sigma2,
        )
        sigma_ale = torch.exp(0.5 * log_sigma2)

        physics_params = F.softplus(self.physics_head(h)) + eps
        q_max = physics_params[:, 0:1]
        n = physics_params[:, 1:2]
        raw_temperature_K = self.raw_temperature_from_scaled_x(x)
        arrhenius_output = self.arrhenius_head(h, raw_temperature_K)

        return {
            "q_hat": q_hat,
            "log_sigma2": log_sigma2,
            "sigma_ale": sigma_ale,
            "q_max": q_max,
            "K": arrhenius_output["K"],
            "K0": arrhenius_output["K0"],
            "n": n,
            "delta_H": arrhenius_output["delta_H"],
            "delta_H_kJmol": arrhenius_output["delta_H_kJmol"],
        }


# Quick architecture check using one batch from the data pipeline.
def smoke_test_arrhenius_architecture() -> None:
    pipeline = build_data_pipeline(batch_size=8)
    batch = next(iter(pipeline["train_loader"]))
    base_config = ArrheniusConfig()
    config = ArrheniusConfig(
        temperature_center=float(
            pipeline["scaler"].center_[base_config.temperature_feature_idx]
        ),
        temperature_scale=float(
            pipeline["scaler"].scale_[base_config.temperature_feature_idx]
        ),
    )
    model = ArrheniusPINN(config)
    output = model(batch["x"], batch["raw_pressure"])
    raw_temperature_K = model.raw_temperature_from_scaled_x(batch["x"])

    print("x:", batch["x"].shape)
    print("raw_temperature_K:", raw_temperature_K.shape)
    print("q_hat:", output["q_hat"].shape)
    print("q_max:", output["q_max"].shape)
    print("K:", output["K"].shape)
    print("K0:", output["K0"].shape)
    print("n:", output["n"].shape)
    print("delta_H_kJmol:", output["delta_H_kJmol"].shape)
    print(
        "temperature range:",
        float(raw_temperature_K.min()),
        float(raw_temperature_K.max()),
    )
    print("max delta_H_kJmol:", float(output["delta_H_kJmol"].max().detach()))


if __name__ == "__main__":
    smoke_test_arrhenius_architecture()
