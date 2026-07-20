from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F

from arrhenius_variant import ArrheniusConfig, ArrheniusHead, verify_arrhenius_structure
from data_pipeline import build_data_pipeline
from sips_gauge_variant import verify_sips_structure
from soft_baseline import ResidualBlock


@dataclass
class AllHardConfig(ArrheniusConfig):
    pressure_dependent_feature_indices: tuple[int, ...] = (4, 6, 10)


# Encoder that removes pressure-derived features before the neural network.
class CompositionEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        comp_feature_indices: tuple[int, ...],
        hidden_dims: tuple[int, ...],
        dropout: float,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.comp_feature_indices = comp_feature_indices

        blocks = []
        current_dim = len(comp_feature_indices)

        for hidden_dim in hidden_dims:
            blocks.append(
                ResidualBlock(
                    input_dim=current_dim,
                    output_dim=hidden_dim,
                    dropout=dropout,
                )
            )
            current_dim = hidden_dim

        self.network = nn.Sequential(*blocks)
        self.output_dim = current_dim

    def split_composition(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] == len(self.comp_feature_indices):
            return x

        if x.shape[1] != self.input_dim:
            raise ValueError(
                f"Expected either {self.input_dim} full features or "
                f"{len(self.comp_feature_indices)} composition features."
            )

        return x[:, self.comp_feature_indices]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_comp = self.split_composition(x)

        return self.network(x_comp)


# Shared V2/V8 model: Sips gauge output with Arrhenius K(T).
class AllHardSipsArrheniusPINN(nn.Module):
    def __init__(self, config: Optional[AllHardConfig] = None):
        super().__init__()
        self.config = config or AllHardConfig()
        excluded_indices = set(self.config.pressure_dependent_feature_indices)
        self.comp_feature_indices = tuple(
            idx for idx in range(self.config.input_dim) if idx not in excluded_indices
        )

        if self.config.temperature_feature_idx not in self.comp_feature_indices:
            raise ValueError(
                "Temperature must be available in the composition encoder."
            )

        self.encoder = CompositionEncoder(
            input_dim=self.config.input_dim,
            comp_feature_indices=self.comp_feature_indices,
            hidden_dims=self.config.hidden_dims,
            dropout=self.config.dropout,
        )
        hidden_dim = self.encoder.output_dim

        self.head_qmax = nn.Linear(hidden_dim, 1)
        self.head_n = nn.Linear(hidden_dim, 1)
        self.head_logvar = nn.Linear(hidden_dim, 1)
        self.arrhenius_head = ArrheniusHead(hidden_dim, self.config)

    def split_composition(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder.split_composition(x)

    def raw_temperature_from_scaled_x(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] == self.config.input_dim:
            temperature_idx = self.config.temperature_feature_idx
            scaled_temperature = x[
                :,
                temperature_idx : temperature_idx + 1,
            ]
        elif x.shape[1] == len(self.comp_feature_indices):
            temperature_comp_idx = self.comp_feature_indices.index(
                self.config.temperature_feature_idx
            )
            scaled_temperature = x[:, temperature_comp_idx : temperature_comp_idx + 1]
        else:
            raise ValueError("x must contain either full or composition features.")

        return (
            scaled_temperature * self.config.temperature_scale
            + self.config.temperature_center
        )

    def predict_parameters_from_comp(
        self,
        x_comp: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        eps = self.config.softplus_epsilon
        h = self.encoder(x_comp)
        q_max = F.softplus(self.head_qmax(h)) + eps
        n = F.softplus(self.head_n(h)) + eps
        log_sigma2 = torch.clamp(
            self.head_logvar(h),
            min=self.config.min_log_sigma2,
            max=self.config.max_log_sigma2,
        )
        sigma_ale = torch.exp(0.5 * log_sigma2)
        raw_temperature_K = self.raw_temperature_from_scaled_x(x_comp)
        arrhenius_output = self.arrhenius_head(h, raw_temperature_K)

        return {
            "q_max": q_max,
            "K": arrhenius_output["K"],
            "K0": arrhenius_output["K0"],
            "n": n,
            "log_sigma2": log_sigma2,
            "sigma_ale": sigma_ale,
            "delta_H": arrhenius_output["delta_H"],
            "delta_H_kJmol": arrhenius_output["delta_H_kJmol"],
        }

    def forward_from_composition(
        self,
        x_comp: torch.Tensor,
        raw_pressure: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if raw_pressure.ndim == 1:
            raw_pressure = raw_pressure.reshape(-1, 1)

        if raw_pressure.shape[1] != 1:
            raise ValueError("raw_pressure must have shape (batch_size, 1).")

        if not (raw_pressure > 0).all():
            raise ValueError("Sips requires P > 0.")

        output = self.predict_parameters_from_comp(x_comp)
        eps = self.config.softplus_epsilon
        log_KP_n = output["n"] * (
            torch.log(torch.clamp(output["K"], min=eps))
            + torch.log(torch.clamp(raw_pressure, min=eps))
        )
        log_KP_n = torch.clamp(
            log_KP_n,
            min=self.config.min_arrhenius_exponent,
            max=self.config.max_arrhenius_exponent,
        )
        saturation_fraction = torch.sigmoid(log_KP_n)
        saturation_fraction = torch.clamp(
            saturation_fraction,
            min=eps,
            max=1.0 - eps,
        )
        q_hat = output["q_max"] * saturation_fraction

        return {
            "q_hat": q_hat,
            "log_sigma2": output["log_sigma2"],
            "sigma_ale": output["sigma_ale"],
            "q_max": output["q_max"],
            "K": output["K"],
            "K0": output["K0"],
            "n": output["n"],
            "delta_H": output["delta_H"],
            "delta_H_kJmol": output["delta_H_kJmol"],
        }

    def forward(
        self,
        x: torch.Tensor,
        raw_pressure: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        x_comp = self.split_composition(x)

        return self.forward_from_composition(x_comp, raw_pressure)


V2AllHardPINN = AllHardSipsArrheniusPINN
V8HardSipsArrheniusPINN = AllHardSipsArrheniusPINN


# Quick architecture check for the shared V2/V8 hard constraints.
def smoke_test_all_hard_variant() -> None:
    pipeline = build_data_pipeline(batch_size=8)
    base_config = AllHardConfig()
    temperature_idx = base_config.temperature_feature_idx
    config = AllHardConfig(
        temperature_center=float(pipeline["scaler"].center_[temperature_idx]),
        temperature_scale=float(pipeline["scaler"].scale_[temperature_idx]),
    )
    model = AllHardSipsArrheniusPINN(config)
    batch = next(iter(pipeline["train_loader"]))
    output = model(batch["x"], batch["raw_pressure"])
    sips_verification = verify_sips_structure(
        model,
        pipeline["train_loader"],
        n_comp=16,
        n_P=8,
    )
    arrhenius_verification = verify_arrhenius_structure(
        model,
        pipeline["train_loader"],
        n_samples=16,
        required_fraction_in_range=0.0,
    )

    print("x:", batch["x"].shape)
    print("composition features:", len(model.comp_feature_indices))
    print("q_hat:", output["q_hat"].shape)
    print("q_max:", output["q_max"].shape)
    print("K:", output["K"].shape)
    print("K0:", output["K0"].shape)
    print("n:", output["n"].shape)
    print("delta_H:", output["delta_H"].shape)
    print("min q_hat:", float(output["q_hat"].min().detach()))
    print(
        "min qmax_minus_q:", float((output["q_max"] - output["q_hat"]).min().detach())
    )
    print("sips_min_dq_dP:", sips_verification["min_dq_dP"])
    print(
        "arrhenius_identity_error:",
        arrhenius_verification["max_arrhenius_identity_error"],
    )


if __name__ == "__main__":
    smoke_test_all_hard_variant()
