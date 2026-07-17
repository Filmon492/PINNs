from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F

from soft_baseline import ResidualBlock, V1ModelConfig


# Linear layer whose effective weights are always non-negative.
class PositiveLinear(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        bias: bool = True,
    ):
        super().__init__()
        self.weight_raw = nn.Parameter(torch.empty(output_dim, input_dim))

        if bias:
            self.bias = nn.Parameter(torch.zeros(output_dim))
        else:
            self.register_parameter("bias", None)

        nn.init.normal_(self.weight_raw, mean=-3.0, std=0.05)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = F.softplus(self.weight_raw)

        return F.linear(x, weight, self.bias)


# Pressure-only pathway with non-negative weights and monotone activations.
class MonotonePressurePathway(nn.Module):
    def __init__(self, hidden_dims: tuple[int, ...] = (64, 32, 16)):
        super().__init__()
        if not hidden_dims:
            raise ValueError("hidden_dims must contain at least one layer.")

        dims = (1, *hidden_dims)
        self.layers = nn.ModuleList(
            PositiveLinear(dims[idx], dims[idx + 1]) for idx in range(len(dims) - 1)
        )
        self.output_dim = hidden_dims[-1]

    def forward(self, raw_pressure: torch.Tensor) -> torch.Tensor:
        if raw_pressure.ndim == 1:
            raw_pressure = raw_pressure.reshape(-1, 1)

        if raw_pressure.shape[1] != 1:
            raise ValueError("raw_pressure must have shape (batch_size, 1).")

        if not (raw_pressure > 0).all():
            raise ValueError("raw_pressure must be positive.")

        h = raw_pressure
        for layer in self.layers:
            h = F.softplus(layer(h))

        return h


# Output head where the pressure contribution cannot decrease q_hat.
class MonotoneAdsorptionHead(nn.Module):
    def __init__(
        self,
        comp_dim: int,
        pressure_dim: int,
        softplus_epsilon: float = 1e-6,
    ):
        super().__init__()
        self.comp_head = nn.Linear(comp_dim, 1)
        self.pressure_head = PositiveLinear(pressure_dim, 1, bias=False)
        self.softplus_epsilon = softplus_epsilon

    def forward(
        self,
        h_comp: torch.Tensor,
        h_pressure: torch.Tensor,
    ) -> torch.Tensor:
        z_q = self.comp_head(h_comp) + self.pressure_head(h_pressure)

        return F.softplus(z_q) + self.softplus_epsilon


# V4 hard-monotonicity PINN with pressure entering only the monotone pathway.
class HardMonotonicityPINN(nn.Module):
    def __init__(self, config: Optional[V1ModelConfig] = None):
        super().__init__()
        self.config = config or V1ModelConfig()
        self.comp_feature_indices = tuple(
            idx
            for idx in range(self.config.input_dim)
            if idx != self.config.pressure_feature_idx
        )

        comp_input_dim = len(self.comp_feature_indices)
        blocks = []
        current_dim = comp_input_dim

        for hidden_dim in self.config.hidden_dims:
            blocks.append(
                ResidualBlock(
                    input_dim=current_dim,
                    output_dim=hidden_dim,
                    dropout=self.config.dropout,
                )
            )
            current_dim = hidden_dim

        self.comp_encoder = nn.Sequential(*blocks)
        self.pressure_pathway = MonotonePressurePathway()
        combined_dim = current_dim + self.pressure_pathway.output_dim

        self.mean_head = MonotoneAdsorptionHead(
            comp_dim=current_dim,
            pressure_dim=self.pressure_pathway.output_dim,
            softplus_epsilon=self.config.softplus_epsilon,
        )
        self.logvar_head = nn.Linear(combined_dim, 1)
        self.physics_head = nn.Linear(combined_dim, 4)

    def forward(
        self,
        x: torch.Tensor,
        raw_pressure: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if x.shape[1] != self.config.input_dim:
            raise ValueError(f"Expected {self.config.input_dim} input features.")

        if not (raw_pressure > 0).all():
            raise ValueError("raw_pressure must be positive.")

        x_comp = x[:, self.comp_feature_indices]
        h_comp = self.comp_encoder(x_comp)
        h_pressure = self.pressure_pathway(raw_pressure)
        h = torch.cat([h_comp, h_pressure], dim=1)

        q_hat = self.mean_head(h_comp, h_pressure)
        log_sigma2 = torch.clamp(
            self.logvar_head(h),
            min=self.config.min_log_sigma2,
            max=self.config.max_log_sigma2,
        )
        sigma_ale = torch.exp(0.5 * log_sigma2)

        eps = self.config.softplus_epsilon
        physics_params = F.softplus(self.physics_head(h)) + eps
        q_max = physics_params[:, 0:1]
        K = physics_params[:, 1:2]
        n = physics_params[:, 2:3]
        delta_H = -self.config.enthalpy_scale_j_per_mol * physics_params[:, 3:4]

        return {
            "q_hat": q_hat,
            "log_sigma2": log_sigma2,
            "sigma_ale": sigma_ale,
            "q_max": q_max,
            "K": K,
            "n": n,
            "delta_H": delta_H,
        }


# Quick architecture check on one batch from the data pipeline.
def smoke_test_hard_monotonicity_architecture() -> None:
    from data_pipeline import build_data_pipeline

    pipeline = build_data_pipeline(batch_size=8)
    batch = next(iter(pipeline["train_loader"]))
    model = HardMonotonicityPINN()
    output = model(batch["x"], batch["raw_pressure"])

    print("x:", batch["x"].shape)
    print("raw_pressure:", batch["raw_pressure"].shape)
    print("q_hat:", output["q_hat"].shape)
    print("q_max:", output["q_max"].shape)
    print("K:", output["K"].shape)
    print("n:", output["n"].shape)
    print("delta_H:", output["delta_H"].shape)
    print("composition features:", len(model.comp_feature_indices))


if __name__ == "__main__":
    smoke_test_hard_monotonicity_architecture()
