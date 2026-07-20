from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F

from data_pipeline import build_data_pipeline
from hard_bounds_variant import verify_bounds_output
from hard_monotonicity_variant import MonotonePressurePathway, PositiveLinear
from soft_baseline import ResidualBlock, V1ModelConfig


@dataclass
class JointVariantConfig(V1ModelConfig):
    pressure_dependent_feature_indices: tuple[int, ...] = (4, 6, 10)


# Output head that guarantees both 0 < q_hat < q_max and dq_hat/dP >= 0.
class HardBoundsMonotoneHead(nn.Module):
    def __init__(
        self,
        comp_dim: int,
        pressure_dim: int,
        softplus_epsilon: float = 1e-6,
    ):
        super().__init__()
        self.head_qmax = nn.Linear(comp_dim, 1)
        self.head_q_comp = nn.Linear(comp_dim, 1)
        self.head_q_pressure = PositiveLinear(pressure_dim, 1, bias=False)
        self.softplus_epsilon = softplus_epsilon

    def forward(
        self,
        h_comp: torch.Tensor,
        h_pressure: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q_max = F.softplus(self.head_qmax(h_comp)) + self.softplus_epsilon
        z_q = self.head_q_comp(h_comp) + self.head_q_pressure(h_pressure)
        q_hat = q_max * torch.sigmoid(z_q)

        return q_hat, q_max


# V7 joint variant: hard bounds plus hard pressure monotonicity.
class HardBoundsMonotonicityPINN(nn.Module):
    def __init__(self, config: Optional[JointVariantConfig] = None):
        super().__init__()
        self.config = config or JointVariantConfig()
        excluded_indices = set(self.config.pressure_dependent_feature_indices)
        self.comp_feature_indices = tuple(
            idx for idx in range(self.config.input_dim) if idx not in excluded_indices
        )

        blocks = []
        current_dim = len(self.comp_feature_indices)

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
        pressure_dim = self.pressure_pathway.output_dim
        combined_dim = current_dim + pressure_dim

        self.bounds_mono_head = HardBoundsMonotoneHead(
            comp_dim=current_dim,
            pressure_dim=pressure_dim,
            softplus_epsilon=self.config.softplus_epsilon,
        )
        self.logvar_head = nn.Linear(combined_dim, 1)
        self.physics_head = nn.Linear(combined_dim, 3)

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

        q_hat, q_max = self.bounds_mono_head(h_comp, h_pressure)
        log_sigma2 = torch.clamp(
            self.logvar_head(h),
            min=self.config.min_log_sigma2,
            max=self.config.max_log_sigma2,
        )
        sigma_ale = torch.exp(0.5 * log_sigma2)

        eps = self.config.softplus_epsilon
        physics_params = F.softplus(self.physics_head(h)) + eps
        K = physics_params[:, 0:1]
        n = physics_params[:, 1:2]
        delta_H = -self.config.enthalpy_scale_j_per_mol * physics_params[:, 2:3]

        return {
            "q_hat": q_hat,
            "log_sigma2": log_sigma2,
            "sigma_ale": sigma_ale,
            "q_max": q_max,
            "K": K,
            "n": n,
            "delta_H": delta_H,
        }


# Quick V7 architecture and structural guarantee check.
def smoke_test_joint_variants() -> None:
    pipeline = build_data_pipeline(batch_size=8)
    batch = next(iter(pipeline["train_loader"]))
    model = HardBoundsMonotonicityPINN()
    raw_pressure = batch["raw_pressure"].detach().clone().requires_grad_(True)
    output = model(batch["x"], raw_pressure)
    dq_dP = torch.autograd.grad(
        output["q_hat"].sum(),
        raw_pressure,
        create_graph=False,
        retain_graph=False,
    )[0]
    verify_bounds_output(output, context="V7 smoke test batch")

    print("x:", batch["x"].shape)
    print("composition features:", len(model.comp_feature_indices))
    print("q_hat:", output["q_hat"].shape)
    print("q_max:", output["q_max"].shape)
    print("min q_hat:", float(output["q_hat"].min().detach()))
    print(
        "min qmax_minus_q:", float((output["q_max"] - output["q_hat"]).min().detach())
    )
    print("min dq_dP:", float(dq_dP.min().detach()))


if __name__ == "__main__":
    smoke_test_joint_variants()
