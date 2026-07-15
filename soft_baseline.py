from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F

from data_pipeline import build_data_pipeline

R_GAS = 8.314


@dataclass
class V1ModelConfig:
    input_dim: int = 12
    hidden_dims: tuple[int, ...] = (512, 1024, 512, 256, 128)
    dropout: float = 0.1
    temperature_feature_idx: int = 3
    pressure_feature_idx: int = 4
    softplus_epsilon: float = 1e-6
    min_log_sigma2: float = -10.0
    max_log_sigma2: float = 6.0
    enthalpy_scale_j_per_mol: float = 10_000.0


# Residual block with LayerNorm and Swish/SiLU activation.
class ResidualBlock(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, dropout: float):
        super().__init__()
        self.main = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

        if input_dim == output_dim:
            self.skip = nn.Identity()
        else:
            self.skip = nn.Linear(input_dim, output_dim)

        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.activation(self.main(x) + self.skip(x)))


# V1 soft-constraint PINN baseline architecture.
class SoftBaselinePINN(nn.Module):
    def __init__(self, config: Optional[V1ModelConfig] = None):
        super().__init__()
        self.config = config or V1ModelConfig()

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
        self.physics_head = nn.Linear(current_dim, 4)

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


# Compute Gaussian data loss with learned aleatoric uncertainty.
def heteroscedastic_nll_loss(
    q_true: torch.Tensor,
    q_hat: torch.Tensor,
    log_sigma2: torch.Tensor,
) -> torch.Tensor:
    sigma2 = torch.exp(log_sigma2)
    squared_error = (q_true - q_hat) ** 2

    return torch.mean(squared_error / (2.0 * sigma2) + 0.5 * log_sigma2)


# Compute the Sips adsorption curve from predicted physics parameters.
def sips_isotherm(
    q_max: torch.Tensor,
    K: torch.Tensor,
    n: torch.Tensor,
    raw_pressure: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    KP = torch.clamp(K * raw_pressure, min=eps)
    KP_n = torch.exp(n * torch.log(KP))

    return q_max * KP_n / (1.0 + KP_n)


# Penalize disagreement between q_hat and the predicted Sips curve.
def sips_loss(
    output: dict[str, torch.Tensor],
    raw_pressure: torch.Tensor,
) -> torch.Tensor:
    q_sips = sips_isotherm(
        q_max=output["q_max"],
        K=output["K"],
        n=output["n"],
        raw_pressure=raw_pressure,
    )

    return F.mse_loss(output["q_hat"], q_sips)


# Penalize negative adsorption and adsorption above q_max.
def bounds_loss(output: dict[str, torch.Tensor]) -> torch.Tensor:
    q_hat = output["q_hat"]
    q_max = output["q_max"]

    lower_violation = F.relu(-q_hat)
    upper_violation = F.relu(q_hat - q_max)

    return torch.mean(lower_violation + upper_violation)


# Penalize negative adsorption-pressure slopes.
def monotonicity_loss(
    output: dict[str, torch.Tensor],
    x: torch.Tensor,
    pressure_feature_idx: int,
) -> torch.Tensor:
    if not x.requires_grad:
        raise ValueError("x must require gradients for monotonicity_loss.")

    gradient = torch.autograd.grad(
        output["q_hat"].sum(),
        x,
        create_graph=True,
        retain_graph=True,
    )[0]
    pressure_gradient = gradient[:, pressure_feature_idx : pressure_feature_idx + 1]

    return F.relu(-pressure_gradient).mean()


# Penalize deviations from the van't Hoff temperature relation.
def vant_hoff_loss(
    output: dict[str, torch.Tensor],
    x: torch.Tensor,
    temperature_feature_idx: int,
    temperature_center: float,
    temperature_scale: float,
) -> torch.Tensor:
    if not x.requires_grad:
        raise ValueError("x must require gradients for vant_hoff_loss.")

    if temperature_scale == 0:
        raise ValueError("temperature_scale must be nonzero.")

    ln_K = torch.log(torch.clamp(output["K"], min=1e-8))
    gradient = torch.autograd.grad(
        ln_K.sum(),
        x,
        create_graph=True,
        retain_graph=True,
    )[0]
    scaled_temperature = x[:, temperature_feature_idx : temperature_feature_idx + 1]
    raw_temperature = scaled_temperature * temperature_scale + temperature_center
    d_lnK_d_scaled_T = gradient[
        :,
        temperature_feature_idx : temperature_feature_idx + 1,
    ]
    d_lnK_d_T = d_lnK_d_scaled_T / temperature_scale
    d_lnK_d_invT = d_lnK_d_T * (-(raw_temperature**2))
    target = -output["delta_H"] / R_GAS

    return F.mse_loss(d_lnK_d_invT, target)


# Combine data loss and all soft physics penalties.
def compute_v1_losses(
    output: dict[str, torch.Tensor],
    y_log: torch.Tensor,
    raw_pressure: torch.Tensor,
    lambda_p: float,
    x: torch.Tensor,
    config: V1ModelConfig,
    temperature_center: float,
    temperature_scale: float,
) -> dict[str, torch.Tensor]:
    q_true = torch.exp(y_log)
    data_loss = heteroscedastic_nll_loss(
        q_true=q_true,
        q_hat=output["q_hat"],
        log_sigma2=output["log_sigma2"],
    )
    sips_penalty = sips_loss(output, raw_pressure)
    bounds_penalty = bounds_loss(output)
    mono_penalty = monotonicity_loss(
        output,
        x,
        pressure_feature_idx=config.pressure_feature_idx,
    )
    vH_penalty = vant_hoff_loss(
        output,
        x,
        temperature_feature_idx=config.temperature_feature_idx,
        temperature_center=temperature_center,
        temperature_scale=temperature_scale,
    )
    physics_loss = sips_penalty + bounds_penalty + mono_penalty + vH_penalty
    total_loss = data_loss + lambda_p * physics_loss

    return {
        "total_loss": total_loss,
        "data_loss": data_loss,
        "physics_loss": physics_loss,
        "sips_loss": sips_penalty,
        "bounds_loss": bounds_penalty,
        "mono_loss": mono_penalty,
        "vH_loss": vH_penalty,
    }


# Quick architecture check using one batch from the data pipeline.
def smoke_test_model() -> None:
    pipeline = build_data_pipeline(batch_size=8)
    batch = next(iter(pipeline["train_loader"]))
    model = SoftBaselinePINN()

    x = batch["x"].detach().clone().requires_grad_(True)
    output = model(x, batch["raw_pressure"])
    losses = compute_v1_losses(
        output,
        y_log=batch["y"],
        raw_pressure=batch["raw_pressure"],
        lambda_p=0.1,
        x=x,
        config=model.config,
        temperature_center=float(
            pipeline["scaler"].center_[model.config.temperature_feature_idx]
        ),
        temperature_scale=float(
            pipeline["scaler"].scale_[model.config.temperature_feature_idx]
        ),
    )

    print("x:", batch["x"].shape)
    print("q_hat:", output["q_hat"].shape)
    print("log_sigma2:", output["log_sigma2"].shape)
    print("q_max:", output["q_max"].shape)
    print("K:", output["K"].shape)
    print("n:", output["n"].shape)
    print("delta_H:", output["delta_H"].shape)
    print("total_loss:", float(losses["total_loss"].detach()))
    print("data_loss:", float(losses["data_loss"].detach()))
    print("physics_loss:", float(losses["physics_loss"].detach()))
    print("mono_loss:", float(losses["mono_loss"].detach()))
    print("vH_loss:", float(losses["vH_loss"].detach()))


if __name__ == "__main__":
    smoke_test_model()
