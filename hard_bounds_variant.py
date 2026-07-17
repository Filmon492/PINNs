from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F

from data_pipeline import build_data_pipeline
from soft_baseline import ResidualBlock, V1ModelConfig


# Output head that guarantees 0 < q_hat < q_max by construction.
class HardBoundsHead(nn.Module):
    def __init__(self, input_dim: int, softplus_epsilon: float = 1e-6):
        super().__init__()
        self.head_qmax = nn.Linear(input_dim, 1)
        self.head_q = nn.Linear(input_dim, 1)
        self.softplus_epsilon = softplus_epsilon

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        q_max = F.softplus(self.head_qmax(h)) + self.softplus_epsilon
        z_q = self.head_q(h)
        q_hat = q_max * torch.sigmoid(z_q)

        return q_hat, q_max


# V3 hard-bounds PINN: same encoder as V1, but hard-bounded adsorption output.
class HardBoundsPINN(nn.Module):
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
        self.bounds_head = HardBoundsHead(
            input_dim=current_dim,
            softplus_epsilon=self.config.softplus_epsilon,
        )
        self.logvar_head = nn.Linear(current_dim, 1)
        self.physics_head = nn.Linear(current_dim, 3)

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
        q_hat, q_max = self.bounds_head(h)
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


# Verify that a model output satisfies the hard physical bounds.
def verify_bounds_output(
    output: dict[str, torch.Tensor],
    context: str = "model output",
) -> None:
    q_hat = output["q_hat"]
    q_max = output["q_max"]

    lower_violations = q_hat <= 0
    upper_violations = q_hat >= q_max

    if lower_violations.any() or upper_violations.any():
        raise ValueError(
            f"Hard bounds failed for {context}: "
            f"lower violations={int(lower_violations.sum().item())}, "
            f"upper violations={int(upper_violations.sum().item())}, "
            f"min q_hat={float(q_hat.min().item())}, "
            f"max q_hat={float(q_hat.max().item())}, "
            f"min q_max={float(q_max.min().item())}."
        )


# Check hard bounds over batches from a DataLoader.
def verify_bounds(
    model: HardBoundsPINN,
    loader,
    device: str = "cpu",
    max_batches: Optional[int] = None,
) -> None:
    device = torch.device(device)
    model = model.to(device)
    model.eval()

    with torch.inference_mode():
        for batch_idx, batch in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            x = batch["x"].to(device)
            raw_pressure = batch["raw_pressure"].to(device)
            output = model(x, raw_pressure)
            verify_bounds_output(output, context=f"batch {batch_idx}")


# Stress-test hard bounds on random inputs and extreme raw pressures.
def verify_bounds_on_random_inputs(
    model: HardBoundsPINN,
    n_samples: int = 1000,
    device: str = "cpu",
) -> None:
    device = torch.device(device)
    model = model.to(device)
    model.eval()

    x = torch.randn(n_samples, model.config.input_dim, device=device)
    pressure_feature = torch.linspace(-20.0, 20.0, n_samples, device=device)
    x[:, model.config.pressure_feature_idx] = pressure_feature
    raw_pressure = torch.logspace(-6, 3, n_samples, device=device).reshape(-1, 1)

    with torch.inference_mode():
        output = model(x, raw_pressure)
        verify_bounds_output(output, context="1000 random stress inputs")


# Quick architecture and hard-bounds check.
def smoke_test_hard_bounds() -> None:
    pipeline = build_data_pipeline(batch_size=8)
    batch = next(iter(pipeline["train_loader"]))
    model = HardBoundsPINN()

    output = model(batch["x"], batch["raw_pressure"])
    verify_bounds_output(output, context="first training batch")
    verify_bounds(model, pipeline["train_loader"], max_batches=2)
    verify_bounds_on_random_inputs(model)

    print("x:", batch["x"].shape)
    print("q_hat:", output["q_hat"].shape)
    print("q_max:", output["q_max"].shape)
    print("Bounds verified for first batches and 1000 random stress inputs.")


if __name__ == "__main__":
    smoke_test_hard_bounds()
