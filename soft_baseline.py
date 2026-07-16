from dataclasses import dataclass, replace
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F

from data_pipeline import build_data_pipeline

R_GAS = 8.314
DEFAULT_LAMBDA_P_VALUES = (0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0)


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


@dataclass
class TrainingConfig:
    lambda_p: float = 0.1
    lr: float = 1e-3
    weight_decay: float = 1e-5
    max_epochs: int = 1000
    patience: int = 100
    gradient_clip_norm: float = 5.0
    device: str = "cpu"


# Make repeated training runs comparable by using the same random start.
def set_torch_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


# Run one training pass over all batches in train_loader.
def train_one_epoch(
    model: SoftBaselinePINN,
    train_loader,
    optimizer: torch.optim.Optimizer,
    training_config: TrainingConfig,
    temperature_center: float,
    temperature_scale: float,
) -> dict[str, float]:
    model.train()
    device = torch.device(training_config.device)
    loss_totals: dict[str, float] = {}
    n_batches = 0

    for batch in train_loader:
        x = batch["x"].to(device).detach().clone().requires_grad_(True)
        y_log = batch["y"].to(device)
        raw_pressure = batch["raw_pressure"].to(device)

        output = model(x, raw_pressure)
        losses = compute_v1_losses(
            output,
            y_log=y_log,
            raw_pressure=raw_pressure,
            lambda_p=training_config.lambda_p,
            x=x,
            config=model.config,
            temperature_center=temperature_center,
            temperature_scale=temperature_scale,
        )

        optimizer.zero_grad()
        losses["total_loss"].backward()

        if training_config.gradient_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=training_config.gradient_clip_norm,
            )

        optimizer.step()

        for name, value in losses.items():
            loss_totals[name] = loss_totals.get(name, 0.0) + float(value.detach())

        n_batches += 1

    if n_batches == 0:
        raise ValueError("train_loader produced no batches.")

    return {name: value / n_batches for name, value in loss_totals.items()}


# Evaluate one full validation pass without updating model parameters.
def validate_one_epoch(
    model: SoftBaselinePINN,
    val_loader,
    training_config: TrainingConfig,
    temperature_center: float,
    temperature_scale: float,
) -> dict[str, float]:
    model.eval()
    device = torch.device(training_config.device)
    loss_totals: dict[str, float] = {}
    n_batches = 0

    for batch in val_loader:
        x = batch["x"].to(device).detach().clone().requires_grad_(True)
        y_log = batch["y"].to(device)
        raw_pressure = batch["raw_pressure"].to(device)

        output = model(x, raw_pressure)
        losses = compute_v1_losses(
            output,
            y_log=y_log,
            raw_pressure=raw_pressure,
            lambda_p=training_config.lambda_p,
            x=x,
            config=model.config,
            temperature_center=temperature_center,
            temperature_scale=temperature_scale,
        )

        for name, value in losses.items():
            loss_totals[name] = loss_totals.get(name, 0.0) + float(value.detach())

        n_batches += 1

    if n_batches == 0:
        raise ValueError("val_loader produced no batches.")

    return {name: value / n_batches for name, value in loss_totals.items()}


# Train V1 across many epochs with validation tracking and early stopping.
def train_v1_model(
    model: SoftBaselinePINN,
    train_loader,
    val_loader,
    training_config: TrainingConfig,
    temperature_center: float,
    temperature_scale: float,
) -> tuple[SoftBaselinePINN, list[dict[str, float]]]:
    device = torch.device(training_config.device)
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config.lr,
        weight_decay=training_config.weight_decay,
    )

    best_val_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, training_config.max_epochs + 1):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            training_config,
            temperature_center,
            temperature_scale,
        )
        val_metrics = validate_one_epoch(
            model,
            val_loader,
            training_config,
            temperature_center,
            temperature_scale,
        )

        val_loss = val_metrics["total_loss"]
        improved = val_loss < best_val_loss

        if improved:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
        else:
            epochs_without_improvement += 1

        history_row = {
            "epoch": float(epoch),
            "best_epoch": float(best_epoch),
            "best_val_loss": best_val_loss,
        }
        history_row.update(
            {f"train_{name}": value for name, value in train_metrics.items()}
        )
        history_row.update(
            {f"val_{name}": value for name, value in val_metrics.items()}
        )
        history.append(history_row)

        if epochs_without_improvement >= training_config.patience:
            break

    if best_state is None:
        raise ValueError("Training did not produce a valid model state.")

    model.load_state_dict(best_state)

    return model, history


# Train one V1 model per lambda_p value and summarize validation performance.
def run_lambda_sweep(
    train_loader,
    val_loader,
    temperature_center: float,
    temperature_scale: float,
    lambda_p_values: tuple[float, ...] = DEFAULT_LAMBDA_P_VALUES,
    model_config: Optional[V1ModelConfig] = None,
    training_config: Optional[TrainingConfig] = None,
    seed: int = 42,
) -> tuple[list[dict[str, float]], dict[str, float]]:
    if not lambda_p_values:
        raise ValueError("lambda_p_values must contain at least one value.")

    results: list[dict[str, float]] = []
    base_model_config = model_config or V1ModelConfig()
    base_training_config = training_config or TrainingConfig()

    for lambda_p in lambda_p_values:
        set_torch_seed(seed)
        model = SoftBaselinePINN(base_model_config)
        current_training_config = replace(
            base_training_config,
            lambda_p=float(lambda_p),
        )

        _, history = train_v1_model(
            model,
            train_loader,
            val_loader,
            current_training_config,
            temperature_center,
            temperature_scale,
        )

        best_epoch = int(history[-1]["best_epoch"])
        best_row = history[best_epoch - 1]
        results.append(
            {
                "lambda_p": float(lambda_p),
                "epochs_trained": float(len(history)),
                "best_epoch": best_row["epoch"],
                "best_val_loss": best_row["val_total_loss"],
                "best_val_data_loss": best_row["val_data_loss"],
                "best_val_physics_loss": best_row["val_physics_loss"],
                "best_val_sips_loss": best_row["val_sips_loss"],
                "best_val_bounds_loss": best_row["val_bounds_loss"],
                "best_val_mono_loss": best_row["val_mono_loss"],
                "best_val_vH_loss": best_row["val_vH_loss"],
            }
        )

    best_result = min(results, key=lambda row: row["best_val_loss"])

    return results, best_result


# Build the data pipeline and run the V1 soft-baseline lambda_p experiment.
def run_soft_baseline_exp(
    batch_size: int = 64,
    lambda_p_values: tuple[float, ...] = DEFAULT_LAMBDA_P_VALUES,
    model_config: Optional[V1ModelConfig] = None,
    training_config: Optional[TrainingConfig] = None,
    seed: int = 42,
    verbose: bool = True,
) -> tuple[list[dict[str, float]], dict[str, float]]:
    resolved_model_config = model_config or V1ModelConfig()
    resolved_training_config = training_config or TrainingConfig()
    pipeline = build_data_pipeline(batch_size=batch_size)
    scaler = pipeline["scaler"]
    temperature_idx = resolved_model_config.temperature_feature_idx
    temperature_center = float(scaler.center_[temperature_idx])
    temperature_scale = float(scaler.scale_[temperature_idx])

    results, best_result = run_lambda_sweep(
        train_loader=pipeline["train_loader"],
        val_loader=pipeline["val_loader"],
        temperature_center=temperature_center,
        temperature_scale=temperature_scale,
        lambda_p_values=lambda_p_values,
        model_config=resolved_model_config,
        training_config=resolved_training_config,
        seed=seed,
    )

    if verbose:
        print("lambda_p sweep results")
        print("lambda_p | epochs | best_epoch | val_total | val_data | val_physics")
        for row in results:
            print(
                f"{row['lambda_p']:>8.3g} | "
                f"{row['epochs_trained']:>6.0f} | "
                f"{row['best_epoch']:>10.0f} | "
                f"{row['best_val_loss']:>9.4g} | "
                f"{row['best_val_data_loss']:>8.4g} | "
                f"{row['best_val_physics_loss']:>11.4g}"
            )

        print()
        print(f"Best lambda_p: {best_result['lambda_p']}")
        print(f"Best validation loss: {best_result['best_val_loss']:.6g}")

    return results, best_result


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
