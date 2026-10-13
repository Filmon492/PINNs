from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from data_pipeline import build_data_pipeline
from result_paths import ensure_step_results_dir
from soft_baseline import (
    R_GAS,
    ResidualBlock,
    TrainingConfig,
    V1ModelConfig,
    bounds_loss,
    heteroscedastic_nll_loss,
    set_torch_seed,
    sips_loss,
    summarize_validation_metrics,
)

DEFAULT_PRESSURE_DEPENDENT_FEATURE_INDICES = (4, 6, 10)


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


@dataclass
class HardMonotonicityConfig(V1ModelConfig):
    pressure_dependent_feature_indices: tuple[int, ...] = (
        DEFAULT_PRESSURE_DEPENDENT_FEATURE_INDICES
    )


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
        self.config = config or HardMonotonicityConfig()
        pressure_dependent_indices = set(
            getattr(
                self.config,
                "pressure_dependent_feature_indices",
                DEFAULT_PRESSURE_DEPENDENT_FEATURE_INDICES,
            )
        )
        pressure_dependent_indices.add(self.config.pressure_feature_idx)

        invalid_indices = [
            idx
            for idx in pressure_dependent_indices
            if idx < 0 or idx >= self.config.input_dim
        ]
        if invalid_indices:
            raise ValueError(
                f"Pressure-dependent feature indices are out of range: {invalid_indices}"
            )

        self.pressure_dependent_feature_indices = tuple(
            sorted(pressure_dependent_indices)
        )
        self.comp_feature_indices = tuple(
            idx
            for idx in range(self.config.input_dim)
            if idx not in self.pressure_dependent_feature_indices
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

        self.mean_head = MonotoneAdsorptionHead(
            comp_dim=current_dim,
            pressure_dim=self.pressure_pathway.output_dim,
            softplus_epsilon=self.config.softplus_epsilon,
        )
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

        x_comp = x[:, self.comp_feature_indices]
        h_comp = self.comp_encoder(x_comp)
        h_pressure = self.pressure_pathway(raw_pressure)

        q_hat = self.mean_head(h_comp, h_pressure)
        log_sigma2 = torch.clamp(
            self.logvar_head(h_comp),
            min=self.config.min_log_sigma2,
            max=self.config.max_log_sigma2,
        )
        sigma_ale = torch.exp(0.5 * log_sigma2)

        eps = self.config.softplus_epsilon
        physics_params = F.softplus(self.physics_head(h_comp)) + eps
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


# Penalize deviations from the van't Hoff relation without pressure gradients.
def vant_hoff_loss_comp_only(
    output: dict[str, torch.Tensor],
    x_comp: torch.Tensor,
    temperature_comp_idx: int,
    temperature_center: float,
    temperature_scale: float,
) -> torch.Tensor:
    if not x_comp.requires_grad:
        raise ValueError("x_comp must require gradients for vant_hoff_loss_comp_only.")

    if temperature_scale == 0:
        raise ValueError("temperature_scale must be nonzero.")

    ln_K = torch.log(torch.clamp(output["K"], min=1e-8))
    gradient = torch.autograd.grad(
        ln_K.sum(),
        x_comp,
        create_graph=True,
        retain_graph=True,
    )[0]
    scaled_temperature = x_comp[:, temperature_comp_idx : temperature_comp_idx + 1]
    raw_temperature = scaled_temperature * temperature_scale + temperature_center
    d_lnK_d_scaled_T = gradient[
        :,
        temperature_comp_idx : temperature_comp_idx + 1,
    ]
    d_lnK_d_T = d_lnK_d_scaled_T / temperature_scale
    d_lnK_d_invT = d_lnK_d_T * (-(raw_temperature**2))
    target = -output["delta_H"] / R_GAS

    return F.mse_loss(d_lnK_d_invT, target)


# Combine data loss with V4 soft physics penalties, excluding monotonicity_loss.
def compute_v4_losses(
    output: dict[str, torch.Tensor],
    y_log: torch.Tensor,
    raw_pressure: torch.Tensor,
    lambda_p: float,
    x_comp: torch.Tensor,
    model: HardMonotonicityPINN,
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
    temperature_comp_idx = model.comp_feature_indices.index(
        model.config.temperature_feature_idx
    )
    vH_penalty = vant_hoff_loss_comp_only(
        output,
        x_comp=x_comp,
        temperature_comp_idx=temperature_comp_idx,
        temperature_center=temperature_center,
        temperature_scale=temperature_scale,
    )
    physics_loss = sips_penalty + bounds_penalty + vH_penalty
    total_loss = data_loss + lambda_p * physics_loss

    return {
        "total_loss": total_loss,
        "data_loss": data_loss,
        "physics_loss": physics_loss,
        "sips_loss": sips_penalty,
        "bounds_loss": bounds_penalty,
        "vH_loss": vH_penalty,
    }


# Build model inputs that keep the composition branch differentiable.
def prepare_v4_inputs(
    x: torch.Tensor,
    model: HardMonotonicityPINN,
) -> tuple[torch.Tensor, torch.Tensor]:
    x_for_model = x.detach().clone()
    x_comp = x_for_model[:, model.comp_feature_indices].detach().clone()
    x_comp.requires_grad_(True)
    x_for_model[:, model.comp_feature_indices] = x_comp

    return x_for_model, x_comp


# Check the hard monotonicity guarantee by differentiating q_hat with respect to P.
def verify_monotonicity(
    model: HardMonotonicityPINN,
    train_loader,
    n_comp: int = 500,
    n_P: int = 50,
    pressure_min: float = 0.01,
    pressure_max: float = 15.0,
    tolerance: float = 1e-6,
    device: str = "cpu",
    random_state: int = 42,
    comp_batch_size: int = 64,
) -> dict[str, float]:
    if n_comp <= 0:
        raise ValueError("n_comp must be positive.")

    if n_P <= 1:
        raise ValueError("n_P must be greater than 1.")

    if pressure_min <= 0 or pressure_max <= pressure_min:
        raise ValueError("Pressure range must satisfy 0 < pressure_min < pressure_max.")

    device_obj = torch.device(device)
    model = model.to(device_obj)
    was_training = model.training
    model.eval()

    x_batches = []
    for batch in train_loader:
        x_batches.append(batch["x"])

    if not x_batches:
        raise ValueError("train_loader produced no batches.")

    x_reference = torch.cat(x_batches, dim=0)
    if len(x_reference) == 0:
        raise ValueError("No reference compositions are available.")

    generator = torch.Generator()
    generator.manual_seed(random_state)
    n_selected = min(n_comp, len(x_reference))
    selected_indices = torch.randperm(len(x_reference), generator=generator)[
        :n_selected
    ]
    x_reference = x_reference[selected_indices].to(device_obj)
    pressure_values = torch.linspace(
        pressure_min,
        pressure_max,
        n_P,
        device=device_obj,
    )

    total_count = 0
    negative_count = 0
    min_gradient = float("inf")

    try:
        for start in range(0, n_selected, comp_batch_size):
            x_chunk = x_reference[start : start + comp_batch_size]
            chunk_size = len(x_chunk)
            x_grid = x_chunk.repeat_interleave(n_P, dim=0)
            raw_pressure = pressure_values.repeat(chunk_size).reshape(-1, 1)
            raw_pressure = raw_pressure.detach().clone().requires_grad_(True)

            output = model(x_grid, raw_pressure)
            dq_dP = torch.autograd.grad(
                output["q_hat"].sum(),
                raw_pressure,
                create_graph=False,
                retain_graph=False,
            )[0]
            negative_gradients = dq_dP < -tolerance

            total_count += int(dq_dP.numel())
            negative_count += int(negative_gradients.sum().item())
            min_gradient = min(min_gradient, float(dq_dP.min().detach().cpu().item()))
    finally:
        model.train(was_training)

    fraction_negative = negative_count / total_count
    max_violation_magnitude = max(0.0, -min_gradient)
    metrics = {
        "n_comp": float(n_selected),
        "n_P": float(n_P),
        "min_dq_dP": min_gradient,
        "max_violation_magnitude": max_violation_magnitude,
        "fraction_negative": fraction_negative,
    }

    if negative_count:
        raise ValueError(
            "Hard monotonicity failed: "
            f"{negative_count}/{total_count} gradients were below -{tolerance}. "
            f"Minimum dq/dP={min_gradient:.6g}."
        )

    return metrics


# Run one V4 training pass over all batches.
def train_one_epoch_v4(
    model: HardMonotonicityPINN,
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
        x, x_comp = prepare_v4_inputs(batch["x"].to(device), model)
        y_log = batch["y"].to(device)
        raw_pressure = batch["raw_pressure"].to(device)

        output = model(x, raw_pressure)
        losses = compute_v4_losses(
            output,
            y_log=y_log,
            raw_pressure=raw_pressure,
            lambda_p=training_config.lambda_p,
            x_comp=x_comp,
            model=model,
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


# Evaluate V4 on validation batches without updating model parameters.
def validate_one_epoch_v4(
    model: HardMonotonicityPINN,
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
        x, x_comp = prepare_v4_inputs(batch["x"].to(device), model)
        y_log = batch["y"].to(device)
        raw_pressure = batch["raw_pressure"].to(device)

        output = model(x, raw_pressure)
        losses = compute_v4_losses(
            output,
            y_log=y_log,
            raw_pressure=raw_pressure,
            lambda_p=training_config.lambda_p,
            x_comp=x_comp,
            model=model,
            temperature_center=temperature_center,
            temperature_scale=temperature_scale,
        )

        for name, value in losses.items():
            loss_totals[name] = loss_totals.get(name, 0.0) + float(value.detach())

        n_batches += 1

    if n_batches == 0:
        raise ValueError("val_loader produced no batches.")

    return {name: value / n_batches for name, value in loss_totals.items()}


# Train V4 across many epochs with validation tracking and early stopping.
def train_v4_model(
    model: HardMonotonicityPINN,
    train_loader,
    val_loader,
    training_config: TrainingConfig,
    temperature_center: float,
    temperature_scale: float,
) -> tuple[HardMonotonicityPINN, list[dict[str, float]]]:
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
        train_metrics = train_one_epoch_v4(
            model,
            train_loader,
            optimizer,
            training_config,
            temperature_center,
            temperature_scale,
        )
        val_metrics = validate_one_epoch_v4(
            model,
            val_loader,
            training_config,
            temperature_center,
            temperature_scale,
        )
        val_summary = summarize_validation_metrics(
            model,
            val_loader,
            device=training_config.device,
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
        history_row.update(
            {
                "val_RMSE": val_summary["accuracy"]["RMSE"],
                "val_MAE": val_summary["accuracy"]["MAE"],
                "val_R2": val_summary["accuracy"]["R2"],
                "val_neg_rate": val_summary["physical"]["neg_rate"],
                "val_mono_rate": val_summary["physical"]["mono_rate"],
                "val_sat_rate": val_summary["physical"]["sat_rate"],
                "val_violation_rate": val_summary["physical"]["violation_rate"],
            }
        )
        history.append(history_row)

        if epochs_without_improvement >= training_config.patience:
            break

    if best_state is None:
        raise ValueError("Training did not produce a valid V4 model state.")

    model.load_state_dict(best_state)

    return model, history


# Train V4 and evaluate it on T1/T2/T3 with the unified harness.
def evaluate_hard_monotonicity_model(
    pipeline: dict,
    model_config: Optional[V1ModelConfig] = None,
    training_config: Optional[TrainingConfig] = None,
    tier_names: tuple[str, ...] = ("T1", "T2", "T3"),
    n_bootstrap: int = 100,
    n_mc: int = 100,
    seed: int = 42,
    verify_n_comp: int = 500,
    verify_n_P: int = 50,
) -> tuple[HardMonotonicityPINN, list[dict[str, float]], dict[str, dict], dict]:
    from unified_eval_harness import evaluate_model

    resolved_model_config = model_config or HardMonotonicityConfig()
    resolved_training_config = training_config or TrainingConfig()
    scaler = pipeline["scaler"]
    temperature_idx = resolved_model_config.temperature_feature_idx
    temperature_center = float(scaler.center_[temperature_idx])
    temperature_scale = float(scaler.scale_[temperature_idx])

    set_torch_seed(seed)
    model = HardMonotonicityPINN(resolved_model_config)
    model, history = train_v4_model(
        model,
        pipeline["train_loader"],
        pipeline["val_loader"],
        resolved_training_config,
        temperature_center,
        temperature_scale,
    )
    monotonicity_verification = verify_monotonicity(
        model,
        pipeline["train_loader"],
        n_comp=verify_n_comp,
        n_P=verify_n_P,
        device=resolved_training_config.device,
        random_state=seed,
    )
    evaluation_results = evaluate_model(
        model,
        pipeline,
        tier_names=tier_names,
        device=resolved_training_config.device,
        n_bootstrap=n_bootstrap,
        n_mc=n_mc,
        hard_constraints=("monotonicity",),
        random_state=seed,
    )

    return model, history, evaluation_results, monotonicity_verification


# Run the full V4 hard-monotonicity experiment and save reusable artifacts.
def run_and_save_hard_monotonicity_exp(
    batch_size: int = 64,
    model_config: Optional[V1ModelConfig] = None,
    training_config: Optional[TrainingConfig] = None,
    tier_names: tuple[str, ...] = ("T1", "T2", "T3"),
    results_dir=Path("results"),
    checkpoint_dir=Path("checkpoints"),
    n_bootstrap: int = 100,
    n_mc: int = 100,
    seed: int = 42,
    verify_n_comp: int = 500,
    verify_n_P: int = 50,
    verbose: bool = True,
) -> dict[str, object]:
    from unified_eval_harness import save_results_csv

    resolved_model_config = model_config or HardMonotonicityConfig()
    resolved_training_config = training_config or TrainingConfig()
    pipeline = build_data_pipeline(batch_size=batch_size)

    results_dir = ensure_step_results_dir(results_dir, step=5)
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    model, history, evaluation_results, monotonicity_verification = (
        evaluate_hard_monotonicity_model(
            pipeline,
            model_config=resolved_model_config,
            training_config=resolved_training_config,
            tier_names=tier_names,
            n_bootstrap=n_bootstrap,
            n_mc=n_mc,
            seed=seed,
            verify_n_comp=verify_n_comp,
            verify_n_P=verify_n_P,
        )
    )

    history_path = results_dir / "hard_monotonicity_training_history.csv"
    pd.DataFrame(history).to_csv(history_path, index=False)

    verification_path = results_dir / "hard_monotonicity_verification.csv"
    pd.DataFrame([monotonicity_verification]).to_csv(verification_path, index=False)

    evaluation_metrics_path = results_dir / "hard_monotonicity_eval_metrics.csv"
    evaluation_table = save_results_csv(
        evaluation_results,
        evaluation_metrics_path,
        model_name="V4_hard_monotonicity",
    )

    checkpoint_path = checkpoint_dir / "hard_monotonicity_v4.pt"
    torch.save(
        {
            "model_state_dict": {
                name: value.detach().cpu() for name, value in model.state_dict().items()
            },
            "model_config": asdict(resolved_model_config),
            "training_config": asdict(resolved_training_config),
            "feature_names": pipeline["feature_names"],
            "t2_rank": pipeline["t2_rank"],
            "seed": seed,
            "hard_constraints": ["monotonicity"],
            "monotonicity_verification": monotonicity_verification,
        },
        checkpoint_path,
    )

    if verbose:
        print(f"Saved V4 training history: {history_path}")
        print(f"Saved V4 monotonicity verification: {verification_path}")
        print(f"Saved V4 evaluation metrics: {evaluation_metrics_path}")
        print(f"Saved V4 checkpoint: {checkpoint_path}")

    results_dict = {"V4": evaluation_results}

    return {
        "model": model,
        "history": history,
        "evaluation_results": evaluation_results,
        "results_dict": results_dict,
        "evaluation_table": evaluation_table,
        "monotonicity_verification": monotonicity_verification,
        "paths": {
            "training_history": history_path,
            "monotonicity_verification": verification_path,
            "evaluation_metrics": evaluation_metrics_path,
            "checkpoint": checkpoint_path,
        },
    }


# Quick architecture, loss, training, and monotonicity check.
def smoke_test_hard_monotonicity() -> None:
    pipeline = build_data_pipeline(batch_size=8)
    batch = next(iter(pipeline["train_loader"]))
    model = HardMonotonicityPINN()
    training_config = TrainingConfig(lambda_p=0.1, max_epochs=1, patience=1)
    temperature_center = float(
        pipeline["scaler"].center_[model.config.temperature_feature_idx]
    )
    temperature_scale = float(
        pipeline["scaler"].scale_[model.config.temperature_feature_idx]
    )
    x_for_model, x_comp = prepare_v4_inputs(batch["x"], model)
    output = model(x_for_model, batch["raw_pressure"])
    losses = compute_v4_losses(
        output,
        y_log=batch["y"],
        raw_pressure=batch["raw_pressure"],
        lambda_p=0.1,
        x_comp=x_comp,
        model=model,
        temperature_center=temperature_center,
        temperature_scale=temperature_scale,
    )
    verification = verify_monotonicity(
        model,
        pipeline["train_loader"],
        n_comp=16,
        n_P=8,
    )
    trained_model, history = train_v4_model(
        HardMonotonicityPINN(),
        pipeline["train_loader"],
        pipeline["val_loader"],
        training_config,
        temperature_center,
        temperature_scale,
    )
    trained_verification = verify_monotonicity(
        trained_model,
        pipeline["train_loader"],
        n_comp=16,
        n_P=8,
    )

    print("x:", batch["x"].shape)
    print("raw_pressure:", batch["raw_pressure"].shape)
    print("q_hat:", output["q_hat"].shape)
    print("q_max:", output["q_max"].shape)
    print("K:", output["K"].shape)
    print("n:", output["n"].shape)
    print("delta_H:", output["delta_H"].shape)
    print("composition features:", len(model.comp_feature_indices))
    print("total_loss:", float(losses["total_loss"].detach()))
    print("physics_loss:", float(losses["physics_loss"].detach()))
    print("min_dq_dP:", verification["min_dq_dP"])
    print("fraction_negative:", verification["fraction_negative"])
    print("epochs_trained:", len(history))
    print("train_total_loss:", history[-1]["train_total_loss"])
    print("val_total_loss:", history[-1]["val_total_loss"])
    print("trained_min_dq_dP:", trained_verification["min_dq_dP"])
    print("Hard monotonicity verified before and after one training epoch.")


if __name__ == "__main__":
    smoke_test_hard_monotonicity()
