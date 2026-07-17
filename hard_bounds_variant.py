from dataclasses import asdict
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from data_pipeline import build_data_pipeline
from soft_baseline import (
    ResidualBlock,
    TrainingConfig,
    V1ModelConfig,
    heteroscedastic_nll_loss,
    monotonicity_loss,
    set_torch_seed,
    sips_loss,
    summarize_validation_metrics,
    vant_hoff_loss,
)


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


# Combine data loss with V3 soft physics penalties, excluding bounds_loss.
def compute_v3_losses(
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
    physics_loss = sips_penalty + mono_penalty + vH_penalty
    total_loss = data_loss + lambda_p * physics_loss

    return {
        "total_loss": total_loss,
        "data_loss": data_loss,
        "physics_loss": physics_loss,
        "sips_loss": sips_penalty,
        "mono_loss": mono_penalty,
        "vH_loss": vH_penalty,
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


# Run one V3 training pass over all batches.
def train_one_epoch_v3(
    model: HardBoundsPINN,
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

    for batch_idx, batch in enumerate(train_loader):
        x = batch["x"].to(device).detach().clone().requires_grad_(True)
        y_log = batch["y"].to(device)
        raw_pressure = batch["raw_pressure"].to(device)

        output = model(x, raw_pressure)
        verify_bounds_output(output, context=f"training batch {batch_idx}")
        losses = compute_v3_losses(
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


# Evaluate V3 on validation batches without updating model parameters.
def validate_one_epoch_v3(
    model: HardBoundsPINN,
    val_loader,
    training_config: TrainingConfig,
    temperature_center: float,
    temperature_scale: float,
) -> dict[str, float]:
    model.eval()
    device = torch.device(training_config.device)
    loss_totals: dict[str, float] = {}
    n_batches = 0

    for batch_idx, batch in enumerate(val_loader):
        x = batch["x"].to(device).detach().clone().requires_grad_(True)
        y_log = batch["y"].to(device)
        raw_pressure = batch["raw_pressure"].to(device)

        output = model(x, raw_pressure)
        verify_bounds_output(output, context=f"validation batch {batch_idx}")
        losses = compute_v3_losses(
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


# Train V3 across many epochs with validation tracking and early stopping.
def train_v3_model(
    model: HardBoundsPINN,
    train_loader,
    val_loader,
    training_config: TrainingConfig,
    temperature_center: float,
    temperature_scale: float,
) -> tuple[HardBoundsPINN, list[dict[str, float]]]:
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
        train_metrics = train_one_epoch_v3(
            model,
            train_loader,
            optimizer,
            training_config,
            temperature_center,
            temperature_scale,
        )
        val_metrics = validate_one_epoch_v3(
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
        raise ValueError("Training did not produce a valid V3 model state.")

    model.load_state_dict(best_state)

    return model, history


# Verify hard bounds on all requested pipeline tiers.
def verify_bounds_on_pipeline(
    model: HardBoundsPINN,
    pipeline: dict,
    tier_names: tuple[str, ...] = ("train", "validation", "T1", "T2", "T3"),
    device: str = "cpu",
) -> None:
    loader_keys = {
        "train": "train_loader",
        "validation": "val_loader",
        "val": "val_loader",
        "T1": "T1_loader",
        "T2": "T2_loader",
        "T3": "T3_loader",
    }

    for tier_name in tier_names:
        loader_key = loader_keys.get(tier_name, f"{tier_name}_loader")

        if loader_key not in pipeline:
            raise KeyError(f"Pipeline does not contain {loader_key}.")

        verify_bounds(
            model,
            pipeline[loader_key],
            device=device,
        )


# Train V3 and evaluate it on T1/T2/T3 with the unified harness.
def evaluate_hard_bounds_model(
    pipeline: dict,
    model_config: Optional[V1ModelConfig] = None,
    training_config: Optional[TrainingConfig] = None,
    tier_names: tuple[str, ...] = ("T1", "T2", "T3"),
    n_bootstrap: int = 100,
    n_mc: int = 100,
    seed: int = 42,
) -> tuple[HardBoundsPINN, list[dict[str, float]], dict[str, dict]]:
    from unified_eval_harness import evaluate_model

    resolved_model_config = model_config or V1ModelConfig()
    resolved_training_config = training_config or TrainingConfig()
    scaler = pipeline["scaler"]
    temperature_idx = resolved_model_config.temperature_feature_idx
    temperature_center = float(scaler.center_[temperature_idx])
    temperature_scale = float(scaler.scale_[temperature_idx])

    set_torch_seed(seed)
    model = HardBoundsPINN(resolved_model_config)
    model, history = train_v3_model(
        model,
        pipeline["train_loader"],
        pipeline["val_loader"],
        resolved_training_config,
        temperature_center,
        temperature_scale,
    )
    verify_bounds_on_pipeline(
        model,
        pipeline,
        tier_names=("train", "validation", *tier_names),
        device=resolved_training_config.device,
    )
    evaluation_results = evaluate_model(
        model,
        pipeline,
        tier_names=tier_names,
        device=resolved_training_config.device,
        n_bootstrap=n_bootstrap,
        n_mc=n_mc,
        hard_constraints=("nonnegative", "saturation"),
        random_state=seed,
    )

    return model, history, evaluation_results


# Run the full V3 hard-bounds experiment and save reusable artifacts.
def run_and_save_hard_bounds_exp(
    batch_size: int = 64,
    model_config: Optional[V1ModelConfig] = None,
    training_config: Optional[TrainingConfig] = None,
    tier_names: tuple[str, ...] = ("T1", "T2", "T3"),
    results_dir=Path("results"),
    checkpoint_dir=Path("checkpoints"),
    n_bootstrap: int = 100,
    n_mc: int = 100,
    seed: int = 42,
    verbose: bool = True,
) -> dict[str, object]:
    from unified_eval_harness import save_results_csv

    resolved_model_config = model_config or V1ModelConfig()
    resolved_training_config = training_config or TrainingConfig()
    pipeline = build_data_pipeline(batch_size=batch_size)

    results_dir = Path(results_dir)
    checkpoint_dir = Path(checkpoint_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    model, history, evaluation_results = evaluate_hard_bounds_model(
        pipeline,
        model_config=resolved_model_config,
        training_config=resolved_training_config,
        tier_names=tier_names,
        n_bootstrap=n_bootstrap,
        n_mc=n_mc,
        seed=seed,
    )

    history_path = results_dir / "hard_bounds_training_history.csv"
    pd.DataFrame(history).to_csv(history_path, index=False)

    evaluation_metrics_path = results_dir / "hard_bounds_eval_metrics.csv"
    evaluation_table = save_results_csv(
        evaluation_results,
        evaluation_metrics_path,
        model_name="V3_hard_bounds",
    )

    checkpoint_path = checkpoint_dir / "hard_bounds_v3.pt"
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
            "hard_constraints": ["nonnegative", "saturation"],
        },
        checkpoint_path,
    )

    if verbose:
        print(f"Saved V3 training history: {history_path}")
        print(f"Saved V3 evaluation metrics: {evaluation_metrics_path}")
        print(f"Saved V3 checkpoint: {checkpoint_path}")

    results_dict = {"V3": evaluation_results}

    return {
        "model": model,
        "history": history,
        "evaluation_results": evaluation_results,
        "results_dict": results_dict,
        "evaluation_table": evaluation_table,
        "paths": {
            "training_history": history_path,
            "evaluation_metrics": evaluation_metrics_path,
            "checkpoint": checkpoint_path,
        },
    }


# Quick architecture and hard-bounds check.
def smoke_test_hard_bounds() -> None:
    pipeline = build_data_pipeline(batch_size=8)
    batch = next(iter(pipeline["train_loader"]))
    model = HardBoundsPINN()
    training_config = TrainingConfig(lambda_p=0.1, max_epochs=1, patience=1)

    x = batch["x"].detach().clone().requires_grad_(True)
    output = model(x, batch["raw_pressure"])
    losses = compute_v3_losses(
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
    verify_bounds_output(output, context="first training batch")
    verify_bounds(model, pipeline["train_loader"], max_batches=2)
    verify_bounds_on_random_inputs(model)

    trained_model, history = train_v3_model(
        HardBoundsPINN(),
        pipeline["train_loader"],
        pipeline["val_loader"],
        training_config,
        temperature_center=float(
            pipeline["scaler"].center_[model.config.temperature_feature_idx]
        ),
        temperature_scale=float(
            pipeline["scaler"].scale_[model.config.temperature_feature_idx]
        ),
    )
    verify_bounds(trained_model, pipeline["val_loader"], max_batches=2)

    print("x:", batch["x"].shape)
    print("q_hat:", output["q_hat"].shape)
    print("q_max:", output["q_max"].shape)
    print("total_loss:", float(losses["total_loss"].detach()))
    print("physics_loss:", float(losses["physics_loss"].detach()))
    print("epochs_trained:", len(history))
    print("train_total_loss:", history[-1]["train_total_loss"])
    print("val_total_loss:", history[-1]["val_total_loss"])
    print("val_sat_rate:", history[-1]["val_sat_rate"])
    print("Bounds verified for first batches and 1000 random stress inputs.")


if __name__ == "__main__":
    smoke_test_hard_bounds()
