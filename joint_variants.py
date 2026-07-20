from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from data_pipeline import build_data_pipeline
from hard_bounds_variant import verify_bounds_output
from hard_monotonicity_variant import (
    MonotonePressurePathway,
    PositiveLinear,
    vant_hoff_loss_comp_only,
    verify_monotonicity,
)
from soft_baseline import (
    ResidualBlock,
    TrainingConfig,
    V1ModelConfig,
    heteroscedastic_nll_loss,
    set_torch_seed,
    sips_loss,
    summarize_validation_metrics,
)


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

        self.bounds_mono_head = HardBoundsMonotoneHead(
            comp_dim=current_dim,
            pressure_dim=pressure_dim,
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

        x_comp = x[:, self.comp_feature_indices]
        h_comp = self.comp_encoder(x_comp)
        h_pressure = self.pressure_pathway(raw_pressure)

        q_hat, q_max = self.bounds_mono_head(h_comp, h_pressure)
        log_sigma2 = torch.clamp(
            self.logvar_head(h_comp),
            min=self.config.min_log_sigma2,
            max=self.config.max_log_sigma2,
        )
        sigma_ale = torch.exp(0.5 * log_sigma2)

        eps = self.config.softplus_epsilon
        physics_params = F.softplus(self.physics_head(h_comp)) + eps
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


# Combine data loss with V7 soft physics penalties, excluding hard constraints.
def compute_v7_losses(
    output: dict[str, torch.Tensor],
    y_log: torch.Tensor,
    raw_pressure: torch.Tensor,
    lambda_p: float,
    x_comp: torch.Tensor,
    model: HardBoundsMonotonicityPINN,
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
    physics_loss = sips_penalty + vH_penalty
    total_loss = data_loss + lambda_p * physics_loss

    return {
        "total_loss": total_loss,
        "data_loss": data_loss,
        "physics_loss": physics_loss,
        "sips_loss": sips_penalty,
        "vH_loss": vH_penalty,
    }


# Build model inputs that keep the composition branch differentiable.
def prepare_v7_inputs(
    x: torch.Tensor,
    model: HardBoundsMonotonicityPINN,
) -> tuple[torch.Tensor, torch.Tensor]:
    x_for_model = x.detach().clone()
    x_comp = x_for_model[:, model.comp_feature_indices].detach().clone()
    x_comp.requires_grad_(True)
    x_for_model[:, model.comp_feature_indices] = x_comp

    return x_for_model, x_comp


# Verify V7 hard bounds on DataLoader batches.
def verify_bounds_on_loader(
    model: HardBoundsMonotonicityPINN,
    loader,
    device: str = "cpu",
    max_batches: Optional[int] = None,
) -> None:
    device_obj = torch.device(device)
    model = model.to(device_obj)
    model.eval()

    with torch.inference_mode():
        for batch_idx, batch in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            x = batch["x"].to(device_obj)
            raw_pressure = batch["raw_pressure"].to(device_obj)
            output = model(x, raw_pressure)
            verify_bounds_output(output, context=f"V7 batch {batch_idx}")


# Run one V7 training pass over all batches.
def train_one_epoch_v7(
    model: HardBoundsMonotonicityPINN,
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
        x, x_comp = prepare_v7_inputs(batch["x"].to(device), model)
        y_log = batch["y"].to(device)
        raw_pressure = batch["raw_pressure"].to(device)

        output = model(x, raw_pressure)
        losses = compute_v7_losses(
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


# Evaluate V7 on validation batches without updating model parameters.
def validate_one_epoch_v7(
    model: HardBoundsMonotonicityPINN,
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
        x, x_comp = prepare_v7_inputs(batch["x"].to(device), model)
        y_log = batch["y"].to(device)
        raw_pressure = batch["raw_pressure"].to(device)

        output = model(x, raw_pressure)
        losses = compute_v7_losses(
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


# Train V7 across many epochs with validation tracking and early stopping.
def train_v7_model(
    model: HardBoundsMonotonicityPINN,
    train_loader,
    val_loader,
    training_config: TrainingConfig,
    temperature_center: float,
    temperature_scale: float,
) -> tuple[HardBoundsMonotonicityPINN, list[dict[str, float]]]:
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
        train_metrics = train_one_epoch_v7(
            model,
            train_loader,
            optimizer,
            training_config,
            temperature_center,
            temperature_scale,
        )
        val_metrics = validate_one_epoch_v7(
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
        raise ValueError("Training did not produce a valid V7 model state.")

    model.load_state_dict(best_state)

    return model, history


# Train V7 and evaluate it on T1/T2/T3 with the unified harness.
def evaluate_joint_v7_model(
    pipeline: dict,
    model_config: Optional[JointVariantConfig] = None,
    training_config: Optional[TrainingConfig] = None,
    tier_names: tuple[str, ...] = ("T1", "T2", "T3"),
    n_bootstrap: int = 100,
    n_mc: int = 100,
    seed: int = 42,
    verify_n_comp: int = 500,
    verify_n_P: int = 50,
) -> tuple[
    HardBoundsMonotonicityPINN,
    list[dict[str, float]],
    dict[str, dict],
    dict[str, float],
]:
    from unified_eval_harness import evaluate_model

    resolved_model_config = model_config or JointVariantConfig()
    resolved_training_config = training_config or TrainingConfig()
    scaler = pipeline["scaler"]
    temperature_idx = resolved_model_config.temperature_feature_idx
    temperature_center = float(scaler.center_[temperature_idx])
    temperature_scale = float(scaler.scale_[temperature_idx])

    set_torch_seed(seed)
    model = HardBoundsMonotonicityPINN(resolved_model_config)
    model, history = train_v7_model(
        model,
        pipeline["train_loader"],
        pipeline["val_loader"],
        resolved_training_config,
        temperature_center,
        temperature_scale,
    )
    verify_bounds_on_loader(
        model,
        pipeline["train_loader"],
        device=resolved_training_config.device,
    )
    joint_verification = verify_monotonicity(
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
        hard_constraints=("nonnegative", "saturation", "monotonicity"),
        random_state=seed,
    )

    return model, history, evaluation_results, joint_verification


# Run the full V7 joint hard-bounds plus monotonicity experiment.
def run_and_save_joint_v7_exp(
    batch_size: int = 64,
    model_config: Optional[JointVariantConfig] = None,
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

    resolved_model_config = model_config or JointVariantConfig()
    resolved_training_config = training_config or TrainingConfig()
    pipeline = build_data_pipeline(batch_size=batch_size)

    results_dir = Path(results_dir)
    checkpoint_dir = Path(checkpoint_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    model, history, evaluation_results, joint_verification = evaluate_joint_v7_model(
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

    history_path = results_dir / "joint_v7_training_history.csv"
    pd.DataFrame(history).to_csv(history_path, index=False)

    verification_path = results_dir / "joint_v7_verification.csv"
    pd.DataFrame([joint_verification]).to_csv(verification_path, index=False)

    evaluation_metrics_path = results_dir / "joint_v7_eval_metrics.csv"
    evaluation_table = save_results_csv(
        evaluation_results,
        evaluation_metrics_path,
        model_name="V7_hard_bounds_monotonicity",
    )

    checkpoint_path = checkpoint_dir / "joint_v7.pt"
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
            "hard_constraints": ["nonnegative", "saturation", "monotonicity"],
            "joint_verification": joint_verification,
        },
        checkpoint_path,
    )

    if verbose:
        print(f"Saved V7 training history: {history_path}")
        print(f"Saved V7 joint verification: {verification_path}")
        print(f"Saved V7 evaluation metrics: {evaluation_metrics_path}")
        print(f"Saved V7 checkpoint: {checkpoint_path}")

    results_dict = {"V7": evaluation_results}

    return {
        "model": model,
        "history": history,
        "evaluation_results": evaluation_results,
        "results_dict": results_dict,
        "evaluation_table": evaluation_table,
        "joint_verification": joint_verification,
        "paths": {
            "training_history": history_path,
            "joint_verification": verification_path,
            "evaluation_metrics": evaluation_metrics_path,
            "checkpoint": checkpoint_path,
        },
    }


# Quick V7 architecture, loss, training, and structural guarantee check.
def smoke_test_joint_variants() -> None:
    pipeline = build_data_pipeline(batch_size=8)
    batch = next(iter(pipeline["train_loader"]))
    model = HardBoundsMonotonicityPINN()
    training_config = TrainingConfig(lambda_p=0.1, max_epochs=1, patience=1)
    temperature_center = float(
        pipeline["scaler"].center_[model.config.temperature_feature_idx]
    )
    temperature_scale = float(
        pipeline["scaler"].scale_[model.config.temperature_feature_idx]
    )
    raw_pressure = batch["raw_pressure"].detach().clone().requires_grad_(True)
    x, x_comp = prepare_v7_inputs(batch["x"], model)
    output = model(x, raw_pressure)
    losses = compute_v7_losses(
        output,
        y_log=batch["y"],
        raw_pressure=raw_pressure,
        lambda_p=0.1,
        x_comp=x_comp,
        model=model,
        temperature_center=temperature_center,
        temperature_scale=temperature_scale,
    )
    dq_dP = torch.autograd.grad(
        output["q_hat"].sum(),
        raw_pressure,
        create_graph=False,
        retain_graph=False,
    )[0]
    verify_bounds_output(output, context="V7 smoke test batch")
    monotonicity_verification = verify_monotonicity(
        model,
        pipeline["train_loader"],
        n_comp=16,
        n_P=8,
    )
    trained_model, history = train_v7_model(
        HardBoundsMonotonicityPINN(),
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
    print("composition features:", len(model.comp_feature_indices))
    print("q_hat:", output["q_hat"].shape)
    print("q_max:", output["q_max"].shape)
    print("total_loss:", float(losses["total_loss"].detach()))
    print("physics_loss:", float(losses["physics_loss"].detach()))
    print("sips_loss:", float(losses["sips_loss"].detach()))
    print("vH_loss:", float(losses["vH_loss"].detach()))
    print("min q_hat:", float(output["q_hat"].min().detach()))
    print(
        "min qmax_minus_q:", float((output["q_max"] - output["q_hat"]).min().detach())
    )
    print("min dq_dP:", float(dq_dP.min().detach()))
    print("verification_min_dq_dP:", monotonicity_verification["min_dq_dP"])
    print("epochs_trained:", len(history))
    print("train_total_loss:", history[-1]["train_total_loss"])
    print("val_total_loss:", history[-1]["val_total_loss"])
    print("trained_min_dq_dP:", trained_verification["min_dq_dP"])


if __name__ == "__main__":
    smoke_test_joint_variants()
