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
    monotonicity_loss,
    summarize_validation_metrics,
)


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
        clamped_exponent = torch.clamp(
            exponent,
            min=self.config.min_arrhenius_exponent,
            max=self.config.max_arrhenius_exponent,
        )
        log_K0 = torch.log(torch.clamp(K0, min=eps))
        log_K_unclamped = log_K0 + exponent
        log_K = log_K0 + clamped_exponent
        K = torch.exp(log_K)

        return {
            "K": K,
            "K0": K0,
            "log_K": log_K,
            "log_K_unclamped": log_K_unclamped,
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
            "log_K": arrhenius_output["log_K"],
            "log_K_unclamped": arrhenius_output["log_K_unclamped"],
            "n": n,
            "delta_H": arrhenius_output["delta_H"],
            "delta_H_kJmol": arrhenius_output["delta_H_kJmol"],
        }


# Stable Sips penalty for V6, avoiding overflow in (K P)^n.
def stable_sips_loss(
    output: dict[str, torch.Tensor],
    raw_pressure: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    log_KP_n = output["n"] * (
        torch.log(torch.clamp(output["K"], min=eps))
        + torch.log(torch.clamp(raw_pressure, min=eps))
    )
    log_KP_n = torch.clamp(log_KP_n, min=-50.0, max=50.0)
    q_sips = output["q_max"] * torch.sigmoid(log_KP_n)

    return F.mse_loss(output["q_hat"], q_sips)


# Combine data loss with V6 soft physics penalties, excluding van't Hoff loss.
def compute_v6_losses(
    output: dict[str, torch.Tensor],
    y_log: torch.Tensor,
    raw_pressure: torch.Tensor,
    lambda_p: float,
    x: torch.Tensor,
    config: ArrheniusConfig,
) -> dict[str, torch.Tensor]:
    q_true = torch.exp(y_log)
    data_loss = heteroscedastic_nll_loss(
        q_true=q_true,
        q_hat=output["q_hat"],
        log_sigma2=output["log_sigma2"],
    )
    sips_penalty = stable_sips_loss(output, raw_pressure)
    bounds_penalty = bounds_loss(output)
    mono_penalty = monotonicity_loss(
        output,
        x,
        pressure_feature_idx=config.pressure_feature_idx,
    )
    physics_loss = sips_penalty + bounds_penalty + mono_penalty
    total_loss = data_loss + lambda_p * physics_loss

    return {
        "total_loss": total_loss,
        "data_loss": data_loss,
        "physics_loss": physics_loss,
        "sips_loss": sips_penalty,
        "bounds_loss": bounds_penalty,
        "mono_loss": mono_penalty,
    }


# Verify the Arrhenius identity and enthalpy range on training samples.
def verify_arrhenius_structure(
    model: ArrheniusPINN,
    train_loader,
    n_samples: int = 100,
    tolerance: float = 1e-4,
    expected_dH_range_kJmol: tuple[float, float] = (-60.0, -5.0),
    required_fraction_in_range: float = 0.95,
    device: str = "cpu",
    random_state: int = 42,
) -> dict[str, float]:
    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")

    device_obj = torch.device(device)
    original_dtype = next(model.parameters()).dtype
    verification_dtype = torch.float64
    model = model.to(device_obj, dtype=verification_dtype)
    was_training = model.training
    model.eval()

    x_batches = []
    for batch in train_loader:
        x_batches.append(batch["x"])

    if not x_batches:
        raise ValueError("train_loader produced no batches.")

    x_all = torch.cat(x_batches, dim=0)
    generator = torch.Generator()
    generator.manual_seed(random_state)
    n_selected = min(n_samples, len(x_all))
    selected_indices = torch.randperm(len(x_all), generator=generator)[:n_selected]

    x = x_all[selected_indices].to(device_obj, dtype=verification_dtype)

    try:
        h = model.encoder(x).detach()
        raw_temperature = model.raw_temperature_from_scaled_x(x).detach()
        inverse_temperature = (1.0 / raw_temperature).detach().clone()
        inverse_temperature.requires_grad_(True)
        varied_temperature = 1.0 / inverse_temperature
        output = model.arrhenius_head(h, varied_temperature)
        ln_K = output.get(
            "log_K_unclamped",
            torch.log(torch.clamp(output["K"], min=1e-8)),
        )
        d_lnK_d_invT = torch.autograd.grad(
            ln_K.sum(),
            inverse_temperature,
            create_graph=False,
            retain_graph=False,
        )[0]
        expected_slope = -output["delta_H"] / R_GAS
        max_identity_error = float(
            torch.max(torch.abs(d_lnK_d_invT - expected_slope)).detach().cpu().item()
        )
        delta_H_kJmol = output["delta_H_kJmol"].detach().cpu()
    finally:
        model.train(was_training)
        model.to(dtype=original_dtype)

    lower, upper = expected_dH_range_kJmol
    in_range = (delta_H_kJmol >= lower) & (delta_H_kJmol <= upper)
    fraction_in_range = float(in_range.float().mean().item())
    metrics = {
        "n_samples": float(n_selected),
        "max_arrhenius_identity_error": max_identity_error,
        "delta_H_kJmol_mean": float(delta_H_kJmol.mean().item()),
        "delta_H_kJmol_std": float(delta_H_kJmol.std(unbiased=False).item()),
        "delta_H_kJmol_min": float(delta_H_kJmol.min().item()),
        "delta_H_kJmol_max": float(delta_H_kJmol.max().item()),
        "fraction_delta_H_in_expected_range": fraction_in_range,
    }

    if max_identity_error > tolerance:
        raise ValueError(
            "Arrhenius identity check failed: "
            f"max error {max_identity_error:.6g} exceeds {tolerance:.6g}."
        )

    if fraction_in_range < required_fraction_in_range:
        raise ValueError(
            "Arrhenius enthalpy range check failed: "
            f"{fraction_in_range:.3f} in range {expected_dH_range_kJmol}, "
            f"required {required_fraction_in_range:.3f}."
        )

    return metrics


# Summarize predicted enthalpy by inferred coal rank.
def compute_enthalpy_by_rank(
    model: ArrheniusPINN,
    pipeline: dict,
    split_names: tuple[str, ...] = ("train", "validation", "T1", "T2", "T3"),
    device: str = "cpu",
) -> pd.DataFrame:
    suffixes = {
        "train": "train",
        "validation": "val",
        "val": "val",
        "T1": "T1",
        "T2": "T2",
        "T3": "T3",
    }
    rows = []
    model = model.to(device)
    was_training = model.training
    model.eval()

    try:
        with torch.inference_mode():
            for split_name in split_names:
                suffix = suffixes[split_name]
                split_key = "validation" if split_name == "val" else split_name
                split_df = pipeline["splits"][split_key].reset_index(drop=True)
                x = torch.tensor(
                    pipeline["arrays"][f"X_{suffix}"],
                    dtype=torch.float32,
                    device=device,
                )
                raw_pressure = torch.tensor(
                    pipeline["arrays"][f"raw_pressure_{suffix}"],
                    dtype=torch.float32,
                    device=device,
                )
                output = model(x, raw_pressure)
                delta_H_values = output["delta_H_kJmol"].cpu().numpy().reshape(-1)

                for coal_rank, delta_H_kJmol in zip(
                    split_df["coal_rank_inferred"],
                    delta_H_values,
                ):
                    rows.append(
                        {
                            "split": split_key,
                            "coal_rank": coal_rank,
                            "delta_H_kJmol": float(delta_H_kJmol),
                        }
                    )
    finally:
        model.train(was_training)

    enthalpy_table = pd.DataFrame(rows)

    return (
        enthalpy_table.groupby(["split", "coal_rank"], sort=True)
        .agg(
            n_samples=("delta_H_kJmol", "size"),
            delta_H_kJmol_mean=("delta_H_kJmol", "mean"),
            delta_H_kJmol_std=("delta_H_kJmol", "std"),
            delta_H_kJmol_min=("delta_H_kJmol", "min"),
            delta_H_kJmol_max=("delta_H_kJmol", "max"),
        )
        .reset_index()
    )


# Run one V6 training pass over all batches.
def train_one_epoch_v6(
    model: ArrheniusPINN,
    train_loader,
    optimizer: torch.optim.Optimizer,
    training_config: TrainingConfig,
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
        losses = compute_v6_losses(
            output,
            y_log=y_log,
            raw_pressure=raw_pressure,
            lambda_p=training_config.lambda_p,
            x=x,
            config=model.config,
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


# Evaluate V6 on validation batches without updating model parameters.
def validate_one_epoch_v6(
    model: ArrheniusPINN,
    val_loader,
    training_config: TrainingConfig,
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
        losses = compute_v6_losses(
            output,
            y_log=y_log,
            raw_pressure=raw_pressure,
            lambda_p=training_config.lambda_p,
            x=x,
            config=model.config,
        )

        for name, value in losses.items():
            loss_totals[name] = loss_totals.get(name, 0.0) + float(value.detach())

        n_batches += 1

    if n_batches == 0:
        raise ValueError("val_loader produced no batches.")

    return {name: value / n_batches for name, value in loss_totals.items()}


# Train V6 across many epochs with validation tracking and early stopping.
def train_v6_model(
    model: ArrheniusPINN,
    train_loader,
    val_loader,
    training_config: TrainingConfig,
) -> tuple[ArrheniusPINN, list[dict[str, float]]]:
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
        train_metrics = train_one_epoch_v6(
            model,
            train_loader,
            optimizer,
            training_config,
        )
        val_metrics = validate_one_epoch_v6(
            model,
            val_loader,
            training_config,
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
        raise ValueError("Training did not produce a valid V6 model state.")

    model.load_state_dict(best_state)

    return model, history


# Train V6 and evaluate it on T1/T2/T3 with the unified harness.
def evaluate_arrhenius_model(
    pipeline: dict,
    model_config: Optional[ArrheniusConfig] = None,
    training_config: Optional[TrainingConfig] = None,
    tier_names: tuple[str, ...] = ("T1", "T2", "T3"),
    n_bootstrap: int = 100,
    n_mc: int = 100,
    seed: int = 42,
    verify_n_samples: int = 100,
    required_fraction_in_range: float = 0.95,
    strict_parameter_ranges: bool = False,
) -> tuple[ArrheniusPINN, list[dict[str, float]], dict[str, dict], dict]:
    from soft_baseline import set_torch_seed
    from unified_eval_harness import evaluate_model

    base_model_config = model_config or ArrheniusConfig()
    scaler = pipeline["scaler"]
    temperature_idx = base_model_config.temperature_feature_idx
    resolved_model_config = ArrheniusConfig(
        **{
            **asdict(base_model_config),
            "temperature_center": float(scaler.center_[temperature_idx]),
            "temperature_scale": float(scaler.scale_[temperature_idx]),
        }
    )
    resolved_training_config = training_config or TrainingConfig()

    set_torch_seed(seed)
    model = ArrheniusPINN(resolved_model_config)
    model, history = train_v6_model(
        model,
        pipeline["train_loader"],
        pipeline["val_loader"],
        resolved_training_config,
    )
    arrhenius_verification = verify_arrhenius_structure(
        model,
        pipeline["train_loader"],
        n_samples=verify_n_samples,
        required_fraction_in_range=(
            required_fraction_in_range if strict_parameter_ranges else 0.0
        ),
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
        hard_constraints=(),
        random_state=seed,
    )

    return model, history, evaluation_results, arrhenius_verification


# Run the full V6 Arrhenius experiment and save reusable artifacts.
def run_and_save_arrhenius_exp(
    batch_size: int = 64,
    model_config: Optional[ArrheniusConfig] = None,
    training_config: Optional[TrainingConfig] = None,
    tier_names: tuple[str, ...] = ("T1", "T2", "T3"),
    results_dir=Path("results"),
    checkpoint_dir=Path("checkpoints"),
    n_bootstrap: int = 100,
    n_mc: int = 100,
    seed: int = 42,
    verify_n_samples: int = 100,
    required_fraction_in_range: float = 0.95,
    strict_parameter_ranges: bool = False,
    verbose: bool = True,
) -> dict[str, object]:
    from unified_eval_harness import save_results_csv

    resolved_training_config = training_config or TrainingConfig()
    pipeline = build_data_pipeline(batch_size=batch_size)
    results_dir = ensure_step_results_dir(results_dir, step=7)
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    model, history, evaluation_results, arrhenius_verification = (
        evaluate_arrhenius_model(
            pipeline,
            model_config=model_config,
            training_config=resolved_training_config,
            tier_names=tier_names,
            n_bootstrap=n_bootstrap,
            n_mc=n_mc,
            seed=seed,
            verify_n_samples=verify_n_samples,
            required_fraction_in_range=required_fraction_in_range,
            strict_parameter_ranges=strict_parameter_ranges,
        )
    )

    history_path = results_dir / "arrhenius_training_history.csv"
    pd.DataFrame(history).to_csv(history_path, index=False)

    verification_path = results_dir / "arrhenius_verification.csv"
    pd.DataFrame([arrhenius_verification]).to_csv(verification_path, index=False)

    evaluation_metrics_path = results_dir / "arrhenius_eval_metrics.csv"
    evaluation_table = save_results_csv(
        evaluation_results,
        evaluation_metrics_path,
        model_name="V6_arrhenius",
    )

    enthalpy_summary = compute_enthalpy_by_rank(
        model,
        pipeline,
        device=resolved_training_config.device,
    )
    enthalpy_summary_path = results_dir / "arrhenius_enthalpy_by_rank.csv"
    enthalpy_summary.to_csv(enthalpy_summary_path, index=False)

    if strict_parameter_ranges:
        if (
            arrhenius_verification["fraction_delta_H_in_expected_range"]
            < required_fraction_in_range
        ):
            raise ValueError("V6 enthalpy range verification failed.")

    checkpoint_path = checkpoint_dir / "arrhenius_v6.pt"
    torch.save(
        {
            "model_state_dict": {
                name: value.detach().cpu() for name, value in model.state_dict().items()
            },
            "model_config": asdict(model.config),
            "training_config": asdict(resolved_training_config),
            "feature_names": pipeline["feature_names"],
            "t2_rank": pipeline["t2_rank"],
            "seed": seed,
            "hard_constraints": ["van_t_hoff"],
            "arrhenius_verification": arrhenius_verification,
            "strict_parameter_ranges": strict_parameter_ranges,
        },
        checkpoint_path,
    )

    if verbose:
        print(f"Saved V6 training history: {history_path}")
        print(f"Saved V6 Arrhenius verification: {verification_path}")
        print(f"Saved V6 evaluation metrics: {evaluation_metrics_path}")
        print(f"Saved V6 enthalpy summary: {enthalpy_summary_path}")
        print(f"Saved V6 checkpoint: {checkpoint_path}")

    results_dict = {"V6": evaluation_results}

    return {
        "model": model,
        "history": history,
        "evaluation_results": evaluation_results,
        "results_dict": results_dict,
        "evaluation_table": evaluation_table,
        "arrhenius_verification": arrhenius_verification,
        "enthalpy_summary": enthalpy_summary,
        "paths": {
            "training_history": history_path,
            "arrhenius_verification": verification_path,
            "evaluation_metrics": evaluation_metrics_path,
            "enthalpy_summary": enthalpy_summary_path,
            "checkpoint": checkpoint_path,
        },
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
    x = batch["x"].detach().clone().requires_grad_(True)
    output = model(x, batch["raw_pressure"])
    losses = compute_v6_losses(
        output,
        y_log=batch["y"],
        raw_pressure=batch["raw_pressure"],
        lambda_p=0.1,
        x=x,
        config=config,
    )
    raw_temperature_K = model.raw_temperature_from_scaled_x(x)
    verification = verify_arrhenius_structure(
        model,
        pipeline["train_loader"],
        n_samples=16,
        required_fraction_in_range=0.0,
    )
    enthalpy_summary = compute_enthalpy_by_rank(
        model,
        pipeline,
        split_names=("validation",),
    )
    training_config = TrainingConfig(lambda_p=0.1, max_epochs=1, patience=1)
    trained_model, history = train_v6_model(
        ArrheniusPINN(config),
        pipeline["train_loader"],
        pipeline["val_loader"],
        training_config,
    )

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
        float(raw_temperature_K.min().detach()),
        float(raw_temperature_K.max().detach()),
    )
    print("max delta_H_kJmol:", float(output["delta_H_kJmol"].max().detach()))
    print("total_loss:", float(losses["total_loss"].detach()))
    print("physics_loss:", float(losses["physics_loss"].detach()))
    print("max identity error:", verification["max_arrhenius_identity_error"])
    print("fraction dH in range:", verification["fraction_delta_H_in_expected_range"])
    print("enthalpy rows:", len(enthalpy_summary))
    print("epochs_trained:", len(history))
    print("val_total_loss:", history[-1]["val_total_loss"])
    print(
        "trained parameters:",
        sum(parameter.numel() for parameter in trained_model.parameters()),
    )


if __name__ == "__main__":
    smoke_test_arrhenius_architecture()
