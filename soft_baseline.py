from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from data_pipeline import build_data_pipeline

R_GAS = 8.314
DEFAULT_LAMBDA_P_VALUES = (0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0)
DEFAULT_RANDOM_SEEDS = (0, 1, 2)


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


# Combine validation accuracy, physical violations, and optional UQ metrics.
def summarize_validation_metrics(
    model: SoftBaselinePINN,
    val_loader,
    device: str,
    n_mc: int = 0,
) -> dict[str, dict[str, float]]:
    from unified_eval_harness import (
        collect_mc_dropout_predictions,
        collect_predictions,
        compute_accuracy_metrics,
        compute_physical_violation_metrics,
        compute_uq_metrics,
    )

    predictions = collect_predictions(
        model,
        val_loader,
        device=device,
    )
    accuracy_metrics = compute_accuracy_metrics(
        predictions["q_true"],
        predictions["q_hat"],
    )
    physical_metrics = compute_physical_violation_metrics(predictions)

    finite_violation_rates = [
        value for value in physical_metrics.values() if pd.notna(value)
    ]
    if finite_violation_rates:
        physical_metrics["violation_rate"] = float(max(finite_violation_rates))
    else:
        physical_metrics["violation_rate"] = float("nan")

    uq_metrics = {}
    if n_mc >= 2:
        mc_predictions = collect_mc_dropout_predictions(
            model,
            val_loader,
            device=device,
            n_mc=n_mc,
        )
        uq_metrics = compute_uq_metrics(
            predictions["q_true"],
            mc_predictions["q_hat_mc"],
            mc_predictions.get("sigma_ale_mc"),
        )

    return {
        "accuracy": accuracy_metrics,
        "physical": physical_metrics,
        "uq": uq_metrics,
    }


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
        raise ValueError("Training did not produce a valid model state.")

    model.load_state_dict(best_state)

    return model, history


# Train one V1 model per lambda_p value and select by validation RMSE.
def run_lambda_sweep(
    train_loader,
    val_loader,
    temperature_center: float,
    temperature_scale: float,
    lambda_p_values: tuple[float, ...] = DEFAULT_LAMBDA_P_VALUES,
    model_config: Optional[V1ModelConfig] = None,
    training_config: Optional[TrainingConfig] = None,
    seed: int = 42,
    sweep_n_mc: int = 20,
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

        model, history = train_v1_model(
            model,
            train_loader,
            val_loader,
            current_training_config,
            temperature_center,
            temperature_scale,
        )

        best_epoch = int(history[-1]["best_epoch"])
        best_row = history[best_epoch - 1]
        val_summary = summarize_validation_metrics(
            model,
            val_loader,
            device=current_training_config.device,
            n_mc=sweep_n_mc,
        )
        val_accuracy = val_summary["accuracy"]
        val_physical = val_summary["physical"]
        val_uq = val_summary["uq"]
        results.append(
            {
                "lambda_p": float(lambda_p),
                "epochs_trained": float(len(history)),
                "best_epoch": best_row["epoch"],
                "selection_val_RMSE": val_accuracy["RMSE"],
                "selection_val_MAE": val_accuracy["MAE"],
                "selection_val_R2": val_accuracy["R2"],
                "selection_val_neg_rate": val_physical["neg_rate"],
                "selection_val_mono_rate": val_physical["mono_rate"],
                "selection_val_sat_rate": val_physical["sat_rate"],
                "selection_val_violation_rate": val_physical["violation_rate"],
                "selection_val_ECE": val_uq.get("ECE", float("nan")),
                "best_val_loss": best_row["val_total_loss"],
                "best_val_data_loss": best_row["val_data_loss"],
                "best_val_physics_loss": best_row["val_physics_loss"],
                "best_val_sips_loss": best_row["val_sips_loss"],
                "best_val_bounds_loss": best_row["val_bounds_loss"],
                "best_val_mono_loss": best_row["val_mono_loss"],
                "best_val_vH_loss": best_row["val_vH_loss"],
            }
        )

    best_result = min(results, key=lambda row: row["selection_val_RMSE"])

    return results, best_result


# Aggregate fold/seed sweep results and select lambda_p by mean validation RMSE.
def aggregate_cv_lambda_results(
    cv_results: list[dict[str, float]],
) -> tuple[list[dict[str, float]], dict[str, float]]:
    if not cv_results:
        raise ValueError("cv_results must contain at least one result row.")

    results_table = pd.DataFrame(cv_results)
    summary_rows: list[dict[str, float]] = []
    metrics_to_summarize = [
        "selection_val_RMSE",
        "selection_val_MAE",
        "selection_val_R2",
        "selection_val_neg_rate",
        "selection_val_mono_rate",
        "selection_val_sat_rate",
        "selection_val_violation_rate",
        "selection_val_ECE",
        "best_val_loss",
        "best_val_data_loss",
        "best_val_physics_loss",
        "epochs_trained",
    ]

    for lambda_p, group in results_table.groupby("lambda_p", sort=True):
        summary_row = {
            "lambda_p": float(lambda_p),
            "n_runs": float(len(group)),
        }

        for metric_name in metrics_to_summarize:
            summary_row[f"{metric_name}_mean"] = float(group[metric_name].mean())
            if len(group) > 1:
                summary_row[f"{metric_name}_std"] = float(group[metric_name].std())
            else:
                summary_row[f"{metric_name}_std"] = 0.0

        summary_rows.append(summary_row)

    best_summary = min(
        summary_rows,
        key=lambda row: row["selection_val_RMSE_mean"],
    )

    return summary_rows, best_summary


# Run the blueprint lambda_p sweep across CV folds and random seeds.
def run_cv_lambda_sweep(
    pipeline: dict,
    lambda_p_values: tuple[float, ...] = DEFAULT_LAMBDA_P_VALUES,
    seeds: tuple[int, ...] = DEFAULT_RANDOM_SEEDS,
    model_config: Optional[V1ModelConfig] = None,
    training_config: Optional[TrainingConfig] = None,
    max_folds: Optional[int] = None,
    sweep_n_mc: int = 20,
) -> tuple[list[dict[str, float]], list[dict[str, float]], dict[str, float]]:
    if not seeds:
        raise ValueError("seeds must contain at least one random seed.")

    if "cv_folds" not in pipeline:
        raise KeyError("Pipeline does not contain cv_folds.")

    cv_folds = pipeline["cv_folds"]
    if max_folds is not None:
        cv_folds = cv_folds[:max_folds]

    if not cv_folds:
        raise ValueError("No CV folds are available for the sweep.")

    resolved_model_config = model_config or V1ModelConfig()
    resolved_training_config = training_config or TrainingConfig()
    cv_results: list[dict[str, float]] = []

    for seed in seeds:
        for fold in cv_folds:
            scaler = fold["scaler"]
            temperature_idx = resolved_model_config.temperature_feature_idx
            temperature_center = float(scaler.center_[temperature_idx])
            temperature_scale = float(scaler.scale_[temperature_idx])
            fold_results, _ = run_lambda_sweep(
                train_loader=fold["train_loader"],
                val_loader=fold["val_loader"],
                temperature_center=temperature_center,
                temperature_scale=temperature_scale,
                lambda_p_values=lambda_p_values,
                model_config=resolved_model_config,
                training_config=resolved_training_config,
                seed=seed,
                sweep_n_mc=sweep_n_mc,
            )

            for result in fold_results:
                cv_results.append(
                    {
                        "seed": float(seed),
                        "fold": float(fold["fold"]),
                        **result,
                    }
                )

    cv_summary, best_summary = aggregate_cv_lambda_results(cv_results)

    return cv_results, cv_summary, best_summary


# Build the data pipeline and run the V1 soft-baseline lambda_p experiment.
def run_soft_baseline_exp(
    batch_size: int = 64,
    lambda_p_values: tuple[float, ...] = DEFAULT_LAMBDA_P_VALUES,
    model_config: Optional[V1ModelConfig] = None,
    training_config: Optional[TrainingConfig] = None,
    seed: int = 42,
    sweep_n_mc: int = 20,
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
        sweep_n_mc=sweep_n_mc,
    )

    if verbose:
        print("lambda_p sweep results")
        print("lambda_p | epochs | best_epoch | val_RMSE | violation | ECE | val_total")
        for row in results:
            print(
                f"{row['lambda_p']:>8.3g} | "
                f"{row['epochs_trained']:>6.0f} | "
                f"{row['best_epoch']:>10.0f} | "
                f"{row['selection_val_RMSE']:>8.4g} | "
                f"{row['selection_val_violation_rate']:>9.4g} | "
                f"{row['selection_val_ECE']:>3.4g} | "
                f"{row['best_val_loss']:>9.4g}"
            )

        print()
        print(f"Best lambda_p: {best_result['lambda_p']}")
        print(f"Best validation RMSE: {best_result['selection_val_RMSE']:.6g}")

    return results, best_result


# Save the lambda_p sensitivity trade-off plot requested by the blueprint.
def save_lambda_tradeoff_plot(
    cv_summary: list[dict[str, float]],
    output_path: Path,
) -> Path:
    import os

    summary_table = pd.DataFrame(cv_summary).sort_values("lambda_p")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(output_path.parent / ".matplotlib"))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x_positions = list(range(len(summary_table)))
    x_labels = [f"{value:g}" for value in summary_table["lambda_p"]]

    fig, left_axis = plt.subplots(figsize=(8, 4.5))

    left_axis.plot(
        x_positions,
        summary_table["selection_val_RMSE_mean"],
        marker="o",
        color="tab:blue",
        label="Validation RMSE",
    )

    violation_column = "selection_val_violation_rate_mean"
    if violation_column not in summary_table.columns:
        physical_columns = [
            column_name
            for column_name in (
                "selection_val_neg_rate_mean",
                "selection_val_mono_rate_mean",
                "selection_val_sat_rate_mean",
            )
            if column_name in summary_table.columns
        ]
        if physical_columns:
            violation_column = "_available_violation_rate_mean"
            summary_table[violation_column] = summary_table[physical_columns].max(
                axis=1
            )

    right_axis = None
    if violation_column in summary_table.columns:
        right_axis = left_axis.twinx()
        right_axis.plot(
            x_positions,
            summary_table[violation_column],
            marker="s",
            color="tab:red",
            label="Violation rate",
        )
        right_axis.set_ylabel("Physical violation rate", color="tab:red")
    else:
        left_axis.text(
            0.02,
            0.95,
            "Physical violation rate unavailable in this summary",
            transform=left_axis.transAxes,
            va="top",
            color="tab:red",
            fontsize=9,
        )

    left_axis.set_xlabel("lambda_p")
    left_axis.set_ylabel("Validation RMSE", color="tab:blue")
    left_axis.set_xticks(x_positions)
    left_axis.set_xticklabels(x_labels)
    left_axis.grid(True, axis="y", alpha=0.3)

    handles_left, labels_left = left_axis.get_legend_handles_labels()
    if right_axis is None:
        handles_right, labels_right = [], []
    else:
        handles_right, labels_right = right_axis.get_legend_handles_labels()
    left_axis.legend(
        handles_left + handles_right,
        labels_left + labels_right,
        loc="best",
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    return output_path


# Train the selected soft baseline and evaluate it on T1/T2/T3 with the harness.
def evaluate_best_soft_baseline(
    pipeline: dict,
    best_result: dict[str, float],
    model_config: Optional[V1ModelConfig] = None,
    training_config: Optional[TrainingConfig] = None,
    tier_names: tuple[str, ...] = ("T1", "T2", "T3"),
    n_bootstrap: int = 100,
    n_mc: int = 100,
    seed: int = 42,
) -> tuple[SoftBaselinePINN, list[dict[str, float]], dict[str, dict]]:
    from unified_eval_harness import evaluate_model

    resolved_model_config = model_config or V1ModelConfig()
    resolved_training_config = training_config or TrainingConfig()
    selected_training_config = replace(
        resolved_training_config,
        lambda_p=float(best_result["lambda_p"]),
    )
    scaler = pipeline["scaler"]
    temperature_idx = resolved_model_config.temperature_feature_idx
    temperature_center = float(scaler.center_[temperature_idx])
    temperature_scale = float(scaler.scale_[temperature_idx])

    set_torch_seed(seed)
    model = SoftBaselinePINN(resolved_model_config)
    model, history = train_v1_model(
        model,
        pipeline["train_loader"],
        pipeline["val_loader"],
        selected_training_config,
        temperature_center,
        temperature_scale,
    )
    evaluation_results = evaluate_model(
        model,
        pipeline,
        tier_names=tier_names,
        device=selected_training_config.device,
        n_bootstrap=n_bootstrap,
        n_mc=n_mc,
        random_state=seed,
    )

    return model, history, evaluation_results


# Run the full V1 soft-baseline experiment and save reusable artifacts.
def run_and_save_soft_baseline_exp(
    batch_size: int = 64,
    lambda_p_values: tuple[float, ...] = DEFAULT_LAMBDA_P_VALUES,
    seeds: tuple[int, ...] = DEFAULT_RANDOM_SEEDS,
    model_config: Optional[V1ModelConfig] = None,
    training_config: Optional[TrainingConfig] = None,
    tier_names: tuple[str, ...] = ("T1", "T2", "T3"),
    results_dir=Path("results"),
    checkpoint_dir=Path("checkpoints"),
    n_bootstrap: int = 100,
    n_mc: int = 100,
    sweep_n_mc: int = 20,
    seed: int = 42,
    max_folds: Optional[int] = None,
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

    cv_results, cv_summary, best_result = run_cv_lambda_sweep(
        pipeline=pipeline,
        lambda_p_values=lambda_p_values,
        seeds=seeds,
        model_config=resolved_model_config,
        training_config=resolved_training_config,
        max_folds=max_folds,
        sweep_n_mc=sweep_n_mc,
    )
    cv_sweep_path = results_dir / "soft_baseline_cv_lambda_sweep.csv"
    cv_summary_path = results_dir / "soft_baseline_cv_lambda_summary.csv"
    tradeoff_plot_path = results_dir / "soft_baseline_lambda_tradeoff.png"
    pd.DataFrame(cv_results).to_csv(cv_sweep_path, index=False)
    pd.DataFrame(cv_summary).to_csv(cv_summary_path, index=False)
    save_lambda_tradeoff_plot(cv_summary, tradeoff_plot_path)

    model, history, evaluation_results = evaluate_best_soft_baseline(
        pipeline,
        best_result,
        model_config=resolved_model_config,
        training_config=resolved_training_config,
        tier_names=tier_names,
        n_bootstrap=n_bootstrap,
        n_mc=n_mc,
        seed=seed,
    )

    history_path = results_dir / "soft_baseline_training_history.csv"
    pd.DataFrame(history).to_csv(history_path, index=False)

    evaluation_metrics_path = results_dir / "soft_baseline_eval_metrics.csv"
    evaluation_table = save_results_csv(
        evaluation_results,
        evaluation_metrics_path,
        model_name="V1_soft_baseline",
    )

    selected_training_config = replace(
        resolved_training_config,
        lambda_p=float(best_result["lambda_p"]),
    )
    checkpoint_path = checkpoint_dir / "soft_baseline_v1.pt"
    torch.save(
        {
            "model_state_dict": {
                name: value.detach().cpu() for name, value in model.state_dict().items()
            },
            "model_config": asdict(resolved_model_config),
            "training_config": asdict(selected_training_config),
            "best_result": best_result,
            "feature_names": pipeline["feature_names"],
            "t2_rank": pipeline["t2_rank"],
            "lambda_p_values": list(lambda_p_values),
            "seeds": list(seeds),
            "max_folds": max_folds,
            "sweep_n_mc": sweep_n_mc,
            "seed": seed,
        },
        checkpoint_path,
    )

    if verbose:
        print(f"Saved CV lambda sweep: {cv_sweep_path}")
        print(f"Saved CV lambda summary: {cv_summary_path}")
        print(f"Saved lambda trade-off plot: {tradeoff_plot_path}")
        print(f"Saved training history: {history_path}")
        print(f"Saved evaluation metrics: {evaluation_metrics_path}")
        print(f"Saved checkpoint: {checkpoint_path}")

    return {
        "model": model,
        "cv_results": cv_results,
        "cv_summary": cv_summary,
        "best_result": best_result,
        "history": history,
        "evaluation_results": evaluation_results,
        "evaluation_table": evaluation_table,
        "paths": {
            "cv_lambda_sweep": cv_sweep_path,
            "cv_lambda_summary": cv_summary_path,
            "lambda_tradeoff_plot": tradeoff_plot_path,
            "training_history": history_path,
            "evaluation_metrics": evaluation_metrics_path,
            "checkpoint": checkpoint_path,
        },
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
