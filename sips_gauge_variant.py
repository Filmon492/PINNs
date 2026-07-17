from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from data_pipeline import build_data_pipeline
from soft_baseline import (
    R_GAS,
    ResidualBlock,
    TrainingConfig,
    V1ModelConfig,
    heteroscedastic_nll_loss,
    set_torch_seed,
    summarize_validation_metrics,
)


@dataclass
class SipsGaugeConfig(V1ModelConfig):
    pressure_dependent_feature_indices: tuple[int, ...] = (4, 6, 10)
    q_max_range_m3t: tuple[float, float] = (10.0, 40.0)
    K_range_MPa_inv: tuple[float, float] = (0.1, 5.0)
    n_range: tuple[float, float] = (0.3, 2.0)


# V5 gauge model: composition predicts Sips parameters; pressure enters formula only.
class SipsGaugePINN(nn.Module):
    def __init__(self, config: Optional[SipsGaugeConfig] = None):
        super().__init__()
        self.config = config or SipsGaugeConfig()
        excluded_indices = set(self.config.pressure_dependent_feature_indices)
        self.comp_feature_indices = tuple(
            idx for idx in range(self.config.input_dim) if idx not in excluded_indices
        )

        if self.config.temperature_feature_idx not in self.comp_feature_indices:
            raise ValueError(
                "Temperature must be available in the composition encoder."
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

        self.encoder = nn.Sequential(*blocks)
        self.head_qmax = nn.Linear(current_dim, 1)
        self.head_K = nn.Linear(current_dim, 1)
        self.head_n = nn.Linear(current_dim, 1)
        self.head_logvar = nn.Linear(current_dim, 1)
        self.head_delta_H = nn.Linear(current_dim, 1)

    def split_composition(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] == len(self.comp_feature_indices):
            return x

        if x.shape[1] != self.config.input_dim:
            raise ValueError(
                f"Expected either {self.config.input_dim} full features or "
                f"{len(self.comp_feature_indices)} composition features."
            )

        return x[:, self.comp_feature_indices]

    def predict_parameters_from_comp(
        self,
        x_comp: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if x_comp.shape[1] != len(self.comp_feature_indices):
            raise ValueError(
                f"Expected {len(self.comp_feature_indices)} composition features."
            )

        eps = self.config.softplus_epsilon
        h = self.encoder(x_comp)
        q_max = F.softplus(self.head_qmax(h)) + eps
        K = F.softplus(self.head_K(h)) + eps
        n = F.softplus(self.head_n(h)) + eps
        log_sigma2 = torch.clamp(
            self.head_logvar(h),
            min=self.config.min_log_sigma2,
            max=self.config.max_log_sigma2,
        )
        sigma_ale = torch.exp(0.5 * log_sigma2)
        delta_H = -self.config.enthalpy_scale_j_per_mol
        delta_H = delta_H * (F.softplus(self.head_delta_H(h)) + eps)

        return {
            "q_max": q_max,
            "K": K,
            "n": n,
            "log_sigma2": log_sigma2,
            "sigma_ale": sigma_ale,
            "delta_H": delta_H,
        }

    def forward_from_composition(
        self,
        x_comp: torch.Tensor,
        raw_pressure: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if raw_pressure.ndim == 1:
            raw_pressure = raw_pressure.reshape(-1, 1)

        if raw_pressure.shape[1] != 1:
            raise ValueError("raw_pressure must have shape (batch_size, 1).")

        if not (raw_pressure > 0).all():
            raise ValueError("Sips requires P > 0.")

        output = self.predict_parameters_from_comp(x_comp)
        eps = self.config.softplus_epsilon
        log_KP_n = output["n"] * (
            torch.log(torch.clamp(output["K"], min=eps))
            + torch.log(torch.clamp(raw_pressure, min=eps))
        )
        log_KP_n = torch.clamp(log_KP_n, min=-50.0, max=50.0)
        saturation_fraction = torch.sigmoid(log_KP_n)
        saturation_fraction = torch.clamp(
            saturation_fraction,
            min=eps,
            max=1.0 - eps,
        )
        q_hat = output["q_max"] * saturation_fraction

        return {
            "q_hat": q_hat,
            "log_sigma2": output["log_sigma2"],
            "sigma_ale": output["sigma_ale"],
            "q_max": output["q_max"],
            "K": output["K"],
            "n": output["n"],
            "delta_H": output["delta_H"],
        }

    def forward(
        self,
        x: torch.Tensor,
        raw_pressure: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        x_comp = self.split_composition(x)

        return self.forward_from_composition(x_comp, raw_pressure)


# Build differentiable composition inputs for temperature-gradient penalties.
def prepare_v5_inputs(
    x: torch.Tensor,
    model: SipsGaugePINN,
) -> torch.Tensor:
    x_comp = model.split_composition(x).detach().clone()
    x_comp.requires_grad_(True)

    return x_comp


# Penalize deviations from the van't Hoff relation through K(T).
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


# Combine data loss with V5 soft van't Hoff penalty only.
def compute_v5_losses(
    output: dict[str, torch.Tensor],
    y_log: torch.Tensor,
    lambda_p: float,
    x_comp: torch.Tensor,
    model: SipsGaugePINN,
    temperature_center: float,
    temperature_scale: float,
) -> dict[str, torch.Tensor]:
    q_true = torch.exp(y_log)
    data_loss = heteroscedastic_nll_loss(
        q_true=q_true,
        q_hat=output["q_hat"],
        log_sigma2=output["log_sigma2"],
    )
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
    physics_loss = vH_penalty
    total_loss = data_loss + lambda_p * physics_loss

    return {
        "total_loss": total_loss,
        "data_loss": data_loss,
        "physics_loss": physics_loss,
        "vH_loss": vH_penalty,
    }


# Verify V5's structural bounds and monotonicity guarantees.
def verify_sips_structure(
    model: SipsGaugePINN,
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
    lower_violation_count = 0
    upper_violation_count = 0
    negative_gradient_count = 0
    min_q_hat = float("inf")
    min_qmax_minus_q = float("inf")
    min_gradient = float("inf")

    try:
        for start in range(0, n_selected, comp_batch_size):
            x_chunk = x_reference[start : start + comp_batch_size]
            x_comp = model.split_composition(x_chunk)
            chunk_size = len(x_comp)
            x_comp_grid = x_comp.repeat_interleave(n_P, dim=0)
            raw_pressure = pressure_values.repeat(chunk_size).reshape(-1, 1)
            raw_pressure = raw_pressure.detach().clone().requires_grad_(True)

            output = model.forward_from_composition(x_comp_grid, raw_pressure)
            q_hat = output["q_hat"]
            q_max = output["q_max"]
            dq_dP = torch.autograd.grad(
                q_hat.sum(),
                raw_pressure,
                create_graph=False,
                retain_graph=False,
            )[0]

            lower_violations = q_hat <= 0
            upper_violations = q_hat >= q_max
            negative_gradients = dq_dP < -tolerance

            total_count += int(q_hat.numel())
            lower_violation_count += int(lower_violations.sum().item())
            upper_violation_count += int(upper_violations.sum().item())
            negative_gradient_count += int(negative_gradients.sum().item())
            min_q_hat = min(min_q_hat, float(q_hat.min().detach().cpu().item()))
            qmax_minus_q = q_max - q_hat
            min_qmax_minus_q = min(
                min_qmax_minus_q,
                float(qmax_minus_q.min().detach().cpu().item()),
            )
            min_gradient = min(min_gradient, float(dq_dP.min().detach().cpu().item()))
    finally:
        model.train(was_training)

    metrics = {
        "n_comp": float(n_selected),
        "n_P": float(n_P),
        "min_q_hat": min_q_hat,
        "min_qmax_minus_q": min_qmax_minus_q,
        "min_dq_dP": min_gradient,
        "lower_violation_rate": lower_violation_count / total_count,
        "upper_violation_rate": upper_violation_count / total_count,
        "fraction_negative_dq_dP": negative_gradient_count / total_count,
        "max_monotonicity_violation_magnitude": max(0.0, -min_gradient),
    }

    if lower_violation_count or upper_violation_count or negative_gradient_count:
        raise ValueError(
            "Sips structural verification failed: "
            f"lower={lower_violation_count}, upper={upper_violation_count}, "
            f"negative gradients={negative_gradient_count}."
        )

    return metrics


# Run one V5 training pass over all batches.
def train_one_epoch_v5(
    model: SipsGaugePINN,
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
        x_comp = prepare_v5_inputs(batch["x"].to(device), model)
        y_log = batch["y"].to(device)
        raw_pressure = batch["raw_pressure"].to(device)

        output = model.forward_from_composition(x_comp, raw_pressure)
        losses = compute_v5_losses(
            output,
            y_log=y_log,
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


# Evaluate V5 on validation batches without updating model parameters.
def validate_one_epoch_v5(
    model: SipsGaugePINN,
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
        x_comp = prepare_v5_inputs(batch["x"].to(device), model)
        y_log = batch["y"].to(device)
        raw_pressure = batch["raw_pressure"].to(device)

        output = model.forward_from_composition(x_comp, raw_pressure)
        losses = compute_v5_losses(
            output,
            y_log=y_log,
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


# Train V5 across many epochs with validation tracking and early stopping.
def train_v5_model(
    model: SipsGaugePINN,
    train_loader,
    val_loader,
    training_config: TrainingConfig,
    temperature_center: float,
    temperature_scale: float,
) -> tuple[SipsGaugePINN, list[dict[str, float]]]:
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
        train_metrics = train_one_epoch_v5(
            model,
            train_loader,
            optimizer,
            training_config,
            temperature_center,
            temperature_scale,
        )
        val_metrics = validate_one_epoch_v5(
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
        raise ValueError("Training did not produce a valid V5 model state.")

    model.load_state_dict(best_state)

    return model, history


def _split_suffix(split_name: str) -> str:
    suffixes = {
        "train": "train",
        "validation": "val",
        "val": "val",
        "T1": "T1",
        "T2": "T2",
        "T3": "T3",
    }
    if split_name not in suffixes:
        raise KeyError(f"Unsupported split for parameter analysis: {split_name}.")

    return suffixes[split_name]


# Collect q_max, K, and n predictions together with inferred coal rank.
def collect_parameter_predictions(
    model: SipsGaugePINN,
    pipeline: dict,
    split_name: str,
    device: str = "cpu",
) -> pd.DataFrame:
    suffix = _split_suffix(split_name)
    arrays = pipeline["arrays"]
    split_key = "validation" if split_name == "val" else split_name
    split_df = pipeline["splits"][split_key].reset_index(drop=True)

    x = torch.tensor(arrays[f"X_{suffix}"], dtype=torch.float32, device=device)
    raw_pressure = torch.tensor(
        arrays[f"raw_pressure_{suffix}"],
        dtype=torch.float32,
        device=device,
    )

    model = model.to(device)
    was_training = model.training
    model.eval()

    try:
        with torch.inference_mode():
            output = model(x, raw_pressure)
    finally:
        model.train(was_training)

    if len(split_df) != output["q_max"].numel():
        raise ValueError("Split dataframe and model output lengths do not match.")

    return pd.DataFrame(
        {
            "split": split_key,
            "coal_rank": split_df["coal_rank_inferred"].to_numpy(),
            "q_max": output["q_max"].detach().cpu().numpy().reshape(-1),
            "K": output["K"].detach().cpu().numpy().reshape(-1),
            "n": output["n"].detach().cpu().numpy().reshape(-1),
        }
    )


# Summarize interpretable Sips parameters by coal rank.
def compute_parameter_analysis(
    model: SipsGaugePINN,
    pipeline: dict,
    split_names: tuple[str, ...] = ("train", "validation", "T1", "T2", "T3"),
    device: str = "cpu",
    strict_ranges: bool = False,
) -> pd.DataFrame:
    parameter_frames = [
        collect_parameter_predictions(model, pipeline, split_name, device=device)
        for split_name in split_names
    ]
    parameter_table = pd.concat(parameter_frames, ignore_index=True)
    summary = (
        parameter_table.groupby(["split", "coal_rank"], sort=True)
        .agg(
            n_samples=("q_max", "size"),
            q_max_mean=("q_max", "mean"),
            q_max_std=("q_max", "std"),
            K_mean=("K", "mean"),
            K_std=("K", "std"),
            n_mean=("n", "mean"),
            n_std=("n", "std"),
        )
        .reset_index()
    )
    qmax_min, qmax_max = model.config.q_max_range_m3t
    K_min, K_max = model.config.K_range_MPa_inv
    n_min, n_max = model.config.n_range
    summary["q_max_mean_in_range"] = summary["q_max_mean"].between(qmax_min, qmax_max)
    summary["K_mean_in_range"] = summary["K_mean"].between(K_min, K_max)
    summary["n_mean_in_range"] = summary["n_mean"].between(n_min, n_max)
    summary["all_means_in_range"] = (
        summary["q_max_mean_in_range"]
        & summary["K_mean_in_range"]
        & summary["n_mean_in_range"]
    )

    if strict_ranges and not summary["all_means_in_range"].all():
        failed = summary.loc[
            ~summary["all_means_in_range"],
            ["split", "coal_rank", "q_max_mean", "K_mean", "n_mean"],
        ]
        raise ValueError(
            "Some rank-level Sips parameter means are outside literature ranges:\n"
            f"{failed.to_string(index=False)}"
        )

    return summary


# Train V5 and evaluate it on T1/T2/T3 with the unified harness.
def evaluate_sips_gauge_model(
    pipeline: dict,
    model_config: Optional[SipsGaugeConfig] = None,
    training_config: Optional[TrainingConfig] = None,
    tier_names: tuple[str, ...] = ("T1", "T2", "T3"),
    n_bootstrap: int = 100,
    n_mc: int = 100,
    seed: int = 42,
    verify_n_comp: int = 500,
    verify_n_P: int = 50,
) -> tuple[SipsGaugePINN, list[dict[str, float]], dict[str, dict], dict]:
    from unified_eval_harness import evaluate_model

    resolved_model_config = model_config or SipsGaugeConfig()
    resolved_training_config = training_config or TrainingConfig()
    scaler = pipeline["scaler"]
    temperature_idx = resolved_model_config.temperature_feature_idx
    temperature_center = float(scaler.center_[temperature_idx])
    temperature_scale = float(scaler.scale_[temperature_idx])

    set_torch_seed(seed)
    model = SipsGaugePINN(resolved_model_config)
    model, history = train_v5_model(
        model,
        pipeline["train_loader"],
        pipeline["val_loader"],
        resolved_training_config,
        temperature_center,
        temperature_scale,
    )
    structural_verification = verify_sips_structure(
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

    return model, history, evaluation_results, structural_verification


# Run the full V5 Sips gauge experiment and save reusable artifacts.
def run_and_save_sips_gauge_exp(
    batch_size: int = 64,
    model_config: Optional[SipsGaugeConfig] = None,
    training_config: Optional[TrainingConfig] = None,
    tier_names: tuple[str, ...] = ("T1", "T2", "T3"),
    results_dir=Path("results"),
    checkpoint_dir=Path("checkpoints"),
    n_bootstrap: int = 100,
    n_mc: int = 100,
    seed: int = 42,
    verify_n_comp: int = 500,
    verify_n_P: int = 50,
    strict_parameter_ranges: bool = False,
    verbose: bool = True,
) -> dict[str, object]:
    from unified_eval_harness import save_results_csv

    resolved_model_config = model_config or SipsGaugeConfig()
    resolved_training_config = training_config or TrainingConfig()
    pipeline = build_data_pipeline(batch_size=batch_size)

    results_dir = Path(results_dir)
    checkpoint_dir = Path(checkpoint_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    model, history, evaluation_results, structural_verification = (
        evaluate_sips_gauge_model(
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

    history_path = results_dir / "sips_gauge_training_history.csv"
    pd.DataFrame(history).to_csv(history_path, index=False)

    verification_path = results_dir / "sips_gauge_verification.csv"
    pd.DataFrame([structural_verification]).to_csv(verification_path, index=False)

    evaluation_metrics_path = results_dir / "sips_gauge_eval_metrics.csv"
    evaluation_table = save_results_csv(
        evaluation_results,
        evaluation_metrics_path,
        model_name="V5_sips_gauge",
    )

    parameter_analysis = compute_parameter_analysis(
        model,
        pipeline,
        device=resolved_training_config.device,
        strict_ranges=strict_parameter_ranges,
    )
    parameter_analysis_path = results_dir / "sips_gauge_parameter_analysis.csv"
    parameter_analysis.to_csv(parameter_analysis_path, index=False)

    checkpoint_path = checkpoint_dir / "sips_gauge_v5.pt"
    torch.save(
        {
            "model_state_dict": {
                name: value.detach().cpu() for name, value in model.state_dict().items()
            },
            "model_config": asdict(resolved_model_config),
            "training_config": asdict(resolved_training_config),
            "feature_names": pipeline["feature_names"],
            "composition_feature_names": [
                pipeline["feature_names"][idx] for idx in model.comp_feature_indices
            ],
            "t2_rank": pipeline["t2_rank"],
            "seed": seed,
            "hard_constraints": ["nonnegative", "saturation", "monotonicity"],
            "structural_verification": structural_verification,
            "strict_parameter_ranges": strict_parameter_ranges,
        },
        checkpoint_path,
    )

    if verbose:
        print(f"Saved V5 training history: {history_path}")
        print(f"Saved V5 structural verification: {verification_path}")
        print(f"Saved V5 evaluation metrics: {evaluation_metrics_path}")
        print(f"Saved V5 parameter analysis: {parameter_analysis_path}")
        print(f"Saved V5 checkpoint: {checkpoint_path}")

    results_dict = {"V5": evaluation_results}
    paths = {
        "training_history": history_path,
        "structural_verification": verification_path,
        "evaluation_metrics": evaluation_metrics_path,
        "parameter_analysis": parameter_analysis_path,
        "checkpoint": checkpoint_path,
    }

    return {
        "model": model,
        "history": history,
        "evaluation_results": evaluation_results,
        "results_dict": results_dict,
        "evaluation_table": evaluation_table,
        "structural_verification": structural_verification,
        "parameter_analysis": parameter_analysis,
        "paths": paths,
    }


# Quick architecture, loss, structural, and one-epoch training check.
def smoke_test_sips_gauge() -> None:
    pipeline = build_data_pipeline(batch_size=8)
    batch = next(iter(pipeline["train_loader"]))
    model = SipsGaugePINN()
    training_config = TrainingConfig(lambda_p=0.1, max_epochs=1, patience=1)
    temperature_center = float(
        pipeline["scaler"].center_[model.config.temperature_feature_idx]
    )
    temperature_scale = float(
        pipeline["scaler"].scale_[model.config.temperature_feature_idx]
    )

    x_comp = prepare_v5_inputs(batch["x"], model)
    output = model.forward_from_composition(x_comp, batch["raw_pressure"])
    losses = compute_v5_losses(
        output,
        y_log=batch["y"],
        lambda_p=0.1,
        x_comp=x_comp,
        model=model,
        temperature_center=temperature_center,
        temperature_scale=temperature_scale,
    )
    verification = verify_sips_structure(
        model,
        pipeline["train_loader"],
        n_comp=16,
        n_P=8,
    )
    trained_model, history = train_v5_model(
        SipsGaugePINN(),
        pipeline["train_loader"],
        pipeline["val_loader"],
        training_config,
        temperature_center,
        temperature_scale,
    )
    trained_verification = verify_sips_structure(
        trained_model,
        pipeline["train_loader"],
        n_comp=16,
        n_P=8,
    )
    parameter_analysis = compute_parameter_analysis(
        trained_model,
        pipeline,
        split_names=("validation",),
    )

    print("x:", batch["x"].shape)
    print("x_comp:", x_comp.shape)
    print("q_hat:", output["q_hat"].shape)
    print("composition features:", len(model.comp_feature_indices))
    print("total_loss:", float(losses["total_loss"].detach()))
    print("min_q_hat:", verification["min_q_hat"])
    print("min_qmax_minus_q:", verification["min_qmax_minus_q"])
    print("fraction_negative_dq_dP:", verification["fraction_negative_dq_dP"])
    print("epochs_trained:", len(history))
    print("val_total_loss:", history[-1]["val_total_loss"])
    print("trained_min_dq_dP:", trained_verification["min_dq_dP"])
    print("parameter_analysis_rows:", len(parameter_analysis))
    print("Sips gauge structure verified before and after one training epoch.")


if __name__ == "__main__":
    smoke_test_sips_gauge()
