from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from arrhenius_variant import (
    ArrheniusConfig,
    ArrheniusHead,
    compute_enthalpy_by_rank,
    verify_arrhenius_structure,
)
from data_pipeline import build_data_pipeline
from sips_gauge_variant import verify_sips_structure
from soft_baseline import (
    ResidualBlock,
    TrainingConfig,
    heteroscedastic_nll_loss,
    summarize_validation_metrics,
)


@dataclass
class AllHardConfig(ArrheniusConfig):
    pressure_dependent_feature_indices: tuple[int, ...] = (4, 6, 10)


# Encoder that removes pressure-derived features before the neural network.
class CompositionEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        comp_feature_indices: tuple[int, ...],
        hidden_dims: tuple[int, ...],
        dropout: float,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.comp_feature_indices = comp_feature_indices

        blocks = []
        current_dim = len(comp_feature_indices)

        for hidden_dim in hidden_dims:
            blocks.append(
                ResidualBlock(
                    input_dim=current_dim,
                    output_dim=hidden_dim,
                    dropout=dropout,
                )
            )
            current_dim = hidden_dim

        self.network = nn.Sequential(*blocks)
        self.output_dim = current_dim

    def split_composition(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] == len(self.comp_feature_indices):
            return x

        if x.shape[1] != self.input_dim:
            raise ValueError(
                f"Expected either {self.input_dim} full features or "
                f"{len(self.comp_feature_indices)} composition features."
            )

        return x[:, self.comp_feature_indices]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_comp = self.split_composition(x)

        return self.network(x_comp)


# Shared V2/V8 model: Sips gauge output with Arrhenius K(T).
class AllHardSipsArrheniusPINN(nn.Module):
    def __init__(self, config: Optional[AllHardConfig] = None):
        super().__init__()
        self.config = config or AllHardConfig()
        excluded_indices = set(self.config.pressure_dependent_feature_indices)
        self.comp_feature_indices = tuple(
            idx for idx in range(self.config.input_dim) if idx not in excluded_indices
        )

        if self.config.temperature_feature_idx not in self.comp_feature_indices:
            raise ValueError(
                "Temperature must be available in the composition encoder."
            )

        self.encoder = CompositionEncoder(
            input_dim=self.config.input_dim,
            comp_feature_indices=self.comp_feature_indices,
            hidden_dims=self.config.hidden_dims,
            dropout=self.config.dropout,
        )
        hidden_dim = self.encoder.output_dim

        self.head_qmax = nn.Linear(hidden_dim, 1)
        self.head_n = nn.Linear(hidden_dim, 1)
        self.head_logvar = nn.Linear(hidden_dim, 1)
        self.arrhenius_head = ArrheniusHead(hidden_dim, self.config)

    def split_composition(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder.split_composition(x)

    def raw_temperature_from_scaled_x(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] == self.config.input_dim:
            temperature_idx = self.config.temperature_feature_idx
            scaled_temperature = x[
                :,
                temperature_idx : temperature_idx + 1,
            ]
        elif x.shape[1] == len(self.comp_feature_indices):
            temperature_comp_idx = self.comp_feature_indices.index(
                self.config.temperature_feature_idx
            )
            scaled_temperature = x[:, temperature_comp_idx : temperature_comp_idx + 1]
        else:
            raise ValueError("x must contain either full or composition features.")

        return (
            scaled_temperature * self.config.temperature_scale
            + self.config.temperature_center
        )

    def predict_parameters_from_comp(
        self,
        x_comp: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        eps = self.config.softplus_epsilon
        h = self.encoder(x_comp)
        q_max = F.softplus(self.head_qmax(h)) + eps
        n = F.softplus(self.head_n(h)) + eps
        log_sigma2 = torch.clamp(
            self.head_logvar(h),
            min=self.config.min_log_sigma2,
            max=self.config.max_log_sigma2,
        )
        sigma_ale = torch.exp(0.5 * log_sigma2)
        raw_temperature_K = self.raw_temperature_from_scaled_x(x_comp)
        arrhenius_output = self.arrhenius_head(h, raw_temperature_K)

        return {
            "q_max": q_max,
            "K": arrhenius_output["K"],
            "K0": arrhenius_output["K0"],
            "log_K": arrhenius_output["log_K"],
            "log_K_unclamped": arrhenius_output["log_K_unclamped"],
            "n": n,
            "log_sigma2": log_sigma2,
            "sigma_ale": sigma_ale,
            "delta_H": arrhenius_output["delta_H"],
            "delta_H_kJmol": arrhenius_output["delta_H_kJmol"],
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

        # Use float64 in the Sips gauge to avoid false decreases near saturation.
        q_max = output["q_max"].to(torch.float64)
        log_K = output["log_K_unclamped"].to(torch.float64)
        n = output["n"].to(torch.float64)
        pressure = raw_pressure.to(torch.float64)
        log_KP_n = n * (log_K + torch.log(torch.clamp(pressure, min=eps)))
        log_KP_n = torch.clamp(
            log_KP_n,
            min=self.config.min_arrhenius_exponent,
            max=self.config.max_arrhenius_exponent,
        )
        saturation_fraction = torch.sigmoid(log_KP_n)
        saturation_fraction = torch.clamp(
            saturation_fraction,
            min=eps,
            max=1.0 - eps,
        )
        q_hat = q_max * saturation_fraction

        return {
            "q_hat": q_hat,
            "log_sigma2": output["log_sigma2"],
            "sigma_ale": output["sigma_ale"],
            "q_max": q_max,
            "K": output["K"],
            "K0": output["K0"],
            "log_K": output["log_K"],
            "log_K_unclamped": output["log_K_unclamped"],
            "n": output["n"],
            "delta_H": output["delta_H"],
            "delta_H_kJmol": output["delta_H_kJmol"],
        }

    def forward(
        self,
        x: torch.Tensor,
        raw_pressure: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        x_comp = self.split_composition(x)

        return self.forward_from_composition(x_comp, raw_pressure)


V2AllHardPINN = AllHardSipsArrheniusPINN
V8HardSipsArrheniusPINN = AllHardSipsArrheniusPINN


# Compute the all-hard objective; physics constraints are enforced by architecture.
def compute_all_hard_losses(
    output: dict[str, torch.Tensor],
    y_log: torch.Tensor,
) -> dict[str, torch.Tensor]:
    q_true = torch.exp(y_log)
    data_loss = heteroscedastic_nll_loss(
        q_true=q_true,
        q_hat=output["q_hat"],
        log_sigma2=output["log_sigma2"],
    )

    return {
        "total_loss": data_loss,
        "data_loss": data_loss,
    }


# Run one all-hard training pass over all batches.
def train_one_epoch_all_hard(
    model: AllHardSipsArrheniusPINN,
    train_loader,
    optimizer: torch.optim.Optimizer,
    training_config: TrainingConfig,
) -> dict[str, float]:
    model.train()
    device = torch.device(training_config.device)
    loss_totals: dict[str, float] = {}
    n_batches = 0

    for batch in train_loader:
        x = batch["x"].to(device)
        y_log = batch["y"].to(device)
        raw_pressure = batch["raw_pressure"].to(device)

        output = model(x, raw_pressure)
        losses = compute_all_hard_losses(output, y_log)

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


# Evaluate all-hard validation batches without updating model parameters.
def validate_one_epoch_all_hard(
    model: AllHardSipsArrheniusPINN,
    val_loader,
    training_config: TrainingConfig,
) -> dict[str, float]:
    model.eval()
    device = torch.device(training_config.device)
    loss_totals: dict[str, float] = {}
    n_batches = 0

    with torch.inference_mode():
        for batch in val_loader:
            x = batch["x"].to(device)
            y_log = batch["y"].to(device)
            raw_pressure = batch["raw_pressure"].to(device)

            output = model(x, raw_pressure)
            losses = compute_all_hard_losses(output, y_log)

            for name, value in losses.items():
                loss_totals[name] = loss_totals.get(name, 0.0) + float(value.detach())

            n_batches += 1

    if n_batches == 0:
        raise ValueError("val_loader produced no batches.")

    return {name: value / n_batches for name, value in loss_totals.items()}


# Train the shared V2/V8 model with validation tracking and early stopping.
def train_all_hard_model(
    model: AllHardSipsArrheniusPINN,
    train_loader,
    val_loader,
    training_config: TrainingConfig,
) -> tuple[AllHardSipsArrheniusPINN, list[dict[str, float]]]:
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
        train_metrics = train_one_epoch_all_hard(
            model,
            train_loader,
            optimizer,
            training_config,
        )
        val_metrics = validate_one_epoch_all_hard(
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
        raise ValueError("Training did not produce a valid all-hard model state.")

    model.load_state_dict(best_state)

    return model, history


# Train the shared V2/V8 model and evaluate it on all requested tiers.
def evaluate_all_hard_model(
    pipeline: dict,
    model_config: Optional[AllHardConfig] = None,
    training_config: Optional[TrainingConfig] = None,
    tier_names: tuple[str, ...] = ("T1", "T2", "T3"),
    n_bootstrap: int = 100,
    n_mc: int = 100,
    seed: int = 42,
    verify_n_comp: int = 500,
    verify_n_P: int = 50,
    verify_n_samples: int = 100,
    required_fraction_in_range: float = 0.95,
    strict_parameter_ranges: bool = False,
) -> tuple[
    AllHardSipsArrheniusPINN,
    list[dict[str, float]],
    dict[str, dict],
    dict[str, float],
    dict[str, float],
]:
    from soft_baseline import set_torch_seed
    from unified_eval_harness import evaluate_model

    base_model_config = model_config or AllHardConfig()
    scaler = pipeline["scaler"]
    temperature_idx = base_model_config.temperature_feature_idx
    resolved_model_config = AllHardConfig(
        **{
            **asdict(base_model_config),
            "temperature_center": float(scaler.center_[temperature_idx]),
            "temperature_scale": float(scaler.scale_[temperature_idx]),
        }
    )
    resolved_training_config = training_config or TrainingConfig()

    set_torch_seed(seed)
    model = AllHardSipsArrheniusPINN(resolved_model_config)
    model, history = train_all_hard_model(
        model,
        pipeline["train_loader"],
        pipeline["val_loader"],
        resolved_training_config,
    )
    sips_verification = verify_sips_structure(
        model,
        pipeline["train_loader"],
        n_comp=verify_n_comp,
        n_P=verify_n_P,
        device=resolved_training_config.device,
        random_state=seed,
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
        hard_constraints=("nonnegative", "saturation", "monotonicity"),
        random_state=seed,
    )

    return model, history, evaluation_results, sips_verification, arrhenius_verification


# Run the full shared V2/V8 experiment and save reusable artifacts.
def run_and_save_all_hard_exp(
    batch_size: int = 64,
    model_config: Optional[AllHardConfig] = None,
    training_config: Optional[TrainingConfig] = None,
    tier_names: tuple[str, ...] = ("T1", "T2", "T3"),
    results_dir=Path("results"),
    checkpoint_dir=Path("checkpoints"),
    n_bootstrap: int = 100,
    n_mc: int = 100,
    seed: int = 42,
    verify_n_comp: int = 500,
    verify_n_P: int = 50,
    verify_n_samples: int = 100,
    required_fraction_in_range: float = 0.95,
    strict_parameter_ranges: bool = False,
    verbose: bool = True,
) -> dict[str, object]:
    from unified_eval_harness import save_results_csv

    resolved_training_config = training_config or TrainingConfig()
    pipeline = build_data_pipeline(batch_size=batch_size)

    results_dir = Path(results_dir)
    checkpoint_dir = Path(checkpoint_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    (
        model,
        history,
        evaluation_results,
        sips_verification,
        arrhenius_verification,
    ) = evaluate_all_hard_model(
        pipeline,
        model_config=model_config,
        training_config=resolved_training_config,
        tier_names=tier_names,
        n_bootstrap=n_bootstrap,
        n_mc=n_mc,
        seed=seed,
        verify_n_comp=verify_n_comp,
        verify_n_P=verify_n_P,
        verify_n_samples=verify_n_samples,
        required_fraction_in_range=required_fraction_in_range,
        strict_parameter_ranges=strict_parameter_ranges,
    )

    history_path = results_dir / "all_hard_training_history.csv"
    pd.DataFrame(history).to_csv(history_path, index=False)

    sips_verification_path = results_dir / "all_hard_sips_verification.csv"
    pd.DataFrame([sips_verification]).to_csv(sips_verification_path, index=False)

    arrhenius_verification_path = results_dir / "all_hard_arrhenius_verification.csv"
    pd.DataFrame([arrhenius_verification]).to_csv(
        arrhenius_verification_path,
        index=False,
    )

    evaluation_metrics_path = results_dir / "all_hard_eval_metrics.csv"
    evaluation_table = save_results_csv(
        evaluation_results,
        evaluation_metrics_path,
        model_name="V2_V8_all_hard",
    )

    enthalpy_summary = compute_enthalpy_by_rank(
        model,
        pipeline,
        device=resolved_training_config.device,
    )
    enthalpy_summary_path = results_dir / "all_hard_enthalpy_by_rank.csv"
    enthalpy_summary.to_csv(enthalpy_summary_path, index=False)

    if strict_parameter_ranges:
        if (
            arrhenius_verification["fraction_delta_H_in_expected_range"]
            < required_fraction_in_range
        ):
            raise ValueError("All-hard enthalpy range verification failed.")

    checkpoint_path = checkpoint_dir / "all_hard_v2_v8.pt"
    torch.save(
        {
            "model_state_dict": {
                name: value.detach().cpu() for name, value in model.state_dict().items()
            },
            "model_config": asdict(model.config),
            "training_config": asdict(resolved_training_config),
            "feature_names": pipeline["feature_names"],
            "composition_feature_names": [
                pipeline["feature_names"][idx] for idx in model.comp_feature_indices
            ],
            "t2_rank": pipeline["t2_rank"],
            "seed": seed,
            "hard_constraints": [
                "nonnegative",
                "saturation",
                "monotonicity",
                "sips",
                "van_t_hoff",
            ],
            "sips_verification": sips_verification,
            "arrhenius_verification": arrhenius_verification,
            "strict_parameter_ranges": strict_parameter_ranges,
            "reported_as": ["V2", "V8"],
        },
        checkpoint_path,
    )

    if verbose:
        print(f"Saved all-hard training history: {history_path}")
        print(f"Saved all-hard Sips verification: {sips_verification_path}")
        print(f"Saved all-hard Arrhenius verification: {arrhenius_verification_path}")
        print(f"Saved all-hard evaluation metrics: {evaluation_metrics_path}")
        print(f"Saved all-hard enthalpy summary: {enthalpy_summary_path}")
        print(f"Saved all-hard checkpoint: {checkpoint_path}")

    results_dict = {
        "V2": evaluation_results,
        "V8": evaluation_results,
    }

    return {
        "model": model,
        "history": history,
        "evaluation_results": evaluation_results,
        "results_dict": results_dict,
        "evaluation_table": evaluation_table,
        "sips_verification": sips_verification,
        "arrhenius_verification": arrhenius_verification,
        "enthalpy_summary": enthalpy_summary,
        "paths": {
            "training_history": history_path,
            "sips_verification": sips_verification_path,
            "arrhenius_verification": arrhenius_verification_path,
            "evaluation_metrics": evaluation_metrics_path,
            "enthalpy_summary": enthalpy_summary_path,
            "checkpoint": checkpoint_path,
        },
    }


# Quick architecture check for the shared V2/V8 hard constraints.
def smoke_test_all_hard_variant() -> None:
    pipeline = build_data_pipeline(batch_size=8)
    base_config = AllHardConfig()
    temperature_idx = base_config.temperature_feature_idx
    config = AllHardConfig(
        temperature_center=float(pipeline["scaler"].center_[temperature_idx]),
        temperature_scale=float(pipeline["scaler"].scale_[temperature_idx]),
    )
    model = AllHardSipsArrheniusPINN(config)
    training_config = TrainingConfig(lambda_p=0.0, max_epochs=1, patience=1)
    batch = next(iter(pipeline["train_loader"]))
    output = model(batch["x"], batch["raw_pressure"])
    losses = compute_all_hard_losses(output, batch["y"])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config.lr,
        weight_decay=training_config.weight_decay,
    )
    train_metrics = train_one_epoch_all_hard(
        model,
        pipeline["train_loader"],
        optimizer,
        training_config,
    )
    val_metrics = validate_one_epoch_all_hard(
        model,
        pipeline["val_loader"],
        training_config,
    )
    trained_model, history = train_all_hard_model(
        AllHardSipsArrheniusPINN(config),
        pipeline["train_loader"],
        pipeline["val_loader"],
        training_config,
    )
    sips_verification = verify_sips_structure(
        trained_model,
        pipeline["train_loader"],
        n_comp=16,
        n_P=8,
    )
    arrhenius_verification = verify_arrhenius_structure(
        trained_model,
        pipeline["train_loader"],
        n_samples=16,
        required_fraction_in_range=0.0,
    )
    (
        evaluated_model,
        eval_history,
        evaluation_results,
        eval_sips_verification,
        eval_arrhenius_verification,
    ) = evaluate_all_hard_model(
        pipeline,
        model_config=config,
        training_config=training_config,
        n_bootstrap=2,
        n_mc=2,
        verify_n_comp=16,
        verify_n_P=8,
        verify_n_samples=16,
    )
    saved_results = run_and_save_all_hard_exp(
        batch_size=8,
        model_config=config,
        training_config=training_config,
        results_dir=Path("/private/tmp/uioenergy_all_hard_results"),
        checkpoint_dir=Path("/private/tmp/uioenergy_all_hard_checkpoints"),
        n_bootstrap=2,
        n_mc=2,
        verify_n_comp=16,
        verify_n_P=8,
        verify_n_samples=16,
        verbose=False,
    )

    print("x:", batch["x"].shape)
    print("composition features:", len(model.comp_feature_indices))
    print("q_hat:", output["q_hat"].shape)
    print("q_max:", output["q_max"].shape)
    print("K:", output["K"].shape)
    print("K0:", output["K0"].shape)
    print("n:", output["n"].shape)
    print("delta_H:", output["delta_H"].shape)
    print("total_loss:", float(losses["total_loss"].detach()))
    print("data_loss:", float(losses["data_loss"].detach()))
    print("train_total_loss:", train_metrics["total_loss"])
    print("val_total_loss:", val_metrics["total_loss"])
    print("epochs_trained:", len(history))
    print("best_val_loss:", history[-1]["best_val_loss"])
    print("eval_epochs_trained:", len(eval_history))
    print("eval_tiers:", sorted(evaluation_results))
    print("eval_model_type:", evaluated_model.__class__.__name__)
    print("saved_paths:", sorted(saved_results["paths"]))
    print("min q_hat:", float(output["q_hat"].min().detach()))
    print(
        "min qmax_minus_q:", float((output["q_max"] - output["q_hat"]).min().detach())
    )
    print("sips_min_dq_dP:", sips_verification["min_dq_dP"])
    print("eval_sips_min_dq_dP:", eval_sips_verification["min_dq_dP"])
    print(
        "arrhenius_identity_error:",
        arrhenius_verification["max_arrhenius_identity_error"],
    )
    print(
        "eval_arrhenius_identity_error:",
        eval_arrhenius_verification["max_arrhenius_identity_error"],
    )


if __name__ == "__main__":
    smoke_test_all_hard_variant()
