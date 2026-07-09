# PINNs for Methane Adsorption in Coal

This project investigates physics-informed neural networks (PINNs) for modeling methane adsorption in coal samples. The goal is to compare soft physics penalties with hard-constrained neural network architectures under interpolation and extrapolation tests.

## Project Status

The project currently has two main parts implemented:

- Step 1: data pipeline with leakage-safe splits
- Step 2: unified evaluation harness

Step 3, the soft baseline model and lambda-p sensitivity sweep, is the next implementation stage.

## Dataset

The raw dataset contains methane adsorption measurements with:

- moisture content
- ash content
- volatile matter
- temperature
- pressure
- methane adsorption capacity

The target variable is:

```text
CH4_adsorption_m3t
```

## Data Pipeline

The data pipeline is implemented in:

```text
data_pipeline.py
```

It currently performs:

- CSV loading and column renaming
- basic validation
- pressure positivity check
- temperature conversion from Celsius to Kelvin
- experiment ID inference from repeated coal-condition curves
- 12-feature engineering
- inferred coal-rank calculation on a dry, ash-free basis
- leakage-safe train/validation/test splits
- robust scaling fitted on training data only
- log transform of the adsorption target
- PyTorch `Dataset` and `DataLoader` creation
- 5-fold `GroupKFold` cross-validation

The model input tensor `x` contains 12 numeric features:

```text
moisture_wt
ash_wt
volatile_wt
temperature_K
pressure_MPa
Tr
Pr
FC
OM
beta
PT
MV
```

Raw pressure is also preserved separately as:

```text
raw_pressure
```

This is needed for physics-based model layers and Sips-type constraints.

## Evaluation Tiers

The pipeline currently builds three evaluation tiers:

```text
T1: interpolation-style held-out experiments
T2: inferred coal-rank extrapolation
T3: high-pressure extrapolation
```

T2 is based on:

```text
coal_rank_inferred
```

This rank is inferred from dry, ash-free volatile matter and fixed carbon. It is not a directly measured coal-rank label, so this should be stated clearly in any manuscript.

The current default T2 held-out rank is:

```text
Subbituminous / Lignite
```

## Evaluation Harness

The unified evaluation harness is implemented in:

```text
unified_eval_harness.py
```

It evaluates every model with the same logic across:

```text
T1
T2
T3
```

The harness computes:

- accuracy metrics: `R2`, `RMSE`, `MAE`, `MaxAE`, relative error
- bootstrap confidence intervals
- physical violation rates: `neg_rate`, `sat_rate`, `mono_rate`
- uncertainty metrics with MC Dropout
- computational metrics
- statistical comparison helpers
- CSV export
- LaTeX table export

## Running Checks

Use:

```bash
.venv/bin/python data_pipeline.py
.venv/bin/python unified_eval_harness.py
```

Style checks:

```bash
RUFF_CACHE_DIR=/private/tmp/ruff-cache .venv/bin/ruff check data_pipeline.py unified_eval_harness.py
.venv/bin/black --check data_pipeline.py unified_eval_harness.py
```

## Next Step

The next planned implementation step is:

```text
Step 3: Soft baseline PINN + lambda_p sensitivity sweep
```

This will add:

- V1 soft-constrained model architecture
- data loss and physics penalty losses
- training loop with early stopping
- lambda-p sweep
- connection to the unified evaluation harness
