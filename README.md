# PINNs for Methane Adsorption in Coal

This project investigates physics-informed neural networks (PINNs) for modeling methane adsorption in coal samples. The goal is to compare soft physics penalties with hard-constrained neural network architectures under interpolation and extrapolation tests.

## Project Status

The blueprint implementation is complete through Step 11:

- Step 1: data pipeline with leakage-safe splits
- Step 2: unified evaluation harness
- Step 3: soft baseline model and lambda-p sensitivity sweep
- Step 4: hard bounds variant
- Step 5: hard monotonicity variant
- Step 6: hard Sips gauge variant
- Step 7: hard van't Hoff / Arrhenius variant
- Step 8: joint hard-constraint variants
- Step 9: ablation and interaction analysis
- Step 10: uncertainty calibration
- Step 11: interpretability and physical parameter validation

The numerical results summarized in `report.tex` and `report.pdf` are from the
full training and evaluation protocol. Earlier leaner runs were used only as
implementation checks while building the pipeline.

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

The pipeline builds three evaluation tiers:

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

The default T2 held-out rank is:

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

## Full-Protocol Result Summary

The full soft-baseline physics-weight sweep selected:

```text
lambda_p = 0.05
validation score = 4.916
validation RMSE = 4.897
validation violation rate = 0.004
```

The main RMSE winners by evaluation tier were:

```text
T1 interpolation:          V4 hard monotonicity, RMSE = 5.213
T2 coal-rank extrapolation: V2/V8 all-hard combined, RMSE = 5.056
T3 pressure extrapolation:  V7 hard bounds + monotonicity, RMSE = 1.192
```

T3 contains only three samples, so T3 results are treated as descriptive
pressure-extrapolation evidence rather than strong statistical evidence.

The hard-constrained variants passed their intended architecture-level physical
verification checks. In particular:

- V4 and V7 passed pressure-monotonicity checks with zero violation rate.
- V5 and V2/V8 satisfied Sips bounds and pressure monotonicity with zero lower,
  upper, and slope violation rates.
- V6 and V2/V8 satisfied the Arrhenius identity to numerical precision.

Uncertainty calibration was evaluated with 100 Monte Carlo dropout samples per
model and split. The V7 hard-bounds and monotonicity model gave the lowest
validation ECE, while the V2/V8 all-hard combined variants gave the lowest T1
and T2 ECE values in the full-protocol table.

The main conclusion is that hard constraints improve physical validity by
construction, while predictive accuracy and calibration benefits depend on the
evaluation tier.

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

## Results Layout

Generated result files are organized by blueprint step:

```text
results/step03_soft_baseline/
results/step04_hard_bounds/
results/step05_hard_monotonicity/
results/step06_sips_gauge/
results/step07_arrhenius/
results/step08_joint_variants/
results/step09_ablation/
results/step10_uq_calibration/
results/step11_interpretability/
```

The experiment scripts now save directly into these step-specific folders by
default. Steps 1 and 2 do not currently generate standalone result files.

The generated `results/` and `checkpoints/` folders are ignored by Git because
they contain generated artifacts and large model checkpoint files. For review,
the project includes `SUPERVISOR_REVIEW.md`, which maps each blueprint step to
the corresponding script and saved output folder. A separate supervisor review
zip package can be shared through OneDrive when the generated outputs and
checkpoints are needed.
