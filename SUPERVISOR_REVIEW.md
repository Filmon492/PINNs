# Supervisor review guide

This package contains the project scripts, full data, saved full-protocol
results, model checkpoints, and the current report files.

## Protocol status

The numerical results reported in `report.tex` and `report.pdf` are from the
full training and evaluation protocol. The early leaner runs were used as
implementation checks while building the pipeline and are not used as reported
results or as a formal comparison in the manuscript.

A fair reduced-vs-full computational-cost comparison has not yet been performed.
The report therefore treats computational cost as further work rather than as a
main result.

## Main files

- `data/coal_gas_20250826.csv`: raw dataset.
- `report.tex`, `report.pdf`: current manuscript/report.
- `researchblueprint.tex`, `researchblueprint.pdf`: original blueprint.
- `README.md`: project overview.
- `requirements.txt`, `pyproject.toml`: Python environment setup.

## Step-to-script map

| Blueprint step | Main script | Saved outputs |
|---|---|---|
| Step 1: data pipeline | `data_pipeline.py` | split summaries produced from code |
| Step 2: evaluation harness | `unified_eval_harness.py` | used by all later variants |
| Step 3: soft baseline | `soft_baseline.py` | `results/step03_soft_baseline/` |
| Step 4: hard bounds | `hard_bounds_variant.py` | `results/step04_hard_bounds/` |
| Step 5: hard monotonicity | `hard_monotonicity_variant.py` | `results/step05_hard_monotonicity/` |
| Step 6: hard Sips gauge | `sips_gauge_variant.py` | `results/step06_sips_gauge/` |
| Step 7: hard Van't Hoff / Arrhenius | `arrhenius_variant.py` | `results/step07_arrhenius/` |
| Step 8: joint hard variants | `joint_variants.py`, `all_hard_variant.py` | `results/step08_joint_variants/` |
| Step 9: ablation and interaction | `ablation_analysis.py` | `results/step09_ablation/` |
| Step 10: uncertainty calibration | `uq_calibration.py` | `results/step10_uq_calibration/` |
| Step 11: interpretability | `interpretability_analysis.py` | `results/step11_interpretability/` |

## Checkpoints

Saved trained model states are in `checkpoints/`:

- `soft_baseline_v1.pt`
- `hard_bounds_v3.pt`
- `hard_monotonicity_v4.pt`
- `sips_gauge_v5.pt`
- `arrhenius_v6.pt`
- `joint_v7.pt`
- `all_hard_v2_v8.pt`

The `results/` and `checkpoints/` folders are intentionally ignored by Git
because they are generated artifacts. They are included in the review package
for inspection.
