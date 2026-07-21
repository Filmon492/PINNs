from pathlib import Path

RESULTS_ROOT = Path("results")
STEP_RESULT_FOLDERS = {
    3: "step03_soft_baseline",
    4: "step04_hard_bounds",
    5: "step05_hard_monotonicity",
    6: "step06_sips_gauge",
    7: "step07_arrhenius",
    8: "step08_joint_variants",
    9: "step09_ablation",
    10: "step10_uq_calibration",
    11: "step11_interpretability",
}


# Resolve a step-specific result directory from a root or direct step path.
def resolve_step_results_dir(results_dir=RESULTS_ROOT, step: int = 3) -> Path:
    if step not in STEP_RESULT_FOLDERS:
        raise KeyError(f"No result folder is configured for Step {step}.")

    results_dir = Path(results_dir)
    step_folder = STEP_RESULT_FOLDERS[step]

    if results_dir.name == step_folder:
        return results_dir

    return results_dir / step_folder


# Create and return the directory used for one blueprint step.
def ensure_step_results_dir(results_dir=RESULTS_ROOT, step: int = 3) -> Path:
    step_dir = resolve_step_results_dir(results_dir, step)
    step_dir.mkdir(parents=True, exist_ok=True)

    return step_dir


# Locate a result file in the organized step folder, with old flat path fallback.
def find_step_result_file(
    filename: str,
    results_dir=RESULTS_ROOT,
    step: int = 3,
) -> Path:
    results_dir = Path(results_dir)
    organized_path = resolve_step_results_dir(results_dir, step) / filename

    if organized_path.exists():
        return organized_path

    legacy_path = results_dir / filename

    if legacy_path.exists():
        return legacy_path

    return organized_path
