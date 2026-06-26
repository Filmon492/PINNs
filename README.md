# PINNs for Methane Adsorption in Coal

This project investigates physics-informed neural networks (PINNs) for modeling methane adsorption in coal samples. The goal is to compare standard data-driven models with physics-informed and hard-constrained models for adsorption prediction.

## Project Status

This project is currently under development.

Completed so far:

- Data loading from CSV
- Column renaming and basic validation
- Temperature conversion from Celsius to Kelvin
- Inferred experiment IDs from repeated coal-condition curves
- 12-feature engineering
- Experiment-level train/test split to avoid leakage

## Dataset

The dataset contains methane adsorption measurements with the following raw variables:

- Moisture content
- Ash content
- Volatile matter
- Temperature
- Pressure
- Methane adsorption capacity

The target variable is:

```text
CH4_adsorption_m3t
