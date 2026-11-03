# Operations

## Local Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e .[dev]
```

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.venv\Scripts\Activate.ps1
```

## Common Commands

Validate the dataset:

```powershell
python -m src validate --input data/raw/raw_dataset.csv
```

The default checks training input, including a complete `risk_level` target. For unlabeled incoming data, select the intended operation:

```powershell
python -m src validate --input data/raw/incoming.csv --purpose prediction
python -m src validate --input data/raw/incoming.csv --purpose anomaly
```

Prediction, anomaly detection, and profiling select telemetry-only validation automatically. They accept missing labels and single-class batches. Numeric infinities, missing timestamps, and invalid physical ranges are rejected before execution; see the Dataset Card for bounds. `failure_risk` is optional for every operation.

Train and generate artifacts:

```powershell
python -m src train --input data/raw/raw_dataset.csv
```

Run anomaly detection:

```powershell
python -m src anomaly --input data/raw/raw_dataset.csv --contamination 0.05
```

Run tests:

```powershell
python -m pytest
```

Create a data quality profile:

```powershell
python -m src profile --input data/raw/raw_dataset.csv
```

Run batch prediction after training:

```powershell
python -m src predict --input data/raw/raw_dataset.csv --model models/random_forest_model.pkl
```

Use the selected model filename reported by training. New artifacts contain the complete fitted preprocessing and classifier pipeline and infer their input schema from the artifact. `--metadata` is only used for older bare classifiers. Retrain old models to obtain training-fitted preprocessing; the compatibility path retains batch-dependent preprocessing for those old artifacts.

Run lint and security-oriented static checks:

```powershell
python -m ruff check src tests scripts
```

Generate a larger deterministic synthetic dataset:

```powershell
python scripts/generate_synthetic_data.py --output data/raw/synthetic_dataset.csv --rows 250 --seed 42
python -m src validate --input data/raw/synthetic_dataset.csv
```

Build the Python package:

```powershell
python -m build --no-isolation
```

Build and smoke-test the runtime container:

```powershell
docker build -t env-telemetry-risk-pipeline:local .
docker run --rm env-telemetry-risk-pipeline:local
```

## Artifacts

- `data/processed/cleaned_dataset.csv`
- `data/processed/featured_dataset.csv`
- `data/outputs/metrics.json`
- `data/outputs/model_comparison.json`
- `data/outputs/model_metadata.json`
- `data/outputs/data_profile.json`
- `data/outputs/data_profile.md`
- `data/outputs/predictions.csv`
- `data/outputs/batch_predictions.csv`
- `data/outputs/training_summary.md`
- `data/outputs/anomalies.csv`
- `data/outputs/anomaly_summary.json`
- `models/*.pkl`
- `reports/figures/*.png`

For supervised training, `cleaned_dataset.csv` retains missing telemetry values. `featured_dataset.csv` contains unscaled engineered features computed with the winning pipeline's training-fitted imputer, plus the original labels and timestamps for audit. This export is generated after fitting and does not participate in model selection.

## Production Hardening Ideas

- Add environment-based configuration for paths and model settings.
- Publish container images to a registry after protected-branch CI passes.
- Persist logs to file for long-running or scheduled jobs.
- Add hyperparameter search within training-only cross-validation for stronger model selection.
