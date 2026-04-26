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

## Artifacts

- `data/processed/cleaned_dataset.csv`
- `data/processed/featured_dataset.csv`
- `data/outputs/metrics.json`
- `data/outputs/model_comparison.json`
- `data/outputs/predictions.csv`
- `data/outputs/training_summary.md`
- `data/outputs/anomalies.csv`
- `data/outputs/anomaly_summary.json`
- `models/*.pkl`
- `reports/figures/*.png`

## Production Hardening Ideas

- Add environment-based configuration for paths and model settings.
- Add linting and formatting checks to CI.
- Persist logs to file for long-running or scheduled jobs.
- Add cross-validation or hyperparameter search for stronger model selection.
