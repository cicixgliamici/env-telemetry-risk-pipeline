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
python -m src train --input data/raw/raw_dataset.csv --output-dir runs
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
python -m src predict --input data/raw/raw_dataset.csv --model "runs/<run_id>/models/<selected_model>_model.pkl"
```

Use the `model_artifact` path printed by training instead of the placeholders.
Training also prints the absolute `run_dir` and unique `run_id`. New artifacts contain the complete fitted preprocessing and classifier pipeline and infer their input schema from the artifact. `--metadata` is only used for older bare classifiers. Retrain old models to obtain training-fitted preprocessing; the compatibility path retains batch-dependent preprocessing for those old artifacts.

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

Training and anomaly detection create distinct folders under `--output-dir`
(default: `runs/`, relative to the working directory). Paths below use `<run>` for
the absolute `run_dir` returned by the command:

- `<run>/run.json`: operation, identity, start/end timestamps, and status
- `<run>/config.json`: input path and effective execution settings
- `<run>/environment.json`: Python/platform and installed package versions
- `<run>/requirements.txt`: exact version snapshot of the installed environment
- `<run>/processed/cleaned_dataset.csv` (training)
- `<run>/processed/featured_dataset.csv` (training)
- `<run>/outputs/metrics.json` (training)
- `<run>/outputs/model_comparison.json` (training)
- `<run>/outputs/model_metadata.json` (training)
- `<run>/outputs/predictions.csv` (training)
- `<run>/outputs/training_summary.md` (training)
- `<run>/outputs/anomalies.csv` (anomaly)
- `<run>/outputs/anomaly_summary.json` (anomaly)
- `<run>/models/*.pkl`
- `<run>/figures/*.png`

Standalone commands retain these defaults, configurable via `profile --output-dir`
and `predict --output`:

- `data/outputs/data_profile.json`
- `data/outputs/data_profile.md`
- `data/outputs/batch_predictions.csv`

Each new run gets a separate directory, including repeated or concurrent executions.
`run.json` moves from `running` to `completed` only after all artifacts are written.
Exceptions leave status `failed` and retain partial artifacts for diagnosis. A process
interrupted outside Python exception handling may leave status `running`.
Previous fixed-directory artifacts are not migrated or deleted.

To reproduce an environment, use the recorded Python/platform in a clean virtual
environment, install `<run>/requirements.txt`, then install the same source revision
with `pip install --no-deps -e .` and rerun the command using the recorded configuration
and identical input data. The snapshot includes all installed distributions except
this local project; it is not a cross-platform lockfile and does not archive wheels,
source code, or data. Prefer running experiments in a dedicated virtual environment.

For supervised training, `cleaned_dataset.csv` retains missing telemetry values. `featured_dataset.csv` contains unscaled model features computed with the winning pipeline's training-fitted imputer, plus the original labels and timestamps for audit. For raw-only or dummy winners it contains raw imputed inputs; otherwise it also includes engineered features. This export is generated after fitting and does not participate in model selection.

`training_summary.md` compares five candidates on identical training folds, reports feature-ablation deltas and the dummy gap, and separates winner-only test metrics. `model_metadata.json` records the actual `feature_set`, normalized data fingerprint, class counts, and environment versions. The fingerprint identifies the table, not its original source. See `docs/EXPERIMENT_RESULTS.md` for a fixed synthetic reproduction protocol and observed results.

## Production Hardening Ideas

- Add environment-based configuration for paths and model settings.
- Publish container images to a registry after protected-branch CI passes.
- Persist logs to file for long-running or scheduled jobs.
- Add hyperparameter search within training-only cross-validation for stronger model selection.
