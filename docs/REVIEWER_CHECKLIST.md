# Reviewer Checklist

Use this checklist to evaluate the project quickly and consistently.

## Fast Verification

```bash
pip install -e .[dev]
python -m ruff check src tests scripts
python -m pytest -q
python -m src validate --input data/raw/raw_dataset.csv
python -m src profile --input data/raw/raw_dataset.csv
```

## Full Local Flow

```bash
python scripts/generate_synthetic_data.py --output data/raw/synthetic_dataset.csv --rows 250 --seed 42
python -m src validate --input data/raw/synthetic_dataset.csv
python -m src profile --input data/raw/synthetic_dataset.csv
python -m src train --input data/raw/synthetic_dataset.csv
python -m src predict --input data/raw/synthetic_dataset.csv --model models/random_forest_model.pkl
python -m src anomaly --input data/raw/synthetic_dataset.csv --contamination 0.05
python -m build --no-isolation
```

If the selected model is not `random_forest`, use the model name shown in `data/outputs/metrics.json`.

## What To Inspect

- `docs/MODELING_NOTES.md`: mathematical background and reviewer questions.
- `docs/MODEL_CARD.md`: intended use, evaluation, limitations, and monitoring.
- `docs/DATASET_CARD.md`: schema, synthetic data, and leakage notes.
- `.github/workflows/ci.yml`: test matrix, package build, synthetic data validation, and container smoke test.
- `data/outputs/model_metadata.json`: selected model, feature columns, excluded columns, and run config.
- `data/outputs/data_profile.md`: row count, target balance, missing values, numeric summary, and warnings.

## Review Signals

- Validation runs before modeling.
- `failure_risk` is excluded to avoid leakage.
- The model comparison is metric-driven.
- Cross-validation is reported for more stable review evidence.
- Batch prediction uses saved metadata feature order.
- Synthetic data is deterministic and tested.
- Docker runs as a non-root user.
- CI checks lint, tests, package build, synthetic data, and container smoke.

## Good Questions To Ask

- Why is `failure_risk` excluded from features?
- Why weighted F1 instead of only accuracy?
- What does cross-validation add beyond one train/test split?
- How does batch prediction preserve the training feature order?
- Why is synthetic data useful but not proof of production performance?
- What would need to change before real deployment?
