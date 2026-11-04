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
python -m src predict --input data/raw/synthetic_dataset.csv --model "runs/<run_id>/models/<selected_model>_model.pkl"
python -m src anomaly --input data/raw/synthetic_dataset.csv --contamination 0.05
python -m build --no-isolation
```

Replace prediction placeholders with the exact `model_artifact` path printed by training.
Each training/anomaly invocation prints its `run_id` and absolute `run_dir`;
`--output-dir` changes the parent directory without reusing an existing run.

## What To Inspect

- `docs/MODELING_NOTES.md`: mathematical background and reviewer questions.
- `docs/MODEL_CARD.md`: intended use, evaluation, limitations, and monitoring.
- `docs/DATASET_CARD.md`: schema, synthetic data, and leakage notes.
- `docs/EXPERIMENT_RESULTS.md`: reproducible 1,000-row benchmark, baseline, ablation, and limits.
- `runs/<run_id>/outputs/training_summary.md`: CV table, raw/engineered deltas, dummy gap, and winner-only test metrics.
- `.github/workflows/ci.yml`: test matrix, package build, synthetic data validation, and container smoke test.
- `runs/<run_id>/outputs/model_metadata.json`: selected model, feature columns, excluded columns, and run identity.
- `runs/<run_id>/run.json`, `config.json`, `environment.json`, and `requirements.txt`: status, settings, and environment snapshot.
- `data/outputs/data_profile.md`: row count, target balance, missing values, numeric summary, and warnings.

## Review Signals

- Validation runs before modeling.
- `failure_risk` is excluded to avoid leakage.
- The model comparison is metric-driven.
- Identical folds compare raw and engineered inputs; the majority-class dummy can win.
- Macro F1 and class-1 recall expose weaknesses hidden by weighted averages.
- Cross-validation is reported for more stable review evidence.
- Batch prediction uses the fitted pipeline's own feature order.
- Synthetic data is deterministic and tested.
- Docker runs as a non-root user.
- CI checks lint, tests, package build, synthetic data, and container smoke.

## Good Questions To Ask

- Why is `failure_risk` excluded from features?
- Why weighted F1 instead of only accuracy?
- Does feature engineering help every classifier and every metric?
- Does the selected model beat the majority rule, and does it maximize risk recall?
- What does cross-validation add beyond one train/test split?
- How does batch prediction preserve the training feature order?
- Why is synthetic data useful but not proof of production performance?
- What would need to change before real deployment?
