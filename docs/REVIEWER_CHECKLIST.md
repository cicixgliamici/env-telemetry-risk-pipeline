# Reviewer Checklist

Use this checklist to evaluate the project quickly and consistently.

## Fast Verification

```bash
pip install -e .[dev]
python -m ruff check src tests scripts
python -m ruff format --check src tests scripts
python -m pytest -q
python -m src validate --input data/raw/raw_dataset.csv
python -m src profile --input data/raw/raw_dataset.csv
```

## Full Local Flow

```bash
python -m src review --rows 1000 --seed 42 --repeats 5 --output-dir runs
python -m build --no-isolation
```

Open the `REVIEW_SUMMARY.md` path printed by the review command. Its relative links
remain valid when sharing the entire directory. It includes random, temporal, and
cost-policy training, robustness on the temporal classifier, batch scoring with the
cost-policy model, profiling, and anomaly detection. Each child has its own lifecycle
manifest, settings, environment, and outputs. Use `--rows 250 --repeats 1` for a faster
smoke run. Small or unusual generated datasets can fail early temporal class checks;
the command never silently switches evaluation protocol.

The review seed controls generated data and stress draws; model and split seeds
remain the recorded project constant (42). The command does not run tests or builds.

## Suggested Reading Order

1. [Review status](REVIEW_STATUS.md) and [roadmap](ROADMAP.md): completed scope and checks.
2. [Dataset card](DATASET_CARD.md) and [model card](MODEL_CARD.md): task and limitations.
3. Generated `REVIEW_SUMMARY.md`: measured evidence and links to the full reports.
4. [Architecture](ARCHITECTURE.md), code, and tests: inspect the implementation behind the claims.

## What To Inspect

- `docs/MODELING_NOTES.md`: mathematical background and reviewer questions.
- `docs/MODEL_CARD.md`: intended use, evaluation, limitations, and monitoring.
- `docs/DATASET_CARD.md`: schema, synthetic data, and leakage notes.
- `docs/EXPERIMENT_RESULTS.md`: reproducible 1,000-row benchmark, baseline, ablation, and limits.
- `docs/ROBUSTNESS_RESULTS.md`: frozen-model stress protocol, measured synthetic deltas, and uncertainty limits.
- `docs/DECISION_POLICY.md`: validation-only risk cutoffs, relative error costs, and precision/recall tradeoffs.
- `runs/<run_id>/outputs/training_summary.md`: CV table, raw/engineered deltas, dummy gap, and winner-only test metrics.
- `.github/workflows/ci.yml`: test matrix, package build, synthetic data validation, and container smoke test.
- `runs/<run_id>/outputs/model_metadata.json`: selected model, feature columns, excluded columns, and run identity.
- `runs/<run_id>/outputs/evaluation_protocol.json`: exact holdout/CV row membership, class counts, timestamps, and exclusion gaps.
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
- Temporal evaluation keeps whole timestamp groups together and trains each fold strictly before its validation window.
- Robustness experiments preserve source models/labels and distinguish sensor errors from changed case mix.
- Cost-based cutoffs use only training-validation probabilities and travel with the saved pipeline.
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
