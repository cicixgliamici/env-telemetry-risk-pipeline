# Frozen Model Robustness Experiment

## Question and Fixed Protocol

This experiment measures a selected classifier's sensitivity to missing telemetry,
measurement errors, and a changed mix of held-out cases. It uses synthetic data to
test this implementation; its labels are not observed environmental failures.

Run date: 2026-11-04. The generator produced 1,000 rows with seed 42 (620 class-0,
380 class-1). Temporal evaluation with test fraction 0.2 and one excluded timestamp
used 799 training rows, one gap row, and the latest 200 rows for testing. Training-only
expanding-window CV selected the engineered-feature logistic regression. Its parameters,
imputer, and scaler remained frozen throughout the robustness experiment. The test
period contained 81 risk cases (40.5%).

Scenario definitions, severities, and five stochastic repetitions with seed 42 were
fixed before stress scoring. Their output was not used to select another model,
retune parameters, or change thresholds. Baseline predictions reproduce the training
run's saved test predictions. The original stratified benchmark in
[Experiment Results](EXPERIMENT_RESULTS.md) uses a different protocol and selected
model; its scores should not be treated as the baseline for this experiment.

## Scenarios

| Scenario | Transformation |
| --- | --- |
| Baseline | Original held-out measurements and labels |
| Missing 10% / 30% | Independently mask each telemetry cell with probability 0.1 / 0.3 |
| Noise 0.1 / 0.3 std | Independent zero-mean Gaussian noise, scaled by each feature's training standard deviation |
| Sensor bias 0.5 std | Add half a training standard deviation to humidity, temperature, and pollution |
| High-pollution mix 3x | Sample 200 existing test cases with replacement; cases above the training pollution upper quartile have three times the sampling weight |

Missingness/noise/bias model sensor degradation for the same underlying cases, so
labels stay fixed and are never regenerated from corrupted features. Noise and bias
are clipped to the existing broad physical validation bounds; pressure remains
positive. Original missing values remain missing. Constant or entirely missing
training features have zero noise/bias scale.

Scales, median, and upper-quartile threshold come exclusively from holdout training
rows. The measured pollution threshold was 45.61 proxy units. Missing test pollution
values receive the training median only for deciding case-mix sampling weights;
the frozen model still receives the actual missing measurement.

Missing-value masks are nested across severity levels, and noise levels share the
same Gaussian draws within a repetition. Repetitions use independent deterministic
random streams. Baseline and systematic bias run once because they are deterministic.
The population-mix experiment resamples complete cases, preserving their feature-label
relationship. Its metric delta reflects composition and finite-sample variation,
rather than changed prediction accuracy for the same individual measurements.

## Measured Results

| Scenario | Weighted F1 mean +/- std | Delta from baseline | Risk recall mean | Changed predictions |
| --- | ---: | ---: | ---: | ---: |
| Baseline | 0.8162 +/- 0.0000 | 0.0000 | 0.8272 | 0.0000 |
| Missing 10% | 0.8024 +/- 0.0143 | -0.0138 | 0.7704 | 0.0570 |
| Missing 30% | 0.7521 +/- 0.0258 | -0.0641 | 0.6395 | 0.1660 |
| Noise 0.1 std | 0.8201 +/- 0.0073 | +0.0039 | 0.8272 | 0.0280 |
| Noise 0.3 std | 0.8008 +/- 0.0114 | -0.0154 | 0.7877 | 0.0930 |
| Sensor bias 0.5 std | 0.7158 +/- 0.0000 | -0.1004 | 0.9630 | 0.2050 |
| High-pollution mix 3x | 0.7974 +/- 0.0310 | -0.0188 | 0.8272 | 0.0000 |

Changed predictions compare each scenario case against its own original prediction,
including when a resampled source row appears more than once. The unchanged-case
mix therefore has zero changed predictions while its overall F1 changes.

In this experiment, 30% missingness reduces weighted F1 by 0.0641 and risk recall
from 0.8272 to 0.6395. The sensor bias produces the largest measured F1 reduction
among these fixed scenarios, while increasing risk recall: the classifier flags more
cases as risky, illustrating a precision/recall tradeoff. Small noise slightly improves
F1 here; these finite-sample results do not imply a monotonic degradation law.

The achieved missing fractions were 0.0925 and 0.2923: cellwise Bernoulli masking
does not guarantee exact percentages in each finite batch. The high-pollution mix
raises the mean high-pollution fraction from 0.2450 to 0.4810 and the risk fraction
from 0.4050 to 0.4710. These changes help explain why its F1 delta cannot be
interpreted as a paired measurement-degradation effect.

Standard deviations describe perturbation or resampling variability for one frozen
model and holdout, using population std (`ddof=0`). They are not confidence intervals,
significance tests, retraining variability, or independent-dataset uncertainty.
Unavailable risk recall is `null`; aggregates use only repetitions with observed
risk cases and record their contributing counts. Every other metric uses all cases.

## Reproduction

```bash
python scripts/generate_synthetic_data.py --output data/raw/synthetic_dataset.csv --rows 1000 --seed 42
python -m src train --input data/raw/synthetic_dataset.csv --split-strategy temporal --test-size 0.2 --temporal-gap 1 --output-dir runs
python -m src robustness --training-run "runs/<training_run_id>" --seed 42 --repeats 5 --output-dir runs
```

Replace the final command's placeholder with `run_dir` printed by the training
command. The source run must be a completed supervised run with its cleaned data,
model, metadata, predictions, and evaluation protocol. New robustness runs have their
own lifecycle manifest, configuration, and environment snapshot; source files are
retained unchanged. As with prediction, only load trusted local serialized models.

The experiment checks the cleaned-table fingerprint, disjoint train/test membership,
model feature schema, and reproduction of the original holdout predictions. It stores
SHA-256 identities for the source model, cleaned CSV, and protocol. These hashes
identify artifacts; they do not verify external provenance. A copied source folder
can be evaluated because artifact paths resolve within that folder. Changed data or
an incompatible environment can fail the fingerprint/baseline consistency checks.

Normalized input SHA-256:

```text
2abcb3a4a44bce08d65bef7aaf2a7762daac7ed014861546233b0a4a680e4639
```

Execution environment: Python 3.14.3, NumPy 2.3.5, pandas 3.0.2, scikit-learn 1.8.0.
Full precision outputs are in the generated JSON reports. Runtime artifacts are
ignored by Git; this document preserves the measured protocol and results.

## Artifacts and Review

Each robustness run contains:

- `outputs/robustness_summary.json`: aggregate metrics, stds, deltas, observed repetition counts, and source identities;
- `outputs/robustness_summary.md`: score and sample-composition tables with interpretation;
- `outputs/robustness_repetitions.json`: per-repetition metrics and diagnostics;
- `outputs/robustness_predictions.csv`: scored telemetry, original cleaned row index, scenario, zero-based repetition and case position, labels, and predictions;
- `outputs/stress_profile.json`: training-only feature stds and pollution sampling threshold;
- `figures/robustness_f1.png`: mean weighted F1 and descriptive std by scenario.

Audit source row indices refer to the source run's zero-based cleaned data rows.
The case-mix scenario can repeat them. Recompute all reported classification scores
from the audit's `y_true` and `y_pred`, or re-score its telemetry with the same model.
High-pollution fractions use all cases as denominator; missing pollution values do
not count as high. Entirely missing training pollution makes only the case-mix
scenario unavailable, with an explicit reason; other scenarios still complete.

## Limits and Next Evidence

The generator and engineered features share domain assumptions. Its dataset-wide
quantile target remains an artificial labeling rule even under chronological splitting.
Independent cellwise missingness, Gaussian errors, and a fixed sensor offset do not
cover contiguous outages, correlated failures, delayed labels, or evolving real-world
feature/target relationships. The changed case mix only resamples conditions already
present in this holdout; it cannot reveal performance in unseen conditions.

Reusing these results to improve a model consumes this holdout as development data.
A revised model needs new independent evaluation. A future observational experiment
still requires a sourced/versioned dataset, license, sensor/site definitions,
collection period, label-availability rules, and a defensible observed target.
