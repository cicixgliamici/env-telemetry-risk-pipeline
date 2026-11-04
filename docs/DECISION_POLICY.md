# Cost-Based Risk Decisions

## Objective and Controls

The default `--decision-policy classifier` keeps the estimator's original label
decision. Optional `--decision-policy cost` changes the decision cutoff for the
selected binary risk classifier. It requires exactly two training classes,
including `RISK_LABEL = 1`, and both classes among the validation observations.
Other class codes remain valid for default classification; cost evaluation rejects
test labels outside the policy's two-class vocabulary.

```bash
python scripts/generate_synthetic_data.py --output data/raw/synthetic_dataset.csv --rows 1000 --seed 42
python -m src train --input data/raw/synthetic_dataset.csv --split-strategy temporal --temporal-gap 1 --decision-policy cost --false-negative-cost 5 --false-positive-cost 1 --output-dir runs
```

The error costs are relative weights supplied by the user. Defaults 5 and 1 are
illustrative, do not activate cost selection by themselves, and are not estimates
of environmental, monetary, or operational impact. Choose the ratio from the
intended decision and its consequences before looking at test performance. Costs
must be finite and non-negative, with at least one positive value.

For a fixed set of N validation observations, the selection objective is:

```text
cost per observation = (false_negative_cost * FN + false_positive_cost * FP) / N
```

Every case has equal weight; risk prevalence affects this objective. The policy
does not guarantee a minimum recall or a maximum alert volume.

## Selection Without Test Tuning

The pipeline first selects its classifier using the existing mean CV weighted F1.
It then clones that winner for each of the same training-only folds. Every clone
fits its imputer, feature engineering, scaler, and classifier on the fold's training
rows, and predicts probabilities only for its validation rows. Candidate model
scores and feature-ablation results continue to use original classifier decisions.

The validation probabilities are pooled for decision tuning. Random stratified CV
covers all training rows once. Expanding-window CV leaves initial warm-up history
and some gap observations without validation predictions; only rows actually in
validation windows contribute to cutoff selection. Their row IDs and coverage are
recorded. A validation row cannot contribute more than once.

Cutoff candidates are fixed at 0.00, 0.05, ..., 1.00. The rule predicts class 1
when its probability is **greater than or equal to** the cutoff; otherwise it
predicts the model's other class. Probability columns are located through `classes_`,
so the risk class need not be the second class. Grid values are rounded to their
declared two decimal places before scoring to preserve inclusive boundary behavior.

An additional `never_risk` candidate explicitly predicts the other class for every
case. Its serialized threshold is `null`. This differs from cutoff 1.00 when some
probabilities equal exactly one. Cutoff 0.00 already provides the all-risk decision.
Equal-cost candidates prefer the cutoff closest to 0.50, then the lower cutoff;
`never_risk` comes last in ties.

The lowest pooled validation cost selects the policy. The winner is fitted on the
full holdout training period, and its policy is saved before test evaluation. Test
probabilities are computed once and reused for the two prespecified decision rules:
original classifier and selected policy. Neither the test scores nor a scan of test
thresholds can revise the saved decision.

This optimizes the cutoff of the F1-selected classifier. It does not jointly search
all classifier families for minimum cost. The coarse grid may miss a better continuous
cutoff. The selected validation cost is tuning evidence, not an unbiased estimate
of generalization: both classifier selection and cutoff selection use these folds.

## Persisted Prediction Behavior

`RiskDecisionPipeline` is a scikit-learn `Pipeline` subclass. Its optional learned
`risk_decision_policy_` controls `predict` and accuracy `score`; `predict_proba` remains the classifier's
original probability output. Without the learned policy it uses ordinary pipeline
prediction. Scikit-learn cloning does not carry learned cutoff state into another fold.

The complete saved model applies the same rule when loaded directly, used by the
`predict` CLI, or evaluated by the frozen-model `robustness` command. No scoring-time
sidecar or manually supplied cutoff is required. Existing ordinary pipeline artifacts
retain their original behavior. Cost-policy artifacts identify their format as
`telemetry_pipeline_v2_threshold`; ordinary artifacts retain `telemetry_pipeline_v1`.

Changing a cutoff does not calibrate probabilities. A cutoff of 0.25 is a decision
criterion, not evidence that cases assigned score 0.25 fail 25% of the time. The
empirical CV objective does not rely on a calibrated-probability formula. Calibration
assessment and any future calibration fitting need their own training-only design.

## Measured Synthetic Example

Run date: 2026-11-04. The 1,000-row generator with seed 42 supplied 799 temporal
training rows, one excluded gap row, and the latest 200 test cases, including 81
risk cases. Expanding CV selected engineered-feature logistic regression. The
597 validation predictions covered 74.72% of the holdout training rows. With FN:FP
cost weights 5:1, the selected cutoff was **0.25**.

| Training-validation decision | Relative cost per observation |
| --- | ---: |
| Original classifier | 0.5092 |
| Selected cutoff | 0.3384 |

| Held-out metric | Original classifier | Selected cutoff 0.25 |
| --- | ---: | ---: |
| False negatives | 14 | 5 |
| False positives | 23 | 42 |
| Risk precision | 0.7444 | 0.6441 |
| Risk recall | 0.8272 | 0.9383 |
| Weighted F1 | 0.8162 | 0.7652 |
| Relative cost per observation | 0.4650 | 0.3350 |

The selected policy misses nine fewer risk cases but generates nineteen additional
false alerts. Under the illustrative 5:1 weights, this saves 26 relative cost units
over 200 cases, while reducing weighted F1 and risk precision. This is a measured
tradeoff under one assumed objective, not evidence of a universally better model.
Lower validation cost does not guarantee lower cost on future data.

This is the project's already published synthetic holdout, reused for a comparable
demonstration; it is not a new independent confirmation dataset. The generator's
dataset-wide quantile labels are not observed failures. New real-world claims need
representative sourced data, defensible outcomes, domain-justified costs, and a fresh
independent evaluation after any development informed by these reports.

Normalized table SHA-256:

```text
2abcb3a4a44bce08d65bef7aaf2a7762daac7ed014861546233b0a4a680e4639
```

Environment: Python 3.14.3, NumPy 2.3.5, pandas 3.0.2, scikit-learn 1.8.0. Exact
versions and full-precision metrics are also recorded in the generated run.

## Artifacts to Review

- `outputs/decision_policy.json`: cutoff/rule, risk and other labels, costs, selection objective, validation coverage, and tuning costs;
- `outputs/threshold_comparison.csv`: fixed candidate grid plus never-risk, FN/FP/TP/TN, risk precision/recall, total/per-case cost, and the selected flag;
- `outputs/threshold_validation_predictions.csv`: fold, cleaned row index, validation label, risk probability, original prediction, and selected-rule prediction;
- `figures/threshold_selection.png`: precision, recall, and cost against training-validation cutoff;
- `outputs/metrics.json`: headline test metrics for the selected rule and a separate comparison with the original classifier;
- `outputs/model_metadata.json` and the model artifact: the same persisted policy;
- `outputs/training_summary.md`: a reviewer-facing summary of the objective and held-out tradeoff.

Row indices refer to zero-based rows of the run's `processed/cleaned_dataset.csv`.
The threshold CSV uses an empty threshold field with rule `never_risk` for the
additional no-alert candidate. Risk precision is unavailable when no alerts occur;
risk recall is unavailable when no risk cases are observed. JSON represents these
undefined metrics as `null`, rather than manufacturing a measured zero.
