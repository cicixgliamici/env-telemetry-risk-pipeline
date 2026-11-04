# Temporal Evaluation

## Evaluation Question

Use `--split-strategy temporal` to evaluate how a model trained on earlier
observations predicts later observations. The default `stratified_random` protocol
remains available for reproducing the existing synthetic benchmark. Choose the
protocol before inspecting test scores; repeatedly choosing a protocol or model
after viewing holdout results would compromise that holdout.

```bash
python scripts/generate_synthetic_data.py --output data/raw/synthetic_dataset.csv --rows 1000 --seed 42
python -m src train --input data/raw/synthetic_dataset.csv --split-strategy temporal --test-size 0.2 --temporal-gap 1 --output-dir runs
```

The generator supplies a deterministic implementation check. Its labels use a
dataset-wide quantile of its own risk formula; chronological splitting does not
turn those labels into observed failures or independent evidence of field performance.

## Holdout Boundaries

After validation and deduplication, rows are sorted by timestamp. Equal timestamps
stay together, including when several measurements describe the same instant.
Comparisons use UTC instants; timestamps without a timezone are interpreted as UTC.

`--test-size` must be a finite fraction strictly between zero and one. For temporal
evaluation, the last `ceil(test_size * distinct_timestamp_count)` timestamp groups
become the test set. The resulting fraction of rows may differ from the requested
fraction when group sizes vary. Random stratified splitting retains its existing
row-fraction interpretation.

`--temporal-gap N` excludes the N groups immediately before the test period from
final training. It also excludes N groups between each CV training and validation
window. For example, with ten distinct times, `--test-size 0.2 --temporal-gap 1`
uses times 1 through 7 for training, excludes time 8, and tests on times 9 and 10.
The final training timestamp is always strictly earlier than the first test timestamp.

The default gap is zero. Gaps count distinct timestamps, not rows or elapsed hours:
with irregular sampling, one group is not a fixed physical duration. A gap for
overlapping measurement windows must be chosen from their real overlap and sampling
pattern. This option does not automatically infer those properties.

## Model Selection

The training period is divided into up to three expanding-window folds using
`TimeSeriesSplit` on distinct timestamp groups, then mapping groups back to rows.
At least three distinct training timestamps are required for two folds. Every
candidate receives identical folds. Each validation window occurs strictly after
its training window; training history grows between folds, with no shuffling.

Each fold fits the complete imputer, feature engineering, optional scaler, and
classifier pipeline using only that fold's historical training rows. Weighted F1
averaged across folds selects the winner. The winner is refitted on the complete
holdout training period and evaluated once on the final test period. Timestamp
and target columns remain excluded from model features.

Each fold's training window must include at least two classes so all classifier
families are trainable. If early history has only one class or the requested gap
leaves too little history, training fails explicitly. The implementation does not
shuffle observations, remove difficult folds, or fall back to random validation.
The bundled 12-row example has a long initial run of class 0 and cannot support
this temporal protocol; use a longer representative history.

Validation and test windows may contain a single class or classes that historical
training has never seen. Their class counts are recorded. Class-1 recall is
undefined when no class-1 cases are observed: test recall is `null`, and CV recall
means/stds use only folds with observed risk cases. `recall_risk_observed_folds`
records how many folds contribute; zero contributing folds produces `null` values.
Weighted F1 still uses every fold. Fold standard deviations are descriptive,
not confidence intervals. Fold metrics also depend on changing class prevalence.

## Review Evidence

Each run keeps the chosen strategy, test fraction, gap, and model settings in
`config.json`. `outputs/evaluation_protocol.json` additionally records:

- holdout training/test row indices, class counts, and time ranges;
- excluded holdout gap row indices;
- each CV fold's training/validation row indices, class counts, and time ranges;
- the split/CV strategy and units used for the test fraction and gap.

Row indices refer to zero-based data rows in that run's
`processed/cleaned_dataset.csv`, after deduplication and sorting. They are not
original CSV line numbers. The Markdown training summary includes the holdout
time ranges and fold boundaries. Metrics and model metadata identify the strategy
so temporal and random results cannot silently be presented as the same protocol.

## Remaining Limits

Chronological validation measures performance over the particular periods observed.
It does not establish reliability in future conditions that are absent from those
periods. Data collected at one site may also differ from data at another site.

The current schema has no defined site/sensor identity or label-availability time.
Separating unseen sites requires a grouped evaluation design once those identifiers
and the intended generalization question are defined. Timestamp grouping keeps
simultaneous rows together but does not provide site isolation. Delayed outcomes,
overlapping label horizons, and features unavailable at prediction time require
additional availability rules or purging beyond this timestamp gap.
