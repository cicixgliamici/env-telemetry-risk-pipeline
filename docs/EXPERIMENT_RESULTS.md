# Reproducible Synthetic Experiment

This experiment checks whether the pipeline learns more than the majority-class rule and whether feature engineering helps. It is synthetic evidence about this implementation, not an estimate of environmental failure prediction in the field.

## Data and Protocol

Run date: 2026-11-03. The repository's generator produced 1,000 rows with seed 42: 620 class-0 and 380 class-1 observations. Labels come from the generator's noisy risk formula and dataset-wide quantile threshold, not observed failures. The generator and engineered features share domain assumptions; this favors evaluation within the same artificial world.

```bash
python scripts/generate_synthetic_data.py --output data/raw/synthetic_dataset.csv --rows 1000 --seed 42
python -m src train --input data/raw/synthetic_dataset.csv
```

The split is stratified with seed 42: 800 training rows and 200 held-out test rows. All five candidates use the same three stratified training folds. Each fold fits its own imputer and optional scaler. Random Forest uses 250 trees, depth 8, and balanced class weights. Logistic regression uses balanced weights, standard scaling, and 1,000 maximum iterations. Each family keeps identical estimator settings across its raw and engineered variants. Selection uses training CV weighted F1; only the winner is evaluated on the test split.

Raw variants use eight telemetry inputs. Engineered variants add eight derived features. The `DummyClassifier(strategy="most_frequent")` ignores the measurements. It is a selectable candidate, so a feature-based model is not guaranteed to win. Existing unsuffixed family names denote engineered variants; `_raw` identifies the ablation.

## Training Cross-Validation

| Candidate | Weighted F1 mean +/- std | Macro F1 mean | Class-1 recall mean |
| --- | ---: | ---: | ---: |
| Random Forest, engineered | 0.8044 +/- 0.0310 | 0.7918 | 0.7334 |
| Logistic regression, engineered | 0.7993 +/- 0.0191 | 0.7895 | 0.7960 |
| Random Forest, raw | 0.7770 +/- 0.0201 | 0.7608 | 0.6609 |
| Logistic regression, raw | 0.7982 +/- 0.0246 | 0.7885 | 0.7993 |
| Majority-class dummy | 0.4746 +/- 0.0019 | 0.3827 | 0.0000 |

The selected Random Forest exceeds the dummy by 0.3298 in mean CV weighted F1. Feature engineering changes Random Forest weighted F1 by +0.0274 and class-1 recall by +0.0724. For logistic regression, weighted F1 changes by only +0.0011 and recall declines by 0.0033. Feature engineering therefore does not improve every metric for every model.

Logistic regression has higher mean risk recall than the winner. The selection objective remains weighted F1, which shows a tradeoff rather than establishing that the selected model best serves a false-negative-sensitive application. Fold standard deviations describe variation across these three overlapping training experiments; they are not confidence intervals or statistical significance tests.

## Winner on the Held-Out Test Split

| Metric | Value |
| --- | ---: |
| Accuracy | 0.8200 |
| Weighted F1 | 0.8215 |
| Macro F1 | 0.8125 |
| Class-1 recall | 0.8158 |
| Class-1 support | 76 |
| Total test observations | 200 |

No test scores for alternative candidates were used to revise the selection. The test result is from one fixed split and does not establish stability across independent datasets or future time periods.

## Reproduction Identity

The SHA-256 below identifies the validated, deduplicated, timestamp-sorted table serialized as UTF-8 CSV by the training pipeline. It is not a raw-file checksum or proof of provenance.

```text
2abcb3a4a44bce08d65bef7aaf2a7762daac7ed014861546233b0a4a680e4639
```

Execution environment: Python 3.14.3, NumPy 2.3.5, pandas 3.0.2, scikit-learn 1.8.0. Version differences can change numeric results or normalized CSV serialization. Full precision scores, fold standard deviations, feature deltas, input identity, and environment versions are saved in the ignored runtime JSON artifacts. `training_summary.md` is regenerated for each run; this document preserves the reported experiment for review.

## Evidence Still Needed

The bundled 12-row example has no verified external provenance, and this generated dataset is not a substitute for observational data. No real-world performance claim follows from these results.

A real-data experiment needs a source and version, permission or license, collection dates, sensor and site definitions, mapping to the telemetry schema, and a defensible observed target. Site identifiers or repeated sensor windows may require grouped or chronological validation. Risk labels derived only from the project's own feature formula would demonstrate imitation of that rule rather than observed failure prediction. Calibration, drift, and operational error costs also remain to be assessed.
