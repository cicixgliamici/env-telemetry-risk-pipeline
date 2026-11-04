# Review Roadmap

The six stages prepare an understandable, reproducible research demonstration.
Completion means ready for technical review; synthetic experiments do not establish
performance on observed failures or readiness for operational decisions.

| Stage | Delivered behavior | Evidence |
| --- | --- | --- |
| 1. Prediction correctness | Probability columns follow model class labels; inference preserves scoring rows and fitted preprocessing. | Prediction and leakage tests |
| 2. Reproducible runs | Unique directories, execution status, settings, environment snapshots, and input fingerprints. | Run manifests and model metadata |
| 3. Evaluation design | Random and chronological holdouts, expanding historical folds, whole timestamp groups, exclusion gaps, baseline and feature comparisons. | Evaluation protocol and training reports |
| 4. Robustness | Frozen-model measurement perturbations and case-mix sensitivity with training-only scales and repeated audits. | [Robustness results](ROBUSTNESS_RESULTS.md) |
| 5. Decision policy | Training-validation cost selection with a persisted binary risk cutoff and explicit precision/recall tradeoffs. | [Decision policy](DECISION_POLICY.md) |
| 6. Reviewer handoff | Separate training persistence, labeled charts, formatter gate, installed-package reproduction, and one linked evidence bundle. | [Reviewer checklist](REVIEWER_CHECKLIST.md) and generated `REVIEW_SUMMARY.md` |

## Remaining research and deployment work

1. Obtain sourced observational data with a documented license and defensible risk labels.
2. Reserve fresh external or prospective evaluation data before further tuning.
3. Elicit decision costs from the actual application and assess probability calibration.
4. Assess grouped assets/sites, time dependence, and uncertainty across independent datasets.
5. Add operational monitoring, access controls, deployment configuration, and ownership
   only after a concrete deployment setting is defined.

Hyperparameter search should remain inside training validation. Any repeated inspection
of the current synthetic holdout makes subsequent comparisons exploratory.
