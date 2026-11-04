# Review Status

Prepared on 2026-11-04 for technical review. The six stages in the
[roadmap](ROADMAP.md) are implemented. The project demonstrates a reproducible
synthetic ML workflow; real-world predictive validity remains unestablished.

## Verified locally

- Full test suite: **208 passed**, including split isolation, class-aware prediction,
  persisted decision rules, stress reproducibility, and review report links.
- Ruff lint and format checks: passed across source, tests, and scripts.
- Python compilation and whitespace checks: passed.
- The complete 1,000-row review workflow with seed 42 and five stress repetitions:
  completed; confusion-matrix labels and counts inspected visually.
- Wheel and source distribution: built successfully; wheel Python files match the
  checkout. The source archive includes documentation, helpers, test fixtures, and
  the bundled input dataset.
- Installed-wheel smoke: the full review workflow completed on 250 rows from a
  separate working directory, using the wheel's package files and existing local
  dependencies rather than importing source files from the checkout.

Environment: Windows, Python 3.14.3, NumPy 2.3.5, pandas 3.0.2,
scikit-learn 1.8.0, Matplotlib 3.10.8, pytest 8.2.0, Ruff 0.4.4.
Each runtime bundle records its full installed environment separately.

## Reproduce and inspect

```bash
pip install -e .[dev]
python -m ruff check src tests scripts
python -m ruff format --check src tests scripts
python -m pytest -q
python -m build --no-isolation
python -m src review --rows 1000 --seed 42 --repeats 5 --output-dir runs
```

Open the printed `REVIEW_SUMMARY.md`. The output directory is ignored by Git;
share the entire generated directory to preserve relative report and figure links.
For the detailed reading order, use the [reviewer checklist](REVIEWER_CHECKLIST.md).

## Results to discuss

| Protocol | Selected model | Held-out weighted F1 | Risk recall |
| --- | --- | ---: | ---: |
| Stratified random | Random forest, engineered | 0.8215 | 0.8158 |
| Temporal, one timestamp gap | Logistic regression, engineered | 0.8162 | 0.8272 |
| Temporal, training-selected cost policy | Logistic regression, engineered | 0.7652 | 0.9383 |

The illustrative 5:1 cost policy chooses threshold 0.25. On the shared 200-row
temporal holdout, missed risks decrease from 14 to 5 and false alerts increase
from 23 to 42. Weighted F1 decreases while the weighted error cost decreases
from 0.465 to 0.335 per observation. This is a tradeoff under supplied weights,
not evidence that one rule is universally better. The holdout has already been
inspected, so this comparison is exploratory.

The frozen temporal classifier is sensitive to measurement loss: 30% missingness
reduces mean weighted F1 from 0.8162 to approximately 0.7521 across five draws.
See [robustness results](ROBUSTNESS_RESULTS.md) for other stresses and variability.

## Outstanding external checks and limits

The updated GitHub Actions workflow is configured for Python 3.11/3.12 tests,
lint, formatting, wheel installation outside the checkout, and container smoke
validation. It has not been run remotely for these local changes. Docker is
unavailable on this machine, so no local container success is claimed.

Observational data, sourced labels, application-specific costs, probability
calibration, and fresh external evaluation remain prerequisites for operational
claims. The requirements snapshot is environment evidence rather than a portable
dependency lockfile. All changes remain local; no push was performed.
