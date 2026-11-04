# Publication Checklist

Publish the repository as a reviewable synthetic ML demonstration. Follow the
[reviewer checklist](REVIEWER_CHECKLIST.md) for verification and the
[review status](REVIEW_STATUS.md) for measured results and outstanding limits.

## Before pushing

- Confirm `git status --short` is empty after the phase commits.
- Inspect `git log --oneline origin/main..HEAD` and `git diff --stat origin/main..HEAD`
  to review all unpublished work, including earlier commits.
- Keep generated runs, models, local planning notes, credentials, caches, and build
  products out of Git. `.gitignore` does not remove already tracked files.
- Retain the small bundled input CSV, synthetic generator, tests, license, and
  documentation: these are intentional review materials.

```bash
python -m ruff check src tests scripts
python -m ruff format --check src tests scripts
python -m pytest -q
python -m build --no-isolation
python -m src review --rows 1000 --seed 42 --repeats 5 --output-dir runs
```

The three follow-up commits correspond to frozen-model robustness (phase 4),
training-only cost decisions (phase 5), and reviewer handoff/publication preparation
(phase 6). The earlier phases remain in their existing commits.

## When publication is authorized

1. Refresh remote references and inspect branch divergence before pushing.
2. Push the reviewed branch and verify the remote Python 3.11/3.12, package, and
   container jobs. Local checks do not establish remote CI success.
3. If preparing a release, choose its version and tag explicitly, then attach the
   wheel/source archive produced by successful CI. Dependency snapshots are
   environment records, not portable lockfiles.
4. Share the generated review bundle separately if desired; preserve its complete
   directory so the relative report links remain valid.

No push, tag, package upload, or public release is performed by local commit
preparation. Observational validation and application-specific decision costs remain
future work; avoid claims of operational predictive performance.
