# Sprint 2 Final Validation - Evidence status

- Implemented: sample fixed across seeds, per-seed duplicate-disjointness audit, per-class confusion matrices and precision/recall/F1 inherited from Sprint 2 Part 2, aggregate macro-F1 mean/std/min/max, rare-class warnings.
- Requires user's local run on 10 CSVs; this package does **not** contain measurements for that run.
- Cannot establish original V10 CNN train/test independence, cross-day CIC generalization, or CTU scenario holdout from these inputs.
- Completion criterion: tests pass, experiments finish without leakage assertions, results are reviewed, limitations included in report.
