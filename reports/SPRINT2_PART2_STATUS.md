# Sprint 2 Part 2 status

Implemented: bounded stratified reservoir from actual CIC/CTU CSVs; exact-feature-duplicate grouping with conflict guard; train-only median imputation; RandomForest, XGBoost, LightGBM, diagonal Gaussian anomaly detector (fit benign train, threshold benign validation); per-class metrics and confusion matrices; unit tests.

**Not yet executed on user's local Windows CSV datasets.** Run `python scripts/sprint2_train_evaluate.py` and inspect output. Metrics from this pipeline are internal sampled pilots; they are **not independent holdout** measurements. CNN checkpoint provenance unchanged. CTU temporal/topology analyses are not possible from the selected two-CSV version.
