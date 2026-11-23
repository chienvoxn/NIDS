# Sprint 2: CSV-only data quality, label mapping and checkpoint diagnostics

This patch is additive. It does not edit trained `model.pt`, `preprocessing.joblib`, `deployment_manifest.json` or the Streamlit dashboard. It is tailored to `chienvoxn/NIDS` at commit `158a401` and the 10 locally downloaded CSV files.

## Windows PowerShell (from C:\Projects\NIDS)

```powershell
$env:PYTHONPATH = (Resolve-Path '.\src').Path
python -m pytest -q tests --junitxml=reports/sprint2_pytest.xml
python -m pytest -q test_cic_flow_extractor.py
python scripts/sprint2_audit.py
python scripts/sprint2_diagnostic_eval.py --csv data/demo/cic_pilot_sample.csv
```

If the standalone test file at repository root conflicts with `tests/test_cic_flow_extractor.py`, continue to execute the commands separately as shown. If the checkpoint diagnostic errors out because its saved preprocessing object differs, **do not modify checkpoint or manifest**; share full traceback for a precise adapter fix.

Outputs:
* `reports/sprint2_data_quality.json` — complete streaming dataset-wide label counts and quality on selected numeric columns.
* `reports/sprint2_cnn_diagnostic.json` — accuracy, macro-F1, per-class precision/recall/F1, confusion matrix **on a diagnostic CSV sample only**.
* `reports/sprint2_pytest.xml` — automated tests.

The audit reads the ten files by chunks; expect millions of rows, and allow it to finish. The report counts NaN, Inf and numeric coercion errors on **selected** features. It does **not** estimate duplicates because none of the available CSVs have durable flow identity; comparing all rows naively can be misleading and costly. It does **not** impute, train, refit or generate false timestamps/IPs.

## Important scope limitations and experimental gates

* CIC model uses 60 specific features. Compare every CSV header against the saved manifest; extra CIC columns are ignored, feature order is preserved.
* Normalize the corrupted Web Attack spelling **for metric labels only**, without altering `manifest['label_order']` or class indices.
* The CTU-13 two-CSV re-release has binary `Label`, no IP/timestamp or scenario IDs. It supports binary **flow classification** with a separate trained pipeline, not graph topology/beaconing or CTU cross-scenario holdout.
* A true leakage-free supervised test requires the original train/val/test membership or disjoint data provenance. The saved model has `split_sha256` but **not** sample membership; the supplied demo/sample and the same source CIC archive cannot be assumed an unseen holdout.
* Correct workflow for a **future new experiment**: split data first (by groups/session where available); fit imputers, scalers, SMOTE and threshold selection on train/validation only; freeze configuration; evaluate once on untouched test; record per-class reports and hashes. This patch **does not** fake retraining or independence.
* If you plan to commit results, consider excluding local environment / machine paths. Raw CSV data are ignored by Git.

## Sprint 2 acceptance

1. `pytest` passes including the newly added label-schema tests.
2. Streaming audit finishes all 10 CSVs and confirms unknown_labels empty (otherwise investigate).
3. CIC schema matches 60/60 across 8 files; CTU schema is documented as binary 57 numeric candidates excluding index and label (check real file headers for specifics).
4. Diagnostic saved CNN evaluation runs and exports a 15-class classification report and confusion matrix; this is NOT holdout performance.
5. Reproducible independent test: **open** until split membership or a disjoint holdout is demonstrated.
6. CTU train/validation/test model performance: **open**, no numbers fabricated by this patch.
