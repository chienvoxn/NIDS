# Sprint 1 — Environment & repository audit

Generated (UTC): `2026-11-23T08:29:42.843406+00:00`

> This is a factual environment snapshot, not evidence of independent model accuracy or realtime deployment.

## Environment

- Platform: `Windows-10-10.0.19045-SP0`
- Python: `3.11.9`
- Executable: `C:\Projects\NIDS\.venv\Scripts\python.exe`
- Git HEAD: `unavailable`
- Branch: `unavailable`

## Dependency versions

| Package | Version |
| --- | --- |
| numpy | 2.4.6 |
| pandas | 2.3.3 |
| pyarrow | 26.0.1 |
| scikit-learn | 1.6.1 |
| xgboost | 3.2.0 |
| lightgbm | 4.7.0 |
| networkx | 3.6.1 |
| matplotlib | 3.11.2 |
| torch | 2.15.0 |
| joblib | 1.6.0 |
| pytest | 9.2.0 |
| nbformat | 5.11.1 |
| jupyterlab | 4.6.6 |
| streamlit | 1.67.0 |

- pip check: **ok**

## Git and Git LFS

- Git status: `unavailable`
- Git LFS: `unavailable`

| Artifact | Exists | Bytes | LFS pointer only |
| --- | --- | ---: | --- |
| models/final/model.pt | True | 486457 | False |
| models/final/preprocessing.joblib | True | 5702 | False |
| models/final/deployment_manifest.json | True | 2810 | — |
| data/demo/cic_pilot_sample.csv | True | 1300519 | — |

## Dataset inventory

### CIC

- Status: `missing`
- Path: `C:\Projects\NIDS\data\raw\CIC-IDS2017.zip`

### CTU

- Status: `missing`
- Path: `C:\Projects\NIDS\data\raw\ctu-13-dataset.zip`

## Notebook intermediate artifacts

| Artifact | Locations found |
| --- | --- |
| cic_expanded_splits.parquet | NOT FOUND in scanned directories |
| experiment_manifest.json | NOT FOUND in scanned directories |
| feature_sets.json | NOT FOUND in scanned directories |
| validation_bias_grid.csv | NOT FOUND in scanned directories |
| pilot_test_metrics.json | NOT FOUND in scanned directories |
| baseline_test_metrics.csv | NOT FOUND in scanned directories |
| scenario_holdout_metrics.json | NOT FOUND in scanned directories |
| final_test_metrics.json | NOT FOUND in scanned directories |

## Test evidence

- This script does not execute pytest automatically (to keep the audit reproducible and side-effect-safe).
- Run `python -m pytest -q tests test_cic_flow_extractor.py --junitxml=reports/sprint1_pytest.xml` and attach the output.
- Run `python -m pip freeze > reports/dependency_versions.txt` to capture all installed versions.

## Known limitations / acceptance gates

- Model checkpoint and preprocessing must both be real LFS binaries (not pointer text).
- Existing CIC flow CSV may lack timestamps/IPs/flow IDs: do NOT manufacture temporal windows.
- The PCAP feature extractor does not yet have verified CICFlowMeter parity.
- CTU label counts and class coverage must be audited on the actual local archive.
- Notebook intermediate artifacts from prior Kaggle runs may be absent; absence is not proof of nonexistence elsewhere.
- No evidence of completed Linux realtime capture and 2–3 end-to-end lab scenarios is claimed.
