## Source: SPRINT1_SETUP.md

# Sprint 1 — Setup and evidence capture (Windows PowerShell)

Run from your existing `C:\Projects\NIDS` project. This patch is additive; no model or Streamlit source is replaced.

```powershell
cd C:\Projects\NIDS
# If virtual environment is inactive:
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1

# Apply the deployment compatibility constraint (only if sklearn is not already 1.6.1):
python -m pip install -r requirements.txt -c constraints-windows.txt

$env:PYTHONPATH = (Resolve-Path '.\src').Path
python -m pip check
python -m pytest -q tests test_cic_flow_extractor.py --junitxml=reports/sprint1_pytest.xml
python -m pip freeze > reports/dependency_versions.txt
python scripts/sprint1_audit.py
```

If you have the two original ZIP files, copy them to `data/raw/` and rerun:

```powershell
python scripts/sprint1_audit.py --cic data/raw/CIC-IDS2017.zip --ctu data/raw/ctu-13-dataset.zip
python scripts/run_project.py cic-audit --cic data/raw/CIC-IDS2017.zip
python scripts/run_project.py ctu-audit --ctu data/raw/ctu-13-dataset.zip --scenarios 5,11,12
python scripts/check_m2_schema.py
```

Files produced:
- `reports/environment_audit.md`: environment, dependencies, artifact evidence and blockers
- `reports/environment_audit.json`: machine-readable audit
- `reports/dataset_inventory.json`: ZIP hash, entries, CIC column headers, CTU Parquet list
- `reports/sprint1_pytest.xml`: pytest test result (only after running tests)
- `reports/dependency_versions.txt`: full installed packages (`pip freeze`)

**Warning:** do not commit raw PCAP/ZIP, private host data, or your virtual environment. Review generated reports for sensitive local paths before publishing them. Current `.gitignore` ignores `data/raw/` but does not ignore every report output. Keep the environment-specific reports local unless intentionally sharing them.


---

## Source: SPRINT2_SETUP.md

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


---

## Source: SPRINT2_PART2_SETUP.md

# Sprint 2 Part 2 — CSV-only experimental evaluation

This package **adds** a separate reproducible pilot training script. It does not alter the Streamlit dashboard, existing checkpoint or saved preprocessing.

## Windows / PowerShell

```powershell
cd C:\Projects\NIDS
$env:PYTHONPATH = (Resolve-Path '.\src').Path
python -m pytest -q tests --junitxml=reports/sprint2_part2_pytest.xml
python scripts/sprint2_train_evaluate.py --dataset both --per-class 350
```

Full local CSV locations expected:
- `data/raw/CIC-IDS2017/MachineLearningCVE/*.csv` (8 files)
- `data/raw/CTU-13/CTU13-CSV-Dataset-main/*.csv` (2 files)

Results: `reports/sprint2_part2_metrics.json` includes per-model validation/test Precision, Recall, F1, Accuracy, confusion matrices, class distributions, and split provenance. For a quick check: `--dataset ctu --models rf --per-class 250`. To increase pilot coverage: `--per-class 1000` (more RAM and training time).

## Research integrity limitations

- An **internal** stratified random split of unique *exact feature vectors*. All exact duplicates go into the same split. Conflicting-label exact duplicates are rejected; near-duplicates may still cross folds.
- Each class is *capped* by reservoir sampling. Class priors are not preserved; thus accuracy/FPR are **not** representative of natural traffic prevalence.
- This is **not** a cross-day / cross-scenario / independent external holdout; source CSVs have no IP, timestamp, scenario. Report metrics as `internal sampled pilot only`.
- All models use the **same** internal fold, feature list, training-only median imputer; tree baselines never fit on validation/test. Gaussian mean/variance are fit **only to normal training rows**, threshold (99th percentile) selected **only on normal validation rows**.
- CIC baseline 15-class task and CTU binary task are **different tasks**. Never compare their metrics as if measured on the same benchmark.
- Saved CNN checkpoint metrics remain **diagnostic only** because training row identities for the original CNN have not been recovered. Never claim these new baseline metrics are a fair apples-to-apples comparison with the checkpoint.
- CTU CSV lacks IP/timestamp/scenario/topology labels; it cannot validate beaconing, P2P/centralized botnet classification.
- This run reports evaluation; it does not save deployable baseline models. Model persistence/retraining on a declared frozen split is a separate task.
- Dataset files are ignored by Git. Review JSON reports and run outputs before any commit.


---

## Source: SPRINT2_FINAL_VALIDATION_SETUP.md

# NIDS Sprint 2 - Final Validation (CSV-only)

This patch ADDS validation code; it does not alter your saved CNN checkpoint or the existing Sprint 2 scripts.

## Windows PowerShell

```powershell
cd C:\Projects\NIDS
Expand-Archive -Path "$HOME\Downloads\NIDS_Sprint2_Final_Validation.zip" -DestinationPath "C:\Projects\NIDS" -Force
$env:PYTHONPATH = (Resolve-Path '.\src').Path
python -m pytest -q tests --junitxml=reports/sprint2_final_pytest.xml
python scripts/sprint2_final_validation.py --dataset both --per-class 350 --seeds 42 43 44
```

The second command scans the ten raw CSVs once per dataset, then runs each model for each seed. This can take time on CPU; for a faster first check:

```powershell
python scripts/sprint2_final_validation.py --dataset ctu --models rf --per-class 250 --seeds 42 43
```

Results: `reports/sprint2_final_validation.json` (full per-seed confusion matrices and per-class metrics, sampled class counts, exact duplicate hash checks, stability mean/std). This is a **fixed-sample internal repeated-split evaluation**, **not** an independent holdout or temporal validation.

## Methodological caveats

- CIC's rare categories may have tiny test support (Heartbleed 11 in prior sample, SQL Injection 21, Infiltration 36), yielding uncertain per-class F1.
- Overlap of exact feature fingerprints between folds is tested. Near-duplicates, related hosts, campaigns, and original CNN split overlap are NOT ruled out.
- Sampling is class-balanced, so test prevalence is not representative of deployment prevalence; do not interpret accuracy as real-world attack-detection accuracy.
- Gaussian is binary anomaly detection; comparing its macro-F1 against CIC's multiclass baselines is **not a same-task comparison**.
- CTU CSV version has no trustworthy scenario/time/IP identifiers. No genuine scenario holdout is possible here.
- Do **not** compare these values numerically with the saved CNN V10's separate pilot sample as if test conditions were identical.
- Multi-seed stability varies the split of the *same fixed sample*, not the data acquisition distribution.

## Commit after inspecting outputs

```powershell
git add SPRINT2_FINAL_VALIDATION_SETUP.md src/nids/sprint2_validation.py scripts/sprint2_final_validation.py tests/test_sprint2_validation.py reports/sprint2_final_validation.json
git diff --cached --check
git commit -m "test: validate sprint 2 splits and multi-seed stability"
git push origin main
```

Keep local pytest XML and raw CSVs out of Git.


---

## Source: SPRINT3_SETUP.md

# NIDS Sprint 3: Multi-Scale Residual CNN + SE + MHSA

## Scope
A **new** CNN-family experiment, separate from deployed V10. The pre-existing `models/final` checkpoint, preprocessing, and Streamlit dashboard are unchanged. This is a **CIC-IDS2017 CSV-only sampled internal evaluation**, not temporal/external validation. The saved V10 and prior Sprint 2 baselines were not evaluated on this exact split; do not compare their metric numbers as a fair ranking.

## Architecture
60 numeric columns -> feature tokenizer -> two multi-scale (k=1,3,5) blocks with optional residual skip -> optional SE -> optional MHSA -> attention/mean/max pooling -> classifier. CNNs over tabular columns depend on their arbitrary feature order; an MLP or boosting baseline may outperform them.

Four experiments: `ms_cnn`, `ms_residual`, `ms_residual_se`, `ms_residual_se_mhsa`.

## PowerShell (from C:\Projects\NIDS, active .venv)

```powershell
$env:PYTHONPATH = (Resolve-Path '.\src').Path
python -m pytest -q tests --junitxml=reports/sprint3_pytest.xml
python scripts/sprint3_ablation.py --per-class 150 --epochs 3 --seeds 42
```

If smoke run succeeds, conduct a stronger experiment (more CPU time):

```powershell
python scripts/sprint3_ablation.py --per-class 350 --epochs 12 --seeds 42 43 44
```

Optional train-only class weighting experiment, with **separate** output:

```powershell
python scripts/sprint3_ablation.py --per-class 350 --epochs 12 --seeds 42 43 44 --class-weights --output reports/sprint3_ablation_weighted.json
```

Outputs: `reports/sprint3_ablation.json` and optional weighted variant. This does not save a deployable model: retraining + provenance verification are required before promoting a model into `models/final`.

## Experimental validity
- One fixed per-class reservoir sample across all variants and seeds; one split and preprocessing per seed, shared by all CNN variants.
- Exact-identical feature rows grouped in the same fold; conflicting labels rejected. Near-duplicate and original collection leakage may remain.
- Imputer, signed-log transform and scaler use train only (signed-log formula itself has no fitted parameters). Clip scaled values to [-12,12].
- Model epoch selected using validation macro-F1; test used once, after best epoch selection.
- This first Sprint 3 patch does **not** retrain LightGBM/XGBoost on the matching split and therefore does not establish CNN superiority over baselines.
- Do not use old V10 pilot Accuracy as a fair comparison.


---

