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
