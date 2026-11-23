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
