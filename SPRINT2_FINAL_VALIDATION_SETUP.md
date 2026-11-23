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
