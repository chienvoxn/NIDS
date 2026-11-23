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
