# NIDS demo integration phase 1
Additive patch for commit a12b088. Existing dashboard and training scripts untouched. Extract at repository root. Use trusted local model artifacts only; joblib deserialization executes Python code. Never upload untrusted checkpoint/preprocessing files.

PowerShell:
```powershell
$env:PYTHONPATH = 'src'
python -m pytest -q tests/test_demo_engine.py
python -m streamlit run dashboard/final_app.py
```
Select CIC provisional T01 or CTU final seed 42. CSV must contain the actual training features (unscaled values). CIC final seed 42 is preconfigured but may be unavailable until final export. Confirm output directory in final ZIP; update configs/demo_models.json if needed. Seed 42 is fixed for deployment, not selected using test scores. Stop/restart dashboard after replacing artifacts. Model report, preprocessing and checkpoint must be from the same run; schema checks cannot prove provenance for swapped same-schema weights. Keep original archives.

PCAP mode only provides existing basic flow statistics and graph indicators; no unverified CNN prediction. Existing extractor merges repeated 5-tuples and may undercount beaconing. No live capture or attack generation in this patch. Lab instructions in lab/README.md.
