# Phase 4 — SOC Dashboard

Requires installed Phase 1, 2 and 3. No training, new dataset or PCAP needed.

## Install (PowerShell, repository root)
Stop Streamlit with Ctrl+C. Back up dashboard/final_app.py and existing .streamlit/config.toml if present before extracting the ZIP with -Force. Merge existing Streamlit settings into the provided theme config if needed.

```powershell
$env:PYTHONPATH = "src"
python -m pytest -q tests test_cic_flow_extractor.py
python -m streamlit run dashboard/final_app.py
```

## Acceptance checks
- CSV Detection: CIC provisional T01, data/demo/cic_cnn_demo.csv, Run detection. Check Traffic overview, Evaluation and Alerts & export; filtering and downloading should preserve results for the same input.
- Replay: select PortScan + BENIGN, DoS Hulk + BENIGN or Bot + BENIGN; each uses the Phase 3 mixed_case selection (up to 100 per class, seed 42). Check progress, metrics and export. Existing Normal/PortScan/DoS/Bot/Mixed presets remain available.
- Botnet Topology: select Centralized, P2P or Benign hub without upload. Upload data/demo/beaconing_test.csv to override example; select host pair. For six equally spaced events expect 2 hosts, 1 link, indeterminate topology, mean IAT 10 seconds, std and CV zero. Check timeline and interval plot. The IAT candidate table uses existing botnet implementation; the selected-pair cards use explicitly labeled population standard deviation.
- CTU: use CTU final seed 42 with compatible CTU CSV only.

Charts use Matplotlib and Streamlit native chart controls, not Plotly. The graph is static. The theme file may replace existing configuration: back it up and merge any other settings. No model or algorithm files changed. Python syntax and seven existing Phase 2/3 unit tests checked; actual Streamlit UI and real checkpoint inference not run in the build environment. Replay is not live packet detection. Regular connections or centralized topology do not prove infection.
