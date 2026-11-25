# NIDS Demo Phase 2 — CSV, scenarios and topology
Requires Phase 1 (demo_engine.py and configs/demo_models.json). Replaces only dashboard/final_app.py; adds independent helpers, tests and a passive Linux metadata capture script. No training changes. No PCAP upload.

Windows PowerShell from repository root:

```powershell
$env:PYTHONPATH = "src"
python -m pytest -q tests/test_demo_lab.py
python -m pytest -q tests test_cic_flow_extractor.py
python -m streamlit run dashboard/final_app.py
```

Choose CIC provisional T01, upload data/demo/cic_cnn_demo.csv. Run CSV Detection, then Attack Scenario Replay for Normal, PortScan, DoS, Bot and Mixed. Export predictions and metrics for each separately. Metrics apply only to the selected sample. No FPR is defined for attack-only data; no detection rate is defined for benign-only data. Batch timestamps are processing times, not original traffic times. Replay runs sequential batches without network capture or original time pacing. Topology uses shape heuristics on controlled examples; benign hub is intentionally indistinguishable from a malicious star based on shape alone. Optional metadata CSV must contain src,dst,time (ISO timestamps). IAT candidates do not constitute verified botnet detection.

## Linux (authorized isolated lab only)
Optional passive capture: install Scapy in the project's Linux virtual environment, identify the lab interface, and run with suitable capture permissions. Example (replace eth0):

```bash
sudo .venv/bin/python scripts/live_flow_metadata.py --iface eth0 --seconds 30 --out outputs/live_metadata.csv
```

The script stores no raw packets, only directional 5-second aggregate metadata. It is not a CICFlowMeter implementation, does not perform CNN inference and cannot be used to claim realtime AI detection. Aggregated timestamps are unsuitable for faithful beacon interval measurements; use actual connection timestamps for IAT validation. Complete live feature parity and verify a model on labeled lab traffic before claiming live detection. Keep existing PCAP code/tests for regression unless instructor explicitly waives source-code requirement.

## Final checkpoint
Unpack official CIC results into the repository, verify model registry path and select CIC final seed 42. Repeat tests and scenarios. Do not select seed based on test scores. Do not commit data or checkpoints accidentally.
