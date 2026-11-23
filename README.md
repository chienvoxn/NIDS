# SE-AT-CNN NIDS — implementation starter

This repository is an **auditable working starter**, not a claim that the whole research project or Linux realtime deployment is complete.

## Current data facts

- CIC-IDS2017.zip: eight CICFlowMeter CSVs, **79 columns**, and **no timestamps, source/destination IP or flow IDs** in the uploaded copy. Consequently do **not** claim that arbitrary row order is temporal data. SE-AT-CNN is implemented but must not be trained on fabricated temporal windows.
- ctu-13-dataset.zip: 13 `.binetflow.parquet` files. Pilot scope: scenarios 5,11,12. Requires `pyarrow` to read; confirm schema and labeling before experiments.
- Neither ZIP contains raw PCAP. PCAP-to-flow demonstration requires a separate lab PCAP; this is a limited extractor, not a byte-for-byte CICFlowMeter replacement.

## Setup (Windows / Linux; no Docker required)

```bash
python -m venv .venv
# Linux: source .venv/bin/activate
# Windows PowerShell: .venv\\Scripts\\Activate.ps1
pip install -r requirements.txt
python -m pytest -q tests
```

Keep both ZIP archives **outside Git**, optionally at `data/raw/` (ignored). Explicitly provide paths:

```bash
python scripts/run_project.py inspect --cic /path/CIC-IDS2017.zip --ctu /path/ctu-13-dataset.zip
python scripts/run_project.py cic-audit --cic /path/CIC-IDS2017.zip
python scripts/run_project.py cic-pilot --cic /path/CIC-IDS2017.zip --per-label 300 --out reports/pilot
python scripts/run_project.py ctu-audit --ctu /path/ctu-13-dataset.zip --scenarios 5,11,12
python scripts/run_project.py ctu-graph --ctu /path/ctu-13-dataset.zip --scenarios 5,11,12
python scripts/run_project.py pcap-to-csv --pcap /path/lab.pcap --out reports/lab_flows.csv
```

## Evaluation warning

`cic-pilot` intentionally uses bounded class sampling and stratified random splitting only for **pipeline smoke testing**. Its scores should **not** appear as final independent-test results: training/test leakage through events or duplicates is possible. The final evaluation needs labeled event/host/time metadata, which are absent from the uploaded CIC CSV. Do not train the temporal CNN using order of these CSV rows. Model weights and JSON results are output only after actual training.

## Deliverables status

- [x] ZIP census + CIC schema audit
- [x] Reproducible CIC exploratory pilot RF/XGBoost/LightGBM/Gaussian
- [x] CNN, CNN+SE, CNN+AT, SE-AT-CNN definitions + shape tests
- [x] CTU label/graph/beaconing data adapters (requires Parquet dependency)
- [x] Basic classic PCAP IPv4 TCP/UDP flow extractor with tests
- [ ] CTU schema verification and experimental results
- [ ] Genuine temporal training and leakage-safe event-aware evaluation
- [ ] Validated offline/realtime feature parity on real lab PCAP
- [ ] Live Scapy sniffer, model integration, realtime latency and three lab demos

Original detailed specification: `PROJECT_SPEC_ORIGINAL.md`. Use this README as truthful implementation status.

## Offline extension (2026-11-11)

Nine Jupyter notebooks now reside in `notebooks/`:
`01_CIC_EDA`, `02_CTU_EDA`, `03_Gaussian_Anomaly`, `04_ML_Baselines`,
`05_CNN_Ablation`, `06_CTU_Botnet_Graph`, `07_Topology_Beaconing`,
`08_CTU_Temporal_Training`, `09_Final_Evaluation`.
They use the common `src/nids/` code; no fabricated model scores.

Run:

```bash
python scripts/generate_eda.py --cic /path/CIC-IDS2017.zip
python scripts/run_offline.py --ctu /path/ctu-13-dataset.zip
python scripts/run_offline.py --ctu /path/ctu-13-dataset.zip --graph
python scripts/train_ctu_cnn.py --ctu /path/ctu-13-dataset.zip --epochs 5
python -m pytest -q tests
```

**Important**: `scripts/run_offline.py` and CTU notebooks require `pyarrow`.
CTU experiments are not completed in this delivery because this runtime cannot
install it. See `reports/OFFLINE_STATUS.md` for the precise completed/blocked items.
The CIC CSVs do not support chronological temporal classification; do not claim
that the CTU temporal training constitutes completion of CIC multiclass SE-AT-CNN.
