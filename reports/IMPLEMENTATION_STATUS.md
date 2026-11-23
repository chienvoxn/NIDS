# Verified implementation status — 2026-11-11

## Uploaded archive inventory

- CIC-IDS2017.zip: 8 CSV files, 884,645,759 uncompressed bytes, 79 columns each.
- ctu-13-dataset.zip: 13 Parquet files, 122,260,500 uncompressed bytes. Parquet schema and row-level labels are **not verified** because `pyarrow` / `fastparquet` is unavailable here and package downloads are blocked. They are supported by the included loader once dependencies are installed.
- Neither ZIP contains PCAP. This uploaded CIC CSV copy has **no timestamp, source/destination IP or flow ID**, which prevents defensible causal temporal sequence reconstruction or end-to-end training/realtime feature parity directly from it.

## Full CIC CSV audit

`reports/cic_full_audit.json`: 2,830,743 rows and 15 label strings, including 2,273,097 BENIGN and 557,646 non-BENIGN. Web Attack label contains a replacement Unicode character in uploaded CSV; retain raw label string until harmonization is documented.

## Experiment run

Pilot reservoir cap: 150 examples per class (fewer for very rare labels). Stratified random train/test for **smoke testing only**, not a leakage-safe final benchmark:

| Baseline | Accuracy | Macro-F1 |
|---|---:|---:|
| Random Forest | 0.9015 | 0.8893 |
| XGBoost | 0.8994 | 0.8531 |
| LightGBM | 0.9015 | 0.8989 |

Gaussian normal-only training pilot: precision 0.9643, recall 0.2512, FPR 0.1081. Threshold was fit on train benign quantile rather than held-out validation, so results are exploratory.

Outputs in `reports/pilot/`, including classification reports and actual model artifacts. Do not label these as final testing results.

## Tests

`pytest -q tests`: five tests pass covering four CNN ablations shapes, Gaussian, CTU Background handling, NetworkX graph and generated Ethernet IPv4 TCP packet parsing.

## Remaining work for instructor acceptance

1. Install Parquet dependencies and audit rows/schema/label mapping for CTU-5, CTU-11, CTU-12; verify Background handling and graph features.
2. Obtain a small labeled PCAP from a controlled lab for PCAP-to-flow and offline/replay parity; cannot pretend the CSV archive contains PCAP.
3. Obtain genuinely timestamped flow data for temporal SE-AT-CNN, or request agreement for a feature-axis non-temporal CNN alternative. Do not invent timestamps.
4. Perform event/host-aware split, 5-baseline+4-ablation model training, metrics and stress-test.
5. Implement and validate Linux live sniffer + integrated inference, anomaly/botnet alerts, three authorized lab tests and measured end-to-end latency.
