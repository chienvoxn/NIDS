# Phase 5B — Live feature candidates (no PCAP files)

Requires existing repository `src/nids/cic_flow_extractor.py` and its dependencies. Reuses its provisional formulas, not a new validated CICFlowMeter implementation. No GeneratedLabelledFlows or PCAP dataset required. No changes to CNN/checkpoints, Phase 5A or dashboard.

## Windows: check integration

```powershell
$env:PYTHONPATH = "src"
python -m pytest -q tests/test_live_features.py
```

Expect four passes when existing extractor is importable. Three tests use a fake flow factory to isolate lifecycle, limits and schema; the fourth uses the real extractor. In the build environment only three passed and the fourth skipped (repository module unavailable locally); GitHub source was inspected. No Scapy capture was executed.

## Linux: passive monitoring on an authorized isolated lab interface

Use the project's Linux Python environment and Scapy. Identify the interface with `ip -br link`. Run with authorized packet-capture privileges, substituting your isolated interface:

```bash
sudo env PYTHONPATH=src .venv/bin/python scripts/run_live_features.py --iface LAB_INTERFACE --seconds 60
```

Writes `outputs/live_features/<run_id>/provisional_features.csv` and `summary.json`. No payload or raw packet file is stored. IP addresses and timestamps are sensitive metadata; keep local. Times are epoch seconds. Features follow the checkpoint report order. Missing formula keys raise an error. The capture is passive and bounded to 300 seconds/100000 packets; flow state has separate limits. Idle and max-age expiration occurs when another supported packet arrives; remaining rows at stop are explicitly partial. This is not a sliding-window CNN detector. The synchronous callback can lose packets under load; counters do not measure all kernel drops. Capacity invalidation can lead to later fragments of the same flow appearing as new candidates. TCP RST closes a flow; full TCP FIN lifecycle, capture boundary, UDP lengths, active/idle, subflow, direction and packet-size conventions are not validated against CICFlowMeter. All rows are marked `provisional_not_cnn_ready`. Never feed them to DemoEngine or upload them for CNN detection as if training-compatible.

## Reference comparison (optional, requires independent matched evidence)

The retained MachineLearningCSV contains feature rows but no reliable matching identifiers for newly captured packets. Do not join unrelated rows or invent matches. For an independent compatible reference extractor operating on the same authorized lab traffic, prepare two CSVs with explicitly matched unique `match_id` and all 60 features. Then:

```bash
PYTHONPATH=src python scripts/audit_live_features.py --candidate candidate_matched.csv --reference reference_matched.csv
```

Reports finite comparisons and tolerance counts per feature; missing/nonfinite pairs do not count as matches. This is a diagnostic, not a parity certificate. Reference capture alignment, flow boundary policy and extractor version must be documented separately. No reference data supplied in this phase; no automatic CNN approval. If no compatible live reference is possible, discuss scope with instructor or train a separate model on appropriately labeled live features rather than claiming existing CNN compatibility.

## Next

Validate the capture and reference semantics; only then integrate Final CNN and measure alert latency from actual lab event timestamps. Phase 5A metadata monitoring and Phase 3 CSV replay remain usable independently.
