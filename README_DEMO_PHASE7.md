# Phase 7A/B — Linux lab smoke test (khong PCAP)

Requires existing Phase 1–5D, CIC final seed 42, original nids.cic_flow_extractor. Use an Ubuntu VM with isolated network; all smoke traffic below is restricted to 127.0.0.1. No attack tools, no additional dataset. Copy the project including trusted checkpoint/preprocessing/metrics; do not copy Windows .venv. Linux Python 3.11 recommended to match training dependencies. Install repository requirements appropriate for Linux (do not apply Windows constraints), then verify pip check. Install Scapy if missing. CPU Torch is sufficient. Run from project root. No live verified alerts are enabled.

## 1. Environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install scapy
python -m pip check
export PYTHONPATH=src
python -m pytest -q tests/test_live_monitor.py tests/test_live_features.py tests/test_stream_detection.py tests/test_monitor_view.py tests/test_linux_lab_check.py
ip -br link
```

If Python 3.11 or compatible dependencies are not available, resolve installation before capture. Windows venv is not portable. Use only trusted source/artifacts in the VM. The commands below use sudo only for packet capture, not CNN or Streamlit. No public interface is needed.

## 2. Capture smoke test

Terminal A (from project root, venv active):

```bash
sudo env PYTHONPATH="$PWD/src" "$PWD/.venv/bin/python" scripts/run_live_features.py --iface lo --seconds 30 --max-packets 10000
```

After capture starts, Terminal B (same project root):

```bash
source .venv/bin/activate
export PYTHONPATH=src
python scripts/lab_loopback_smoke.py
```

Expected six benign TCP connections completed; packet count depends on OS and capture behavior. Loopback may duplicate observations. Wait for terminal A to finish; use its printed output directory. If root permissions block reads, change ownership of that particular run directory to the VM user. Do not claim flow counts must be six; flow segmentation and capture duplicates differ. Candidate rows at capture end are partial.

```bash
CAPTURE=$(find outputs/live_features -mindepth 1 -maxdepth 1 -type d | sort | tail -1)
python scripts/check_linux_lab.py --capture "$CAPTURE"
```

## 3. Experimental shadow after capture

```bash
python scripts/run_stream_detection.py --csv "$CAPTURE/provisional_features.csv" --mode shadow --ack-unverified --seconds 10 --batch-size 32 --max-rows 10000
SHADOW=$(find outputs/stream_detection -mindepth 1 -maxdepth 1 -type d | sort | tail -1)
python scripts/check_linux_lab.py --capture "$CAPTURE" --shadow "$SHADOW"
```

Use printed directory paths if running multiple sessions. This sequential smoke test checks ingestion, not concurrent detection latency. Shadow must have zero replay alerts and verified_live=false. Any attack predictions on this benign synthetic traffic are experimental, not evidence of verified FPR. The checker does not certify matching capture/shadow provenance beyond row counts and metadata consistency. For concurrency, start shadow after capture creates CSV and before capture ends, allowing more time than capture duration. Phase 5B may flush flows only at capture end. Do not report this as validated realtime detection.

## 4. Dashboard fix

The supplied dashboard/stream_view.py changes stacked bars to BENIGN + Attack predictions instead of Total + Attack. The maximum height of a full 32-row batch is 32. The supplied run_stream_detection.py normalizes explicit missing numeric tokens in shadow mode only. Backup both files before overwrite. No changes to training or weights.

## 5. Remaining acceptance gates

Independent same-traffic reference flow exporter with known CICFlowMeter version, matched flow boundaries and matching IDs; compare all 60 features with scripts/audit_live_features.py. Schema checks and synthetic tests alone are not semantic parity. Without reference, keep shadow status. No large PCAP dataset is required but an independent reference is still needed. No genuine attacks executed by this package. Actual Linux capture, shadow inference, timing, and labeled live lab scenarios remain unverified. Do not substitute CSV replay or this benign smoke test for live attack evaluation. Record capture start/end, packet/flow counts, dropped/invalidated, source, model hash, and evidence paths. Do not use window aggregates as connection events for beaconing.
