# Phase 5C - CNN stream replay and experimental shadow mode

Requires Phase 1 (DemoEngine), Phase 5B for shadow mode, and CIC final seed 42 artifacts. No PCAP or additional datasets required. Run only in an authorized lab.

## Windows PowerShell

```powershell
$env:PYTHONPATH = "src"
python -m pytest -q tests/test_stream_detection.py
python scripts/run_stream_detection.py --csv data/demo/cic_cnn_demo.csv --batch-size 32 --delay 0.1
```

Replay uses existing training-compatible CSV features, not packet capture. It writes predictions.csv, predictions.jsonl, replay_alerts.jsonl and summary.json under outputs/stream_detection/<run_id>. Processing timestamps are UTC, not original traffic timestamps. Batch processing time includes preprocessing and inference; it is not detection latency. The replay is bounded by max-rows, not seconds. The delay is artificial. The summary complete flag denotes a normal bounded run, not exhaustive capture.

## Linux shadow mode (optional experimental diagnostic)

Start Phase 5B capture on your own isolated lab interface. Once its provisional_features.csv exists, use another terminal (with PYTHONPATH=src and the virtual environment active):

```bash
python scripts/run_stream_detection.py --mode shadow --ack-unverified --csv outputs/live_features/RUN_ID/provisional_features.csv --seconds 60
```

Replace RUN_ID with the actual directory. Shadow reads from the beginning and polls appended single-line rows from the Phase 5B writer. It stops after the configured observation interval or row limit. It is not a guaranteed lossless streaming transport. Predictions are marked shadow_only_unverified; no replay alerts are emitted in this mode. Provisional features are rejected in default replay mode. Only use replay with known training-compatible CSV. There is no verified live alert mode and no automatic parity approval. Independent reference comparisons, coverage of flow boundaries and feature conventions, real capture tests and end-to-end latency evaluation remain necessary. Do not interpret experimental shadow outputs as confirmed attacks.

## Verification

Four tests use a fake engine to verify replay, gate, shadow and batch limits. They do not validate actual CNN inference or Linux capture. Run the replay command above to exercise the actual local checkpoint. Keep outputs local; logs may contain network metadata.
