# Phase 5D - Stream Monitoring
Requires Phase 4.1 and 5C, Streamlit with st.fragment (>=1.37), Plotly. This replaces dashboard/final_app.py with the Phase 4.1 version plus the monitoring route. Back up local edits first.

Run from repository root with PYTHONPATH=src:

- python -m pytest -q tests/test_monitor_view.py
- python -m streamlit run dashboard/final_app.py
- Select Stream Monitoring. Choose Latest or a saved run.
- In another terminal: python scripts/run_stream_detection.py --csv data/demo/cic_cnn_demo.csv --batch-size 32 --delay 0.5

Auto refresh reads snapshots every two seconds. Partial trailing records are ignored until complete. Maximum 100000 records per snapshot; table shows latest 500. A missing summary does not prove a process is running. Replay alerts are simulated; shadow predictions never count as replay alerts. No verified live detection or packet timestamp latency is claimed. No PCAP required.
