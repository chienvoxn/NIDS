# Phase 4.1 — SOC charts
Requires Phase 1–4 already installed. Stop Streamlit, back up dashboard/final_app.py, extract this ZIP into repository root. No checkpoint or training changes. Install Plotly in the existing virtual environment: `python -m pip install "plotly>=5.18,<7"`. Set `$env:PYTHONPATH="src"` in PowerShell, run `python -m pytest -q tests test_cic_flow_extractor.py` (expected 84 tests if previous 81 are present). Run `python -m streamlit run dashboard/final_app.py`.

- CSV Detection: upload existing CIC demo, run detection, inspect annotated binary/multiclass heatmaps and normalization switch.
- Replay: choose PortScan + BENIGN, inspect stacked batch counts and cumulative alerts in the table. Last batch may be smaller.
- Botnet Topology: choose Centralized, then P2P, without uploaded data. Upload beaconing_test.csv and compare with data/demo/beaconing_irregular_test.csv. Both have six events; irregular gaps are 1, 19, 2, 23, 5 seconds (mean 10, population std sqrt(84), CV sqrt(84)/10). Periodicity is a signal, not malware proof. Graph shape remains an undirected heuristic; plotted arrows show observed direction, edge labels count input events. For dense graphs, use degree table.

No PCAP or GeneratedLabelledFlows required. Plotly figure construction and 10 Phase 2–4.1 tests passed in build environment; Streamlit not installed there, so browser testing is still required. Live Linux AI is not included; existing metadata capture is not training-compatible CNN extraction.
