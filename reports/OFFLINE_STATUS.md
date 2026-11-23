# OFFLINE NIDS — execution and limitations

## Data sources inspected

- CIC-IDS2017.zip: 8 CSV files, 2,830,743 labeled records, 15 unique original labels, no timestamp/host/flow ID columns. Full-stream label count from `reports/eda/cic_audit.json`.
- ctu-13-dataset.zip: 13 ZIP entries, all `.binetflow.parquet`; selected scenarios 5,11,12 for the offline experiments. ZIP file inventory, not row-level audit.
- Neither archive includes a raw `.pcap`; no PCAP-to-flow parity result can be asserted.

## Completed with execution evidence

1. Data ZIP inventory and complete CIC label audit.
2. CIC bounded exploratory pilot: RF, XGBoost, LightGBM, Gaussian (original reports in `reports/pilot`). **Random stratified pilot split is NOT an event-disjoint final evaluation**.
3. Implemented SE-AT-CNN architecture and all four ablation configurations. Verified tensor shapes in unit tests.
4. Implemented CTU causal window utilities, history-only graph counts, binary label mapping, scenario-based holdout experiments, and torch temporal training entrypoint. **Not run on provided CTU Parquet**, missing `pyarrow` in runtime.
5. Created nine executable Jupyter notebooks with descriptions and explicit provenance gates.
6. 9/9 tests pass using installed Python dependencies.

## Blockers to genuinely complete ALL offline research requirements

- Parquet decoding: install `pyarrow` in a Kaggle environment, run notebooks 02, 06, 07 and 08, then check scenario labels and availability of normal examples. No results should be reported before that.
- Temporal CIC classification: supplied CIC CSV has no real timestamps, source/destination, or flow identifiers. Cannot create real chronological windows from shuffled CSV rows; obtain original capture/labeled flow metadata to train temporal SE-AT-CNN on CIC as specified. CTU temporal SE-AT-CNN is only a separate botnet pilot, NOT a substitute for CIC multiclass results.
- Leakage-safe CIC evaluation: original event/host/time metadata required for validated event-disjoint evaluation; current benchmark is a reproducible exploratory sanity check only.
- Graph topology correctness: synthetic star, mesh, and benign counterexamples exist, but a validated graph topology classifier with a labeled balanced benchmark has not been trained. CTU graph topology is not intrinsically labeled centralized/P2P.
- Offline vs future realtime feature parity: capture schema validation still requires lab PCAP and live/replay packets; this is a deployment prerequisite, not an offline score.

## Running on Kaggle

Add the code repository as a Kaggle Dataset, add both ZIPs as datasets, and enable GPU for Torch notebooks. In the notebook setup cell override environment variables `CIC_ZIP` and `CTU_ZIP` to Kaggle Input paths. Install dependencies first: `pip install -r requirements.txt`. Run notebook 01 and 02 to inspect actual label coverage before calling CTU training notebooks.

## Next critical experiments

1. CTU audit: `python scripts/run_project.py ctu-audit --ctu /path/ctu-13-dataset.zip --scenarios 5,11,12`.
2. Check verified class coverage for each held-out scenario; if missing, select another set of scenario groups without using test performance to choose the split.
3. Flow-only vs graph-enhanced RF: `python scripts/run_offline.py --ctu /path/ctu-13-dataset.zip --graph` and without `--graph`.
4. Temporal CTU experiment: `python scripts/train_ctu_cnn.py --ctu /path/ctu-13-dataset.zip --epochs 5`.
5. Add original CIC chronological metadata, then implement a rigorous multiclass temporal evaluation. Do not report demo architecture outputs as trained attack-classifier results.
