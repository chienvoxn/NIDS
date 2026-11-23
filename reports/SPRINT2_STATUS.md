# Sprint 2 implementation status

| Work item | Evidence | Status |
|---|---|---|
| 2.1 CIC Web Attack label normalization | `src/nids/sprint2_data.py`, `tests/test_sprint2_data.py` | Implemented; local synthetic tests needed |
| 2.2 Dataset-wide missing/Inf/nonnumeric and class distribution | `scripts/sprint2_audit.py` | Implemented; run on 10 CSVs in user's Windows machine |
| 2.3 Saved CNN 60 features vs CSV | Header test, manifest feature_order enforced in scripts | Implemented; run on all CSVs |
| 2.4 CTU CSV normalization | Binary labels, index drop, no invented time/host fields | Partial; a **separate** CTU training/eval remains |
| 2.5 Train/val/test leakage control | Explicit holdout provenance documentation and guardrails | **Not achieved** for saved checkpoint; need data split IDs |
| 2.6 Saved CNN per-class diagnostic | `scripts/sprint2_diagnostic_eval.py` | Implemented; Windows execution required |
| 2.6 Gaussian / RF / XGBoost / LightGBM, CTU evaluation | Need dedicated, reproducible training with train-only preprocessing and frozen holdout | **Not achieved** |

These statuses distinguish implementation of executable code from an actual successful run on the full private local datasets.
