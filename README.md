# Network Intrusion Detection System (NIDS)

## Extended CNN with Squeeze-and-Excitation and Multi-Head Self-Attention

A research-oriented Network Intrusion Detection System based on an extended Convolutional Neural Network (CNN), integrating Squeeze-and-Excitation (SE) and Multi-Head Self-Attention (MHSA) for network attack classification and botnet detection.

The project includes data analysis, preprocessing, classical machine learning baselines, deep learning experiments, ablation studies, hyperparameter tuning, model deployment packaging, and an interactive Streamlit dashboard.

## 1. Project Objectives

- Analyze network traffic and extract relevant flow-level features.
- Detect anomalous network behavior using Gaussian Anomaly Detection.
- Classify normal traffic and network attacks using machine learning and deep learning.
- Evaluate the contributions of SE and MHSA through ablation studies.
- Investigate class imbalance and train-only SMOTE.
- Optimize model hyperparameters using validation data.
- Evaluate botnet detection and cross-scenario generalization.
- Develop and validate a network intrusion detection prototype.

## 2. Datasets

### CIC-IDS2017

The primary dataset for multiclass network attack classification.

- 15 original traffic labels.
- Normal traffic and multiple network attack categories.
- Eight CICFlowMeter CSV files in the original project archive.
- Feature selection and preprocessing using training data.
- Expanded experimental configuration with 60 selected features.

The available CSV archive does not contain timestamps, source/destination IP addresses, or flow identifiers. Therefore, arbitrary CSV row order must not be interpreted as temporal network traffic.

### CTU-13

The supplementary dataset for binary botnet detection and cross-scenario evaluation.

- 13 botnet traffic scenarios.
- Flow-level botnet classification.
- Scenario-based robustness evaluation.
- Distribution drift and threshold analysis.
- Validation-only CNN hyperparameter tuning.

The available flow-only Parquet files do not provide sufficient metadata for genuine temporal reconstruction, beaconing analysis, or communication graph validation.

Raw PCAP files must be obtained separately for packet-level and realtime experiments.

## 3. Models

### Classical Machine Learning

- Gaussian Anomaly Detection
- Random Forest
- XGBoost
- LightGBM
- K-Nearest Neighbors

### Deep Learning

- CNN
- CNN + SE
- CNN + MHSA
- CNN + SE + MHSA

The proposed architecture combines convolutional feature extraction, channel attention, and multi-head self-attention.

The CIC deployment model uses a feature-tokenized multiscale CNN with SE, MHSA, and train-only SMOTE. This architecture operates on flow-level feature vectors and should not be interpreted as a chronological temporal model.

## 4. Research Notebooks

| Notebook | Purpose |
|---|---|
| `01-cic-eda.ipynb` | CIC-IDS2017 exploratory data analysis |
| `02-ctu-eda.ipynb` | CTU-13 exploratory data analysis |
| `03-gaussian-anomaly.ipynb` | Gaussian anomaly detection |
| `03b-cic-preprocessing-fe.ipynb` | Preprocessing and feature engineering |
| `04-ml-baselines-v2.ipynb` | Classical machine learning baselines |
| `05-cnn-ablation-v10.ipynb` | SMOTE sensitivity and model stability |
| `05-cnn-final.ipynb` | Checkpoint verification and deployment packaging |
| `06-ctu-botnet-graph.ipynb` | CTU-13 botnet detection experiments |
| `07-ctu-13-cross-scenario-drift-botnet-thresh.ipynb` | Distribution drift and threshold analysis |
| `08-ctu-cross-scenario-robus.ipynb` | Cross-scenario robustness evaluation |
| `09-final-experimental-evaluation-evidence-audi.ipynb` | Experimental evaluation and evidence audit |
| `NIDS_CTU13_Hyperparameter_Tuning.ipynb` | Validation-only hyperparameter tuning |

Historical experimental notebooks are preserved in the `backup/before-cleanup` branch.

## 5. CTU-13 Hyperparameter Tuning

Six configurations were evaluated using validation Binary F1 on an NVIDIA Tesla T4 GPU.

| Parameter | Selected value |
|---|---|
| Selected trial | T01 |
| CNN width | 32 |
| Learning rate | 0.001 |
| Dropout | 0.1 |
| Batch size | 512 |
| Class weighting | None |
| Random seed | 42 |
| Maximum epochs | 15 |
| Early stopping patience | 5 |
| Mixed precision | AMP FP16 |
| Validation Binary F1 | 98.4545% |

The configuration was selected using validation data only. This result is not an independent test result and does not establish superiority over classical machine learning baselines.

Detailed experimental evidence is available in `reports/final/07_HYPERPARAMETER_TUNING_CTU.md`.

## 6. Repository Structure

```text
NIDS/
├── .github/workflows/   # Continuous integration
├── dashboard/           # Streamlit dashboard
├── data/                # Local datasets and demo samples
├── models/final/        # Deployment model and preprocessing
├── notebooks/           # Research notebooks
├── outputs/             # Experimental and demo outputs
├── reports/             # Reports and experimental evidence
├── scripts/             # Training, evaluation and audit scripts
├── src/nids/            # Core Python implementation
├── tests/               # Automated tests
├── requirements.txt
├── constraints-windows.txt
├── PROJECT_SPEC_ORIGINAL.md
└── README.md
```

## 7. Installation

### Windows PowerShell

```powershell
git clone https://github.com/chienvoxn/NIDS.git
cd NIDS
git lfs pull

python -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r requirements.txt -c constraints-windows.txt

$env:PYTHONPATH = "src"
```

### Linux

```bash
git clone https://github.com/chienvoxn/NIDS.git
cd NIDS
git lfs pull

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

export PYTHONPATH=src
```

Git LFS must be installed to retrieve the actual deployment model and preprocessing artifacts.

## 8. Dataset Preparation

Store the original datasets locally:

```text
data/raw/
├── CIC-IDS2017.zip
└── ctu-13-dataset.zip
```

Original datasets, large intermediate artifacts, and generated training checkpoints should not be committed directly to Git.

## 9. Run the Dashboard

```bash
python -m streamlit run dashboard/app.py
```

Open:

http://localhost:8501

The dashboard uses the packaged model, preprocessing artifact, and deployment manifest in `models/final/`.

The current dashboard performs flow-level classification. It is not yet evidence of a validated live packet-capture NIDS.

## 10. Testing

Check dependencies:

```bash
python -m pip check
```

Run the automated test suite:

```bash
python -m pytest -q tests test_cic_flow_extractor.py --import-mode=importlib
```

Run the repository audit:

```bash
python scripts/sprint1_audit.py
```

Latest local cleanup verification:

- 69 tests passed.
- 0 failed tests.
- 1 non-blocking timestamp parsing warning.
- No broken package requirements.

GitHub Actions provides automated dependency checks, unit tests, and repository audits.

## 11. Implementation Status

### Completed or Implemented

- [x] Dataset inventory and schema auditing
- [x] CIC-IDS2017 and CTU-13 EDA
- [x] Data preprocessing and feature selection
- [x] Gaussian anomaly detection experiments
- [x] Classical machine learning baselines
- [x] CNN, CNN + SE, CNN + MHSA, and extended CNN implementations
- [x] Preliminary architecture ablation experiments
- [x] SMOTE sensitivity and stability experiments
- [x] CIC deployment model packaging
- [x] CTU-13 cross-scenario robustness experiments
- [x] CTU-13 validation-only hyperparameter tuning
- [x] PCAP-to-flow extractor implementation and unit tests
- [x] Streamlit dashboard implementation
- [x] Automated testing and continuous integration

### Remaining Research and Validation

- [ ] Complete controlled architecture ablation experiments.
- [ ] Complete official CTU-13 CNN evaluation.
- [ ] Establish fair baseline comparisons on equivalent data splits.
- [ ] Validate temporal and beaconing analysis using suitable metadata.
- [ ] Validate centralized and P2P botnet graph analysis.
- [ ] Verify offline and realtime feature compatibility.
- [ ] Integrate live Linux packet capture and model inference.
- [ ] Conduct controlled network laboratory experiments.
- [ ] Measure end-to-end detection latency and false positive rates.
- [ ] Complete the final research report.

## 12. Evaluation Principles

- Fit preprocessing transformations using training data only.
- Apply oversampling to training data only.
- Select hyperparameters and thresholds using validation data.
- Keep test data outside model selection.
- Compare models using equivalent features, splits, and evaluation metrics.
- Distinguish smoke tests, pilot experiments, validation results, and independent test results.
- Do not fabricate temporal information or missing network metadata.
- Report experimental limitations and negative results transparently.

## 13. Project Documentation

- Original specification: `PROJECT_SPEC_ORIGINAL.md`
- Research reports: `reports/final/`
- Experimental evidence: `reports/final/evidence/`
- Model deployment artifacts: `models/final/`
- Historical implementations: `backup/before-cleanup` branch

## 14. Project Status

This repository contains an actively developed network intrusion detection research prototype.

Offline experiments, deployment packaging, and automated testing have been implemented. Independent final evaluation and validated Linux realtime deployment remain ongoing research objectives.

All packet capture and security testing must be conducted in authorized and controlled environments.