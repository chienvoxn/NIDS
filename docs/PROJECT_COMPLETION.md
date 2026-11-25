# Network Intrusion Detection System

## 1. Project objective

Network anomaly detection, multiclass attack classification,
botnet analysis and experimental Linux realtime monitoring.

## 2. Datasets

- CIC-IDS2017: 60 input features, 15 classes.
- CTU-13: 57 input features, binary classification.
- Separate training, validation and test partitions.
- Model preprocessing fitted on training data.

## 3. Implemented models

- Random Forest
- XGBoost
- LightGBM
- KNN
- CNN
- CNN + SE
- CNN + MHSA
- CNN + SE + MHSA
- Gaussian anomaly detection

CNN + MHSA is the primary demonstration model.

## 4. System architecture

Network packets
  -> Scapy capture
  -> Flow and host aggregation
  -> Provisional feature extraction
  -> Trained CNN inference
  -> Gaussian anomaly scoring
  -> Communication graph analysis
  -> Experimental security events
  -> CSV, JSONL and terminal logs

Sliding-window monitoring is available as a separate component.

## 5. Evaluation

CIC CNN + MHSA mean weighted F1: 99.2559%.
CTU CNN + MHSA mean binary F1: 99.1594%.

CNN + MHSA does not outperform the strongest ML baselines.

Three offline scenarios:
- PortScan
- DoS Hulk
- Bot

Full scenario metrics are available in the evaluation evidence.

Saved-prediction replay demonstration:
- PortScan: 10/10 detected.
- DoS Hulk: 10/10 detected.
- Bot: 3/10 detected.

## 6. Linux demonstration

Three successful experimental live fusion sessions:
- 227 captured packet observations.
- 36 processed flows.
- 66/66 validation checks.
- No application or inference queue drops reported.

Final security-event demonstration:
- 76 packet observations.
- 12 processed flows.
- 6 predictions during capture.
- 4 Gaussian anomaly events on benign traffic.
- Mean queue-to-fused-prediction latency: 437.45 ms.
- P95 latency: 1117.96 ms.

## 7. Botnet analysis

CTU binary ML comparison, communication graphs,
beaconing statistics and synthetic centralized/P2P examples.

Graph topology alone does not establish malicious activity.

## 8. Software verification

Full regression before security-event additions:
152 passed, zero failures, zero skipped.

Subsequent security integration test group:
25 passed.

See committed test evidence for actual results.

## 9. Limitations

- Independent CICFlowMeter semantic parity is unverified.
- Live extracted features remain provisional.
- Realtime model outputs are experimental shadow events.
- Attack scenarios were evaluated offline and replayed.
- Live experiments used benign isolated TCP/UDP traffic.
- Verified live attack detection is not established.
- CTU processed data lacks communication endpoint metadata.
- Synthetic topology checks do not prove botnet infection.
- Queue latency excludes flow formation time.
- High-load performance and kernel packet loss are unverified.
- No automated blocking or official attack alerts.

## 10. Evidence

reports/final/evaluation_summary/
reports/final/evidence/scenario_fusion/
reports/final/evidence/final_demo/
reports/final/evidence/linux/

## 11. Implementation files

src/nids/
scripts/
notebooks/
dashboard/
tests/
