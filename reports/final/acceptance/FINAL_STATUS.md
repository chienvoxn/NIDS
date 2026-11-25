# NIDS - Final Implementation Status

## Implemented functions

- PCAP reading and processing.
- Flow-level and host-level analysis.
- Gaussian anomaly detection.
- Multiclass attack classification.
- CNN + MHSA inference.
- CTU-13 botnet classification.
- Centralized/P2P graph analysis.
- Linux realtime packet capture.
- Realtime feature extraction.
- Experimental attack and anomaly events.
- Botnet signal and topology integration.
- CSV and JSONL logging.

## Model evaluation

CIC CNN + MHSA:
- Mean weighted F1: 99.2559%.

CTU CNN + MHSA:
- Mean binary F1: 99.1594%.

## Attack scenario evaluation

| Scenario | Accuracy | Recall | F1 |
|---|---:|---:|---:|
| PortScan | 99.90% | 100% | 99.90% |
| DoS Hulk | 99.70% | 99.60% | 99.70% |
| Bot | 86.00% | 38.91% | 55.75% |

Evaluation mode: offline held-out data.

## Live monitoring

- Four recorded live fusion sessions.
- 385 packet observations.
- 51 processed benign flows.
- Latest ground truth matching: 15/15.
- Latest CNN false positives: 0/15.
- Latest Gaussian false positives: 8/15.
- Latest mean queue-to-fused-prediction latency: 105.24 ms.

## Streaming inference

- Processed flows: 1208.
- Attack predictions: 859.
- Benign predictions: 349.
- Replay alerts: 859.

## Botnet analysis

- Synthetic topology classification: 4/4.
- Dedicated integration tests: 5/5.
- Related regression tests: 23/23.

## Research limitations

- Live attack classification accuracy remains unverified.
- Live botnet infection detection remains unverified.
- CICFlowMeter semantic parity remains unverified.
- Graph topology alone does not prove botnet infection.
- Bot recall requires improvement.
- CNN does not outperform the strongest ML baselines.
- PCAP dataset download is excluded from project scope.
- Final Word report and screenshots require consolidation.

## Next research stage

- Class imbalance handling.
- Minority class recall improvement.
- Macro F1 optimization.
- Comparison with existing CNN + MHSA checkpoint.
