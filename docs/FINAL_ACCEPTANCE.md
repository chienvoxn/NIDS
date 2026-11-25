# NIDS Final Acceptance

## Requirement 6 - Linux Realtime Monitoring

Implemented:
- Linux packet capture using Scapy.
- Flow feature extraction.
- CNN + MHSA inference.
- Gaussian anomaly detection.
- Communication graph analysis.
- Centralized/P2P topology classification.
- Periodic behavior analysis.
- Experimental security events.

Recorded labeled live experiment:
- 158 packet observations.
- 15 processed benign flows.
- 15/15 ground-truth matches.
- Zero CNN false positives.
- Eight Gaussian false positives.
- Mean queue-to-fused-prediction latency: 105.24 ms.

Live attack and botnet detection accuracy remains unverified.

## Requirement 7 - Attack Scenarios

Offline held-out evaluation:

| Attack | Attack samples | Correct type | Recall |
|---|---:|---:|---:|
| PortScan | 1000 | 1000 | 100% |
| DoS Hulk | 1000 | 996 | 99.6% |
| Bot | 293 | 114 | 38.91% |

Saved-prediction alert demonstration:
- PortScan: 10/10.
- DoS Hulk: 10/10.
- Bot: 3/10.

These experiments are offline and replay evaluations,
not verified live network attacks.

## Requirement 8 - Experimental Evidence

Available evidence:
- reports/final/acceptance/
- reports/final/evaluation_summary/
- reports/final/evidence/final_demo/
- reports/final/evidence/labeled_evaluation/
- reports/final/evidence/scenario_fusion/

Previous full regression: 157 passed.
Botnet topology examples: 4/4 correct.
Latest related integration regression: 23 passed.

## Limitations

- Live feature semantic parity remains unverified.
- Communication topology does not prove botnet infection.
- Beaconing heuristics are experimental.
- No verified live attack detection accuracy is claimed.
- Final Word report and screenshots require consolidation.
