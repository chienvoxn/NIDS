# Phase 7G - Continuous Capture and Periodic Expiration

## Environment

- Ubuntu WSL2
- Python 3.11
- Scapy AsyncSniffer
- Interface: nids-veth0
- Isolated network namespace: nids-lab

## Configuration

- Stability runs: 5
- Capture duration: 15 seconds per run
- Timer interval: 250 ms
- Timestamp safety lag: 500 ms
- TCP FIN grace: 1 second

## Results

- Regression tests: 16 passed
- Stability tests: 5/5 passed
- Captured packets: 380
- Packets included in flows: 380
- Bidirectional flows: 60
- TCP flows: 30
- UDP flows: 30
- Timer ticks: 300
- Periodically closed flows: 10
- Unsupported packets: 0
- Application dropped packets: 0
- Late timestamp drops: 0
- Invalidated flows: 0
- Packet accounting: PASS
- Flow accounting: PASS

## Limitations

- Low-volume benign traffic only.
- One application drop was observed in an earlier V4 run.
- The earlier drop was not reproduced in the five stability runs.
- Kernel-level packet loss has not been fully measured.
- Timestamp synchronization is not guaranteed under overload.
- Startup and shutdown failure paths require further testing.
- CICFlowMeter semantic feature parity is not verified.
- CNN inference and live attack alerts remain disabled.
- No PCAP or packet payload was persisted.

## Conclusion

Phase 7G passed the bounded isolated lab stability tests.
This does not certify production readiness or verified live AI detection.
