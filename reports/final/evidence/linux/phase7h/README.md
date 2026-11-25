# Phase 7H - Live Sliding Window

- Window size: 5 seconds
- Stride: 1 second
- Overlap: 80%
- Regression tests: 26/26 passed
- Detailed validation: 16/16 passed
- Captured packets: 75
- Accepted packets: 75
- Emitted windows: 15
- Periodic windows: 14
- Timer ticks: 200
- Application drops: 0
- Late packets: 0
- Flow and host detail validation: PASS
- CNN enabled: false
- Semantic parity: false
- Verified live AI detection: false

Limitations:
Low-volume isolated benign traffic.
Kernel packet loss and overload behavior not verified.
Partial-window handling remains limited.
No PCAP or packet payload persisted.
