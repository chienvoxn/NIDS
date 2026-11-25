# Continuous Linux CNN + MHSA Shadow Inference

- CIC CNN + MHSA seed 42
- CPU inference, two configured CPU threads
- 74 captured packets
- 12 predicted flows
- 6 predictions before capture stopped
- 12 BENIGN predictions
- 0 application and inference drops
- 51 regression tests passed
- 15 pipeline validation checks passed
- No official alerts

Latency measures queue submission to prediction completion.
It excludes flow formation and expiration delays.

The experiment uses low-volume benign lab traffic.
Independent CICFlowMeter parity remains unverified.
No verified live attack detection is claimed.
