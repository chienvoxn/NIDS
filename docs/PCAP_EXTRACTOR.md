## Source: README_M2_2.md

# M2.2 - Provisional flow extractor

This is **not** a CICFlowMeter-compatible extractor, and **must not** be used to make CNN V10 predictions yet.

## Install into repository

- Copy `cic_flow_extractor.py` into `src/nids/cic_flow_extractor.py`.
- Copy `test_cic_flow_extractor.py` into `tests/test_cic_flow_extractor.py`.
- Do **not** replace `src/nids/pcap.py`; this implementation imports `iter_pcap` from it.

## Commands (PowerShell, project root, .venv active)

```powershell
$env:PYTHONPATH = (Resolve-Path '.\src').Path
python -m pytest tests/test_cic_flow_extractor.py -v
python -c "from nids.cic_flow_extractor import audit_manifest; s,m=audit_manifest('models/final/deployment_manifest.json'); print('Provisional:', len(s), 'Missing:', len(m)); print('Missing names:', m)"
```

## Explicit limitations

1. Packet lengths are **IPv4 TCP/UDP payload lengths**, NOT verified to match the reference CICFlowMeter implementation. This may differ significantly from training values.
2. Simple inactive-flow timeout only; no FIN/RST termination, TCP connection tracking, active timeout, or stream reconstruction.
3. Classic PCAP Ethernet/IPv4 only; packets with VLAN tags, IPv6, fragments, truncated captures, and PCAPNG are skipped or rejected.
4. Some 0 values for valid empty-direction statistics are provisional; zero-time rates return NaN rather than fabricated finite values.
5. Flow IAT statistics use population standard deviation, pending matching reference.
6. The 9 unimplemented fields are not filled with zeros. Do not call the CNN inference pathway on output.
7. `extract_flows` keeps the active flows in memory, and is an offline prototype, not a production realtime monitor.
8. The current parser does not attempt to validate IP/TCP checksums, and assumes link type Ethernet.

## Next acceptance gates

- Confirm CICFlowMeter implementation and its exact feature definitions against a common PCAP.
- Implement outstanding fields with tests using known packet timestamps and lengths.
- Add integration tests for flow segmentation, packet replay parity and features relative to reference.
- Export a 60-column dataset only after parity passes; use artifact's exact feature_order.


---

## Source: README_M2_3.md

# M2.3: Provisional 60-feature PCAP extractor

This patch adds nine **provisional** feature formulas to the M2.2 extractor:

- `Active Mean`, `Active Std`, `Active Max`, `Active Min`: durations of bursts separated by gaps strictly greater than 5 seconds, measured in microseconds. Zero-duration single-packet bursts are included.
- `Idle Mean`, `Idle Std`, `Idle Max`, `Idle Min`: inter-burst gaps strictly greater than 5 seconds, measured in microseconds; zeros when no such gap exists.
- `Subflow Bwd Bytes`: mean backward **TCP/UDP payload** bytes per burst (a locally defined approximation, not verified CICFlowMeter semantics).

**WARNING:** `audit_manifest` now reports 60 names present; this checks only schema coverage, NOT correctness or feature parity. The overall extractor uses IPv4 L4 payload length; several CICFlowMeter versions use other packet-length and subflow conventions. It also lacks fully verified flow termination (FIN/RST/timeout), link types beyond Ethernet, VLAN, IPv6 and PCAPNG. DO NOT wire this extractor to V10 for production or claim model accuracy based on its outputs until comparing same-PCAP rows against the reference flow tool.

## Install files (Windows PowerShell, from project root)

Extract `M2_3_provisional_package.zip` **in your project root**, preserving paths:

```powershell
Expand-Archive -Path "$HOME\Downloads\M2_3_provisional_package.zip" -DestinationPath . -Force
$env:PYTHONPATH = (Resolve-Path '.\src').Path
python -m pytest tests/test_cic_flow_extractor.py -v
python -c "from nids.cic_flow_extractor import audit_manifest; a,b=audit_manifest('models/final/deployment_manifest.json'); print(len(a), b)"
```

Expected: **9 passed**; **60 []**. `src/nids/pcap.py` remains unchanged.

## Next checkpoint

Prepare a PCAP file AND a CICFlowMeter CSV generated from the exact same PCAP with known tool version. Compare flow identity/termination first, then packet length/IAT/flags, then Active/Idle/subflow conventions. Treat passing schema checks as insufficient for model inference.


---

