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
