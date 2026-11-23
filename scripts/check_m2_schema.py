"""Read-only M2 schema audit. Python 3.9+, standard library only.
Run from repository root: python scripts/check_m2_schema.py
"""
import argparse
import csv
import json
from pathlib import Path


def resolve(root, relative):
    p = root / relative
    if not p.is_file():
        raise FileNotFoundError(f"Missing file: {p}")
    return p


def header(path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return [c.strip() for c in next(csv.reader(f))]


def main():
    p = argparse.ArgumentParser(description="NIDS M2 schema compatibility audit (no prediction).")
    p.add_argument("--root", type=Path, default=Path.cwd())
    p.add_argument("--manifest", default="models/final/deployment_manifest.json")
    p.add_argument("--csv", default="data/demo/cic_pilot_sample.csv")
    p.add_argument("--pcap-csv", default=None, help="Optional CSV exported by current PCAP extractor")
    p.add_argument("--out", default="reports/m2_schema_audit.json")
    args = p.parse_args()
    root = args.root.resolve()
    manifest = json.loads(resolve(root, args.manifest).read_text(encoding="utf-8-sig"))
    expected = manifest["feature_order"]
    if len(expected) != 60 or len(expected) != len(set(expected)):
        raise ValueError("Invalid manifest feature order: expected 60 distinct names")
    demo = header(resolve(root, args.csv))
    missing_demo = [name for name in expected if name not in demo]
    demo_positions = [demo.index(name) for name in expected if name in demo]
    result = {
        "model": manifest.get("model_name"),
        "manifest_n_features": len(expected),
        "demo_n_columns": len(demo),
        "demo_missing_required": missing_demo,
        "demo_feature_order_matches": not missing_demo and demo_positions == sorted(demo_positions),
        "demo_extras": [x for x in demo if x not in expected],
        "pcap_schema": None,
        "ready_for_realtime_inference": False,
        "status": "AUDIT_ONLY; no raw-PCAP feature parity or replay validation performed",
    }
    if args.pcap_csv:
        actual = header(resolve(root, args.pcap_csv))
        result["pcap_schema"] = {
            "total_fields": len(actual),
            "matched_required": [x for x in expected if x in actual],
            "missing_required": [x for x in expected if x not in actual],
            "extra_fields": [x for x in actual if x not in expected],
            "required_coverage": f"{len([x for x in expected if x in actual])}/60",
        }
    out = root / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("Manifest feature count:", len(expected))
    print("Demo columns:", len(demo))
    print("Demo missing required columns:", len(missing_demo))
    print("Demo feature order preserved:", result["demo_feature_order_matches"])
    if result["pcap_schema"]:
        print("PCAP CSV feature coverage:", result["pcap_schema"]["required_coverage"])
        print("Missing names:", ", ".join(result["pcap_schema"]["missing_required"]))
    print("Report:", out)
    print("Realtime feature parity: NOT VERIFIED")


if __name__ == "__main__":
    main()
