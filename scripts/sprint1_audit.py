#!/usr/bin/env python3
"""Read-only Sprint 1 audit of the NIDS repository (except explicit report outputs).

Run from the repository root after installing requirements:
  python scripts/sprint1_audit.py
  python scripts/sprint1_audit.py --cic data/raw/CIC-IDS2017.zip --ctu data/raw/ctu-13-dataset.zip

Never executes notebook code, unpickles model artifacts, or reads packet payloads.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
import re
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

PACKAGES = ["numpy", "pandas", "pyarrow", "scikit-learn", "xgboost", "lightgbm", "networkx", "matplotlib", "torch", "joblib", "pytest", "nbformat", "jupyterlab", "streamlit"]
REQUIRED_ARTIFACTS = ["models/final/model.pt", "models/final/preprocessing.joblib", "models/final/deployment_manifest.json", "data/demo/cic_pilot_sample.csv"]
EXPECTED_NOTEBOOK_ARTIFACTS = [
    "cic_expanded_splits.parquet", "experiment_manifest.json", "feature_sets.json",
    "validation_bias_grid.csv", "pilot_test_metrics.json", "baseline_test_metrics.csv",
    "scenario_holdout_metrics.json", "final_test_metrics.json",
]


def run(cmd, cwd):
    try:
        r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=120, check=False)
        return {"status": "ok" if r.returncode == 0 else "failed", "returncode": r.returncode, "stdout": r.stdout.strip()[-12000:], "stderr": r.stderr.strip()[-3000:]}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"status": "unavailable", "detail": str(exc)}


def pkg_versions():
    result = {}
    for name in PACKAGES:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def file_info(root, relative):
    p = root / relative
    if not p.is_file():
        return {"exists": False}
    result = {"exists": True, "bytes": p.stat().st_size}
    if p.suffix in {".pt", ".joblib"}:
        with p.open("rb") as handle:
            result["lfs_pointer_only"] = handle.read(40).startswith(b"version https://git-lfs.github.com/spec/")
    return result


def archive_info(path, kind):
    if not path.is_file():
        return {"status": "missing", "path": str(path), "note": "Original archive not available locally"}
    result = {"status": "ok", "path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path), "files": [], "schema": []}
    try:
        with zipfile.ZipFile(path) as z:
            result["corrupt_entry"] = z.testzip()  # CRC audit without extracting to disk
            members = [m for m in z.infolist() if not m.is_dir()]
            for m in members:
                result["files"].append({"name": m.filename, "uncompressed_bytes": m.file_size})
                if kind == "cic" and m.filename.lower().endswith(".csv"):
                    with z.open(m) as f:
                        line = f.readline().decode("utf-8-sig", errors="replace")
                        columns = [s.strip() for s in next(csv.reader([line]))]
                    result["schema"].append({"file": m.filename, "columns": len(columns), "contains_label": "Label" in columns, "contains_timestamp": any(x.lower() in {"timestamp", "time", "starttime"} for x in columns), "contains_src_ip": any(x.lower() in {"source ip", "src ip", "srcaddr"} for x in columns), "contains_dst_ip": any(x.lower() in {"destination ip", "dst ip", "dstaddr"} for x in columns)})
                elif kind == "ctu" and m.filename.endswith(".parquet"):
                    result["schema"].append({"file": m.filename, "format": "parquet", "note": "Row-level schema must be validated with pyarrow in CTU audit command"})
            if result["corrupt_entry"]:
                result["status"] = "invalid_crc"
    except (OSError, zipfile.BadZipFile, UnicodeError, csv.Error) as exc:
        result["status"] = "error"
        result["error"] = str(exc)
    return result


def inventory_artifacts(root):
    found = {}
    for name in EXPECTED_NOTEBOOK_ARTIFACTS:
        locations = []
        for parent in (root / "reports", root / "outputs", root / "data" / "processed", root / "artifacts"):
            if parent.exists():
                locations.extend(str(p.relative_to(root)) for p in parent.rglob(name) if p.is_file())
        found[name] = sorted(set(locations))[:30]
    return found


def md_table(items):
    return "\n".join("| " + " | ".join(str(v).replace("|", "\\|") for v in row) + " |" for row in items)


def report(data):
    packages = data["packages"]
    art = data["artifacts"]
    lines = [
        "# Sprint 1 — Environment & repository audit", "",
        f"Generated (UTC): `{data['generated_utc']}`", "",
        "> This is a factual environment snapshot, not evidence of independent model accuracy or realtime deployment.", "",
        "## Environment", "",
        f"- Platform: `{data['platform']}`", f"- Python: `{data['python']}`", f"- Executable: `{data['executable']}`", f"- Git HEAD: `{data['git']['head'].get('stdout', 'unavailable')}`", f"- Branch: `{data['git']['branch'].get('stdout', 'unavailable')}`", "",
        "## Dependency versions", "",
        md_table([["Package", "Version"], ["---", "---"]] + [[k, v or "NOT INSTALLED"] for k,v in packages.items()]), "",
        f"- pip check: **{data['pip_check']['status']}**", "",
        "## Git and Git LFS", "",
        f"- Git status: `{data['git']['status'].get('stdout', 'unavailable')}`", f"- Git LFS: `{data['git']['lfs_version'].get('stdout', 'unavailable')}`", "",
        md_table([["Artifact", "Exists", "Bytes", "LFS pointer only"], ["---", "---", "---:", "---"]] + [[k, v['exists'], v.get('bytes', '—'), v.get('lfs_pointer_only', '—')] for k,v in art.items()]), "",
        "## Dataset inventory", "",
    ]
    for kind in ("cic", "ctu"):
        inv = data["datasets"][kind]
        lines.extend([f"### {kind.upper()}", "", f"- Status: `{inv['status']}`", f"- Path: `{inv['path']}`"])
        if inv["status"] != "missing":
            lines.extend([f"- SHA256: `{inv.get('sha256','not computed')}`", f"- ZIP entries: {len(inv.get('files', []))}", f"- CRC test: `{inv.get('corrupt_entry')}`"])
            lines.extend(["", md_table([["File", "Columns / Format", "Label present"],["---","---","---"]] + [[x["file"], x.get("columns", x.get("format", "—")), x.get("contains_label", "not checked")] for x in inv.get("schema", [])]), ""])
        lines.append("")
    lines.extend(["## Notebook intermediate artifacts", "", md_table([["Artifact", "Locations found"], ["---", "---"]] + [[k, ", ".join(v) if v else "NOT FOUND in scanned directories"] for k,v in data["notebook_artifacts"].items()]), "", "## Test evidence", "", "- This script does not execute pytest automatically (to keep the audit reproducible and side-effect-safe).", "- Run `python -m pytest -q tests test_cic_flow_extractor.py --junitxml=reports/sprint1_pytest.xml` and attach the output.", "- Run `python -m pip freeze > reports/dependency_versions.txt` to capture all installed versions.", "", "## Known limitations / acceptance gates", "", "- Model checkpoint and preprocessing must both be real LFS binaries (not pointer text).", "- Existing CIC flow CSV may lack timestamps/IPs/flow IDs: do NOT manufacture temporal windows.", "- The PCAP feature extractor does not yet have verified CICFlowMeter parity.", "- CTU label counts and class coverage must be audited on the actual local archive.", "- Notebook intermediate artifacts from prior Kaggle runs may be absent; absence is not proof of nonexistence elsewhere.", "- No evidence of completed Linux realtime capture and 2–3 end-to-end lab scenarios is claimed.", ""])
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument("--cic", type=Path, default=Path("data/raw/CIC-IDS2017.zip"))
    p.add_argument("--ctu", type=Path, default=Path("data/raw/ctu-13-dataset.zip"))
    p.add_argument("--out", type=Path, default=Path("reports"))
    args = p.parse_args()
    root = args.root.resolve()
    out = args.out if args.out.is_absolute() else root / args.out
    out.mkdir(parents=True, exist_ok=True)
    resolve = lambda p: p if p.is_absolute() else root / p
    data = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(), "python": platform.python_version(), "executable": sys.executable,
        "packages": pkg_versions(),
        "git": {"head": run(["git", "rev-parse", "HEAD"], root), "branch": run(["git", "branch", "--show-current"], root), "status": run(["git", "status", "--short"], root), "lfs_version": run(["git", "lfs", "version"], root), "lfs_files": run(["git", "lfs", "ls-files"], root)},
        "pip_check": run([sys.executable, "-m", "pip", "check"], root),
        "artifacts": {x: file_info(root, x) for x in REQUIRED_ARTIFACTS},
        "datasets": {"cic": archive_info(resolve(args.cic), "cic"), "ctu": archive_info(resolve(args.ctu), "ctu")},
        "notebook_artifacts": inventory_artifacts(root),
    }
    (out / "dataset_inventory.json").write_text(json.dumps(data["datasets"], indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (out / "environment_audit.json").write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (out / "environment_audit.md").write_text(report(data), encoding="utf-8")
    print(f"Wrote {out / 'environment_audit.md'}")
    print(f"Wrote {out / 'dataset_inventory.json'}")
    print(f"pip check: {data['pip_check']['status']}")
    for kind, inv in data["datasets"].items():
        print(f"{kind}: {inv['status']}")
    if any(v.get('lfs_pointer_only') for v in data['artifacts'].values()):
        print("WARNING: Git LFS pointers detected instead of model binary")


if __name__ == "__main__":
    main()
