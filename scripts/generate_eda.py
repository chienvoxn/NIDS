"""Reproducible CIC label chart and CTU schema summaries."""

import sys, json, argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nids.datasets import cic_audit, ctu_parquet, ctu_label_audit

p = argparse.ArgumentParser()
p.add_argument("--cic", required=True)
p.add_argument("--ctu")
p.add_argument("--out", default="reports/eda")
a = p.parse_args()
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
audit = cic_audit(a.cic)
(out / "cic_audit.json").write_text(json.dumps(audit, indent=2))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

labels = list(audit["labels"])
vals = list(audit["labels"].values())
plt.figure(figsize=(11, 6))
plt.barh(labels, vals)
plt.xscale("log")
plt.title("CIC-IDS2017 label distribution (log scale)")
plt.tight_layout()
plt.savefig(out / "cic_label_distribution.png", dpi=130)
plt.close()
if a.ctu:
    df = ctu_parquet(a.ctu, (5, 11, 12))
    ctu = ctu_label_audit(df)
    ctu["counts"] = {str(k): int(v) for k, v in ctu["counts"].items()}
    (out / "ctu_label_audit.json").write_text(json.dumps(ctu, indent=2, default=str))
print("EDA saved to", out)
