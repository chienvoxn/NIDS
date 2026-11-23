#!/usr/bin/env python3
"""CTU experiment CLI; requires pyarrow to read supplied Parquet ZIP."""

import argparse, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nids.datasets import ctu_parquet, ctu_label_audit
from nids.offline import ctu_holdout_experiment
from joblib import dump

p = argparse.ArgumentParser()
p.add_argument("--ctu", required=True)
p.add_argument("--out", default="artifacts/botnet")
p.add_argument("--scenarios", default="5,11,12")
p.add_argument("--train", default="5")
p.add_argument("--val", default="11")
p.add_argument("--test", default="12")
p.add_argument("--graph", action="store_true")
a = p.parse_args()
parse = lambda s: tuple(int(i) for i in s.split(","))
df = ctu_parquet(a.ctu, parse(a.scenarios))
model, report = ctu_holdout_experiment(
    df, parse(a.train), parse(a.val), parse(a.test), graph=a.graph
)
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
name = "ctu_rf_graph" if a.graph else "ctu_rf_flow"
dump(model, out / f"{name}.joblib")
(out / f"{name}_metrics.json").write_text(json.dumps(report, indent=2, default=str))
print(
    json.dumps(
        {"model": str(out / f"{name}.joblib"), "results": report["results"]}, indent=2
    )
)
