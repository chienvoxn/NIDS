import argparse, sys, json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nids.datasets import ctu_parquet
from nids.offline import train_ctu_torch_temporal

p = argparse.ArgumentParser()
p.add_argument("--ctu", required=True)
p.add_argument("--out", default="artifacts/botnet")
p.add_argument("--epochs", type=int, default=5)
p.add_argument("--length", type=int, default=16)
a = p.parse_args()
df = ctu_parquet(a.ctu, (5, 11, 12))
_, results = train_ctu_torch_temporal(df, epochs=a.epochs, length=a.length, out=a.out)
print(json.dumps(results, indent=2))
