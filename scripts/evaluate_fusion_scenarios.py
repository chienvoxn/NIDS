"""Evaluate fixed CNN and Gaussian policies on held-out CIC data."""

from pathlib import Path
import json
import time

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from nids.demo_engine import DemoEngine
from nids.gaussian_engine import GaussianEngine


root = Path("outputs/final/cic_final_runs")

cnn = DemoEngine(
    root / "seed_42",
    "cic", "full",
    variant="cnn_mhsa"
)

gaussian = GaussianEngine(
    root / "gaussian", "cic"
)

assert cnn.features == gaussian.features

data = pd.read_parquet(
    "data/processed/final/cic/test.parquet",
    columns=cnn.features + ["target"]
)

data["target"] = data.target.astype(str)

out = Path(
    "reports/final/evidence/scenario_fusion"
)

out.mkdir(parents=True, exist_ok=True)

normal = data[
    data.target.eq("BENIGN")
].sample(n=1000, random_state=42)

rows = []

for scenario in ["PortScan", "DoS Hulk", "Bot"]:

    attack = data[data.target.eq(scenario)]

    assert not attack.empty

    attack = attack.sample(
        n=min(1000, len(attack)),
        random_state=42
    )

    frame = pd.concat(
        [normal, attack],
        ignore_index=True
    )

    truth = frame.target.ne("BENIGN")

    start = time.perf_counter()

    cp = cnn.predict(
        frame, batch_size=128
    )

    cnn_ms = (
        time.perf_counter() - start
    ) * 1000

    start = time.perf_counter()

    gp = gaussian.predict(frame)

    gaussian_ms = (
        time.perf_counter() - start
    ) * 1000

    c = cp.is_attack.to_numpy(dtype=bool)
    g = gp.gaussian_anomaly.to_numpy(dtype=bool)

    for name, pred in {
        "CNN_MHSA": c,
        "Gaussian": g,
        "CNN_OR_Gaussian": c | g,
        "CNN_AND_Gaussian": c & g
    }.items():

        tn, fp, fn, tp = confusion_matrix(
            truth, pred, labels=[0, 1]
        ).ravel()

        rows.append({
            "scenario": scenario,
            "method": name,
            "samples": len(frame),
            "benign_samples": len(normal),
            "attack_samples": len(attack),

            "accuracy": accuracy_score(truth, pred),
            "precision": precision_score(
                truth, pred, zero_division=0
            ),
            "recall": recall_score(
                truth, pred, zero_division=0
            ),
            "f1": f1_score(
                truth, pred, zero_division=0
            ),
            "fpr": fp / (fp + tn),

            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),

            "cnn_batch_ms": cnn_ms,
            "gaussian_batch_ms": gaussian_ms,

            "evaluation_mode":
                "offline_fixed_test_subset"
        })

    pd.DataFrame({
        "target": frame.target,
        "cnn_prediction": cp.predicted,
        "cnn_confidence": cp.confidence,
        "cnn_attack": c,
        "gaussian_score": gp.gaussian_score,
        "gaussian_anomaly": g
    }).to_csv(
        out / f"{scenario.replace(' ', '_')}_predictions.csv",
        index=False
    )

    print("SCENARIO COMPLETED:", scenario)

result = pd.DataFrame(rows)

assert len(result) == 12

result.to_csv(
    out / "scenario_metrics.csv",
    index=False
)

(out / "summary.json").write_text(
    json.dumps({
        "model": cnn.metadata,
        "gaussian_threshold": gaussian.threshold,
        "gaussian_sha256": gaussian.sha256,
        "scenarios": 3,
        "methods": 4,
        "seed": 42,
        "threshold_refitted": False,
        "evaluation_mode":
            "offline_fixed_test_subset",
        "verified_live": False
    }, indent=2) + "\n"
)

print("\n=== SCENARIO BENCHMARK ===")

print(result[[
    "scenario", "method", "samples",
    "accuracy", "precision", "recall",
    "f1", "fpr", "tn", "fp", "fn", "tp"
]].to_string(index=False))

print("\nSCENARIO BENCHMARK - PASS")
