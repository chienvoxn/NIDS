"""Evaluate live predictions against independently recorded lab sessions."""

from pathlib import Path
import json
import pandas as pd

root = Path(
    "reports/final/evidence/labeled_evaluation"
)

truth = pd.DataFrame(
    json.loads(
        (root / "ground_truth.json").read_text()
    )
)

pred = pd.read_csv(
    root / "predictions.csv"
)

keys = [
    "source_ip",
    "source_port",
    "destination_ip",
    "destination_port"
]

assert not truth.duplicated(keys).any()
assert not pred.duplicated(keys).any()

df = truth.merge(
    pred,
    on=keys,
    how="outer",
    indicator=True,
    validate="one_to_one"
)

print("=== MATCHING RESULTS ===")
print(df["_merge"].value_counts())

matched = df[df["_merge"] == "both"].copy()

assert not matched.empty

matched["cnn_correct"] = (
    matched["predicted"] == "BENIGN"
)

results = []

for name, group in matched.groupby("scenario"):

    results.append({
        "scenario": name,
        "matched_flows": len(group),
        "cnn_correct": int(group.cnn_correct.sum()),
        "cnn_false_positives": int(
            (~group.cnn_correct).sum()
        ),
        "cnn_benign_accuracy": float(
            group.cnn_correct.mean()
        ),
        "gaussian_anomalies": int(
            group.gaussian_anomaly.sum()
        ),
        "mean_latency_ms": float(
            group.submit_to_prediction_ms.mean()
        ),
        "p95_latency_ms": float(
            group.submit_to_prediction_ms.quantile(.95)
        )
    })

result = pd.DataFrame(results)

result.to_csv(
    root / "scenario_live_metrics.csv",
    index=False
)

df.to_csv(
    root / "ground_truth_comparison.csv",
    index=False
)

summary = {
    "expected_sessions": len(truth),
    "predicted_flows": len(pred),
    "matched_flows": len(matched),
    "unmatched_truth": int(
        (df["_merge"] == "left_only").sum()
    ),
    "unmatched_predictions": int(
        (df["_merge"] == "right_only").sum()
    ),
    "cnn_false_positives": int(
        (~matched.cnn_correct).sum()
    ),
    "gaussian_anomalies": int(
        matched.gaussian_anomaly.sum()
    ),
    "verified_live_attack_detection": False,
    "semantic_parity": False
}

(root / "evaluation.json").write_text(
    json.dumps(summary, indent=2) + "\n"
)

print("\n=== LIVE SCENARIO METRICS ===")

print(result.to_string(index=False))

print("\n=== EVALUATION SUMMARY ===")

print(json.dumps(summary, indent=2))

print("\nLABELED LIVE EVALUATION COMPLETE")
