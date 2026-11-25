"""Consolidate existing NIDS experimental evidence."""

import json
from pathlib import Path
import pandas as pd

root = Path("reports/final")
evidence = root / "evidence"
out = root / "acceptance"
out.mkdir(parents=True, exist_ok=True)

live = json.loads(
    (evidence / "labeled_evaluation/summary.json").read_text()
)

evaluation = json.loads(
    (evidence / "labeled_evaluation/evaluation.json").read_text()
)

scenario = pd.read_csv(
    evidence / "scenario_fusion/scenario_metrics.csv"
)

topology = pd.read_csv(
    evidence / "linux/phase7p/topology_detection_results.csv"
)

latency = pd.read_csv(
    evidence / "labeled_evaluation/scenario_live_metrics.csv"
)

cnn = scenario[scenario["method"] == "CNN_MHSA"].copy()

cnn.to_csv(
    out / "attack_scenario_results.csv",
    index=False
)

topology.to_csv(
    out / "botnet_topology_results.csv",
    index=False
)

latency.to_csv(
    out / "live_latency_results.csv",
    index=False
)

multiclass = []

for path in sorted(
    (evidence / "scenario_fusion").glob("*_predictions.csv")
):
    df = pd.read_csv(path)

    attack = df[df.target != "BENIGN"]

    correct = int(
        (attack.target == attack.cnn_prediction).sum()
    )

    multiclass.append({
        "scenario": path.stem.replace("_predictions", ""),
        "attack_samples": len(attack),
        "correct_attack_type": correct,
        "attack_type_recall": correct / len(attack),
        "evaluation_mode": "offline"
    })

pd.DataFrame(multiclass).to_csv(
    out / "multiclass_results.csv",
    index=False
)

summary = {
    "linux_capture": bool(live["capture_complete"]),
    "live_pipeline_complete": bool(live["pipeline_complete"]),
    "captured_packets": live["captured"],
    "processed_flows": live["shadow_processed"],
    "matched_ground_truth": evaluation["matched_flows"],
    "cnn_false_positives": evaluation["cnn_false_positives"],
    "gaussian_anomalies": evaluation["gaussian_anomalies"],
    "mean_latency_ms": float(
        (latency.mean_latency_ms * latency.matched_flows).sum()
        / latency.matched_flows.sum()
    ),
    "latency_scope": live["latency_scope"],
    "topology_cases": len(topology),
    "topology_correct": int(topology.topology_match.sum()),
    "offline_attack_scenarios": len(cnn),
    "verified_live_attack_detection": False,
    "verified_live_botnet": False,
    "semantic_parity": False
}

(out / "acceptance_summary.json").write_text(
    json.dumps(summary, indent=2) + "\n",
    encoding="utf-8"
)

print("=== FINAL ACCEPTANCE EVIDENCE ===")
print(json.dumps(summary, indent=2))

print("\n=== ATTACK SCENARIOS ===")
print(cnn.to_string(index=False))

print("\n=== BOTNET TOPOLOGY ===")
print(topology.to_string(index=False))

print("\nACCEPTANCE EVIDENCE GENERATED")
