"""Consolidate existing experimental evidence without retraining."""

from pathlib import Path
import json
import pandas as pd

root = Path("reports/final")
out = root / "evaluation_summary"

out.mkdir(parents=True, exist_ok=True)

# CTU model comparison
ctu = pd.read_csv(
    root / "ctu_final/model_summary.csv"
)

g = json.loads(Path(
    "outputs/final/ctu_final_runs/"
    "gaussian/metrics.json"
).read_text())

gaussian = pd.DataFrame([{
    "Group": "Anomaly",
    "Model": "Gaussian",
    "accuracy": g["test"]["accuracy"],
    "precision": g["test"]["precision"],
    "recall": g["test"]["recall"],
    "f1_score": g["test"]["f1_score"],
    "FPR": g["test"]["false_positive_rate"],
    "ValidationF1": g["validation"]["f1_score"]
}])

ctu = pd.concat(
    [ctu, gaussian],
    ignore_index=True
)

assert len(ctu) == 9
ctu["metric_scope"] = "binary_detection"

ctu.to_csv(
    out / "ctu_botnet_comparison.csv",
    index=False
)

# CIC comparison
cic = pd.read_csv(
    root / "cic_final/model_summary.csv"
)

cg = json.loads(Path(
    "outputs/final/cic_final_runs/gaussian/metrics.json"
).read_text())

cic = pd.concat([
    cic,
    pd.DataFrame([{
        "Group": "Anomaly",
        "Model": "Gaussian",
        "accuracy": cg["test"]["accuracy"],
        "precision": cg["test"]["precision"],
        "recall": cg["test"]["recall"],
        "f1_score": cg["test"]["f1_score"],
        "FPR": cg["test"]["false_positive_rate"],
        "ValidationF1": cg["validation"]["f1_score"]
    }])
], ignore_index=True)

assert len(cic) == 9

cic["metric_scope"] = cic.Model.map(
    lambda x: "binary_detection"
    if x == "Gaussian"
    else "weighted_multiclass"
)

cic.to_csv(
    out / "cic_model_comparison.csv",
    index=False
)

# Three-scenario evaluation
scenarios = pd.read_csv(
    root / "evidence/scenario_fusion/scenario_metrics.csv"
)

assert len(scenarios) == 12

scenarios.to_csv(
    out / "scenario_comparison.csv",
    index=False
)

# Two verified experimental fusion sessions
fusion = root / "evidence/linux/phase7q"

sessions = []

for name in [
    "session1_validation.json",
    "validation.json"
]:

    s = json.loads(
        (fusion / name).read_text()
    )

    assert s["passed"] == s["total"] == 22

    sessions.append({
        "session": name,
        "packets": s["captured_packets"],
        "flows": s["flows"],
        "during_capture": s["predictions_during_capture"],
        "cnn_attack_predictions": s["cnn_attack_predictions"],
        "gaussian_anomalies": s["gaussian_anomalies"],
        **s["latency_ms"],
        "verified_live": False
    })

live = pd.DataFrame(sessions)

live.to_csv(
    out / "live_fusion_comparison.csv",
    index=False
)

summary = {
    "ctu_models": len(ctu),
    "cic_models": len(cic),
    "scenario_results": len(scenarios),
    "fusion_sessions": len(live),
    "captured_packets": int(live.packets.sum()),
    "processed_flows": int(live.flows.sum()),
    "predictions_during_capture": int(
        live.during_capture.sum()
    ),
    "gaussian_anomalies": int(
        live.gaussian_anomalies.sum()
    ),
    "cnn_attack_predictions": int(
        live.cnn_attack_predictions.sum()
    ),
    "semantic_parity": False,
    "verified_live_attack_detection": False,
    "retrained_models": False
}

(out / "summary.json").write_text(
    json.dumps(summary, indent=2) + "\n"
)

print("=== CTU COMPARISON ===")
print(ctu.to_string(index=False))

print("\n=== LIVE FUSION ===")
print(live.to_string(index=False))

print("\n=== FINAL EVALUATION SUMMARY ===")
print(json.dumps(summary, indent=2))

print("\nEVALUATION CONSOLIDATION - PASS")
