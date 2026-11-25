from pathlib import Path
import pandas as pd

from nids.topology_detection import classify_topology

root = Path("reports/final/evidence/linux/phase7p")

df = pd.read_csv(root / "topology_comparison.csv")

expected = {
    "centralized_star": "CENTRALIZED",
    "p2p_cycle": "P2P",
    "p2p_mesh": "P2P",
    "benign_hub": "CENTRALIZED"
}

df["expected_topology"] = df.topology.map(expected)

assert df.expected_topology.notna().all()

df["predicted_topology"] = [
    classify_topology(
        r.nodes,
        r.edges,
        r.max_degree,
        r.mean_degree
    )
    for r in df.itertuples()
]

df["topology_match"] = (
    df.expected_topology == df.predicted_topology
)

df["verified_botnet"] = False

df.to_csv(
    root / "topology_detection_results.csv",
    index=False
)

print(df[[
    "topology",
    "expected_topology",
    "predicted_topology",
    "topology_match",
    "verified_botnet"
]].to_string(index=False))

assert df.topology_match.all()

print("\nTOPOLOGY CLASSIFICATION - PASS")
