#!/usr/bin/env python3
import argparse, sys, json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nids.datasets import inspect_archive, cic_audit, ctu_parquet, ctu_label_audit
from nids.training import run_pilot
from nids.pcap import extract_pcap, save_flows
from nids.botnet import (
    resolve_columns,
    binary_labels,
    communication_graph,
    graph_features,
    beaconing_features,
)

p = argparse.ArgumentParser()
p.add_argument(
    "action",
    choices=[
        "inspect",
        "cic-audit",
        "cic-pilot",
        "ctu-audit",
        "ctu-graph",
        "pcap-to-csv",
    ],
)
p.add_argument("--cic", default="/mnt/data/CIC-IDS2017.zip")
p.add_argument("--ctu", default="/mnt/data/ctu-13-dataset.zip")
p.add_argument("--scenarios", default="5,11,12")
p.add_argument("--per-label", type=int, default=300)
p.add_argument("--out", default="reports/pilot")
p.add_argument("--pcap")
a = p.parse_args()
if a.action == "inspect":
    print(
        json.dumps(
            {"cic": inspect_archive(a.cic), "ctu": inspect_archive(a.ctu)}, indent=2
        )
    )
elif a.action == "cic-audit":
    print(json.dumps(cic_audit(a.cic), indent=2))
elif a.action == "cic-pilot":
    print(json.dumps(run_pilot(a.cic, a.out, a.per_label), indent=2, default=int))
elif a.action in ("ctu-audit", "ctu-graph"):
    df = ctu_parquet(a.ctu, tuple(int(x) for x in a.scenarios.split(",")))
    if a.action == "ctu-audit":
        audit = ctu_label_audit(df)
        audit["counts"] = {str(k): v for k, v in audit["counts"].items()}
        print(json.dumps(audit, indent=2, default=str))
    else:
        cols = resolve_columns(df)
        print("columns", cols)
        labeled = binary_labels(df[cols["label"]])
        print(
            "verified normal/botnet/unknown",
            labeled.value_counts(dropna=False).to_dict(),
        )
        for scenario, part in df.groupby("scenario_id"):
            graph = communication_graph(part.head(30000))
            print(
                "scenario",
                scenario,
                "nodes",
                graph.number_of_nodes(),
                "edges",
                graph.number_of_edges(),
            )
            print(
                "graph feature preview", graph_features(graph).head().to_dict("records")
            )
            print(
                "beacon preview",
                beaconing_features(part.head(30000)).head().to_dict("records"),
            )
elif a.action == "pcap-to-csv":
    if not a.pcap:
        p.error("--pcap is required")
    rows = extract_pcap(a.pcap)
    save_flows(rows, a.out)
    print(json.dumps({"flows": len(rows), "output": a.out}))
