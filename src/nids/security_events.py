"""Experimental security events for the NIDS demonstration."""

import json
from pathlib import Path


def build_event(row, mode="shadow"):

    cnn = bool(row.get("is_attack", False))
    gaussian = bool(row.get("gaussian_anomaly", False))

    if not cnn and not gaussian:
        return None

    if cnn and gaussian:
        severity = "HIGH"
        detector = "CNN_MHSA_GAUSSIAN"
    elif cnn:
        severity = "MEDIUM"
        detector = "CNN_MHSA"
    else:
        severity = "LOW"
        detector = "GAUSSIAN"

    return {
        "timestamp": row.get("processed_at_utc"),
        "source_ip": row.get("source_ip"),
        "destination_ip": row.get("destination_ip"),
        "source_port": row.get("source_port"),
        "destination_port": row.get("destination_port"),
        "protocol": row.get("protocol"),

        "attack_type": (
            str(row.get("predicted"))
            if cnn else "ANOMALY"
        ),

        "cnn_confidence": float(
            row.get("confidence", 0)
        ),

        "gaussian_score": row.get("gaussian_score"),
        "gaussian_threshold": row.get("gaussian_threshold"),

        "detector": detector,
        "severity": severity,

        "graph_nodes": row.get("graph_nodes"),
        "graph_edges": row.get("graph_edges"),
        "pair_connections": row.get("pair_connections"),
        "pair_iat_cv": row.get("pair_iat_cv"),

        "communication_topology": row.get("communication_topology"),
        "beaconing_candidate": row.get("beaconing_candidate"),
        "cnn_bot_signal": row.get("cnn_bot_signal"),
        "botnet_status": row.get("botnet_status"),

        "mode": mode,
        "verified": False,
        "official_alert": False,
        "event_status": "experimental_detection"
    }


def write_event(path, event):

    if event is None:
        return False

    path = Path(path)

    path.parent.mkdir(
        parents=True, exist_ok=True
    )

    with path.open("a", encoding="utf-8") as f:

        f.write(
            json.dumps(event, default=str) + "\n"
        )

    print(
        f"[{event['severity']}] "
        f"{event['timestamp']} | "
        f"{event['source_ip']} -> "
        f"{event['destination_ip']} | "
        f"{event['attack_type']} | "
        f"CNN confidence="
        f"{event['cnn_confidence']:.6f} | "
        f"{event['mode'].upper()}",
        flush=True
    )

    return True
