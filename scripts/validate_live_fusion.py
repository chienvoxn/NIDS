"""Validate continuous experimental fusion evidence."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def validate(root, expected=12):

    root = Path(root)

    s = json.loads(
        (root / "summary.json").read_text()
    )

    if s["captured"] <= 0 or s["rows"] <= 0:
        raise ValueError("Empty capture: experiment failed")

    df = pd.read_csv(root / "predictions.csv")

    features = pd.read_csv(
        root / "provisional_features.csv"
    )

    c = df.is_attack
    g = df.gaussian_anomaly

    states = {
        (False, False): "no_model_signal",
        (True, False): "cnn_signal",
        (False, True): "gaussian_signal",
        (True, True): "corroborated_signal"
    }

    expected_states = [
        states[(bool(x), bool(y))]
        for x, y in zip(c, g)
    ]

    times = pd.to_datetime(
        df.processed_at_utc, utc=True
    ).map(lambda t: t.timestamp())

    during = int((
        (times >= s["capture_started_at_epoch"])
        & (times < s["capture_stopped_at_epoch"])
    ).sum())

    latency = df.submit_to_prediction_ms

    checks = {
        "capture_complete":
            s["capture_complete"] is True,

        "pipeline_complete":
            s["pipeline_complete"] is True,

        "nonempty_capture":
            s["captured"] > 0,

        "all_components_enabled":
            all(s[k] is True for k in [
                "cnn_enabled",
                "gaussian_enabled",
                "botnet_metadata_enabled",
                "fusion_enabled"
            ]),

        "flow_accounting":
            len(df) == len(features)
            == s["rows"]
            == s["shadow_processed"]
            == expected,

        "unique_flow_ids":
            df.flow_row.is_unique
            and set(df.flow_row)
            == set(range(1, expected + 1)),

        "inference_during_capture":
            during > 0,

        "confidence_valid":
            df.confidence.between(0, 1).all(),

        "gaussian_scores_valid":
            np.isfinite(df.gaussian_score).all()
            and df.gaussian_score.ge(0).all(),

        "gaussian_threshold":
            np.allclose(
                df.gaussian_threshold,
                920.5196377825823,
                atol=1e-9, rtol=0
            )
            and s["gaussian_threshold"]
            == 920.5196377825823,

        "gaussian_decision":
            (g == (
                df.gaussian_score
                > df.gaussian_threshold
            )).all(),

        "fusion_decision":
            df.fusion_status.tolist()
            == expected_states,

        "graph_evidence":
            df.graph_nodes.ge(2).all()
            and df.graph_edges.ge(1).all(),

        "bounded_history":
            df.history_flows.between(1, 512).all(),

        "no_application_drops":
            s["dropped"] == 0
            and s["invalidated"] == 0,

        "no_inference_drops":
            s["shadow_dropped"] == 0
            and s["shadow_unprocessed"] == 0,

        "no_worker_errors":
            s["shadow_error"] is None,

        "valid_latency":
            np.isfinite(latency).all()
            and latency.ge(0).all(),

        "correct_model":
            s["model"]["variant"] == "cnn_mhsa"
            and s["model"]["seed"] == 42,

        "shadow_status":
            df.alert_status.eq(
                "shadow_only_unverified"
            ).all(),

        "no_official_alerts":
            not df.official_alert.any()
            and s["official_alerts"] == 0,

        "unverified_status":
            not df.verified_live.any()
            and not df.verified_botnet.any()
            and s["verified_live"] is False
            and s["semantic_parity"] is False
    }

    checks = {
        k: bool(v)
        for k, v in checks.items()
    }

    result = {
        "checks": checks,
        "passed": sum(checks.values()),
        "total": len(checks),
        "captured_packets": s["captured"],
        "flows": len(df),
        "predictions_during_capture": during,
        "cnn_attack_predictions": int(c.sum()),
        "gaussian_anomalies": int(g.sum()),

        "fusion_counts": {
            str(k): int(v)
            for k, v in
            df.fusion_status.value_counts().items()
        },

        "latency_ms": {
            "mean": float(latency.mean()),
            "median": float(latency.median()),
            "p95": float(latency.quantile(0.95)),
            "max": float(latency.max())
        },

        "verified_live": False,
        "semantic_parity": False
    }

    print(df[[
        "flow_row",
        "protocol",
        "predicted",
        "confidence",
        "gaussian_score",
        "gaussian_anomaly",
        "fusion_status",
        "submit_to_prediction_ms"
    ]].to_string(index=False))

    return result


def main():

    p = argparse.ArgumentParser()

    p.add_argument("--run", required=True)
    p.add_argument("--expected-flows", type=int, default=12)
    p.add_argument("--out", required=True)

    a = p.parse_args()

    result = validate(a.run, a.expected_flows)

    path = Path(a.out)

    path.parent.mkdir(parents=True, exist_ok=True)

    path.write_text(
        json.dumps(result, indent=2) + "\n"
    )

    print(json.dumps(result, indent=2))

    print(
        "\nVALIDATION:",
        result["passed"],
        "/",
        result["total"]
    )

    if not all(result["checks"].values()):
        raise SystemExit("LIVE FUSION - FAIL")

    print("LIVE FUSION - PASS")


if __name__ == "__main__":
    main()
