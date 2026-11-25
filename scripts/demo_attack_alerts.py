"""Replay saved held-out model predictions as demo security events."""

import argparse
import json
from pathlib import Path
from datetime import datetime, timezone
import time

import pandas as pd

from nids.security_events import build_event, write_event


def main():

    p = argparse.ArgumentParser()

    p.add_argument(
        "--scenario",
        choices=["PortScan", "DoS_Hulk", "Bot", "all"],
        default="all"
    )

    p.add_argument("--limit", type=int, default=10)
    p.add_argument("--delay", type=float, default=0.1)

    a = p.parse_args()

    if a.limit <= 0 or a.delay < 0:
        raise ValueError("Invalid limit or delay")

    root = Path(
        "reports/final/evidence/scenario_fusion"
    )

    out = Path("outputs/final_demo")
    out.mkdir(parents=True, exist_ok=True)

    alerts = out / "replay_security_events.jsonl"
    alerts.write_text("")

    scenarios = (
        ["PortScan", "DoS_Hulk", "Bot"]
        if a.scenario == "all"
        else [a.scenario]
    )

    results = []

    for name in scenarios:

        df = pd.read_csv(
            root / f"{name}_predictions.csv"
        )

        attack = df[df.target.ne("BENIGN")]

        # Fixed first N attack samples, not selected by prediction.
        sample = attack.head(a.limit)

        detected = 0
        events = 0

        print(f"\n=== SCENARIO: {name} ===")

        for _, r in sample.iterrows():

            row = {
                "processed_at_utc":
                    datetime.now(timezone.utc).isoformat(),

                # Original IP metadata unavailable.
                "source_ip": None,
                "destination_ip": None,

                "predicted": r.cnn_prediction,
                "confidence": float(r.cnn_confidence),
                "is_attack": bool(r.cnn_attack),

                "gaussian_score": float(r.gaussian_score),
                "gaussian_anomaly": bool(r.gaussian_anomaly),
                "gaussian_threshold": 920.5196377825823
            }

            event = build_event(row, mode="replay")

            detected += int(row["is_attack"])

            if event is not None:

                event["scenario"] = name
                event["ground_truth"] = str(r.target)

                events += int(
                    write_event(alerts, event)
                )

            else:
                print(
                    "[NO SIGNAL]",
                    r.target,
                    "->",
                    r.cnn_prediction,
                    flush=True
                )

            time.sleep(a.delay)

        results.append({
            "scenario": name,
            "samples": len(sample),
            "cnn_detected": detected,
            "cnn_missed": len(sample) - detected,
            "experimental_events": events,
            "mode": "saved_prediction_replay",
            "verified_live": False
        })

    result = pd.DataFrame(results)

    result.to_csv(
        out / "demo_results.csv",
        index=False
    )

    print("\n=== DEMO SUMMARY ===")
    print(result.to_string(index=False))

    print("\nSaved:", alerts)


if __name__ == "__main__":
    main()
