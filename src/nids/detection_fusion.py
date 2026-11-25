"""Experimental fusion of independent network detection signals."""

from collections import deque
import math

import pandas as pd

from nids.botnet import (
    beaconing_features,
    communication_graph
)

from nids.stream_detection import PROVISIONAL


class DetectionFusion:

    def __init__(self, gaussian, history_size=512):

        if history_size < 5:
            raise ValueError("Invalid history capacity")

        self.gaussian = gaussian

        self.history = deque(
            maxlen=history_size
        )

    def enrich(self, frame, predictions):

        if len(frame) != len(predictions):
            raise ValueError("Prediction alignment mismatch")

        if not frame.feature_status.eq(PROVISIONAL).all():
            raise ValueError("Provisional features required")

        if not predictions.alert_status.eq(
            "shadow_only_unverified"
        ).all():
            raise ValueError("Shadow predictions required")

        scores = self.gaussian.predict(frame)

        result = predictions.reset_index(drop=True).copy()

        for col in scores:
            result[col] = scores[col].to_numpy()

        extra = []

        for row, prediction in zip(
            frame.to_dict("records"),
            result.to_dict("records")
        ):

            src = str(row["source_ip"])
            dst = str(row["destination_ip"])
            start = float(row["start_time"])

            if not math.isfinite(start):
                raise ValueError("Invalid flow timestamp")

            self.history.append({
                "src": src,
                "dst": dst,
                "timestamp": start
            })

            recent = pd.DataFrame(self.history)

            graph = communication_graph(recent)

            beacons = beaconing_features(
                recent, min_events=5
            )

            pair = beacons[
                (beacons.src == src)
                & (beacons.dst == dst)
            ]

            cv = None
            iat = None

            if not pair.empty:

                value = pair.iloc[0]

                if pd.notna(value.iat_cv):
                    cv = float(value.iat_cv)

                iat = float(value.iat_mean_s)

            cnn = bool(prediction["is_attack"])
            gaussian = bool(prediction["gaussian_anomaly"])

            status = {
                (False, False): "no_model_signal",
                (True, False): "cnn_signal",
                (False, True): "gaussian_signal",
                (True, True): "corroborated_signal"
            }[(cnn, gaussian)]

            extra.append({
                "fusion_status": status,
                "history_flows": len(recent),
                "graph_nodes": graph.number_of_nodes(),
                "graph_edges": graph.number_of_edges(),
                "pair_connections": graph[src][dst]["count"],
                "source_unique_peers": len(
                    set(graph.predecessors(src))
                    | set(graph.successors(src))
                ),
                "pair_iat_mean_s": iat,
                "pair_iat_cv": cv,
                "verified_botnet": False,
                "verified_live": False,
                "official_alert": False
            })

        return pd.concat(
            [result, pd.DataFrame(extra)],
            axis=1
        )
