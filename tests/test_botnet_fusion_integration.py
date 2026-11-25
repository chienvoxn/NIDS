import pandas as pd

from nids.detection_fusion import DetectionFusion
from nids.stream_detection import PROVISIONAL
from nids.security_events import build_event


class GaussianStub:

    def predict(self, frame):
        return pd.DataFrame({
            "gaussian_anomaly": [False] * len(frame),
            "gaussian_score": [0.0] * len(frame),
            "gaussian_threshold": [1.0] * len(frame)
        })


def evaluate(edges, labels=None, timestamps=None):

    n = len(edges)

    frame = pd.DataFrame({
        "source_ip": [e[0] for e in edges],
        "destination_ip": [e[1] for e in edges],
        "start_time": (
            timestamps if timestamps is not None
            else list(range(n))
        ),
        "feature_status": [PROVISIONAL] * n
    })

    labels = labels or ["BENIGN"] * n

    predictions = pd.DataFrame({
        "predicted": labels,
        "is_attack": [v != "BENIGN" for v in labels],
        "confidence": [0.99] * n,
        "alert_status": ["shadow_only_unverified"] * n
    })

    fusion = DetectionFusion(GaussianStub())

    return fusion.enrich(frame, predictions)


def test_centralized_topology():
    edges = [
        ("hub", f"node{i}") for i in range(12)
    ]

    result = evaluate(edges)

    assert result.iloc[-1].communication_topology == "CENTRALIZED"
    assert not result.verified_botnet.any()


def test_p2p_topology():
    edges = [
        (f"node{i}", f"node{(i+1)%13}")
        for i in range(13)
    ]

    result = evaluate(edges)

    assert result.iloc[-1].communication_topology == "P2P"
    assert not result.verified_botnet.any()


def test_periodic_behavior():
    result = evaluate(
        [("client", "server")] * 6,
        timestamps=[0, 5, 10, 15, 20, 25]
    )

    last = result.iloc[-1]

    assert last.beaconing_candidate
    assert last.botnet_status == "PERIODIC_BEHAVIOR_CANDIDATE"
    assert not last.verified_botnet


def test_cnn_bot_signal():
    result = evaluate(
        [("client", "server")],
        labels=["Bot"]
    )

    last = result.iloc[-1]

    assert last.cnn_bot_signal
    assert last.botnet_status == "CNN_BOT_SIGNAL"

    event = build_event(last.to_dict())

    assert event["attack_type"] == "Bot"
    assert event["botnet_status"] == "CNN_BOT_SIGNAL"
    assert not event["official_alert"]


def test_normal_traffic():
    result = evaluate(
        [("client", "server")]
    )

    last = result.iloc[-1]

    assert not last.cnn_bot_signal
    assert not last.beaconing_candidate
    assert last.botnet_status == "NO_BOTNET_SIGNAL"
    assert build_event(last.to_dict()) is None
