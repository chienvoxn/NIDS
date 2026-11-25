import pandas as pd
import pytest

from nids.detection_fusion import DetectionFusion
from nids.stream_detection import PROVISIONAL


class FakeGaussian:

    def predict(self, frame):

        return pd.DataFrame({
            "gaussian_score": frame.score.to_numpy(),
            "gaussian_threshold": 10,
            "gaussian_anomaly": frame.score.to_numpy() > 10
        })


def inputs():

    frame = pd.DataFrame({
        "source_ip": ["10.0.0.1"] * 4,
        "destination_ip": ["10.0.0.2"] * 4,
        "start_time": [1, 2, 3, 4],
        "feature_status": [PROVISIONAL] * 4,
        "score": [1, 1, 20, 20]
    })

    predictions = pd.DataFrame({
        "is_attack": [False, True, False, True],
        "alert_status": ["shadow_only_unverified"] * 4
    })

    return frame, predictions


def test_fusion_signals():

    frame, pred = inputs()

    result = DetectionFusion(
        FakeGaussian()
    ).enrich(frame, pred)

    assert result.fusion_status.tolist() == [
        "no_model_signal",
        "cnn_signal",
        "gaussian_signal",
        "corroborated_signal"
    ]


def test_no_official_alerts():

    frame, pred = inputs()

    result = DetectionFusion(
        FakeGaussian()
    ).enrich(frame, pred)

    assert not result.official_alert.any()
    assert not result.verified_live.any()
    assert not result.verified_botnet.any()


def test_history_capacity():

    frame, pred = inputs()

    engine = DetectionFusion(
        FakeGaussian(), history_size=5
    )

    engine.enrich(frame, pred)
    result = engine.enrich(frame, pred)

    assert len(engine.history) == 5
    assert result.history_flows.max() == 5


def test_graph_evidence():

    frame, pred = inputs()

    result = DetectionFusion(
        FakeGaussian()
    ).enrich(frame, pred)

    assert result.graph_nodes.eq(2).all()
    assert result.graph_edges.eq(1).all()
    assert result.pair_connections.tolist() == [1, 2, 3, 4]


def test_wrong_feature_status():

    frame, pred = inputs()

    frame["feature_status"] = "unmarked"

    with pytest.raises(ValueError):
        DetectionFusion(
            FakeGaussian()
        ).enrich(frame, pred)


def test_wrong_alert_status():

    frame, pred = inputs()

    pred["alert_status"] = "verified"

    with pytest.raises(ValueError):
        DetectionFusion(
            FakeGaussian()
        ).enrich(frame, pred)
