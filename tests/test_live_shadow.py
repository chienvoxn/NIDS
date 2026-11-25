import pandas as pd
import pytest

from nids.live_shadow import LiveShadowWorker
from nids.stream_detection import PROVISIONAL


class FakeEngine:

    metadata = {
        "seed": 42,
        "dataset": "cic",
        "variant": "cnn_mhsa"
    }

    def predict(self, frame, batch_size=32):

        return pd.DataFrame({
            "predicted": ["BENIGN"] * len(frame),
            "confidence": [0.99] * len(frame),
            "is_attack": [False] * len(frame)
        })


def candidate():

    return {
        "feature_status": PROVISIONAL
    }


def test_worker_prediction(tmp_path):

    worker = LiveShadowWorker(
        FakeEngine(),
        tmp_path
    )

    with worker:

        for i in range(12):
            worker.submit(candidate(), i + 1)

    stats = worker.finish()

    assert stats["shadow_processed"] == 12
    assert stats["shadow_dropped"] == 0
    assert stats["shadow_complete"] is True
    assert stats["verified_live"] is False
    assert stats["official_alerts"] == 0

    df = pd.read_csv(
        tmp_path / "predictions.csv"
    )

    assert len(df) == 12
    assert list(df.flow_row) == list(range(1, 13))

    assert df.alert_status.eq(
        "shadow_only_unverified"
    ).all()

    assert df.submit_to_prediction_ms.ge(0).all()


def test_reject_unmarked_flow(tmp_path):

    worker = LiveShadowWorker(
        FakeEngine(),
        tmp_path
    )

    with pytest.raises(ValueError):
        worker.submit({}, 1)


def test_bounded_queue(tmp_path):

    worker = LiveShadowWorker(
        FakeEngine(),
        tmp_path,
        capacity=2
    )

    worker.submit(candidate(), 1)
    worker.submit(candidate(), 2)
    worker.submit(candidate(), 3)

    assert worker.submitted == 2
    assert worker.dropped == 1

    with worker:
        pass

    stats = worker.finish()

    assert stats["shadow_processed"] == 2
    assert not stats["shadow_complete"]


def test_worker_failure(tmp_path):

    class BrokenEngine(FakeEngine):

        def predict(self, frame, batch_size=32):
            raise RuntimeError("Test inference failure")

    worker = LiveShadowWorker(
        BrokenEngine(),
        tmp_path
    )

    with worker:

        worker.submit(candidate(), 1)

    stats = worker.finish()

    assert stats["shadow_error"] is not None
    assert stats["shadow_unprocessed"] == 1
    assert not stats["shadow_complete"]
    assert stats["official_alerts"] == 0
