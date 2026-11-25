from nids.security_events import build_event, write_event
import json


def sample(cnn=False, gaussian=False):
    return {
        "is_attack": cnn,
        "gaussian_anomaly": gaussian,
        "predicted": "PortScan" if cnn else "BENIGN",
        "confidence": 0.99,
        "source_ip": "10.203.0.2",
        "destination_ip": "10.203.0.1",
        "processed_at_utc": "2026-11-25T18:00:00+00:00"
    }


def test_no_signal():
    assert build_event(sample()) is None


def test_cnn_event():
    e = build_event(sample(cnn=True))
    assert e["attack_type"] == "PortScan"
    assert e["severity"] == "MEDIUM"
    assert e["detector"] == "CNN_MHSA"


def test_gaussian_event():
    e = build_event(sample(gaussian=True))
    assert e["attack_type"] == "ANOMALY"
    assert e["severity"] == "LOW"


def test_combined_event():
    e = build_event(sample(True, True))
    assert e["severity"] == "HIGH"
    assert e["verified"] is False
    assert e["official_alert"] is False


def test_event_jsonl(tmp_path):
    path = tmp_path / "events.jsonl"
    e = build_event(sample(True), mode="replay")

    assert write_event(path, e)

    result = json.loads(path.read_text())

    assert result["mode"] == "replay"
    assert result["source_ip"] == "10.203.0.2"
    assert result["cnn_confidence"] == 0.99
