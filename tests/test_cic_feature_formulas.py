import json
import math
from pathlib import Path

import pytest

from nids.cic_flow_extractor import Flow, PacketInfo


def make_packet(t, reverse, payload, flags, window, proto=6):
    src, dst = "10.203.0.2", "10.203.0.1"
    sport, dport = 40000, 18080

    if reverse:
        src, dst = dst, src
        sport, dport = dport, sport

    header = 20 if proto == 6 else 8

    return PacketInfo(
        t, src, dst, sport, dport, proto,
        payload, 20 + header + payload,
        header, flags, window
    )


def test_all_60_feature_formulas():

    packets = [
        make_packet(0, False, 10, 2, 100),
        make_packet(1, True, 20, 16, 200),
        make_packet(2, False, 30, 24, 300),
        make_packet(8, True, 40, 17, 400)
    ]

    f = Flow(packets[0], packets[0].t)

    for p in packets:
        f.add(p)

    actual = f.features()

    expected = {
        "Destination Port": 18080,
        "Flow Duration": 8000000,
        "Total Fwd Packets": 2,
        "Total Backward Packets": 2,
        "Total Length of Fwd Packets": 40,
        "Total Length of Bwd Packets": 60,

        "Fwd Packet Length Max": 30,
        "Fwd Packet Length Min": 10,
        "Fwd Packet Length Mean": 20,
        "Fwd Packet Length Std": 10,

        "Bwd Packet Length Max": 40,
        "Bwd Packet Length Min": 20,
        "Bwd Packet Length Mean": 30,
        "Bwd Packet Length Std": 10,

        "Flow Bytes/s": 12.5,
        "Flow Packets/s": 0.5,

        "Flow IAT Mean": 8000000 / 3,
        "Flow IAT Std": math.sqrt(50) / 3 * 1000000,
        "Flow IAT Max": 6000000,
        "Flow IAT Min": 1000000,

        "Fwd IAT Total": 2000000,
        "Fwd IAT Mean": 2000000,
        "Fwd IAT Std": 0,
        "Fwd IAT Max": 2000000,
        "Fwd IAT Min": 2000000,

        "Bwd IAT Total": 7000000,
        "Bwd IAT Mean": 7000000,
        "Bwd IAT Std": 0,
        "Bwd IAT Max": 7000000,
        "Bwd IAT Min": 7000000,

        "Fwd PSH Flags": 1,
        "Fwd Header Length": 40,
        "Bwd Header Length": 40,

        "Fwd Packets/s": 0.25,
        "Bwd Packets/s": 0.25,

        "Min Packet Length": 10,
        "Max Packet Length": 40,
        "Packet Length Mean": 25,
        "Packet Length Std": math.sqrt(125),
        "Packet Length Variance": 125,

        "FIN Flag Count": 1,
        "RST Flag Count": 0,
        "PSH Flag Count": 1,
        "ACK Flag Count": 3,
        "URG Flag Count": 0,

        "Down/Up Ratio": 1,
        "Average Packet Size": 25,

        "Init_Win_bytes_forward": 100,
        "Init_Win_bytes_backward": 200,

        "Subflow Bwd Bytes": 30,

        "Active Mean": 1000000,
        "Active Std": 1000000,
        "Active Max": 2000000,
        "Active Min": 0,

        "Idle Mean": 6000000,
        "Idle Std": 0,
        "Idle Max": 6000000,
        "Idle Min": 6000000,

        "act_data_pkt_fwd": 2,
        "min_seg_size_forward": 20
    }

    assert len(expected) == 60
    assert set(actual) == set(expected)

    for name, value in expected.items():
        assert actual[name] == pytest.approx(value), name


def test_udp_initial_window_missing():

    p = make_packet(
        0, False, 23, 0, None, proto=17
    )

    f = Flow(p, p.t)
    f.add(p)

    x = f.features()

    assert math.isnan(x["Init_Win_bytes_forward"])
    assert math.isnan(x["Init_Win_bytes_backward"])

    assert x["Total Fwd Packets"] == 1
    assert x["Total Backward Packets"] == 0
    assert x["Fwd Header Length"] == 8


def test_zero_duration_rates():

    p = make_packet(0, False, 10, 2, 100)

    f = Flow(p, p.t)
    f.add(p)

    x = f.features()

    assert x["Flow Duration"] == 0

    for name in [
        "Flow Bytes/s",
        "Flow Packets/s",
        "Fwd Packets/s",
        "Bwd Packets/s"
    ]:
        assert math.isnan(x[name])


def test_no_backward_packets():

    p = make_packet(0, False, 10, 2, 100)

    f = Flow(p, p.t)
    f.add(p)

    x = f.features()

    assert x["Total Backward Packets"] == 0
    assert x["Total Length of Bwd Packets"] == 0
    assert x["Bwd Packet Length Mean"] == 0
    assert x["Down/Up Ratio"] == 0


def test_training_feature_schema():

    path = Path(
        "outputs/final/cic_final_runs/"
        "seed_42/cnn_metrics.json"
    )

    if not path.exists():
        pytest.skip("Local training artifacts unavailable")

    features = json.loads(path.read_text())["features"]

    p = make_packet(0, False, 10, 2, 100)

    f = Flow(p, p.t)
    f.add(p)

    assert len(features) == 60
    assert len(set(features)) == 60
    assert set(features) == set(f.features())
