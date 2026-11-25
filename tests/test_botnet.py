import pandas as pd
import pytest
import networkx as nx

from nids.botnet import (
    resolve_columns,
    beaconing_features,
    communication_graph,
    graph_features,
    topology_lab_cases,
    binary_labels
)


def traffic():

    return pd.DataFrame({
        "source_ip": ["10.203.0.2"] * 6,
        "destination_ip": ["10.203.0.1"] * 6,
        "start_time": [
            1700000000 + i * 5
            for i in range(6)
        ],
        "source_port": list(range(40000, 40006)),
        "destination_port": [18080] * 6
    })


def test_live_schema():

    c = resolve_columns(traffic())

    assert c["src"] == "source_ip"
    assert c["dst"] == "destination_ip"
    assert c["time"] == "start_time"
    assert c["sport"] == "source_port"
    assert c["dport"] == "destination_port"


def test_epoch_beaconing():

    result = beaconing_features(traffic())

    assert len(result) == 1

    r = result.iloc[0]

    assert r.connections == 6
    assert r.iat_mean_s == pytest.approx(5)
    assert r.iat_std_s == 0
    assert r.iat_cv == 0


def test_datetime_beaconing():

    df = traffic()

    df["start_time"] = pd.to_datetime(
        df["start_time"], unit="s", utc=True
    )

    r = beaconing_features(df).iloc[0]

    assert r.iat_mean_s == pytest.approx(5)


def test_insufficient_events():

    result = beaconing_features(
        traffic().iloc[:3]
    )

    assert result.empty
    assert "iat_cv" in result.columns


def test_communication_graph():

    g = communication_graph(traffic())

    assert g.number_of_nodes() == 2
    assert g.number_of_edges() == 1

    assert g[
        "10.203.0.2"
    ][
        "10.203.0.1"
    ]["count"] == 6


def test_graph_features():

    g = communication_graph(traffic())

    df = graph_features(g).set_index("node")

    assert df.loc["10.203.0.2", "out_degree"] == 1
    assert df.loc["10.203.0.1", "in_degree"] == 1


def test_topology_cases():

    cases = topology_lab_cases(12)

    assert max(dict(
        cases["centralized_star"].degree()
    ).values()) == 12

    assert all(
        degree == 2
        for _, degree in cases["p2p_cycle"].degree()
    )

    assert nx.is_isomorphic(
        cases["centralized_star"],
        cases["benign_hub"]
    )


def test_background_is_unknown():

    y = binary_labels(pd.Series([
        "From-Botnet",
        "Normal",
        "Background"
    ]))

    assert y.iloc[0] == 1
    assert y.iloc[1] == 0
    assert pd.isna(y.iloc[2])
