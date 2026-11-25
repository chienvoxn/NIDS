import pytest

from nids.sliding_window import SlidingWindowMonitor


def add(m, t, reverse=False, size=100):

    src, dst = "10.0.0.1", "10.0.0.2"
    sp, dp = 12345, 80

    if reverse:
        src, dst = dst, src
        sp, dp = dp, sp

    return m.add(
        t, src, dst, sp, dp, 6, size
    )


def test_overlapping_windows():

    m = SlidingWindowMonitor(window=5, stride=1)

    for t in range(5):
        assert not add(m, t)

    first = add(m, 5)[0]
    second = add(m, 6)[0]

    assert first["packet_count"] == 5
    assert second["packet_count"] == 5

    assert first["ip_bytes"] == 500
    assert second["ip_bytes"] == 500

    assert first["window_start"] != second["window_start"]


def test_window_boundary():

    m = SlidingWindowMonitor()

    add(m, 0)
    add(m, 4.999)

    rows = add(m, 5)

    assert len(rows) == 1
    assert rows[0]["packet_count"] == 2

    rows = m.advance(6)

    assert rows[0]["packet_count"] == 2


def test_flow_and_host_aggregation():

    m = SlidingWindowMonitor()

    add(m, 0, size=100)
    add(m, 1, reverse=True, size=200)

    row = m.advance(5)[0]

    assert row["flow_count"] == 2
    assert row["host_count"] == 2
    assert row["packet_count"] == 2
    assert row["ip_bytes"] == 300

    hosts = {
        h["host"]: h
        for h in row["hosts"]
    }

    assert hosts["10.0.0.1"]["out_packets"] == 1
    assert hosts["10.0.0.1"]["in_packets"] == 1

    assert hosts["10.0.0.1"]["out_bytes"] == 100
    assert hosts["10.0.0.1"]["in_bytes"] == 200

    assert hosts["10.0.0.1"]["unique_peers"] == 1


def test_empty_windows():

    m = SlidingWindowMonitor()

    add(m, 0)

    rows = m.advance(7)

    assert len(rows) == 3

    assert [
        r["packet_count"] for r in rows
    ] == [1, 0, 0]


def test_late_packet():

    m = SlidingWindowMonitor()

    add(m, 0)
    m.advance(5)

    assert not add(m, 4)

    assert m.late == 1
    assert m.accepted == 1


def test_memory_capacity():

    m = SlidingWindowMonitor(max_packets=2)

    add(m, 0)
    add(m, 1)
    add(m, 2)

    assert len(m.buffer) == 2
    assert m.dropped == 1

    m.advance(7)

    assert len(m.buffer) == 0


def test_partial_flush():

    m = SlidingWindowMonitor()

    add(m, 0)
    add(m, 1)

    rows = m.flush()

    assert len(rows) == 1
    assert rows[0]["partial"] is True
    assert rows[0]["packet_count"] == 2

    assert not m.buffer
    assert not m.flush()


def test_tumbling_compatibility():

    m = SlidingWindowMonitor(window=5, stride=5)

    for t in range(10):
        add(m, t)

    rows = m.advance(10)

    assert len(rows) == 1
    assert rows[0]["packet_count"] == 5


def test_invalid_configuration():

    with pytest.raises(ValueError):
        SlidingWindowMonitor(window=0)

    with pytest.raises(ValueError):
        SlidingWindowMonitor(window=5, stride=6)

    with pytest.raises(ValueError):
        SlidingWindowMonitor(max_packets=0)


def test_catchup_limit():

    m = SlidingWindowMonitor(
        max_windows_per_advance=10
    )

    add(m, 0)

    with pytest.raises(RuntimeError):
        m.advance(1000)

    assert m.clock == 0
