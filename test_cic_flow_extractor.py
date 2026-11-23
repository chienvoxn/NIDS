"""Synthetic regression tests for provisional M2.2 extractor (pytest)."""
import ipaddress
import math
import struct

import pytest

from nids.cic_flow_extractor import extract_flows, parse_packet


def packet(src, dst, sport, dport, *, payload=b'', flags=0x10, window=1024, proto=6):
    s = ipaddress.ip_address(src).packed
    d = ipaddress.ip_address(dst).packed
    if proto == 6:
        layer4 = struct.pack('!HHIIHHHH', sport, dport, 0, 0, (5 << 12) | flags, window, 0, 0) + payload
    else:
        layer4 = struct.pack('!HHHH', sport, dport, 8 + len(payload), 0) + payload
    length = 20 + len(layer4)
    ip = struct.pack('!BBHHHBBH4s4s', 0x45, 0, length, 1, 0, 64, proto, 0, s, d)
    return b'\x00' * 12 + b'\x08\x00' + ip + layer4


def test_bidirectional_direction_and_microseconds():
    packets = [
        (1.0, packet('10.0.0.1', '10.0.0.2', 123, 80, payload=b'ab', flags=2)),
        (1.1, packet('10.0.0.2', '10.0.0.1', 80, 123, payload=b'xyz', flags=18)),
        (1.3, packet('10.0.0.1', '10.0.0.2', 123, 80, payload=b'q', flags=24)),
    ]
    rows = extract_flows(packets)
    assert len(rows) == 1
    feat = rows[0].features()
    assert feat['Destination Port'] == 80
    assert feat['Flow Duration'] == pytest.approx(300000)
    assert feat['Total Fwd Packets'] == 2
    assert feat['Total Backward Packets'] == 1
    assert feat['Total Length of Fwd Packets'] == 3
    assert feat['Total Length of Bwd Packets'] == 3
    assert feat['Fwd IAT Total'] == pytest.approx(300000)
    assert feat['ACK Flag Count'] == 2
    assert feat['Init_Win_bytes_backward'] == 1024


def test_timeout_splits_reused_flow():
    raw = packet('10.0.0.1', '10.0.0.2', 123, 80)
    flows = extract_flows([(1.0, raw), (2.0, raw), (200.0, raw)], idle_timeout=120)
    assert len(flows) == 2
    assert [len(f.all_packets) for f in flows] == [2, 1]


def test_non_monotonic_packet_times_rejected():
    raw = packet('10.0.0.1', '10.0.0.2', 123, 80)
    with pytest.raises(ValueError, match='nondecreasing'):
        extract_flows([(2.0, raw), (1.0, raw)])


def test_udp_and_zero_duration():
    raw = packet('10.0.0.1', '10.0.0.2', 123, 53, proto=17, payload=b'ab')
    p = parse_packet(1.0, raw)
    assert p is not None and p.payload_len == 2
    flow = extract_flows([(1.0, raw)])[0]
    assert math.isnan(flow.features()['Flow Bytes/s'])


def test_fragmented_ipv4_not_interpreted_as_complete():
    raw = bytearray(packet('10.0.0.1', '10.0.0.2', 123, 80))
    raw[14+6:14+8] = b'\x20\x00' # MF bit
    assert parse_packet(1.0, bytes(raw)) is None
