"""M2.3 provisional bidirectional TCP/UDP flow statistics.

This module does NOT claim CICFlowMeter feature parity and deliberately refuses
CNN-ready export. Packet length currently means IPv4 L4 payload bytes; this
convention MUST be validated against the CICFlowMeter version used in training.

Inputs: classic Ethernet/IPv4 PCAP only, using the existing nids.pcap.iter_pcap.
Packets must be timestamp ordered. Unsupported packets are skipped explicitly.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import ipaddress
import json
import math
import struct
from typing import Iterable

from nids.pcap import iter_pcap


@dataclass(frozen=True)
class PacketInfo:
    t: float
    src: str
    dst: str
    sport: int
    dport: int
    proto: int
    payload_len: int
    ip_len: int
    l4_header_len: int
    flags: int
    tcp_window: int | None


def parse_packet(t: float, raw: bytes) -> PacketInfo | None:
    """Parse complete, untagged Ethernet/IPv4 TCP or UDP packet."""
    if len(raw) < 34 or raw[12:14] != b'\x08\x00':
        return None
    start = 14
    vihl = raw[start]
    if vihl >> 4 != 4:
        return None
    ihl = (vihl & 15) * 4
    if ihl < 20 or len(raw) < start + ihl:
        return None
    ip_len = struct.unpack_from('!H', raw, start + 2)[0]
    proto = raw[start + 9]
    if proto not in (6, 17) or ip_len < ihl:
        return None
    frag = struct.unpack_from('!H', raw, start + 6)[0]
    if frag & 0x3fff:  # reject fragmented datagrams, including first fragments with MF
        return None
    if len(raw) < start + ip_len:  # no partial/truncated packet interpretation
        return None
    l4 = start + ihl
    if proto == 6:
        if ip_len < ihl + 20:
            return None
        sport, dport = struct.unpack_from('!HH', raw, l4)
        hlen = (raw[l4 + 12] >> 4) * 4
        if hlen < 20 or ip_len < ihl + hlen:
            return None
        flags = raw[l4 + 13]
        window = struct.unpack_from('!H', raw, l4 + 14)[0]
    else:
        if ip_len < ihl + 8:
            return None
        sport, dport = struct.unpack_from('!HH', raw, l4)
        hlen, flags, window = 8, 0, None
        udp_len = struct.unpack_from('!H', raw, l4 + 4)[0]
        if udp_len < 8 or udp_len > ip_len - ihl:
            return None
    payload = ip_len - ihl - hlen
    return PacketInfo(t, str(ipaddress.ip_address(raw[start+12:start+16])),
                      str(ipaddress.ip_address(raw[start+16:start+20])),
                      sport, dport, proto, payload, ip_len, hlen, flags, window)


def _stats(values: list[float]) -> tuple[float, float, float, float]:
    if not values:
        return 0.0, 0.0, 0.0, 0.0
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    return mean, math.sqrt(variance), min(values), max(values)


def _iat(times: list[float]) -> list[float]:
    return [(times[i] - times[i-1]) * 1e6 for i in range(1, len(times))]


@dataclass
class Flow:
    first: PacketInfo
    last_t: float
    fwd: list[PacketInfo] = field(default_factory=list)
    bwd: list[PacketInfo] = field(default_factory=list)
    all_packets: list[PacketInfo] = field(default_factory=list)

    def add(self, p: PacketInfo) -> None:
        forward = (p.src, p.sport) == (self.first.src, self.first.sport) and (p.dst, p.dport) == (self.first.dst, self.first.dport)
        (self.fwd if forward else self.bwd).append(p)
        self.all_packets.append(p)
        self.last_t = p.t

    def _activity_and_subflows(self, threshold_s: float = 5.0) -> tuple[list[float], list[float], list[list[PacketInfo]]]:
        """Segment packet bursts using a configurable inactivity gap.

        This is an explicit local approximation, NOT validated CICFlowMeter parity.
        Durations returned in microseconds. Single-packet bursts have duration zero.
        """
        if threshold_s <= 0:
            raise ValueError('threshold_s must be > 0')
        active_durations: list[float] = []
        idle_durations: list[float] = []
        bursts: list[list[PacketInfo]] = []
        current: list[PacketInfo] = []
        for pkt in self.all_packets:
            if current and pkt.t - current[-1].t > threshold_s:
                active_durations.append(max(0., (current[-1].t - current[0].t) * 1e6))
                idle_durations.append((pkt.t - current[-1].t) * 1e6)
                bursts.append(current)
                current = []
            current.append(pkt)
        if current:
            active_durations.append(max(0., (current[-1].t - current[0].t) * 1e6))
            bursts.append(current)
        return active_durations, idle_durations, bursts

    def features(self) -> dict[str, float]:
        duration_us = max(0.0, (self.last_t - self.first.t) * 1e6)
        duration_s = duration_us / 1e6
        fwd_len = [float(p.payload_len) for p in self.fwd]
        bwd_len = [float(p.payload_len) for p in self.bwd]
        all_len = [float(p.payload_len) for p in self.all_packets]
        f_mean, f_std, f_min, f_max = _stats(fwd_len)
        b_mean, b_std, b_min, b_max = _stats(bwd_len)
        all_mean, all_std, all_min, all_max = _stats(all_len)
        times = [p.t for p in self.all_packets]
        f_iat = _iat([p.t for p in self.fwd])
        b_iat = _iat([p.t for p in self.bwd])
        a_iat = _iat(times)
        fi_mean, fi_std, fi_min, fi_max = _stats(f_iat)
        bi_mean, bi_std, bi_min, bi_max = _stats(b_iat)
        ai_mean, ai_std, ai_min, ai_max = _stats(a_iat)
        active, idle, bursts = self._activity_and_subflows()
        active_mean, active_std, active_min, active_max = _stats(active)
        idle_mean, idle_std, idle_min, idle_max = _stats(idle)
        # PROVISIONAL: average backward payload bytes per inactivity-defined burst.
        # Subflow semantics differ across CICFlowMeter versions and MUST be verified.
        bwd_endpoint = (self.first.dst, self.first.dport, self.first.src, self.first.sport)
        burst_bwd_bytes = [sum(p.payload_len for p in burst
                               if (p.src, p.sport, p.dst, p.dport) == bwd_endpoint)
                           for burst in bursts]
        subflow_bwd_bytes = (sum(burst_bwd_bytes) / len(bursts)) if bursts else float('nan')
        return {
            'Destination Port': float(self.first.dport),
            'Flow Duration': duration_us,
            'Total Fwd Packets': float(len(self.fwd)),
            'Total Backward Packets': float(len(self.bwd)),
            'Total Length of Fwd Packets': sum(fwd_len),
            'Total Length of Bwd Packets': sum(bwd_len),
            'Fwd Packet Length Max': f_max, 'Fwd Packet Length Min': f_min,
            'Fwd Packet Length Mean': f_mean, 'Fwd Packet Length Std': f_std,
            'Bwd Packet Length Max': b_max, 'Bwd Packet Length Min': b_min,
            'Bwd Packet Length Mean': b_mean, 'Bwd Packet Length Std': b_std,
            'Flow Bytes/s': sum(all_len)/duration_s if duration_s > 0 else float('nan'),
            'Flow Packets/s': len(self.all_packets)/duration_s if duration_s > 0 else float('nan'),
            'Flow IAT Mean': ai_mean, 'Flow IAT Std': ai_std,
            'Flow IAT Max': ai_max, 'Flow IAT Min': ai_min,
            'Fwd IAT Total': sum(f_iat), 'Fwd IAT Mean': fi_mean,
            'Fwd IAT Std': fi_std, 'Fwd IAT Max': fi_max, 'Fwd IAT Min': fi_min,
            'Bwd IAT Total': sum(b_iat), 'Bwd IAT Mean': bi_mean,
            'Bwd IAT Std': bi_std, 'Bwd IAT Max': bi_max, 'Bwd IAT Min': bi_min,
            'Fwd PSH Flags': float(sum(bool(p.flags & 8) for p in self.fwd)),
            'Fwd Header Length': float(sum(p.l4_header_len for p in self.fwd)),
            'Bwd Header Length': float(sum(p.l4_header_len for p in self.bwd)),
            'Fwd Packets/s': len(self.fwd)/duration_s if duration_s > 0 else float('nan'),
            'Bwd Packets/s': len(self.bwd)/duration_s if duration_s > 0 else float('nan'),
            'Min Packet Length': all_min, 'Max Packet Length': all_max,
            'Packet Length Mean': all_mean, 'Packet Length Std': all_std,
            'Packet Length Variance': all_std**2,
            'FIN Flag Count': float(sum(bool(p.flags & 1) for p in self.all_packets)),
            'RST Flag Count': float(sum(bool(p.flags & 4) for p in self.all_packets)),
            'PSH Flag Count': float(sum(bool(p.flags & 8) for p in self.all_packets)),
            'ACK Flag Count': float(sum(bool(p.flags & 16) for p in self.all_packets)),
            'URG Flag Count': float(sum(bool(p.flags & 32) for p in self.all_packets)),
            'Down/Up Ratio': len(self.bwd)/len(self.fwd) if self.fwd else float('nan'),
            'Average Packet Size': sum(all_len)/len(all_len),
            'Init_Win_bytes_forward': float(self.fwd[0].tcp_window) if self.fwd and self.fwd[0].tcp_window is not None else float('nan'),
            'Init_Win_bytes_backward': float(self.bwd[0].tcp_window) if self.bwd and self.bwd[0].tcp_window is not None else float('nan'),
            'Subflow Bwd Bytes': subflow_bwd_bytes,
            'Active Mean': active_mean, 'Active Std': active_std,
            'Active Max': active_max, 'Active Min': active_min,
            'Idle Mean': idle_mean, 'Idle Std': idle_std,
            'Idle Max': idle_max, 'Idle Min': idle_min,
            'act_data_pkt_fwd': float(sum(p.payload_len > 0 for p in self.fwd)),
            'min_seg_size_forward': float(min(p.l4_header_len for p in self.fwd)),
        }


def extract_flows(packets: Iterable[tuple[float, bytes]], idle_timeout: float = 120.0) -> list[Flow]:
    """Return flows, splitting a 5-tuple after idle timeout; timestamps must be ordered.

    NOTE: this is NOT the CICFlowMeter's complete flow-termination policy.
    """
    if idle_timeout <= 0:
        raise ValueError('idle_timeout must be > 0')
    active: dict[tuple, Flow] = {}
    completed: list[Flow] = []
    last_timestamp: float | None = None
    for t, raw in packets:
        if last_timestamp is not None and t < last_timestamp:
            raise ValueError('PCAP timestamps must be nondecreasing')
        last_timestamp = t
        p = parse_packet(t, raw)
        if p is None:
            continue
        a, b = sorted(((p.src, p.sport), (p.dst, p.dport)))
        key = (p.proto, a, b)
        prior = active.get(key)
        if prior is not None and t - prior.last_t > idle_timeout:
            completed.append(active.pop(key))
            prior = None
        if prior is None:
            prior = Flow(first=p, last_t=t)
            active[key] = prior
        prior.add(p)
    completed.extend(active.values())
    return sorted(completed, key=lambda f: f.first.t)


def audit_manifest(manifest_path: str | Path) -> tuple[list[str], list[str]]:
    """Return available formula names; DOES NOT certify CICFlowMeter parity."""
    manifest = json.loads(Path(manifest_path).read_text(encoding='utf-8'))
    names = manifest['feature_order']
    if len(names) != 60 or len(set(names)) != 60:
        raise ValueError('Unexpected feature manifest')
    # Identify supported keys through a deliberately tiny synthetic flow.
    p = PacketInfo(0, '10.0.0.1', '10.0.0.2', 1, 2, 6, 0, 40, 20, 2, 1024)
    f = Flow(p, 0)
    f.add(p)
    implemented = f.features()
    return [n for n in names if n in implemented], [n for n in names if n not in implemented]


def extract_pcap_provisional(path: str | Path) -> list[dict[str, float]]:
    """Return provisional feature vectors; NEVER treat as CNN-ready without parity validation."""
    return [flow.features() for flow in extract_flows(iter_pcap(path))]
