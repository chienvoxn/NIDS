"""Bounded live metadata windows; no packet storage or CNN feature claims."""
from collections import Counter
from datetime import datetime, timezone
import math


def utc(t):
    return datetime.fromtimestamp(t, timezone.utc).isoformat()


class WindowMonitor:
    def __init__(self, seconds=5, max_flows=10000):
        if seconds <= 0 or max_flows < 1:
            raise ValueError('Invalid window limits')
        self.seconds = seconds
        self.max_flows = max_flows
        self.start = None
        self.flows = {}
        self.dropped = 0
        self.late = 0

    def add(self, timestamp, src, dst, sport, dport, protocol, ip_bytes):
        if not math.isfinite(timestamp) or ip_bytes < 0:
            raise ValueError('Invalid packet metadata')
        if self.start is None:
            self.start = math.floor(timestamp / self.seconds) * self.seconds
        if timestamp < self.start:
            self.late += 1
            return None
        closed = self.advance(timestamp)
        key = (src, dst, sport, dport, protocol)
        if key not in self.flows:
            if len(self.flows) >= self.max_flows:
                self.dropped += 1
                return closed
            self.flows[key] = [0, 0, timestamp, timestamp]
        row = self.flows[key]
        row[0] += 1
        row[1] += ip_bytes
        row[2] = min(row[2], timestamp)
        row[3] = max(row[3], timestamp)
        return closed

    def advance(self, timestamp):
        if self.start is None or timestamp < self.start + self.seconds:
            return None
        result = self.flush()
        self.start = math.floor(timestamp / self.seconds) * self.seconds
        return result

    def flush(self):
        if self.start is None or not (self.flows or self.dropped or self.late):
            return None
        flows = []
        hosts = {}
        for (src, dst, sp, dp, proto), (n, b, first, last) in self.flows.items():
            flows.append(dict(src=src, dst=dst, sport=sp, dport=dp, protocol=proto,
                              time=utc(first), last_time=utc(last), packets=n, ip_bytes=b))
            for host, direction, peer in [(src, 'out', dst), (dst, 'in', src)]:
                h = hosts.setdefault(host, dict(in_packets=0, out_packets=0, in_bytes=0, out_bytes=0, peers=set()))
                h[direction+'_packets'] += n
                h[direction+'_bytes'] += b
                h['peers'].add(peer)
        host_rows = [dict(host=k, **{a: b for a,b in v.items() if a != 'peers'}, unique_peers=len(v['peers'])) for k,v in hosts.items()]
        result = dict(window_start=utc(self.start), window_end=utc(self.start+self.seconds),
                      flows=flows, hosts=host_rows, dropped_packets=self.dropped, late_packets=self.late,
                      feature_status='metadata_only', ai_verdict=None)
        self.flows.clear()
        self.dropped = self.late = 0
        return result
