"""Bounded event-time sliding windows for network metadata."""

import math
from collections import deque
from datetime import datetime, timezone


def utc(t):
    return datetime.fromtimestamp(
        t, timezone.utc
    ).isoformat()


class SlidingWindowMonitor:

    def __init__(
        self,
        window=5,
        stride=1,
        max_packets=100000,
        max_windows_per_advance=1000
    ):
        if not (
            math.isfinite(window)
            and math.isfinite(stride)
            and 0 < stride <= window
            and max_packets > 0
            and max_windows_per_advance > 0
        ):
            raise ValueError("Invalid window limits")

        self.window = window
        self.stride = stride
        self.max_packets = max_packets
        self.max_windows_per_advance = max_windows_per_advance

        self.buffer = deque()

        self.start = None
        self.clock = -math.inf

        self.accepted = 0
        self.late = 0
        self.dropped = 0
        self.emitted = 0

    def _snapshot(self, start, end, partial=False):

        flows = {}
        hosts = {}
        packet_count = 0
        byte_count = 0

        for t, src, dst, sp, dp, proto, size in self.buffer:

            if not start <= t < end:
                continue

            packet_count += 1
            byte_count += size

            key = (src, dst, sp, dp, proto)

            row = flows.setdefault(
                key,
                {
                    "src": src,
                    "dst": dst,
                    "sport": sp,
                    "dport": dp,
                    "protocol": proto,
                    "packets": 0,
                    "ip_bytes": 0
                }
            )

            row["packets"] += 1
            row["ip_bytes"] += size

            for host, direction, peer in [
                (src, "out", dst),
                (dst, "in", src)
            ]:
                h = hosts.setdefault(
                    host,
                    {
                        "host": host,
                        "in_packets": 0,
                        "out_packets": 0,
                        "in_bytes": 0,
                        "out_bytes": 0,
                        "peers": set()
                    }
                )

                h[direction + "_packets"] += 1
                h[direction + "_bytes"] += size
                h["peers"].add(peer)

        host_rows = []

        for h in hosts.values():
            row = {
                k: v for k, v in h.items()
                if k != "peers"
            }

            row["unique_peers"] = len(h["peers"])
            host_rows.append(row)

        self.emitted += 1

        return {
            "window_start": utc(start),
            "window_end": utc(end),
            "window_seconds": self.window,
            "stride_seconds": self.stride,
            "partial": partial,
            "packet_count": packet_count,
            "ip_bytes": byte_count,
            "flow_count": len(flows),
            "host_count": len(host_rows),
            "flows": list(flows.values()),
            "hosts": host_rows,
            "late_packets_total": self.late,
            "dropped_packets_total": self.dropped,
            "feature_status": "metadata_only",
            "ai_verdict": None
        }

    def _prune(self):
        while (
            self.buffer
            and self.buffer[0][0] < self.start
        ):
            self.buffer.popleft()

    def advance(self, t):

        if not math.isfinite(t) or t < self.clock:
            raise ValueError("Invalid event time")

        if self.start is None:
            self.clock = t
            return []

        if t >= self.start + self.window:
            count = math.floor(
                (t - self.start - self.window)
                / self.stride
            ) + 1

            if count > self.max_windows_per_advance:
                raise RuntimeError(
                    "Window catch-up limit exceeded"
                )

        self.clock = t
        rows = []

        while self.start + self.window <= t:

            end = self.start + self.window

            rows.append(
                self._snapshot(self.start, end)
            )

            self.start += self.stride
            self._prune()

        return rows

    def add(
        self,
        timestamp,
        src,
        dst,
        sport,
        dport,
        protocol,
        ip_bytes
    ):

        if (
            not math.isfinite(timestamp)
            or ip_bytes < 0
        ):
            raise ValueError("Invalid packet metadata")

        if timestamp < self.clock:
            self.late += 1
            return []

        if self.start is None:
            self.start = (
                math.floor(timestamp / self.stride)
                * self.stride
            )

        rows = self.advance(timestamp)

        if len(self.buffer) >= self.max_packets:
            self.dropped += 1
            return rows

        self.buffer.append(
            (
                timestamp,
                src,
                dst,
                sport,
                dport,
                protocol,
                ip_bytes
            )
        )

        self.accepted += 1

        return rows

    def flush(self):

        if not self.buffer:
            return []

        row = self._snapshot(
            self.start,
            self.start + self.window,
            partial=True
        )

        self.buffer.clear()
        self.start = None

        return [row]

    def summary(self):

        return {
            "accepted_packets": self.accepted,
            "late_packets": self.late,
            "dropped_packets": self.dropped,
            "emitted_windows": self.emitted,
            "buffered_packets": len(self.buffer),
            "window_seconds": self.window,
            "stride_seconds": self.stride,
            "cnn_enabled": False
        }
