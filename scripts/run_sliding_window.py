"""Live sliding-window monitoring for an authorized network."""

import argparse
import csv
import heapq
import json
import math
import queue
import threading
import time
from pathlib import Path

from scapy.all import AsyncSniffer, IP, TCP, UDP

from nids.sliding_window import SlidingWindowMonitor


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--iface", required=True)
    parser.add_argument("--seconds", type=int, default=20)
    parser.add_argument("--max-packets", type=int, default=10000)
    parser.add_argument("--out", type=Path,
                        default=Path("outputs/sliding_window"))

    a = parser.parse_args()

    if not 1 <= a.seconds <= 300:
        parser.error("seconds must be between 1 and 300")

    if not 1 <= a.max_packets <= 100000:
        parser.error("invalid packet limit")

    monitor = SlidingWindowMonitor(
        window=5,
        stride=1,
        max_packets=100000
    )

    pending = queue.Queue(maxsize=8192)
    heap = []
    ready = threading.Event()

    stats = {
        "captured": 0,
        "unsupported": 0,
        "queue_dropped": 0,
        "reorder_dropped": 0,
        "timer_ticks": 0,
        "periodic_windows": 0,
        "capture_complete": False
    }

    def receive(pkt):

        stats["captured"] += 1

        if IP not in pkt:
            stats["unsupported"] += 1
            return

        ip = pkt[IP]

        if ip.frag or int(ip.flags) & 1:
            stats["unsupported"] += 1
            return

        transport = ip.payload

        if not isinstance(transport, (TCP, UDP)):
            stats["unsupported"] += 1
            return

        t = float(pkt.time)

        if not math.isfinite(t) or ip.len is None:
            stats["unsupported"] += 1
            return

        item = (
            t,
            str(ip.src),
            str(ip.dst),
            int(transport.sport),
            int(transport.dport),
            int(ip.proto),
            int(ip.len)
        )

        try:
            pending.put_nowait(item)
        except queue.Full:
            stats["queue_dropped"] += 1

    out = a.out / str(time.time_ns())
    out.mkdir(parents=True, exist_ok=False)

    with (
        (out / "windows.jsonl").open("w") as wf,
        (out / "accepted_metadata.csv").open(
            "w", newline=""
        ) as pf
    ):

        writer = csv.writer(pf)

        writer.writerow([
            "timestamp", "src", "dst",
            "sport", "dport", "protocol", "ip_bytes"
        ])

        def emit(rows):

            for row in rows:
                wf.write(
                    json.dumps(row, allow_nan=False) + "\n"
                )

            wf.flush()

        def drain_queue():

            for _ in range(8192):

                try:
                    item = pending.get_nowait()
                except queue.Empty:
                    break

                if len(heap) >= 10000:
                    stats["reorder_dropped"] += 1
                    continue

                heapq.heappush(heap, item)

        def process_until(cutoff):

            while heap and heap[0][0] <= cutoff:

                item = heapq.heappop(heap)
                before = monitor.accepted

                emit(monitor.add(*item))

                if monitor.accepted > before:
                    writer.writerow(item)

        sniffer = AsyncSniffer(
            iface=a.iface,
            store=False,
            prn=receive,
            count=a.max_packets,
            started_callback=ready.set
        )

        sniffer.start()

        try:

            if not ready.wait(5):
                if not sniffer.running:
                    sniffer.join()
                raise RuntimeError("Capture startup timeout")

            deadline = time.monotonic() + a.seconds

            while sniffer.running:

                remaining = deadline - time.monotonic()

                if remaining <= 0:
                    break

                time.sleep(min(0.1, remaining))

                drain_queue()

                cutoff = time.time() - 0.5

                process_until(cutoff)

                stats["timer_ticks"] += 1

                if cutoff >= monitor.clock:

                    rows = monitor.advance(cutoff)

                    stats["periodic_windows"] += len(rows)

                    emit(rows)

        finally:

            if sniffer.running:
                sniffer.stop()

            sniffer.join()

        while not pending.empty():
            drain_queue()

        process_until(math.inf)

        emit(
            monitor.advance(
                max(monitor.clock, time.time())
            )
        )

        emit(monitor.flush())

        stats["capture_complete"] = True

    stats.update(monitor.summary())

    stats["verified_live"] = False
    stats["semantic_parity"] = False
    stats["feature_status"] = "metadata_only"

    (out / "summary.json").write_text(
        json.dumps(stats, indent=2)
    )

    print(json.dumps(stats, indent=2))
    print("Saved:", out)


if __name__ == "__main__":
    main()
