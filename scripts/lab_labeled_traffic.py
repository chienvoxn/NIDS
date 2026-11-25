"""Bounded benign traffic patterns for isolated NIDS evaluation."""

import argparse
import json
import socket
import time
from pathlib import Path

HOST = "10.203.0.1"
PORT = 18090


def server():
    with socket.socket() as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((HOST, PORT))
        s.listen(4)
        s.settimeout(20)

        for _ in range(15):
            conn, addr = s.accept()

            with conn:
                conn.settimeout(3)
                data = conn.recv(1024)
                conn.sendall(data)

    print("LAB SERVER COMPLETE")


def client(out):
    records = []

    scenarios = [
        ("BENIGN", 1.0),
        ("CONNECTION_PATTERN", 0.3),
        ("PERIODIC_PATTERN", 0.7)
    ]

    for scenario, interval in scenarios:

        print("SCENARIO:", scenario, flush=True)

        for i in range(5):

            payload = b"NIDS authorized lab test"

            with socket.socket() as s:
                s.settimeout(3)
                s.connect((HOST, PORT))

                src_ip, src_port = s.getsockname()

                started = time.time()

                s.sendall(payload)
                received = s.recv(1024)

                assert received == payload

            records.append({
                "scenario": scenario,
                "ground_truth_attack": False,
                "source_ip": src_ip,
                "source_port": src_port,
                "destination_ip": HOST,
                "destination_port": PORT,
                "started_at_epoch": started,
                "completed_at_epoch": time.time(),
                "success": True
            })

            print(
                scenario,
                i + 1,
                "PASS",
                flush=True
            )

            time.sleep(interval)

    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)

    path.write_text(
        json.dumps(records, indent=2)
    )

    print("LAB TRAFFIC COMPLETE:", len(records))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "mode",
        choices=["server", "client"]
    )

    parser.add_argument(
        "--out",
        default="ground_truth.json"
    )

    args = parser.parse_args()

    if args.mode == "server":
        server()
    else:
        client(args.out)
