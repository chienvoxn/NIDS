"""Bounded asynchronous inference for provisional live flows."""

import queue
import threading
import time
from pathlib import Path

import pandas as pd

from nids.stream_detection import process_batch, PROVISIONAL


class LiveShadowWorker:

    def __init__(self, engine, out, capacity=512, batch_size=32):

        if capacity < 1 or not 1 <= batch_size <= 512:
            raise ValueError("Invalid worker limits")

        self.engine = engine
        self.out = Path(out)
        self.batch_size = batch_size
        self.queue = queue.Queue(maxsize=capacity)
        self.stop_event = threading.Event()

        self.submitted = 0
        self.dropped = 0
        self.processed = 0
        self.attack_predictions = 0
        self.error = None

        self.thread = threading.Thread(
            target=self._run,
            name="live-shadow-worker"
        )

    def __enter__(self):

        self.thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):

        self.finish()

    def submit(self, row, flow_id):

        if row.get("feature_status") != PROVISIONAL:
            raise ValueError("Provisional flow required")

        if self.stop_event.is_set():
            raise RuntimeError("Worker already closed")

        item = (
            flow_id,
            time.monotonic(),
            dict(row)
        )

        try:

            self.queue.put_nowait(item)
            self.submitted += 1

        except queue.Full:

            self.dropped += 1

    def _run(self):

        try:

            self.out.mkdir(
                parents=True,
                exist_ok=True
            )

            with (
                (self.out / "predictions.csv").open(
                    "w", newline=""
                ) as csv_file,
                (self.out / "predictions.jsonl").open(
                    "w"
                ) as json_file
            ):

                header = True

                while (
                    not self.stop_event.is_set()
                    or not self.queue.empty()
                ):

                    try:
                        first = self.queue.get(timeout=0.05)
                    except queue.Empty:
                        continue

                    items = [first]

                    while len(items) < self.batch_size:

                        try:
                            items.append(
                                self.queue.get_nowait()
                            )
                        except queue.Empty:
                            break

                    frame = pd.DataFrame(
                        [item[2] for item in items]
                    )

                    result = process_batch(
                        self.engine,
                        frame,
                        mode="shadow",
                        batch_size=self.batch_size
                    )

                    done = time.monotonic()

                    result["flow_row"] = [
                        item[0] for item in items
                    ]

                    result["submit_to_prediction_ms"] = [
                        (done - item[1]) * 1000
                        for item in items
                    ]

                    result.to_csv(
                        csv_file,
                        index=False,
                        header=header
                    )

                    header = False

                    json_file.write(
                        result.to_json(
                            orient="records",
                            lines=True
                        )
                    )

                    csv_file.flush()
                    json_file.flush()

                    self.processed += len(result)

                    self.attack_predictions += int(
                        result["is_attack"].sum()
                    )

        except Exception as exc:

            self.error = repr(exc)

    def finish(self):

        self.stop_event.set()

        if self.thread.ident is not None:
            self.thread.join()

        return {
            "shadow_submitted": self.submitted,
            "shadow_dropped": self.dropped,
            "shadow_processed": self.processed,
            "shadow_attack_predictions":
                self.attack_predictions,

            "shadow_unprocessed":
                self.submitted - self.processed,

            "shadow_error": self.error,

            "shadow_complete": (
                self.error is None
                and self.processed == self.submitted
                and self.dropped == 0
            ),

            "verified_live": False,
            "semantic_parity": False,
            "official_alerts": 0
        }
