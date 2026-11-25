"""Inference using trusted fitted Gaussian artifacts."""

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


class GaussianEngine:

    def __init__(self, directory, dataset):

        dimensions = {"cic": 60, "ctu": 57}

        if dataset not in dimensions:
            raise ValueError("Unsupported dataset")

        directory = Path(directory)
        path = directory / "gaussian.joblib"

        self.dataset = dataset
        self.sha256 = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()

        artifact = joblib.load(path)

        metrics = json.loads(
            (directory / "metrics.json").read_text()
        )

        self.features = list(artifact["features"])
        self.imputer = artifact["imputer"]
        self.scaler = artifact["scaler"]

        self.center = np.asarray(artifact["center"])
        self.variance = np.asarray(artifact["variance"])

        self.threshold = float(artifact["threshold"])

        n = dimensions[dataset]

        if len(self.features) != n:
            raise ValueError("Invalid feature count")

        if len(set(self.features)) != n:
            raise ValueError("Duplicate features")

        if list(self.imputer.feature_names_in_) != self.features:
            raise ValueError("Preprocessing order mismatch")

        for x in [self.center, self.variance]:

            if x.shape != (n,) or not np.isfinite(x).all():
                raise ValueError("Invalid Gaussian parameters")

        if not (self.variance > 0).all():
            raise ValueError("Invalid variance")

        if not np.isfinite(self.threshold):
            raise ValueError("Invalid threshold")

        if self.threshold != float(metrics["threshold"]):
            raise ValueError("Artifact threshold mismatch")

    def predict(self, frame):

        if frame.empty or frame.columns.duplicated().any():
            raise ValueError("Invalid input frame")

        missing = set(self.features) - set(frame.columns)

        if missing:
            raise ValueError(f"Missing features: {missing}")

        raw = frame[self.features].apply(
            pd.to_numeric, errors="raise"
        )

        raw = raw.replace(
            [np.inf, -np.inf], np.nan
        )

        x = self.imputer.transform(raw)
        x = self.scaler.transform(x).astype(np.float32)

        if not np.isfinite(x).all():
            raise ValueError("Nonfinite Gaussian input")

        scores = np.sum(
            (x - self.center) ** 2 / self.variance,
            axis=1
        )

        if not np.isfinite(scores).all():
            raise ValueError("Nonfinite Gaussian score")

        return pd.DataFrame({
            "gaussian_score": scores,
            "gaussian_threshold": self.threshold,
            "gaussian_anomaly": scores > self.threshold
        })
