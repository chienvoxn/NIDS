import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
import torch

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder

from nids.demo_engine import DemoEngine
from nids.sprint3_residual import MultiScaleResidualCNN


@pytest.fixture
def mhsa_bundle(tmp_path):
    features = [f"f{i}" for i in range(57)]

    frame = pd.DataFrame(
        [[0.0] * 57, [1.0] * 57],
        columns=features
    )

    imputer = SimpleImputer(
        strategy="median",
        keep_empty_features=True
    ).fit(frame)

    scaler = StandardScaler().fit(
        imputer.transform(frame)
    )

    encoder = LabelEncoder().fit(["0", "1"])

    joblib.dump(
        {
            "features": features,
            "imputer": imputer,
            "scaler": scaler,
            "label_encoder": encoder,
        },
        tmp_path / "cnn_preprocessing.joblib"
    )

    cfg = dict(
        n_features=57,
        n_classes=2,
        width=32,
        heads=4,
        blocks=2,
        dropout=0.1,
        residual=True,
        se=False,
        mhsa=True,
    )

    model = MultiScaleResidualCNN(**cfg)

    torch.save(
        {
            **cfg,
            "state_dict": model.state_dict(),
            "dataset": "ctu",
            "variant": "cnn_mhsa",
            "classes": ["0", "1"],
            "seed": 42,
            "best_epoch": 1,
        },
        tmp_path / "cnn_mhsa.pt"
    )

    report = {
        "dataset": "ctu",
        "profile": "full",
        "features": features,
        "classes": ["0", "1"],
        "arguments": {
            "width": 32,
            "dropout": 0.1,
            "seed": 42,
        },
        "split_sha256": "synthetic-test",
        "models": {
            "cnn_mhsa": {
                "configuration": {
                    "se": False,
                    "mhsa": True,
                },
                "best_epoch": 1,
                "validation": {
                    "f1_score": 0.5
                },
            },
        },
    }

    (tmp_path / "cnn_metrics.json").write_text(
        json.dumps(report),
        encoding="utf-8"
    )

    return tmp_path, frame


def test_mhsa_load_and_predict(mhsa_bundle):
    path, frame = mhsa_bundle

    engine = DemoEngine(
        path, "ctu", "full", variant="cnn_mhsa"
    )

    result = engine.predict(frame)

    assert engine.metadata["variant"] == "cnn_mhsa"
    assert engine.metadata["seed"] == 42
    assert engine.checkpoint_path.name == "cnn_mhsa.pt"
    assert len(result) == 2
    assert result.confidence.between(0, 1).all()


def test_mhsa_feature_order(mhsa_bundle):
    path, frame = mhsa_bundle

    engine = DemoEngine(
        path, "ctu", "full", variant="cnn_mhsa"
    )

    assert engine.predict(frame).equals(
        engine.predict(frame[frame.columns[::-1]])
    )


def test_mhsa_missing_features(mhsa_bundle):
    path, frame = mhsa_bundle

    engine = DemoEngine(
        path, "ctu", "full", variant="cnn_mhsa"
    )

    with pytest.raises(ValueError, match="Missing features"):
        engine.predict(frame.iloc[:, :-1])


def test_mhsa_wrong_variant(mhsa_bundle):
    path, _ = mhsa_bundle

    with pytest.raises(ValueError):
        DemoEngine(
            path, "ctu", "full",
            variant="cnn_se_mhsa"
        )


def test_mhsa_real_cic():
    root = Path("outputs/final/cic_final_runs/seed_42")

    if not (root / "cnn_mhsa.pt").is_file():
        pytest.skip("Local CIC MHSA checkpoint unavailable")

    engine = DemoEngine(
        root, "cic", "full", variant="cnn_mhsa"
    )

    assert len(engine.features) == 60
    assert len(engine.labels) == 15
    assert engine.metadata["seed"] == 42

    row = pd.DataFrame(
        [engine.median],
        columns=engine.features
    )

    result = engine.predict(row)

    assert len(result) == 1
    assert result.confidence.between(0, 1).all()


def test_mhsa_preprocessing(mhsa_bundle):
    path, frame = mhsa_bundle

    engine = DemoEngine(
        path, "ctu", "full", variant="cnn_mhsa"
    )

    frame.iloc[0, 0] = np.nan

    x = engine.prep["scaler"].transform(
        engine.prep["imputer"].transform(frame)
    ).astype(np.float32)

    with torch.inference_mode():
        expected = torch.softmax(
            engine.model(torch.from_numpy(x)), 1
        ).numpy().max(1)

    assert np.allclose(
        engine.predict(frame).confidence,
        expected,
        atol=1e-7
    )
