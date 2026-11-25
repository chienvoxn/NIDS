import numpy as np
import pandas as pd
import pytest

from sklearn.metrics import f1_score

from nids.final_data import split_indices
from nids.final_metrics import evaluate
from nids.final_metrics import binary_detection_metrics


def test_split_no_overlap():

    df = pd.DataFrame({
        "row": np.arange(100),
        "target": ["0"] * 60 + ["1"] * 40
    })

    folds = split_indices(df, seed=42)

    train = set(folds["train"])
    val = set(folds["validation"])
    test = set(folds["test"])

    assert not train.intersection(val)
    assert not train.intersection(test)
    assert not val.intersection(test)

    assert len(train | val | test) == 100

    assert len(train) == 70
    assert len(val) == 15
    assert len(test) == 15


def test_split_reproducibility():

    df = pd.DataFrame({
        "row": np.arange(100),
        "target": ["0"] * 60 + ["1"] * 40
    })

    a = split_indices(df, seed=42)
    b = split_indices(df, seed=42)

    for name in a:
        assert np.array_equal(a[name], b[name])


def test_rare_class_rejected():

    df = pd.DataFrame({
        "row": [0, 1, 2],
        "target": ["rare"] * 3
    })

    with pytest.raises(ValueError):
        split_indices(df)


def test_cic_weighted_f1():

    y = [0, 0, 0, 1, 2, 2]
    pred = [0, 0, 1, 1, 2, 0]

    result = evaluate(
        y, pred, "cic", [0, 1, 2]
    )

    expected = f1_score(
        y, pred, average="weighted"
    )

    assert result["f1_average"] == "weighted"
    assert result["f1_score"] == pytest.approx(expected)


def test_ctu_binary_f1():

    y = [0, 0, 1, 1]
    pred = [0, 1, 1, 1]

    result = evaluate(
        y, pred, "ctu", [0, 1]
    )

    assert result["f1_average"] == "binary"

    assert result["f1_score"] == pytest.approx(
        f1_score(y, pred, average="binary")
    )


def test_binary_detection_metrics():

    y = [0, 0, 1, 1]
    pred = [0, 1, 0, 1]

    result = binary_detection_metrics(y, pred)

    assert result["false_positive_rate"] == 0.5
    assert result["detection_rate"] == 0.5
