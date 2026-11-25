import numpy as np
import pandas as pd
import pytest

from scripts.audit_live_features import compare


FEATURES = [f"feature_{i}" for i in range(60)]


def frames():

    data = {
        name: [
            1.0, 2.0, np.nan,
            np.inf, -np.inf
        ]
        for name in FEATURES
    }

    data["match_id"] = [
        "flow_1", "flow_2", "flow_3",
        "flow_4", "flow_5"
    ]

    a = pd.DataFrame(data)

    return a, a.copy()


def test_matching_values():

    a, b = frames()

    report = compare(a, b, FEATURES)

    assert len(report) == 60
    assert report["all_values_match"].all()
    assert (report["total_matches"] == 5).all()
    assert (report["matching_nan"] == 1).all()
    assert (report["matching_posinf"] == 1).all()
    assert (report["matching_neginf"] == 1).all()


def test_numeric_mismatch():

    a, b = frames()

    b.loc[0, FEATURES[0]] = 100

    report = compare(a, b, FEATURES)

    assert report.iloc[0]["finite_mismatches"] == 1
    assert report.iloc[0]["total_mismatches"] == 1
    assert not report.iloc[0]["all_values_match"]


def test_nonfinite_mismatch():

    a, b = frames()

    b.loc[2, FEATURES[0]] = 0
    b.loc[3, FEATURES[0]] = -np.inf

    report = compare(a, b, FEATURES)

    assert report.iloc[0]["nonfinite_mismatches"] == 2
    assert report.iloc[0]["total_mismatches"] == 2


def test_reference_order_independent():

    a, b = frames()

    b = b.iloc[::-1]

    report = compare(a, b, FEATURES)

    assert report["all_values_match"].all()


def test_unmatched_ids_rejected():

    a, b = frames()

    b.loc[0, "match_id"] = "unknown"

    with pytest.raises(ValueError):
        compare(a, b, FEATURES)


def test_duplicate_ids_rejected():

    a, b = frames()

    b.loc[0, "match_id"] = b.loc[1, "match_id"]

    with pytest.raises(ValueError):
        compare(a, b, FEATURES)
