import inspect

from nids.final_baselines import load_data


def test_validation_only_loader_option():
    signature = inspect.signature(load_data)

    assert "include_test" in signature.parameters
    assert signature.parameters["include_test"].default is True

def test_tuning_does_not_read_test(monkeypatch, tmp_path):
    import json
    import pandas as pd
    from nids import final_baselines as module

    monkeypatch.setattr(module, "ROOT", tmp_path)

    folder = tmp_path / "data/processed/final/ctu"
    folder.mkdir(parents=True)

    manifest = {
        "features": ["feature_1", "feature_2"]
    }

    (folder / "manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8"
    )

    frame = pd.DataFrame({
        "feature_1": [1.0, 2.0, 3.0, 4.0],
        "feature_2": [4.0, 3.0, 2.0, 1.0],
        "target": ["0", "1", "0", "1"]
    })

    frame.to_parquet(folder / "train.parquet")
    frame.to_parquet(folder / "validation.parquet")

    assert not (folder / "test.parquet").exists()

    xs, ys, prep, manifest = module.load_data(
        "ctu",
        smoke=False,
        include_test=False
    )

    assert len(xs) == 2
    assert len(ys) == 2
    assert len(xs[0]) == 4
    assert len(xs[1]) == 4

