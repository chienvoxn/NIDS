from pathlib import Path
from zipfile import ZipFile
from collections import Counter
import argparse
import hashlib
import json

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from nids.sprint2_data import canonical_cic_label


ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data/raw"
OUT = ROOT / "data/processed/final"
REPORTS = ROOT / "reports/final"
SEED = 42
CHUNK = 100000


def source_paths(kind):
    if kind == "cic":
        return [
            RAW / "CIC-IDS2017/MachineLearningCSV.zip"
        ]

    return [
        RAW / "CTU-13/CTU13_Normal_Traffic.csv",
        RAW / "CTU-13/CTU13_Attack_Traffic.csv"
    ]


def file_sha256(path):
    h = hashlib.sha256()

    with path.open("rb") as f:
        for block in iter(lambda: f.read(1048576), b""):
            h.update(block)

    return h.hexdigest()


def get_features(kind):
    if kind == "cic":
        manifest = json.loads(
            (ROOT / "models/final/deployment_manifest.json")
            .read_text(encoding="utf-8")
        )
        return manifest["feature_order"]

    columns = pd.read_csv(
        source_paths(kind)[0], nrows=0
    ).columns.str.strip()

    return [
        c for c in columns
        if c not in ("Unnamed: 0", "Label")
    ]


def iter_chunks(kind, features):
    def clean(chunk, expected_label=None):
        chunk.columns = chunk.columns.str.strip()

        required = features + ["Label"]

        if not set(required).issubset(chunk.columns):
            raise ValueError("Missing required columns")

        if chunk.columns.duplicated().any():
            raise ValueError("Duplicate column names")

        x = chunk[features].apply(
            pd.to_numeric, errors="coerce"
        ).astype("float64")

        missing = int(x.isna().sum().sum())
        infinite = int(np.isinf(x.to_numpy()).sum())

        x = x.replace([np.inf, -np.inf], np.nan)
        x = x.mask(x == 0, 0.0)

        if kind == "cic":
            y = chunk["Label"].map(canonical_cic_label)

            if y.isna().any():
                raise ValueError("Missing CIC labels")

            y = y.astype(str)

        else:
            y = pd.to_numeric(
                chunk["Label"], errors="raise"
            )

            if not y.eq(expected_label).all():
                raise ValueError("Unexpected CTU label")

            y = y.astype(int).astype(str)

        x["target"] = y.to_numpy()

        return x, missing, infinite

    if kind == "cic":
        with ZipFile(source_paths(kind)[0]) as z:

            members = sorted([
                n for n in z.namelist()
                if n.lower().endswith(".csv")
            ])

            if len(members) != 8:
                raise ValueError("Expected eight CIC CSVs")

            for member in members:
                with z.open(member) as f:
                    for chunk in pd.read_csv(
                        f,
                        chunksize=CHUNK,
                        encoding="latin1",
                        low_memory=False
                    ):
                        yield clean(chunk)

    else:
        for path, label in zip(source_paths(kind), [0, 1]):
            for chunk in pd.read_csv(
                path,
                chunksize=CHUNK,
                low_memory=False
            ):
                yield clean(chunk, label)


def split_indices(meta, seed=SEED):
    rng = np.random.default_rng(seed)

    folds = {
        "train": [],
        "validation": [],
        "test": []
    }

    for label, group in meta.groupby("target"):
        ids = rng.permutation(
            group["row"].to_numpy()
        )

        n = len(ids)

        if n < 4:
            raise ValueError(
                f"Class {label}: only {n} unique rows"
            )

        n_test = max(1, round(n * 0.15))
        n_val = max(1, round(n * 0.15))

        folds["test"].extend(ids[:n_test])

        folds["validation"].extend(
            ids[n_test:n_test + n_val]
        )

        folds["train"].extend(
            ids[n_test + n_val:]
        )

    return {
        key: np.sort(np.asarray(value, dtype=np.int64))
        for key, value in folds.items()
    }


def prepare(kind):
    print(f"\n===== PREPARING {kind.upper()} =====")

    folder = OUT / kind
    folder.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    features = get_features(kind)

    metadata = []
    counts = Counter()
    missing = 0
    infinite = 0
    offset = 0

    # Pass 1: scan complete data and fingerprint features.
    for chunk, m, i in iter_chunks(kind, features):
        hashes = pd.util.hash_pandas_object(
            chunk[features],
            index=False
        ).to_numpy()

        metadata.append(pd.DataFrame({
            "row": np.arange(offset, offset + len(chunk)),
            "fingerprint": hashes,
            "target": chunk["target"].to_numpy()
        }))

        counts.update(chunk["target"])
        missing += m
        infinite += i
        offset += len(chunk)

        print("Scanned:", offset, flush=True)

    meta = pd.concat(metadata, ignore_index=True)
    del metadata

    group_labels = (
        meta.groupby("fingerprint")["target"]
        .nunique()
    )

    conflicts = group_labels[
        group_labels > 1
    ].index

    conflict_mask = meta["fingerprint"].isin(conflicts)

    conflict_rows = int(conflict_mask.sum())

    valid = meta.loc[~conflict_mask].copy()

    before_dedup = len(valid)

    valid = valid.drop_duplicates(
        subset=["fingerprint"],
        keep="first"
    )

    duplicates = before_dedup - len(valid)

    folds = split_indices(valid)

    # Map original rows to a unique split.
    assignment = np.full(offset, -1, dtype=np.int8)

    names = ["train", "validation", "test"]

    for number, name in enumerate(names):
        indices = folds[name]

        if np.any(assignment[indices] != -1):
            raise AssertionError("Split overlap")

        assignment[indices] = number

    selected = meta.loc[
        assignment[meta["row"].to_numpy()] >= 0
    ]

    if selected["fingerprint"].duplicated().any():
        raise AssertionError("Duplicate fingerprint leakage")

    # Pass 2: export selected rows to disk in chunks.
    writers = {}
    offset2 = 0

    try:
        for chunk, _, _ in iter_chunks(kind, features):

            codes = assignment[
                offset2:offset2 + len(chunk)
            ]

            for number, name in enumerate(names):
                part = chunk.loc[codes == number]

                if part.empty:
                    continue

                table = pa.Table.from_pandas(
                    part,
                    preserve_index=False
                )

                if name not in writers:
                    writers[name] = pq.ParquetWriter(
                        str(folder / f"{name}.parquet"),
                        table.schema,
                        compression="snappy"
                    )

                writers[name].write_table(table)

            offset2 += len(chunk)

    finally:
        for writer in writers.values():
            writer.close()

    if offset2 != offset:
        raise AssertionError("Dataset row count changed")

    for name in names:
        if name not in writers:
            raise AssertionError(f"Empty split: {name}")

    split_counts = {}

    for number, name in enumerate(names):
        part = meta.loc[
            assignment[meta["row"].to_numpy()] == number
        ]

        split_counts[name] = {
            str(k): int(v)
            for k, v in part["target"].value_counts().items()
        }

    np.savez_compressed(
        folder / "split_indices.npz",
        **folds
    )

    report = {
        "dataset": kind,
        "seed": SEED,
        "original_rows": offset,
        "features": features,
        "n_features": len(features),
        "original_class_counts": dict(counts),
        "missing_values": missing,
        "infinite_values": infinite,
        "conflicting_groups": int(len(conflicts)),
        "conflicting_rows_removed": conflict_rows,
        "duplicate_rows_removed": int(duplicates),
        "clean_rows": int(len(valid)),
        "split_sizes": {
            k: int(len(v)) for k, v in folds.items()
        },
        "split_class_counts": split_counts,
        "source_sha256": {
            p.name: file_sha256(p)
            for p in source_paths(kind)
        },
        "split_sha256": {
            k: hashlib.sha256(v.tobytes()).hexdigest()
            for k, v in folds.items()
        },
        "preprocessing_note": (
            "NaN retained for train-only median imputation."
        ),
        "evaluation_note": (
            "Internal stratified split after hash-based "
            "feature deduplication; not an independent "
            "temporal, host or cross-scenario holdout."
        )
    }

    text = json.dumps(
        report,
        indent=2,
        ensure_ascii=False
    )

    (REPORTS / f"{kind}_data_audit.json").write_text(
        text,
        encoding="utf-8"
    )

    (folder / "manifest.json").write_text(
        text,
        encoding="utf-8"
    )

    print(text)
    print(f"{kind.upper()} DATA PREPARATION PASSED")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        choices=["cic", "ctu", "both"],
        default="both"
    )

    args = parser.parse_args()

    datasets = (
        ["ctu", "cic"]
        if args.dataset == "both"
        else [args.dataset]
    )

    for kind in datasets:
        prepare(kind)


if __name__ == "__main__":
    main()
