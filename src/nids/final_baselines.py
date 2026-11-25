from pathlib import Path
from time import perf_counter
import argparse
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from nids.final_metrics import (
    evaluate,
    binary_detection_metrics
)


ROOT = Path(__file__).resolve().parents[2]
SEED = 42


def stratified_cap(df, cap):

    if len(df) <= cap:
        return df

    parts = []

    for _, group in df.groupby("target"):

        n = min(
            len(group),
            max(1, round(cap * len(group) / len(df)))
        )

        parts.append(
            group.sample(n=n, random_state=SEED)
        )

    return pd.concat(parts, ignore_index=True)


def load_data(kind, smoke, include_test=True):

    folder = ROOT / "data/processed/final" / kind

    manifest = json.loads(
        (folder / "manifest.json").read_text(
            encoding="utf-8"
        )
    )

    features = manifest["features"]

    frames = [
        pd.read_parquet(folder / f"{name}.parquet")
        for name in (["train", "validation", "test"] if include_test else ["train", "validation"])
    ]

    if smoke:
        frames = [
            stratified_cap(df, cap)
            for df, cap in zip(
                frames, [15000, 5000, 5000]
            )
        ]

    imputer = SimpleImputer(
        strategy="median",
        keep_empty_features=True
    )

    scaler = StandardScaler()

    encoder = LabelEncoder().fit(
        frames[0]["target"].astype(str)
    )

    xtr = imputer.fit_transform(
        frames[0][features]
    )

    xtr = scaler.fit_transform(xtr)

    xs = [xtr.astype(np.float32)]

    for df in frames[1:]:

        x = imputer.transform(df[features])
        x = scaler.transform(x)

        xs.append(x.astype(np.float32))

    ys = [
        encoder.transform(df["target"].astype(str))
        for df in frames
    ]

    for x in xs:
        if not np.isfinite(x).all():
            raise ValueError(
                "Nonfinite values after preprocessing"
            )

    prep = {
        "features": features,
        "imputer": imputer,
        "scaler": scaler,
        "label_encoder": encoder
    }

    return xs, ys, prep, manifest


def build_models(kind, smoke):

    trees = 30 if smoke else 150
    boosting = 30 if smoke else 200

    return {
        "Random Forest": RandomForestClassifier(
            n_estimators=trees,
            max_depth=20,
            min_samples_leaf=2,
            class_weight=(
                "balanced_subsample"
                if kind == "cic" else None
            ),
            n_jobs=-1,
            random_state=SEED
        ),

        "XGBoost": XGBClassifier(
            n_estimators=boosting,
            max_depth=6,
            learning_rate=0.07,
            tree_method="hist",
            subsample=0.85,
            colsample_bytree=0.85,
            n_jobs=4,
            random_state=SEED
        ),

        "LightGBM": LGBMClassifier(
            n_estimators=boosting,
            num_leaves=31,
            learning_rate=0.07,
            class_weight=(
                "balanced"
                if kind == "cic" else None
            ),
            verbosity=-1,
            n_jobs=4,
            random_state=SEED
        ),

        "KNN": KNeighborsClassifier(
            n_neighbors=5,
            weights="distance",
            algorithm="brute",
            n_jobs=-1
        )
    }


def predict_batches(model, x, batch_size=2048):

    return np.concatenate([
        model.predict(x[start:start + batch_size])
        for start in range(0, len(x), batch_size)
    ])


def run_gaussian(xs, ys, normal_id):

    xtr, xva, xte = xs
    ytr, yva, yte = ys

    normal_train = xtr[ytr == normal_id]
    normal_val = xva[yva == normal_id]

    if len(normal_train) < 3 or len(normal_val) < 3:
        raise ValueError("Insufficient normal samples")

    center = normal_train.mean(axis=0)

    variance = np.maximum(
        normal_train.var(axis=0), 1e-6
    )

    def anomaly_score(x):

        return np.sum(
            (x - center) ** 2 / variance,
            axis=1
        )

    threshold = float(
        np.quantile(
            anomaly_score(normal_val), 0.99
        )
    )

    validation_true = (yva != normal_id).astype(int)

    validation_pred = (
        anomaly_score(xva) > threshold
    ).astype(int)

    test_true = (yte != normal_id).astype(int)

    test_pred = (
        anomaly_score(xte) > threshold
    ).astype(int)

    result = {
        "threshold": threshold,
        "validation": binary_detection_metrics(
            validation_true, validation_pred
        ),
        "test": binary_detection_metrics(
            test_true, test_pred
        )
    }

    artifact = {
        "center": center,
        "variance": variance,
        "threshold": threshold,
        "normal_id": normal_id
    }

    return result, artifact


def run(kind, smoke):

    profile = "smoke" if smoke else "full"

    print(f"\n=== {kind.upper()} / {profile} ===")

    out = ROOT / "outputs/final" / profile / kind
    out.mkdir(parents=True, exist_ok=True)

    xs, ys, prep, manifest = load_data(kind, smoke)

    xtr, xva, xte = xs
    ytr, yva, yte = ys

    classes = prep["label_encoder"].classes_.tolist()

    labels = list(range(len(classes)))

    normal_label = "BENIGN" if kind == "cic" else "0"
    normal_id = classes.index(normal_label)

    print("Train:", len(xtr))
    print("Validation:", len(xva))
    print("Test:", len(xte))
    print("Classes:", classes)

    joblib.dump(
        prep, out / "preprocessing.joblib"
    )

    result = {
        "dataset": kind,
        "profile": profile,
        "classes": classes,
        "features": prep["features"],
        "split_sha256": manifest["split_sha256"],
        "split_sizes": {
            "train": len(xtr),
            "validation": len(xva),
            "test": len(xte)
        },
        "models": {}
    }

    rows = []

    for name, model in build_models(kind, smoke).items():

        print("Training:", name, flush=True)

        train_x = xtr
        train_y = ytr

        # Bound KNN runtime on large CIC training data.
        # Prototype reduction affects only its train set.
        if name == "KNN" and len(xtr) > 20000:

            rng = np.random.default_rng(SEED)
            indices = []

            for label in np.unique(ytr):

                candidates = np.flatnonzero(
                    ytr == label
                )

                n = min(
                    len(candidates),
                    max(
                        1,
                        round(
                            20000 * len(candidates)
                            / len(ytr)
                        )
                    )
                )

                indices.extend(
                    rng.choice(
                        candidates,
                        size=n,
                        replace=False
                    )
                )

            train_x = xtr[indices]
            train_y = ytr[indices]

        started = perf_counter()

        if name == "XGBoost" and kind == "cic":

            weights = compute_sample_weight(
                "balanced", train_y
            )

            model.fit(
                train_x,
                train_y,
                sample_weight=weights
            )

        else:
            model.fit(train_x, train_y)

        train_seconds = perf_counter() - started

        val_pred = predict_batches(model, xva)

        started = perf_counter()

        test_pred = predict_batches(model, xte)

        inference_seconds = perf_counter() - started

        val_metrics = evaluate(
            yva, val_pred, kind, labels
        )

        test_metrics = evaluate(
            yte, test_pred, kind, labels
        )

        binary_metrics = binary_detection_metrics(
            (yte != normal_id).astype(int),
            (test_pred != normal_id).astype(int)
        )

        result["models"][name] = {
            "training_rows": int(len(train_x)),
            "validation": val_metrics,
            "test": test_metrics,
            "training_seconds": train_seconds,
            "inference_seconds": inference_seconds,
            "inference_ms_per_flow": (
                1000 * inference_seconds / len(xte)
            ),
            "false_positive_rate": (
                binary_metrics["false_positive_rate"]
            )
        }

        rows.append({
            "Model": name,
            "Accuracy": test_metrics["accuracy"],
            "Precision": test_metrics["precision"],
            "Recall": test_metrics["recall"],
            "F1-score": test_metrics["f1_score"],
            "FPR": binary_metrics["false_positive_rate"],
            "Validation F1": val_metrics["f1_score"],
            "Training rows": len(train_x),
            "Training seconds": train_seconds
        })

        filename = name.lower().replace(" ", "_")

        joblib.dump(
            model, out / f"{filename}.joblib"
        )

        pd.DataFrame(rows).to_csv(
            out / "baseline_metrics.csv", index=False
        )

        print(
            name,
            "F1-score:",
            round(test_metrics["f1_score"], 6),
            flush=True
        )

    gaussian, artifact = run_gaussian(
        xs, ys, normal_id
    )

    result["gaussian"] = gaussian

    joblib.dump(
        artifact, out / "gaussian.joblib"
    )

    (out / "baseline_metrics.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8"
    )

    print("\nBASELINE RESULTS")

    print(
        pd.DataFrame(rows).round(6)
        .to_string(index=False)
    )

    print("\nGAUSSIAN RESULTS")
    print(json.dumps(gaussian["test"], indent=2))

    print("\nSaved:", out)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        choices=["cic", "ctu", "both"],
        default="both"
    )

    parser.add_argument(
        "--smoke",
        action="store_true"
    )

    args = parser.parse_args()

    datasets = (
        ["ctu", "cic"]
        if args.dataset == "both"
        else [args.dataset]
    )

    for kind in datasets:
        run(kind, args.smoke)


if __name__ == "__main__":
    main()
