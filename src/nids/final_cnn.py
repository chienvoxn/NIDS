from pathlib import Path
from time import perf_counter
import argparse
import copy
import json
import random

import joblib
import numpy as np
import pandas as pd
import torch

from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.utils.class_weight import compute_class_weight

from nids.final_baselines import load_data
from nids.final_metrics import evaluate
from nids.final_metrics import binary_detection_metrics
from nids.sprint3_residual import MultiScaleResidualCNN


ROOT = Path(__file__).resolve().parents[2]

VARIANTS = {
    "cnn": {
        "se": False,
        "mhsa": False
    },
    "cnn_se": {
        "se": True,
        "mhsa": False
    },
    "cnn_mhsa": {
        "se": False,
        "mhsa": True
    },
    "cnn_se_mhsa": {
        "se": True,
        "mhsa": True
    }
}


def seed_everything(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loader(x, y, batch_size, shuffle, seed):

    dataset = TensorDataset(
        torch.from_numpy(x),
        torch.from_numpy(y.astype(np.int64))
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=torch.Generator().manual_seed(seed),
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )


def predict(model, loader, criterion, device):

    model.eval()

    predictions = []
    targets = []

    total_loss = 0.0
    total_rows = 0

    with torch.inference_mode():

        for x, y in loader:

            x = x.to(device)
            y = y.to(device)

            logits = model(x)

            loss = criterion(logits, y)

            total_loss += loss.item() * len(y)
            total_rows += len(y)

            predictions.extend(
                logits.argmax(dim=1).cpu().numpy()
            )

            targets.extend(y.cpu().numpy())

    return (
        np.asarray(targets),
        np.asarray(predictions),
        total_loss / total_rows
    )


def train_model(
    kind, variant, xs, ys, classes,
    args, device, out
):

    seed_everything(args.seed)

    config = VARIANTS[variant]

    model = MultiScaleResidualCNN(
        n_features=xs[0].shape[1],
        n_classes=len(classes),
        width=args.width,
        heads=4,
        blocks=2,
        dropout=args.dropout,
        residual=True,
        **config
    ).to(device)

    train_loader = make_loader(
        xs[0], ys[0],
        args.batch_size, True, args.seed
    )

    val_loader = make_loader(
        xs[1], ys[1],
        args.batch_size, False, args.seed
    )

    test_loader = make_loader(
        xs[2], ys[2],
        args.batch_size, False, args.seed
    )

    labels = list(range(len(classes)))

    weights = None

    if kind == "cic":

        weights = compute_class_weight(
            class_weight="balanced",
            classes=np.asarray(labels),
            y=ys[0]
        )

        weights = torch.tensor(
            weights,
            dtype=torch.float32,
            device=device
        )

    criterion = nn.CrossEntropyLoss(
        weight=weights
    )

    eval_criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2
    )

    best_f1 = -1.0
    best_epoch = 0
    best_state = None

    history = []
    stale = 0

    started = perf_counter()

    for epoch in range(1, args.epochs + 1):

        model.train()

        total_loss = 0.0
        total_rows = 0

        correct = 0

        for x, y in train_loader:

            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad(set_to_none=True)

            logits = model(x)
            loss = criterion(logits, y)

            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss")

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 5.0
            )

            optimizer.step()

            total_loss += loss.item() * len(y)
            total_rows += len(y)

            correct += (
                logits.argmax(dim=1) == y
            ).sum().item()

        yv, pv, val_loss = predict(
            model, val_loader,
            eval_criterion, device
        )

        metrics = evaluate(
            yv, pv, kind, labels
        )

        val_f1 = metrics["f1_score"]

        history.append({
            "epoch": epoch,
            "train_loss": total_loss / total_rows,
            "train_accuracy": correct / total_rows,
            "validation_loss": val_loss,
            "validation_accuracy": metrics["accuracy"],
            "validation_f1": val_f1,
            "learning_rate": optimizer.param_groups[0]["lr"]
        })

        print(
            f"{kind}/{variant} "
            f"Epoch {epoch}/{args.epochs} "
            f"Loss={total_loss/total_rows:.5f} "
            f"ValF1={val_f1:.6f}",
            flush=True
        )

        scheduler.step(val_f1)

        if val_f1 > best_f1:

            best_f1 = val_f1
            best_epoch = epoch

            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }

            stale = 0

        else:

            stale += 1

            if stale >= args.patience:

                print("Early stopping", flush=True)
                break

    training_seconds = perf_counter() - started

    model.load_state_dict(best_state)

    # Test is evaluated only after checkpoint selection.
    yv, pv, _ = predict(
        model, val_loader,
        eval_criterion, device
    )

    validation_metrics = evaluate(
        yv, pv, kind, labels
    )

    if device.type == "cuda":
        torch.cuda.synchronize()

    started = perf_counter()

    yt, pt, _ = predict(
        model, test_loader,
        eval_criterion, device
    )

    if device.type == "cuda":
        torch.cuda.synchronize()

    inference_seconds = perf_counter() - started

    test_metrics = evaluate(
        yt, pt, kind, labels
    )

    normal_id = classes.index(
        "BENIGN" if kind == "cic" else "0"
    )

    detection = binary_detection_metrics(
        (yt != normal_id).astype(int),
        (pt != normal_id).astype(int)
    )

    checkpoint = {
        "state_dict": best_state,
        "dataset": kind,
        "variant": variant,
        "n_features": xs[0].shape[1],
        "n_classes": len(classes),
        "classes": classes,
        "width": args.width,
        "heads": 4,
        "blocks": 2,
        "dropout": args.dropout,
        "residual": True,
        "se": config["se"],
        "mhsa": config["mhsa"],
        "seed": args.seed,
        "best_epoch": best_epoch
    }

    torch.save(
        checkpoint,
        out / f"{variant}.pt"
    )

    pd.DataFrame(history).to_csv(
        out / f"{variant}_history.csv",
        index=False
    )

    return {
        "variant": variant,
        "configuration": config,
        "parameters": sum(
            p.numel() for p in model.parameters()
        ),
        "best_epoch": best_epoch,
        "validation": validation_metrics,
        "test": test_metrics,
        "training_seconds": training_seconds,
        "inference_seconds": inference_seconds,
        "inference_ms_per_flow":
            1000 * inference_seconds / len(yt),
        "false_positive_rate":
            detection["false_positive_rate"],
        "detection_rate":
            detection["detection_rate"]
    }


def run(kind, args, device):

    profile = "smoke" if args.smoke else "full"

    out = ROOT / "outputs/final" / profile / kind
    out.mkdir(parents=True, exist_ok=True)

    xs, ys, prep, manifest = load_data(
        kind, args.smoke
    )

    classes = prep["label_encoder"].classes_.tolist()

    joblib.dump(
        prep,
        out / "cnn_preprocessing.joblib"
    )

    report = {
        "dataset": kind,
        "profile": profile,
        "device": str(device),
        "classes": classes,
        "features": prep["features"],
        "split_sha256": manifest["split_sha256"],
        "arguments": vars(args),
        "split_sizes": {
            name: len(y)
            for name, y in zip(
                ["train", "validation", "test"], ys
            )
        },
        "models": {}
    }

    rows = []

    for variant in args.variants:

        result = train_model(
            kind, variant, xs, ys,
            classes, args, device, out
        )

        report["models"][variant] = result

        m = result["test"]

        rows.append({
            "Model": variant,
            "Accuracy": m["accuracy"],
            "Precision": m["precision"],
            "Recall": m["recall"],
            "F1-score": m["f1_score"],
            "Validation F1":
                result["validation"]["f1_score"],
            "FPR": result["false_positive_rate"],
            "Parameters": result["parameters"],
            "Best epoch": result["best_epoch"],
            "Training seconds": result["training_seconds"],
            "Inference ms/flow":
                result["inference_ms_per_flow"]
        })

        pd.DataFrame(rows).to_csv(
            out / "cnn_metrics.csv",
            index=False
        )

        (out / "cnn_metrics.json").write_text(
            json.dumps(report, indent=2),
            encoding="utf-8"
        )

    print(pd.DataFrame(rows).to_string(index=False))
    print("Saved:", out)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        choices=["cic", "ctu", "both"],
        default="both"
    )

    parser.add_argument(
        "--variants",
        nargs="+",
        choices=list(VARIANTS),
        default=list(VARIANTS)
    )

    parser.add_argument(
        "--smoke", action="store_true"
    )

    parser.add_argument(
        "--epochs", type=int, default=20
    )

    parser.add_argument(
        "--batch-size", type=int, default=256
    )

    parser.add_argument(
        "--width", type=int, default=32
    )

    parser.add_argument(
        "--lr", type=float, default=0.001
    )

    parser.add_argument(
        "--dropout", type=float, default=0.1
    )

    parser.add_argument(
        "--patience", type=int, default=5
    )

    parser.add_argument(
        "--seed", type=int, default=42
    )

    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto"
    )

    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    else:
        device = torch.device(args.device)

    print("Device:", device)

    datasets = (
        ["ctu", "cic"]
        if args.dataset == "both"
        else [args.dataset]
    )

    for kind in datasets:
        run(kind, args, device)


if __name__ == "__main__":
    main()
