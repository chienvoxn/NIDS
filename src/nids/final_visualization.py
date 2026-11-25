from pathlib import Path
import argparse
import json

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports/final/figures"

plt.style.use("ggplot")
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "axes.titlesize": 14
})


def save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
        facecolor="white"
    )
    plt.close(fig)
    print("Saved:", path)


def plot_data(kind):

    folder = OUT / kind
    title = "CIC-IDS2017" if kind == "cic" else "CTU-13"

    report = json.loads(
        (ROOT / "reports/final" /
         f"{kind}_data_audit.json")
        .read_text(encoding="utf-8")
    )

    # 1. Class distribution before and after cleaning
    before = pd.Series(report["original_class_counts"])

    after = pd.DataFrame(
        report["split_class_counts"]
    ).fillna(0).sum(axis=1)

    counts = pd.DataFrame({
        "Before cleaning": before,
        "After cleaning": after
    }).fillna(0).sort_values("Before cleaning")

    fig, ax = plt.subplots(figsize=(12, 7))

    counts.plot.barh(
        ax=ax,
        logx=True,
        color=["#64748b", "#0d9488"]
    )

    ax.set_title(f"{title}: Class Distribution")
    ax.set_xlabel("Number of samples (log scale)")
    ax.set_ylabel("Traffic class")

    save(fig, folder / "01_class_distribution.png")

    # 2. Data cleaning
    values = [
        report["clean_rows"],
        report["duplicate_rows_removed"],
        report["conflicting_rows_removed"]
    ]

    labels = [
        "Retained samples",
        "Duplicate rows removed",
        "Conflicting rows removed"
    ]

    fig, ax = plt.subplots(figsize=(10, 5))

    bars = ax.barh(
        labels, values,
        color=["#0d9488", "#d97706", "#dc2626"]
    )

    ax.bar_label(
        bars,
        labels=[f"{v:,}" for v in values],
        padding=5
    )

    ax.set_xlim(0, max(values) * 1.25)
    ax.set_title(f"{title}: Data Cleaning Results")
    ax.set_xlabel("Number of samples")

    save(fig, folder / "02_data_cleaning.png")

    # 3. Dataset split
    splits = report["split_sizes"]

    fig, ax = plt.subplots(figsize=(8, 7))

    ax.pie(
        list(splits.values()),
        labels=[
            f"{k.title()}\n{v:,}"
            for k, v in splits.items()
        ],
        autopct="%1.1f%%",
        startangle=90,
        colors=["#2563eb", "#d97706", "#0d9488"],
        wedgeprops={"edgecolor": "white"}
    )

    ax.set_title(f"{title}: Dataset Split")

    save(fig, folder / "03_dataset_split.png")

    # 4. Feature correlation on training sample only
    train = pd.read_parquet(
        ROOT / "data/processed/final" /
        kind / "train.parquet"
    )

    train = train.sample(
        n=min(30000, len(train)),
        random_state=42
    )

    features = report["features"]

    corr = train[features].corr()

    fig, ax = plt.subplots(figsize=(19, 17))

    im = ax.imshow(
        corr,
        cmap="coolwarm",
        vmin=-1,
        vmax=1
    )

    ax.set_xticks(range(len(features)))
    ax.set_yticks(range(len(features)))

    ax.set_xticklabels(
        features, rotation=90, fontsize=7
    )

    ax.set_yticklabels(features, fontsize=7)

    fig.colorbar(im, ax=ax, shrink=0.75)

    ax.set_title(
        f"{title}: Feature Correlation "
        "(Training Sample)"
    )

    save(fig, folder / "04_correlation_heatmap.png")

    # 5. Selected feature distributions
    selected = [
        c for c in features
        if c in [
            "Flow Duration",
            "Flow Bytes/s",
            "Flow Packets/s",
            "Flow Byts/s",
            "Flow Pkts/s",
            "Flow IAT Mean"
        ]
    ]

    normal = "BENIGN" if kind == "cic" else "0"

    for i, feature in enumerate(selected):

        fig, ax = plt.subplots(figsize=(9, 5))

        for label, mask in [
            ("Normal", train["target"].astype(str) == normal),
            ("Attack", train["target"].astype(str) != normal)
        ]:

            values = train.loc[mask, feature].dropna()
            values = values[np.isfinite(values)]
            values = np.sign(values) * np.log1p(np.abs(values))

            if len(values):
                ax.hist(
                    values,
                    bins=50,
                    alpha=0.55,
                    density=True,
                    label=label
                )

        ax.set_title(f"{title}: {feature}")
        ax.set_xlabel("Signed log-transformed feature")
        ax.set_ylabel("Density")
        ax.legend()

        save(
            fig,
            folder / f"05_feature_distribution_{i+1}.png"
        )


def plot_confusion(matrix, labels, title, path):

    matrix = np.asarray(matrix)

    denominator = matrix.sum(axis=1, keepdims=True)

    normalized = np.divide(
        matrix,
        denominator,
        out=np.zeros_like(matrix, dtype=float),
        where=denominator != 0
    )

    size = 13 if len(labels) > 2 else 6

    fig, ax = plt.subplots(figsize=(size, size))

    im = ax.imshow(
        normalized,
        cmap="Blues",
        vmin=0,
        vmax=1
    )

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))

    ax.set_xticklabels(
        labels, rotation=45, ha="right"
    )

    ax.set_yticklabels(labels)

    for i in range(len(labels)):
        for j in range(len(labels)):

            ax.text(
                j, i, f"{matrix[i,j]:,}",
                ha="center", va="center",
                fontsize=8,
                color=(
                    "white"
                    if normalized[i,j] > 0.5
                    else "black"
                )
            )

    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Actual label")
    ax.set_title(title)

    fig.colorbar(im, ax=ax, shrink=0.8)

    save(fig, path)


def plot_results(kind, profile):

    title = "CIC-IDS2017" if kind == "cic" else "CTU-13"

    folder = OUT / profile / kind

    source = ROOT / "outputs/final" / profile / kind

    metrics = pd.read_csv(
        source / "baseline_metrics.csv"
    )

    result = json.loads(
        (source / "baseline_metrics.json")
        .read_text(encoding="utf-8")
    )

    # 6. Baseline comparison
    columns = [
        "Accuracy", "Precision",
        "Recall", "F1-score"
    ]

    fig, ax = plt.subplots(figsize=(12, 6))

    (metrics.set_index("Model")[columns] * 100).plot.bar(
        ax=ax,
        color=["#2563eb", "#0d9488", "#d97706", "#7c3aed"]
    )

    ax.set_title(
        f"{title}: Baseline Comparison ({profile})"
    )

    ax.set_ylabel("Score (%)")
    ax.set_ylim(0, 108)
    ax.tick_params(axis="x", rotation=0)
    ax.legend(loc="lower right")

    save(fig, folder / "06_baseline_comparison.png")

    # 7. F1-score comparison
    fig, ax = plt.subplots(figsize=(9, 5))

    scores = metrics["F1-score"] * 100

    bars = ax.bar(
        metrics["Model"], scores,
        color="#2563eb"
    )

    ax.bar_label(bars, fmt="%.2f%%", padding=4)
    ax.set_ylim(0, 108)
    ax.set_ylabel("F1-score (%)")

    ax.set_title(
        f"{title}: F1-score Comparison ({profile})"
    )

    save(fig, folder / "07_f1_comparison.png")

    # 8. False positive rate
    fig, ax = plt.subplots(figsize=(9, 5))

    bars = ax.bar(
        metrics["Model"],
        metrics["FPR"] * 100,
        color="#d97706"
    )

    ax.bar_label(bars, fmt="%.3f%%", padding=4)
    ax.set_ylabel("False Positive Rate (%)")
    ax.set_ylim(
        0, max(0.1, metrics["FPR"].max() * 125)
    )

    ax.set_title(
        f"{title}: False Positive Rate ({profile})"
    )

    save(fig, folder / "08_false_positive_rate.png")

    # 9. Training time
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.bar(
        metrics["Model"],
        metrics["Training seconds"],
        color="#64748b"
    )

    ax.set_ylabel("Training time (seconds)")

    ax.set_title(
        f"{title}: Training Time ({profile})"
    )

    save(fig, folder / "09_training_time.png")

    # 10. Confusion matrix for every baseline
    for name, record in result["models"].items():

        labels = result["classes"]

        if kind == "ctu":
            labels = ["Normal", "Botnet"]

        plot_confusion(
            record["test"]["confusion_matrix"],
            labels,
            f"{title}: {name} ({profile})",
            folder / (
                "10_confusion_" +
                name.lower().replace(" ", "_") +
                ".png"
            )
        )

    # 11. Gaussian confusion matrix
    gaussian = result["gaussian"]["test"]

    plot_confusion(
        gaussian["confusion_matrix"],
        ["Normal", "Attack"],
        f"{title}: Gaussian ({profile})",
        folder / "11_gaussian_confusion.png"
    )

    # 12. Gaussian metrics
    names = [
        "Accuracy", "Precision",
        "Recall", "F1-score"
    ]

    values = [
        gaussian["accuracy"] * 100,
        gaussian["precision"] * 100,
        gaussian["recall"] * 100,
        gaussian["f1_score"] * 100
    ]

    fig, ax = plt.subplots(figsize=(9, 5))

    bars = ax.bar(names, values, color="#0d9488")

    ax.bar_label(bars, fmt="%.2f%%", padding=4)
    ax.set_ylim(0, 108)
    ax.set_ylabel("Score (%)")

    ax.set_title(
        f"{title}: Gaussian Detection ({profile})"
    )

    save(fig, folder / "12_gaussian_metrics.png")


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--profile",
        choices=["smoke", "full"],
        default="smoke"
    )

    args = parser.parse_args()

    for kind in ["cic", "ctu"]:
        plot_data(kind)
        plot_results(kind, args.profile)

    print("\nALL VISUALIZATIONS COMPLETED")
    print("Output:", OUT)


if __name__ == "__main__":
    main()
