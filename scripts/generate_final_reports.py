from pathlib import Path
import json
import pandas as pd

root = Path("reports/final")
root.mkdir(parents=True, exist_ok=True)

def save(name, content):
    path = root / name
    path.write_text(content, encoding="utf-8")
    print("Created:", path)

def table(headers, rows):
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |"
    ]
    for row in rows:
        lines.append(
            "| " + " | ".join(map(str, row)) + " |"
        )
    return "\n".join(lines)

def image(path, caption):
    return f"\n![{caption}]({path})\n\n*{caption}*\n"

audits = {}
results = {}

for kind in ["cic", "ctu"]:
    audits[kind] = json.loads(
        (root / f"{kind}_data_audit.json")
        .read_text(encoding="utf-8")
    )

    results[kind] = json.loads(
        Path(
            f"outputs/final/smoke/{kind}/baseline_metrics.json"
        ).read_text(encoding="utf-8")
    )

# 01 - Environment

save("01_ENVIRONMENT_SETUP.md", """
# 1. Thiet lap moi truong phat trien

- Windows, Visual Studio Code.
- Python 3.11 va Virtual Environment.
- PyTorch, Scikit-learn, XGBoost, LightGBM.
- Pandas, NumPy, Matplotlib, NetworkX.
- Git va GitHub.

## Ket qua kiem thu

58 passed, 1 warning in 5.20s.

GPU Kaggle se duoc su dung cho huan luyen CNN.
""")

# 02 - EDA

parts = [
    "# 2. Dataset and Exploratory Data Analysis",
    "## 2.1. Dataset overview"
]

rows = []

for kind, a in audits.items():
    rows.append([
        kind.upper(),
        a["original_rows"],
        a["n_features"],
        len(a["original_class_counts"])
    ])

parts.append(table(
    ["Dataset", "Samples", "Features", "Classes"],
    rows
))

for kind, a in audits.items():

    parts.append(f"## {kind.upper()} - Class distribution")

    parts.append(table(
        ["Class", "Samples"],
        sorted(
            a["original_class_counts"].items(),
            key=lambda x: x[1],
            reverse=True
        )
    ))

    folder = root / "figures" / kind

    for path in sorted(folder.glob("*.png")):

        if (
            path.name.startswith("01_")
            or path.name.startswith("04_")
            or path.name.startswith("05_")
        ):
            parts.append(image(
                path.relative_to(root).as_posix(),
                path.stem
            ))

save("02_DATASET_EDA.md", "\n\n".join(parts))

# 03 - Preprocessing

parts = [
    "# 3. Data Preprocessing",
    """
## Methodology

1. Validate dataset schema.
2. Normalize class labels.
3. Remove conflicting label groups.
4. Remove duplicate feature rows.
5. Replace infinite values with NaN.
6. Stratified train/validation/test split.
7. Fit imputer and scaler on training data only.
8. Apply class weighting where appropriate.

Random seed: 42.
Split ratio: approximately 70/15/15.
"""
]

for kind, a in audits.items():

    parts.append(f"## {kind.upper()}")

    rows = [
        ["Original rows", a["original_rows"]],
        ["Missing values", a["missing_values"]],
        ["Infinite values", a["infinite_values"]],
        ["Conflicting groups", a["conflicting_groups"]],
        ["Conflicting rows removed",
         a["conflicting_rows_removed"]],
        ["Duplicate rows removed",
         a["duplicate_rows_removed"]],
        ["Clean rows", a["clean_rows"]]
    ]

    rows.extend(a["split_sizes"].items())

    parts.append(table(["Metric", "Value"], rows))

    parts.append(image(
        f"figures/{kind}/02_data_cleaning.png",
        "Data cleaning results"
    ))

    parts.append(image(
        f"figures/{kind}/03_dataset_split.png",
        "Train validation test split"
    ))

save("03_DATA_PREPROCESSING.md", "\n\n".join(parts))

# 04 - Gaussian

parts = [
    "# 4. Gaussian Anomaly Detection",
    """
The diagonal Gaussian model estimates the mean
and variance of normal training traffic.

The anomaly threshold is the 99th percentile of
normal validation anomaly scores.

Results below are smoke experiments, not full
dataset experiments.
"""
]

for kind, result in results.items():

    g = result["gaussian"]["test"]

    parts.append(f"## {kind.upper()}")

    parts.append(table(
        ["Metric", "Value (%)"],
        [
            ["Accuracy", f'{g["accuracy"]*100:.4f}'],
            ["Precision", f'{g["precision"]*100:.4f}'],
            ["Recall", f'{g["recall"]*100:.4f}'],
            ["F1-score", f'{g["f1_score"]*100:.4f}'],
            ["FPR", f'{g["false_positive_rate"]*100:.4f}']
        ]
    ))

    for name in [
        "11_gaussian_confusion",
        "12_gaussian_metrics"
    ]:
        parts.append(image(
            f"figures/smoke/{kind}/{name}.png",
            name
        ))

parts.append("""
## Limitations

The detection recall is low despite a low FPR.
Threshold optimization must use validation data
only. Gaussian should not be the sole filtering
gate before attack classification.
""")

save("04_GAUSSIAN_ANOMALY.md", "\n\n".join(parts))

# 05 - Baselines

parts = [
    "# 5. Machine Learning Baselines",
    """
## Models

- Random Forest
- XGBoost
- LightGBM
- K-Nearest Neighbors

Decision Tree is excluded as a standalone baseline.

CIC-IDS2017 uses weighted Precision, Recall and F1.
CTU-13 uses binary metrics with positive class 1.

All results below are smoke experiments.
"""
]

for kind, result in results.items():

    parts.append(f"## {kind.upper()}")

    rows = []

    for name, record in result["models"].items():

        m = record["test"]

        rows.append([
            name,
            f'{m["accuracy"]*100:.4f}',
            f'{m["precision"]*100:.4f}',
            f'{m["recall"]*100:.4f}',
            f'{m["f1_score"]*100:.4f}'
        ])

    parts.append(table(
        [
            "Model", "Accuracy (%)",
            "Precision (%)", "Recall (%)",
            "F1-score (%)"
        ],
        rows
    ))

    folder = root / "figures" / "smoke" / kind

    for path in sorted(folder.glob("*.png")):

        if path.name.startswith((
            "06_", "07_", "08_", "09_", "10_"
        )):
            parts.append(image(
                path.relative_to(root).as_posix(),
                path.stem
            ))

save("05_ML_BASELINES.md", "\n\n".join(parts))

print("\nREPORT GENERATION COMPLETED")
