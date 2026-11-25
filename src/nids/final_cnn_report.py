from pathlib import Path
import argparse
import json

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from nids.final_visualization import plot_confusion


ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports/final"

DISPLAY = {
    "cnn": "CNN",
    "cnn_se": "CNN + SE",
    "cnn_mhsa": "CNN + MHSA",
    "cnn_se_mhsa": "CNN + SE + MHSA"
}

plt.style.use("ggplot")


def save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print("Saved:", path)


def generate(kind, profile):

    title = "CIC-IDS2017" if kind == "cic" else "CTU-13"

    source = ROOT / "outputs/final" / profile / kind
    folder = REPORT / "figures" / profile / kind / "cnn"

    folder.mkdir(parents=True, exist_ok=True)

    result = json.loads(
        (source / "cnn_metrics.json")
        .read_text(encoding="utf-8")
    )

    baseline_result = json.loads(
        (source / "baseline_metrics.json")
        .read_text(encoding="utf-8")
    )

    for key in ["split_sha256", "split_sizes", "classes", "features"]:
        if result[key] != baseline_result[key]:
            raise ValueError(f"Incompatible experiments: {key}")

    metrics = pd.read_csv(source / "cnn_metrics.csv")
    baseline = pd.read_csv(source / "baseline_metrics.csv")

    metrics["Model"] = metrics["Model"].map(DISPLAY)

    # Learning curves

    curves = [
        ("train_loss", "Training Loss", "01_training_loss"),
        ("validation_loss", "Validation Loss", "02_validation_loss"),
        ("validation_accuracy", "Validation Accuracy", "03_validation_accuracy"),
        ("validation_f1", "Validation F1-score", "04_validation_f1")
    ]

    for column, label, filename in curves:

        fig, ax = plt.subplots(figsize=(10, 6))

        for variant in result["models"]:

            history = pd.read_csv(
                source / f"{variant}_history.csv"
            )

            ax.plot(
                history["epoch"],
                history[column],
                marker="o",
                linewidth=2,
                label=DISPLAY[variant]
            )

        ax.set_title(f"{title}: {label} ({profile})")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(label)
        ax.legend()
        ax.grid(alpha=0.25)

        save(fig, folder / f"{filename}.png")

    # Ablation F1-score

    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.bar(
        metrics["Model"],
        metrics["F1-score"] * 100
    )

    ax.bar_label(bars, fmt="%.2f%%", padding=4)
    ax.set_ylim(0, 108)
    ax.set_ylabel("F1-score (%)")
    ax.set_title(f"{title}: Ablation Study ({profile})")

    save(fig, folder / "05_ablation_f1.png")

    # Classification metrics

    fig, ax = plt.subplots(figsize=(12, 6))

    columns = [
        "Accuracy", "Precision", "Recall", "F1-score"
    ]

    (metrics.set_index("Model")[columns] * 100).plot.bar(
        ax=ax
    )

    ax.set_ylim(0, 108)
    ax.set_ylabel("Score (%)")
    ax.tick_params(axis="x", rotation=0)
    ax.set_title(f"{title}: CNN Metrics ({profile})")

    save(fig, folder / "06_cnn_metrics.png")

    # Confusion matrices

    for variant, record in result["models"].items():

        labels = result["classes"]

        if kind == "ctu":
            labels = ["Normal", "Botnet"]

        plot_confusion(
            record["test"]["confusion_matrix"],
            labels,
            f"{title}: {DISPLAY[variant]} ({profile})",
            folder / f"07_confusion_{variant}.png"
        )

    # Combine baselines and CNN

    combined = pd.concat([
        baseline,
        metrics
    ], ignore_index=True)

    combined.to_csv(
        source / "all_models_metrics.csv",
        index=False
    )

    # Baseline vs CNN

    fig, ax = plt.subplots(figsize=(11, 7))

    bars = ax.barh(
        combined["Model"],
        combined["F1-score"] * 100
    )

    ax.bar_label(bars, fmt="%.2f%%", padding=4)
    ax.set_xlim(0, 110)
    ax.set_xlabel("F1-score (%)")
    ax.set_title(f"{title}: Baseline vs CNN ({profile})")

    save(fig, folder / "08_baseline_vs_cnn.png")

    # False Positive Rate

    fig, ax = plt.subplots(figsize=(11, 7))

    bars = ax.barh(
        combined["Model"],
        combined["FPR"] * 100
    )

    ax.bar_label(bars, fmt="%.3f%%", padding=4)
    ax.set_xlim(0, max(1, combined["FPR"].max() * 120))
    ax.set_xlabel("False Positive Rate (%)")
    ax.set_title(f"{title}: False Positive Rate ({profile})")

    save(fig, folder / "09_false_positive_rate.png")

    # Training time

    fig, ax = plt.subplots(figsize=(11, 7))

    bars = ax.barh(
        metrics["Model"],
        metrics["Training seconds"]
    )

    ax.bar_label(bars, fmt="%.3f s", padding=4)
    ax.set_xlim(0, metrics["Training seconds"].max() * 1.25)
    ax.set_xlabel("Training time (seconds)")
    ax.set_title(f"{title}: CNN Training Time ({profile})")

    save(fig, folder / "10_training_time.png")

    # Inference time

    fig, ax = plt.subplots(figsize=(11, 7))

    bars = ax.barh(
        metrics["Model"],
        metrics["Inference ms/flow"]
    )

    ax.bar_label(bars, fmt="%.4f ms", padding=4)
    ax.set_xlim(0, metrics["Inference ms/flow"].max() * 1.25)
    ax.set_xlabel("Mean evaluation time per flow (ms)")
    ax.set_title(f"{title}: CNN Evaluation Time ({profile})")

    save(fig, folder / "11_inference_time.png")

    # Model complexity

    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.bar(
        metrics["Model"],
        metrics["Parameters"]
    )

    ax.bar_label(bars, fmt="%.0f", padding=4)
    ax.set_ylim(0, metrics["Parameters"].max() * 1.2)
    ax.set_ylabel("Trainable parameters")
    ax.set_title(f"{title}: Model Complexity")

    save(fig, folder / "12_model_complexity.png")

    return metrics, baseline, combined, result, folder


def metric_table(frame):

    lines = [
        "| Mô hình | Accuracy | Precision | Recall | F1-score |",
        "|---|---:|---:|---:|---:|"
    ]

    for _, row in frame.iterrows():

        lines.append(
            f'| {row["Model"]} | '
            f'{row["Accuracy"]*100:.4f}% | '
            f'{row["Precision"]*100:.4f}% | '
            f'{row["Recall"]*100:.4f}% | '
            f'{row["F1-score"]*100:.4f}% |'
        )

    return lines


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--profile",
        choices=["smoke", "full"],
        default="smoke"
    )

    args = parser.parse_args()

    architecture = [
        "# 6. Mô hình CNN mở rộng",
        "",
        "## 6.1. Mục tiêu nghiên cứu",
        "",
        "Xây dựng mô hình phát hiện tấn công mạng dựa trên CNN,",
        "kết hợp Squeeze-and-Excitation và Multi-Head Self-Attention.",
        "",
        "## 6.2. Kiến trúc mô hình",
        "",
        "```mermaid",
        "flowchart TD",
        "A[Input Features] --> B[Feature Tokenizer]",
        "B --> C[Multi-scale Residual CNN]",
        "C --> D[Squeeze-and-Excitation]",
        "D --> E[Multi-Head Self-Attention]",
        "E --> F[Feature Aggregation]",
        "F --> G[Fully Connected Classifier]",
        "```",
        "",
        "CNN sử dụng các kernel kích thước 1, 3 và 5.",
        "SE tái hiệu chỉnh các kênh đặc trưng.",
        "MHSA học quan hệ giữa các feature token.",
        "Attention không được diễn giải là attention theo thời gian",
        "vì dữ liệu CSV không có chuỗi timestamp.",
        "",
        "## 6.3. Cấu hình Ablation",
        "",
        "| Mô hình | Residual CNN | SE | MHSA |",
        "|---|---|---|---|",
        "| CNN | Có | Không | Không |",
        "| CNN + SE | Có | Có | Không |",
        "| CNN + MHSA | Có | Không | Có |",
        "| CNN + SE + MHSA | Có | Có | Có |",
        "",
        "Các mô hình giữ nguyên backbone để đánh giá",
        "đóng góp riêng của SE và MHSA.",
        "",
        f"**Chế độ thực nghiệm: {args.profile}.**",
        ""
    ]

    ablation = [
        "# 7. Thực nghiệm Ablation Study",
        "",
        "Checkpoint được lựa chọn bằng F1-score trên validation.",
        "CIC sử dụng Weighted F1; CTU sử dụng Binary F1.",
        ""
    ]

    comparison = [
        "# 8. So sánh Baseline và mô hình đề xuất",
        "",
        "Các mô hình được đối chiếu theo cùng profile,",
        "schema đặc trưng và thông tin chia tập dữ liệu.",
        ""
    ]

    if args.profile == "smoke":

        warning = (
            "> Đây là kết quả Smoke Test, không phải "
            "kết quả huấn luyện chính thức."
        )

        architecture.append(warning)
        ablation.append(warning)
        comparison.append(warning)

    for kind in ["cic", "ctu"]:

        metrics, baseline, combined, result, folder = generate(
            kind, args.profile
        )

        title = "CIC-IDS2017" if kind == "cic" else "CTU-13"

        proposed = metrics.loc[
            metrics["Model"] == "CNN + SE + MHSA"
        ].iloc[0]

        best_baseline = baseline.loc[
            baseline["F1-score"].idxmax()
        ]

        best_validation = metrics.loc[
            metrics["Validation F1"].idxmax()
        ]

        gap = (
            proposed["F1-score"] -
            best_baseline["F1-score"]
        ) * 100

        architecture.extend([
            "",
            f"## {title}: Cấu hình huấn luyện",
            "",
            f'- Thiết bị: {result["device"]}',
            f'- Số đặc trưng: {len(result["features"])}',
            f'- Số lớp: {len(result["classes"])}',
            f'- Số epoch tối đa: {result["arguments"]["epochs"]}',
            f'- Batch size: {result["arguments"]["batch_size"]}',
            f'- Width: {result["arguments"]["width"]}',
            f'- Learning rate: {result["arguments"]["lr"]}',
            f'- Random seed: {result["arguments"]["seed"]}'
        ])

        ablation.extend([
            "",
            f"## {title}",
            "",
            *metric_table(metrics),
            "",
            "### Nhận xét",
            "",
            f'Cấu hình có Validation F1 cao nhất: '
            f'{best_validation["Model"]}.',
            "",
            f'Mô hình đề xuất đạt Test F1-score '
            f'{proposed["F1-score"]*100:.4f}%.',
            ""
        ])

        for path in sorted(folder.glob("*.png")):

            relative = path.relative_to(REPORT).as_posix()

            if path.name.startswith(("08_", "09_")):

                comparison.extend([
                    "",
                    f"![{path.stem}]({relative})",
                    ""
                ])

            else:

                ablation.extend([
                    "",
                    f"![{path.stem}]({relative})",
                    ""
                ])

        comparison.extend([
            "",
            f"## {title}",
            "",
            *metric_table(combined),
            "",
            "### Phân tích kết quả",
            "",
            f'Baseline có Test F1 cao nhất: '
            f'{best_baseline["Model"]}, '
            f'{best_baseline["F1-score"]*100:.4f}%.',
            "",
            f'Chênh lệch F1 của mô hình đề xuất so với '
            f'baseline cao nhất: {gap:+.4f} điểm phần trăm.',
            "",
            f'FPR của mô hình đề xuất: '
            f'{proposed["FPR"]*100:.4f}%.',
            ""
        ])

        if gap < 0:

            comparison.append(
                "Mô hình đề xuất chưa vượt baseline."
            )

    comparison.extend([
        "",
        "## Giới hạn thực nghiệm",
        "",
        "- Không sử dụng tập test để tối ưu siêu tham số.",
        "- Cần kiểm tra chỉ số từng lớp, đặc biệt lớp hiếm.",
        "- Smoke Test không chứng minh hiệu quả cuối cùng.",
        "- Thời gian đánh giá trung bình theo batch không phải",
        "  độ trễ phát hiện tấn công trong hệ thống realtime.",
        "- Cần thực nghiệm chính thức trước khi kết luận ưu thế CNN."
    ])

    documents = {
        "06_PROPOSED_CNN_SE_MHSA.md": architecture,
        f"07_TUNING_ABLATION_{args.profile.upper()}.md": ablation,
        f"08_MODEL_COMPARISON_{args.profile.upper()}.md": comparison
    }

    for filename, lines in documents.items():

        path = REPORT / filename

        path.write_text(
            "\n".join(lines),
            encoding="utf-8"
        )

        print("Report:", path)

    print("\nALL CNN REPORTS COMPLETED")


if __name__ == "__main__":
    main()
