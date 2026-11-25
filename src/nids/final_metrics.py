import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


def evaluate(y_true, y_pred, kind, labels):

    average = (
        "weighted" if kind == "cic"
        else "binary"
    )

    kwargs = {
        "average": average,
        "zero_division": 0
    }

    if average == "binary":
        kwargs["pos_label"] = 1

    return {
        "accuracy": float(
            accuracy_score(y_true, y_pred)
        ),
        "precision": float(
            precision_score(y_true, y_pred, **kwargs)
        ),
        "recall": float(
            recall_score(y_true, y_pred, **kwargs)
        ),
        "f1_score": float(
            f1_score(y_true, y_pred, **kwargs)
        ),
        "f1_average": average,
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=labels
        ).tolist(),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=labels,
            output_dict=True,
            zero_division=0
        )
    }


def binary_detection_metrics(y_true, y_pred):

    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).ravel()

    result = evaluate(
        y_true, y_pred, "ctu", [0, 1]
    )

    result["false_positive_rate"] = (
        float(fp / (fp + tn))
        if fp + tn else None
    )

    result["detection_rate"] = (
        float(tp / (tp + fn))
        if tp + fn else None
    )

    return result
