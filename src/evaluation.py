"""Evaluation for hierarchical multi-label classification.

Three complementary views are computed, following the module's evaluation
guidance for related label levels:

1. Per-label accuracy / weighted F1  - how good is each level in isolation.
2. Exact-match accuracy              - all three levels correct or nothing.
3. Hierarchical (prefix) accuracy    - partial credit for the levels that
   are correct *before the first mistake*, walking y2 -> y3 -> y4. A row
   with y2 wrong scores 0 even if y3/y4 happen to be right, because a
   mis-routed top category makes the downstream labels meaningless for
   support routing.
"""
import json
import logging

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from src import config

logger = logging.getLogger(__name__)


def hierarchical_prefix_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean over samples of (consecutive correct levels from y2) / n_levels."""
    y_true = np.asarray(y_true, dtype=object)
    y_pred = np.asarray(y_pred, dtype=object)
    n_samples, n_levels = y_true.shape
    scores = np.zeros(n_samples)
    for i in range(n_samples):
        correct = 0
        for j in range(n_levels):
            if y_true[i, j] == y_pred[i, j]:
                correct += 1
            else:
                break
        scores[i] = correct / n_levels
    return float(scores.mean())


def exact_match_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=object)
    y_pred = np.asarray(y_pred, dtype=object)
    return float((y_true == y_pred).all(axis=1).mean())


def evaluate_model(model_name: str, y_true: pd.DataFrame, y_pred: np.ndarray) -> dict:
    """Compute all metrics for one model; return a JSON-serialisable dict."""
    metrics = {"model": model_name}
    y_true_arr = y_true.to_numpy(dtype=object)

    for j, col in enumerate(config.LABEL_COLUMNS):
        metrics[f"{col}_accuracy"] = round(
            accuracy_score(y_true_arr[:, j], y_pred[:, j]), 4)
        metrics[f"{col}_f1_weighted"] = round(
            f1_score(y_true_arr[:, j], y_pred[:, j],
                     average="weighted", zero_division=0), 4)

    metrics["exact_match_accuracy"] = round(
        exact_match_accuracy(y_true_arr, y_pred), 4)
    metrics["hierarchical_prefix_accuracy"] = round(
        hierarchical_prefix_accuracy(y_true_arr, y_pred), 4)
    return metrics


def save_classification_reports(model_name: str, y_true: pd.DataFrame,
                                y_pred: np.ndarray) -> None:
    for j, col in enumerate(config.LABEL_COLUMNS):
        report = classification_report(
            y_true.to_numpy(dtype=object)[:, j], y_pred[:, j],
            zero_division=0, output_dict=True)
        pd.DataFrame(report).T.to_csv(
            config.OUTPUTS_DIR / f"classification_report_{model_name}_{col}.csv")


def save_confusion_matrix_plot(model_name: str, y_true, y_pred, col: str) -> None:
    """Confusion matrix heatmap for one label level (used for y2)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = sorted(set(list(y_true) + list(y_pred)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    for i in range(len(labels)):
        for k in range(len(labels)):
            ax.text(k, i, cm[i, k], ha="center", va="center",
                    color="white" if cm[i, k] > cm.max() / 2 else "black")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"{col} confusion matrix - {model_name}")
    fig.colorbar(im)
    fig.tight_layout()
    path = config.OUTPUTS_DIR / f"confusion_matrix_{model_name}_{col}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", path.name)


def save_error_analysis(model_name: str, test_df: pd.DataFrame,
                        y_pred: np.ndarray, max_rows: int = 15) -> pd.DataFrame:
    """Table of misclassified test messages for the report's error analysis."""
    records = []
    y_true = test_df[config.LABEL_COLUMNS].to_numpy(dtype=object)
    for i in range(len(test_df)):
        wrong_levels = [
            config.LABEL_COLUMNS[j]
            for j in range(len(config.LABEL_COLUMNS))
            if y_true[i, j] != y_pred[i, j]
        ]
        if wrong_levels:
            records.append({
                "ticket_id": test_df.iloc[i][config.TICKET_ID_COL],
                "message_excerpt": test_df.iloc[i]["clean_text"][:120],
                "true_y2/y3/y4": " | ".join(y_true[i]),
                "pred_y2/y3/y4": " | ".join(y_pred[i]),
                "wrong_levels": ",".join(wrong_levels),
            })
    err_df = pd.DataFrame(records)
    path = config.OUTPUTS_DIR / f"error_analysis_{model_name}.csv"
    err_df.head(max_rows).to_csv(path, index=False)
    logger.info("Saved %s (%d misclassified rows in total)", path.name, len(err_df))
    return err_df


def save_metrics(all_metrics: list) -> None:
    path = config.OUTPUTS_DIR / "metrics.json"
    with open(path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    logger.info("Saved %s", path.name)
