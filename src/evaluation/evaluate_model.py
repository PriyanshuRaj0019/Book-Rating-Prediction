from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]

FINAL_MODEL_PATH: Final[Path] = PROJECT_ROOT / "models" / "final_model.joblib"
TRAIN_TEST_SPLIT_PATH: Final[Path] = PROJECT_ROOT / "data" / "processed" / "train_test_split.joblib"

FINAL_EVALUATION_METRICS_PATH: Final[Path] = (
    PROJECT_ROOT / "data" / "processed" / "final_evaluation_metrics.json"
)
CLASSIFICATION_REPORT_PATH: Final[Path] = (
    PROJECT_ROOT / "reports" / "docs" / "classification_report.json"
)
CONFUSION_MATRIX_PATH: Final[Path] = PROJECT_ROOT / "reports" / "figures" / "confusion_matrix.png"
ROC_CURVE_PATH: Final[Path] = PROJECT_ROOT / "reports" / "figures" / "roc_curve.png"

POSITIVE_CLASS_LABEL: Final[int] = 1
CLASS_LABELS: Final[list[int]] = [0, 1]

REQUIRED_SPLIT_KEYS: Final[set[str]] = {
    "x_test_processed",
    "y_test",
}


@dataclass(frozen=True)
class EvaluationMetrics:
    model_path: str
    test_sample_count: int
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float
    confusion_matrix: list[list[int]]
    classification_report_path: str
    confusion_matrix_plot_path: str
    roc_curve_plot_path: str
    evaluation_notes: list[str]


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def load_joblib_file(file_path: Path) -> Any:
    if not file_path.exists():
        raise FileNotFoundError(f"Required file not found: {file_path}")

    return joblib.load(file_path)


def load_test_data(file_path: Path) -> tuple[np.ndarray, np.ndarray]:
    payload = load_joblib_file(file_path)

    if not isinstance(payload, dict):
        raise TypeError(f"Expected dictionary payload in {file_path}")

    missing_keys = REQUIRED_SPLIT_KEYS - set(payload.keys())
    if missing_keys:
        raise ValueError(f"Train-test split is missing keys: {sorted(missing_keys)}")

    x_test = np.asarray(payload["x_test_processed"])
    y_test = np.asarray(payload["y_test"]).astype(int)

    return x_test, y_test


def get_positive_class_scores(model: Any, x_test: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(x_test)

        if probabilities.shape[1] >= 2:
            return np.asarray(probabilities[:, POSITIVE_CLASS_LABEL])

    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(x_test))

    raise TypeError(
        "Model does not expose predict_proba or decision_function, so ROC-AUC cannot be calculated."
    )


def create_classification_report(y_test: np.ndarray, predictions: np.ndarray) -> dict[str, Any]:
    return classification_report(
        y_test,
        predictions,
        labels=CLASS_LABELS,
        target_names=["low_rating", "high_rating"],
        zero_division=0,
        output_dict=True,
    )


def calculate_metrics(
    model: Any,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> tuple[EvaluationMetrics, dict[str, Any], np.ndarray, np.ndarray]:
    predictions = model.predict(x_test)
    positive_class_scores = get_positive_class_scores(model, x_test)

    confusion_matrix_values = confusion_matrix(
        y_test,
        predictions,
        labels=CLASS_LABELS,
    ).astype(int)

    report = create_classification_report(y_test, predictions)

    metrics = EvaluationMetrics(
        model_path=str(FINAL_MODEL_PATH),
        test_sample_count=int(x_test.shape[0]),
        accuracy=round(float(accuracy_score(y_test, predictions)), 4),
        precision=round(
            float(
                precision_score(
                    y_test,
                    predictions,
                    pos_label=POSITIVE_CLASS_LABEL,
                    zero_division=0,
                )
            ),
            4,
        ),
        recall=round(
            float(
                recall_score(
                    y_test,
                    predictions,
                    pos_label=POSITIVE_CLASS_LABEL,
                    zero_division=0,
                )
            ),
            4,
        ),
        f1_score=round(
            float(
                f1_score(
                    y_test,
                    predictions,
                    pos_label=POSITIVE_CLASS_LABEL,
                    zero_division=0,
                )
            ),
            4,
        ),
        roc_auc=round(float(roc_auc_score(y_test, positive_class_scores)), 4),
        confusion_matrix=confusion_matrix_values.tolist(),
        classification_report_path=str(CLASSIFICATION_REPORT_PATH),
        confusion_matrix_plot_path=str(CONFUSION_MATRIX_PATH),
        roc_curve_plot_path=str(ROC_CURVE_PATH),
        evaluation_notes=[
            "The final model is evaluated only on the untouched test set.",
            "The test set was not balanced using SMOTE, ADASYN, or random oversampling.",
            "F1 score is important because the target classes are not perfectly balanced.",
            "ROC-AUC measures how well the model separates low-rating and high-rating classes.",
            "Results are educational because the source dataset is a scraping sandbox with random ratings.",
        ],
    )

    return metrics, report, predictions, positive_class_scores


def save_json(payload: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def save_confusion_matrix_plot(confusion_matrix_values: list[list[int]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    matrix = np.asarray(confusion_matrix_values)

    plt.figure(figsize=(7, 6))
    plt.imshow(matrix, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.colorbar()

    tick_marks = np.arange(len(CLASS_LABELS))
    plt.xticks(tick_marks, ["Low Rating", "High Rating"])
    plt.yticks(tick_marks, ["Low Rating", "High Rating"])

    threshold = matrix.max() / 2

    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            plt.text(
                column_index,
                row_index,
                str(matrix[row_index, column_index]),
                ha="center",
                va="center",
                color="white" if matrix[row_index, column_index] > threshold else "black",
            )

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def save_roc_curve_plot(
    y_test: np.ndarray,
    positive_class_scores: np.ndarray,
    roc_auc: float,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    false_positive_rate, true_positive_rate, _ = roc_curve(
        y_test,
        positive_class_scores,
        pos_label=POSITIVE_CLASS_LABEL,
    )

    plt.figure(figsize=(8, 6))
    plt.plot(false_positive_rate, true_positive_rate, label=f"ROC curve, AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random baseline")
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def main() -> None:
    configure_logging()

    final_model = load_joblib_file(FINAL_MODEL_PATH)
    x_test, y_test = load_test_data(TRAIN_TEST_SPLIT_PATH)

    metrics, report, _, positive_class_scores = calculate_metrics(
        model=final_model,
        x_test=x_test,
        y_test=y_test,
    )

    save_json(asdict(metrics), FINAL_EVALUATION_METRICS_PATH)
    save_json(report, CLASSIFICATION_REPORT_PATH)

    save_confusion_matrix_plot(
        confusion_matrix_values=metrics.confusion_matrix,
        output_path=CONFUSION_MATRIX_PATH,
    )

    save_roc_curve_plot(
        y_test=y_test,
        positive_class_scores=positive_class_scores,
        roc_auc=metrics.roc_auc,
        output_path=ROC_CURVE_PATH,
    )

    logging.info("Final model evaluation completed")
    logging.info("Test samples: %s", metrics.test_sample_count)
    logging.info("Accuracy: %.4f", metrics.accuracy)
    logging.info("Precision: %.4f", metrics.precision)
    logging.info("Recall: %.4f", metrics.recall)
    logging.info("F1 score: %.4f", metrics.f1_score)
    logging.info("ROC-AUC: %.4f", metrics.roc_auc)
    logging.info("Saved evaluation metrics to %s", FINAL_EVALUATION_METRICS_PATH)
    logging.info("Saved classification report to %s", CLASSIFICATION_REPORT_PATH)
    logging.info("Saved confusion matrix plot to %s", CONFUSION_MATRIX_PATH)
    logging.info("Saved ROC curve plot to %s", ROC_CURVE_PATH)


if __name__ == "__main__":
    main()
    
