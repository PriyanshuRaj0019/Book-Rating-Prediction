from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final

import joblib
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]

TRAIN_TEST_SPLIT_PATH: Final[Path] = PROJECT_ROOT / "data" / "processed" / "train_test_split.joblib"
BASELINE_MODEL_PATH: Final[Path] = PROJECT_ROOT / "models" / "baseline_model.joblib"
BASELINE_METRICS_PATH: Final[Path] = PROJECT_ROOT / "data" / "processed" / "baseline_metrics.json"

RANDOM_STATE: Final[int] = 42
POSITIVE_CLASS_LABEL: Final[int] = 1

REQUIRED_SPLIT_KEYS: Final[set[str]] = {
    "x_train_processed",
    "x_test_processed",
    "y_train",
    "y_test",
}


@dataclass(frozen=True)
class BaselineMetrics:
    model_name: str
    strategy: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float
    confusion_matrix: list[list[int]]
    classification_report: dict[str, Any]
    train_sample_count: int
    test_sample_count: int
    notes: list[str]


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def load_train_test_split(file_path: Path) -> dict[str, Any]:
    if not file_path.exists():
        raise FileNotFoundError(f"Train-test split file not found: {file_path}")

    split_data = joblib.load(file_path)

    missing_keys = REQUIRED_SPLIT_KEYS - set(split_data.keys())
    if missing_keys:
        raise ValueError(f"Train-test split is missing keys: {sorted(missing_keys)}")

    return split_data


def create_baseline_model() -> DummyClassifier:
    return DummyClassifier(
        strategy="most_frequent",
        random_state=RANDOM_STATE,
    )


def train_baseline_model(
    model: DummyClassifier,
    x_train: np.ndarray,
    y_train: np.ndarray,
) -> DummyClassifier:
    model.fit(x_train, y_train)
    return model


def calculate_roc_auc(
    model: DummyClassifier,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> float:
    if not hasattr(model, "predict_proba"):
        return 0.0

    predicted_probabilities = model.predict_proba(x_test)

    if predicted_probabilities.shape[1] < 2:
        return 0.0

    return round(
        float(roc_auc_score(y_test, predicted_probabilities[:, POSITIVE_CLASS_LABEL])),
        4,
    )


def evaluate_baseline_model(
    model: DummyClassifier,
    x_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> BaselineMetrics:
    predictions = model.predict(x_test)

    return BaselineMetrics(
        model_name="DummyClassifier",
        strategy="most_frequent",
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
        roc_auc=calculate_roc_auc(model, x_test, y_test),
        confusion_matrix=confusion_matrix(y_test, predictions).astype(int).tolist(),
        classification_report=classification_report(
            y_test,
            predictions,
            zero_division=0,
            output_dict=True,
        ),
        train_sample_count=int(x_train.shape[0]),
        test_sample_count=int(x_test.shape[0]),
        notes=[
            "This baseline always predicts the most frequent class from the training data.",
            "It does not learn relationships between features and target values.",
            "Future trained models must outperform this baseline meaningfully.",
            "The test set is unchanged and was not balanced.",
        ],
    )


def save_model(model: DummyClassifier, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)


def save_metrics(metrics: BaselineMetrics, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(asdict(metrics), file, indent=2)


def main() -> None:
    configure_logging()

    split_data = load_train_test_split(TRAIN_TEST_SPLIT_PATH)

    x_train = np.asarray(split_data["x_train_processed"])
    x_test = np.asarray(split_data["x_test_processed"])
    y_train = np.asarray(split_data["y_train"])
    y_test = np.asarray(split_data["y_test"])

    baseline_model = create_baseline_model()
    trained_model = train_baseline_model(
        model=baseline_model,
        x_train=x_train,
        y_train=y_train,
    )

    metrics = evaluate_baseline_model(
        model=trained_model,
        x_train=x_train,
        x_test=x_test,
        y_test=y_test,
    )

    save_model(trained_model, BASELINE_MODEL_PATH)
    save_metrics(metrics, BASELINE_METRICS_PATH)

    logging.info("Baseline model: %s", metrics.model_name)
    logging.info("Strategy: %s", metrics.strategy)
    logging.info("Accuracy: %.4f", metrics.accuracy)
    logging.info("Precision: %.4f", metrics.precision)
    logging.info("Recall: %.4f", metrics.recall)
    logging.info("F1 score: %.4f", metrics.f1_score)
    logging.info("ROC-AUC: %.4f", metrics.roc_auc)
    logging.info("Saved baseline model to %s", BASELINE_MODEL_PATH)
    logging.info("Saved baseline metrics to %s", BASELINE_METRICS_PATH)


if __name__ == "__main__":
    main()
