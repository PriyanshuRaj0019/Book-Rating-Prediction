from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final

import joblib
import numpy as np
import pandas as pd
from scipy.stats import randint
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.svm import SVC


PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]

TRAIN_TEST_SPLIT_PATH: Final[Path] = PROJECT_ROOT / "data" / "processed" / "train_test_split.joblib"
BALANCED_TRAINING_SETS_PATH: Final[Path] = PROJECT_ROOT / "data" / "processed" / "balanced_training_sets.joblib"

MODEL_COMPARISON_PATH: Final[Path] = PROJECT_ROOT / "data" / "processed" / "model_comparison.csv"
MODEL_TRAINING_RESULTS_PATH: Final[Path] = PROJECT_ROOT / "data" / "processed" / "model_training_results.json"

LOGISTIC_REGRESSION_MODEL_PATH: Final[Path] = PROJECT_ROOT / "models" / "logistic_regression_model.joblib"
RANDOM_FOREST_MODEL_PATH: Final[Path] = PROJECT_ROOT / "models" / "random_forest_model.joblib"
SVM_MODEL_PATH: Final[Path] = PROJECT_ROOT / "models" / "svm_model.joblib"
FINAL_MODEL_PATH: Final[Path] = PROJECT_ROOT / "models" / "final_model.joblib"

RANDOM_STATE: Final[int] = 42
CV_SPLITS: Final[int] = 5
SCORING_METRIC: Final[str] = "f1"
BALANCING_STRATEGY: Final[str] = "smote"
POSITIVE_CLASS_LABEL: Final[int] = 1

REQUIRED_SPLIT_KEYS: Final[set[str]] = {
    "x_test_processed",
    "y_test",
}

REQUIRED_BALANCED_KEYS: Final[set[str]] = {
    BALANCING_STRATEGY,
}


@dataclass(frozen=True)
class ModelResult:
    model_name: str
    search_method: str
    best_cv_f1_score: float
    test_accuracy: float
    test_precision: float
    test_recall: float
    test_f1_score: float
    test_roc_auc: float
    best_parameters: dict[str, Any]
    model_path: str


@dataclass(frozen=True)
class TrainingSummary:
    balancing_strategy: str
    cv_splits: int
    scoring_metric: str
    train_sample_count: int
    test_sample_count: int
    feature_count: int
    model_results: list[ModelResult]
    selected_model_name: str
    selected_model_path: str
    final_model_reason: str


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def load_joblib_payload(file_path: Path) -> dict[str, Any]:
    if not file_path.exists():
        raise FileNotFoundError(f"Required file not found: {file_path}")

    payload = joblib.load(file_path)

    if not isinstance(payload, dict):
        raise TypeError(f"Expected dictionary payload in {file_path}")

    return payload


def load_training_and_test_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    split_payload = load_joblib_payload(TRAIN_TEST_SPLIT_PATH)
    balanced_payload = load_joblib_payload(BALANCED_TRAINING_SETS_PATH)

    missing_split_keys = REQUIRED_SPLIT_KEYS - set(split_payload.keys())
    if missing_split_keys:
        raise ValueError(f"Train-test split is missing keys: {sorted(missing_split_keys)}")

    missing_balanced_keys = REQUIRED_BALANCED_KEYS - set(balanced_payload.keys())
    if missing_balanced_keys:
        raise ValueError(f"Balanced training payload is missing keys: {sorted(missing_balanced_keys)}")

    smote_payload = balanced_payload[BALANCING_STRATEGY]

    x_train = np.asarray(smote_payload["x_train"])
    y_train = np.asarray(smote_payload["y_train"])
    x_test = np.asarray(split_payload["x_test_processed"])
    y_test = np.asarray(split_payload["y_test"])

    return x_train, x_test, y_train, y_test


def create_cross_validator() -> StratifiedKFold:
    return StratifiedKFold(
        n_splits=CV_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )


def build_logistic_regression_search(cv: StratifiedKFold) -> GridSearchCV:
    estimator = LogisticRegression(
        max_iter=2_000,
        random_state=RANDOM_STATE,
        solver="liblinear",
    )

    parameter_grid = {
        "C": [0.01, 0.1, 1.0, 10.0],
        "penalty": ["l1", "l2"],
        "class_weight": [None, "balanced"],
    }

    return GridSearchCV(
        estimator=estimator,
        param_grid=parameter_grid,
        scoring=SCORING_METRIC,
        cv=cv,
        n_jobs=-1,
        refit=True,
    )


def build_random_forest_search(cv: StratifiedKFold) -> RandomizedSearchCV:
    estimator = RandomForestClassifier(
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    parameter_distributions = {
        "n_estimators": randint(100, 401),
        "max_depth": [None, 4, 6, 8, 10, 12],
        "min_samples_split": randint(2, 11),
        "min_samples_leaf": randint(1, 6),
        "max_features": ["sqrt", "log2", None],
        "class_weight": [None, "balanced"],
    }

    return RandomizedSearchCV(
        estimator=estimator,
        param_distributions=parameter_distributions,
        n_iter=20,
        scoring=SCORING_METRIC,
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        refit=True,
    )


def build_svm_search(cv: StratifiedKFold) -> GridSearchCV:
    estimator = SVC(
        probability=True,
        random_state=RANDOM_STATE,
    )

    parameter_grid = {
        "C": [0.1, 1.0, 10.0],
        "kernel": ["linear", "rbf"],
        "gamma": ["scale", "auto"],
        "class_weight": [None, "balanced"],
    }

    return GridSearchCV(
        estimator=estimator,
        param_grid=parameter_grid,
        scoring=SCORING_METRIC,
        cv=cv,
        n_jobs=-1,
        refit=True,
    )


def calculate_roc_auc(model: Any, x_test: np.ndarray, y_test: np.ndarray) -> float:
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(x_test)

        if probabilities.shape[1] >= 2:
            return round(float(roc_auc_score(y_test, probabilities[:, POSITIVE_CLASS_LABEL])), 4)

    if hasattr(model, "decision_function"):
        decision_scores = model.decision_function(x_test)
        return round(float(roc_auc_score(y_test, decision_scores)), 4)

    return 0.0


def evaluate_model(
    model_name: str,
    search_method: str,
    search: GridSearchCV | RandomizedSearchCV,
    x_test: np.ndarray,
    y_test: np.ndarray,
    model_path: Path,
) -> ModelResult:
    best_model = search.best_estimator_
    predictions = best_model.predict(x_test)

    return ModelResult(
        model_name=model_name,
        search_method=search_method,
        best_cv_f1_score=round(float(search.best_score_), 4),
        test_accuracy=round(float(accuracy_score(y_test, predictions)), 4),
        test_precision=round(
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
        test_recall=round(
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
        test_f1_score=round(
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
        test_roc_auc=calculate_roc_auc(best_model, x_test, y_test),
        best_parameters=search.best_params_,
        model_path=str(model_path),
    )


def save_model(model: Any, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)


def save_model_comparison(results: list[ModelResult], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    comparison_dataframe = pd.DataFrame(asdict(result) for result in results)
    comparison_dataframe = comparison_dataframe.sort_values(
        by=["test_f1_score", "test_roc_auc", "test_accuracy"],
        ascending=False,
    )

    comparison_dataframe.to_csv(output_path, index=False)


def choose_final_model(results: list[ModelResult]) -> ModelResult:
    if not results:
        raise ValueError("No model results available for final model selection.")

    return sorted(
        results,
        key=lambda result: (
            result.test_f1_score,
            result.test_roc_auc,
            result.test_accuracy,
            result.best_cv_f1_score,
        ),
        reverse=True,
    )[0]


def create_training_summary(
    x_train: np.ndarray,
    x_test: np.ndarray,
    results: list[ModelResult],
    selected_model: ModelResult,
) -> TrainingSummary:
    return TrainingSummary(
        balancing_strategy=BALANCING_STRATEGY,
        cv_splits=CV_SPLITS,
        scoring_metric=SCORING_METRIC,
        train_sample_count=int(x_train.shape[0]),
        test_sample_count=int(x_test.shape[0]),
        feature_count=int(x_train.shape[1]),
        model_results=results,
        selected_model_name=selected_model.model_name,
        selected_model_path=str(FINAL_MODEL_PATH),
        final_model_reason=(
            f"{selected_model.model_name} was selected because it achieved the strongest "
            "overall ranking using test F1 score first, then ROC-AUC, accuracy, and CV F1 "
            "as tie-breakers. F1 is prioritized because the target classes are not perfectly balanced."
        ),
    )


def save_training_summary(summary: TrainingSummary, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(asdict(summary), file, indent=2)


def train_and_evaluate_models(
    x_train: np.ndarray,
    x_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> tuple[list[ModelResult], dict[str, Any]]:
    cv = create_cross_validator()

    searches: list[tuple[str, str, GridSearchCV | RandomizedSearchCV, Path]] = [
        (
            "Logistic Regression",
            "GridSearchCV",
            build_logistic_regression_search(cv),
            LOGISTIC_REGRESSION_MODEL_PATH,
        ),
        (
            "Random Forest",
            "RandomizedSearchCV",
            build_random_forest_search(cv),
            RANDOM_FOREST_MODEL_PATH,
        ),
        (
            "Support Vector Machine",
            "GridSearchCV",
            build_svm_search(cv),
            SVM_MODEL_PATH,
        ),
    ]

    results: list[ModelResult] = []
    trained_models: dict[str, Any] = {}

    for model_name, search_method, search, model_path in searches:
        logging.info("Training %s using %s", model_name, search_method)
        search.fit(x_train, y_train)

        best_model = search.best_estimator_
        save_model(best_model, model_path)

        result = evaluate_model(
            model_name=model_name,
            search_method=search_method,
            search=search,
            x_test=x_test,
            y_test=y_test,
            model_path=model_path,
        )

        results.append(result)
        trained_models[model_name] = best_model

        logging.info(
            "%s | CV F1: %.4f | Test F1: %.4f | ROC-AUC: %.4f",
            model_name,
            result.best_cv_f1_score,
            result.test_f1_score,
            result.test_roc_auc,
        )

    return results, trained_models


def main() -> None:
    configure_logging()

    x_train, x_test, y_train, y_test = load_training_and_test_data()

    results, trained_models = train_and_evaluate_models(
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
    )

    selected_model = choose_final_model(results)
    final_estimator = trained_models[selected_model.model_name]

    save_model(final_estimator, FINAL_MODEL_PATH)
    save_model_comparison(results, MODEL_COMPARISON_PATH)

    training_summary = create_training_summary(
        x_train=x_train,
        x_test=x_test,
        results=results,
        selected_model=selected_model,
    )

    save_training_summary(training_summary, MODEL_TRAINING_RESULTS_PATH)

    logging.info("Selected final model: %s", selected_model.model_name)
    logging.info("Final model reason: %s", training_summary.final_model_reason)
    logging.info("Saved final model to %s", FINAL_MODEL_PATH)
    logging.info("Saved model comparison to %s", MODEL_COMPARISON_PATH)
    logging.info("Saved model training results to %s", MODEL_TRAINING_RESULTS_PATH)


if __name__ == "__main__":
    main()



