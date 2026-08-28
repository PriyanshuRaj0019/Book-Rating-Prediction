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
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline


PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]

TRAIN_TEST_SPLIT_PATH: Final[Path] = PROJECT_ROOT / "data" / "processed" / "train_test_split.joblib"
FINAL_MODEL_PATH: Final[Path] = PROJECT_ROOT / "models" / "final_model.joblib"
FITTED_PREPROCESSING_PIPELINE_PATH: Final[Path] = PROJECT_ROOT / "models" / "fitted_preprocessing_pipeline.joblib"

INTERPRETATION_OUTPUT_PATH: Final[Path] = PROJECT_ROOT / "data" / "processed" / "model_interpretation.json"
FEATURE_IMPORTANCE_PLOT_PATH: Final[Path] = PROJECT_ROOT / "reports" / "figures" / "feature_importance.png"

TOP_FEATURE_COUNT: Final[int] = 15
RANDOM_STATE: Final[int] = 42
N_REPEATS: Final[int] = 20
SCORING_METRIC: Final[str] = "f1"

REQUIRED_SPLIT_KEYS: Final[set[str]] = {
    "x_test_processed",
    "y_test",
}


@dataclass(frozen=True)
class FeatureImportanceRecord:
    rank: int
    feature_name: str
    importance_mean: float
    importance_std: float


@dataclass(frozen=True)
class InterpretationSummary:
    model_path: str
    preprocessing_pipeline_path: str
    test_sample_count: int
    feature_count: int
    scoring_metric: str
    method: str
    top_features: list[FeatureImportanceRecord]
    saved_plot_path: str
    business_interpretation: list[str]
    limitations: list[str]


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
    y_test = np.asarray(payload["y_test"])

    return x_test, y_test


def extract_feature_names(
    preprocessing_pipeline: Pipeline,
    feature_count: int,
) -> list[str]:
    try:
        preprocessor = preprocessing_pipeline.named_steps["preprocessor"]
        feature_selector = preprocessing_pipeline.named_steps["feature_selector"]

        transformed_feature_names = np.asarray(preprocessor.get_feature_names_out())
        selected_feature_mask = feature_selector.get_support()

        selected_feature_names = transformed_feature_names[selected_feature_mask]

        if len(selected_feature_names) == feature_count:
            return [clean_feature_name(feature_name) for feature_name in selected_feature_names]

    except Exception as exc:
        logging.warning("Could not extract fitted pipeline feature names: %s", exc)

    return [f"feature_{index}" for index in range(feature_count)]


def clean_feature_name(feature_name: str) -> str:
    cleaned_name = feature_name.replace("numeric__", "")
    cleaned_name = cleaned_name.replace("categorical__", "")
    cleaned_name = cleaned_name.replace("book_category_", "category_")
    cleaned_name = cleaned_name.replace("availability_status_", "availability_")
    cleaned_name = cleaned_name.replace("price_band_", "price_band_")
    return cleaned_name


def calculate_permutation_importance(
    model: Any,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    result = permutation_importance(
        estimator=model,
        X=x_test,
        y=y_test,
        scoring=SCORING_METRIC,
        n_repeats=N_REPEATS,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    return result.importances_mean, result.importances_std


def create_importance_records(
    feature_names: list[str],
    importance_mean: np.ndarray,
    importance_std: np.ndarray,
) -> list[FeatureImportanceRecord]:
    importance_dataframe = pd.DataFrame(
        {
            "feature_name": feature_names,
            "importance_mean": importance_mean,
            "importance_std": importance_std,
        }
    )

    importance_dataframe = importance_dataframe.sort_values(
        by="importance_mean",
        ascending=False,
    ).head(TOP_FEATURE_COUNT)

    records: list[FeatureImportanceRecord] = []

    for rank, (_, row) in enumerate(importance_dataframe.iterrows(), start=1):
        records.append(
            FeatureImportanceRecord(
                rank=rank,
                feature_name=str(row["feature_name"]),
                importance_mean=round(float(row["importance_mean"]), 6),
                importance_std=round(float(row["importance_std"]), 6),
            )
        )

    return records


def save_feature_importance_plot(
    top_features: list[FeatureImportanceRecord],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plot_dataframe = pd.DataFrame(asdict(feature) for feature in top_features)
    plot_dataframe = plot_dataframe.sort_values(by="importance_mean", ascending=True)

    plt.figure(figsize=(12, 8))
    plt.barh(
        plot_dataframe["feature_name"],
        plot_dataframe["importance_mean"],
        xerr=plot_dataframe["importance_std"],
    )
    plt.title("Top Feature Importance by Permutation")
    plt.xlabel("Mean decrease in F1 score")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def create_business_interpretation(top_features: list[FeatureImportanceRecord]) -> list[str]:
    if not top_features:
        return ["No meaningful feature importance values were produced."]

    highest_feature = top_features[0]

    return [
        f"The most influential feature for the final model is {highest_feature.feature_name}.",
        "Higher permutation importance means model performance drops more when that feature is shuffled.",
        "Price-related and title-derived features help the model estimate whether a book belongs to the high-rating class.",
        "Categorical features such as category, availability, and price band provide additional signal after one-hot encoding.",
        "The interpretation is educational because the source website uses randomly assigned ratings and prices.",
    ]


def create_interpretation_summary(
    x_test: np.ndarray,
    top_features: list[FeatureImportanceRecord],
) -> InterpretationSummary:
    return InterpretationSummary(
        model_path=str(FINAL_MODEL_PATH),
        preprocessing_pipeline_path=str(FITTED_PREPROCESSING_PIPELINE_PATH),
        test_sample_count=int(x_test.shape[0]),
        feature_count=int(x_test.shape[1]),
        scoring_metric=SCORING_METRIC,
        method="Permutation importance on the untouched test set",
        top_features=top_features,
        saved_plot_path=str(FEATURE_IMPORTANCE_PLOT_PATH),
        business_interpretation=create_business_interpretation(top_features),
        limitations=[
            "Permutation importance measures association with model predictions, not causation.",
            "Feature importance can vary across train-test splits.",
            "Highly correlated features can share or dilute importance.",
            "The dataset is from a scraping sandbox, so business interpretation is not real market evidence.",
        ],
    )


def save_interpretation_summary(
    summary: InterpretationSummary,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(asdict(summary), file, indent=2)


def main() -> None:
    configure_logging()

    final_model = load_joblib_file(FINAL_MODEL_PATH)
    preprocessing_pipeline = load_joblib_file(FITTED_PREPROCESSING_PIPELINE_PATH)
    x_test, y_test = load_test_data(TRAIN_TEST_SPLIT_PATH)

    feature_names = extract_feature_names(
        preprocessing_pipeline=preprocessing_pipeline,
        feature_count=x_test.shape[1],
    )

    importance_mean, importance_std = calculate_permutation_importance(
        model=final_model,
        x_test=x_test,
        y_test=y_test,
    )

    top_features = create_importance_records(
        feature_names=feature_names,
        importance_mean=importance_mean,
        importance_std=importance_std,
    )

    save_feature_importance_plot(top_features, FEATURE_IMPORTANCE_PLOT_PATH)

    summary = create_interpretation_summary(
        x_test=x_test,
        top_features=top_features,
    )
    save_interpretation_summary(summary, INTERPRETATION_OUTPUT_PATH)

    logging.info("Model interpreted using permutation importance")
    logging.info("Test samples: %s", summary.test_sample_count)
    logging.info("Feature count: %s", summary.feature_count)
    logging.info("Top feature: %s", top_features[0].feature_name if top_features else "None")
    logging.info("Saved interpretation summary to %s", INTERPRETATION_OUTPUT_PATH)
    logging.info("Saved feature importance plot to %s", FEATURE_IMPORTANCE_PLOT_PATH)


if __name__ == "__main__":
    main()
