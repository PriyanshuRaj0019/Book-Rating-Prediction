from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import ADASYN, SMOTE, RandomOverSampler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.features.build_features import TARGET_COLUMN, build_preprocessing_pipeline


PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]

CLEANED_DATA_PATH: Final[Path] = PROJECT_ROOT / "data" / "processed" / "books_cleaned.csv"
TRAIN_TEST_SPLIT_PATH: Final[Path] = PROJECT_ROOT / "data" / "processed" / "train_test_split.joblib"
BALANCED_TRAINING_SETS_PATH: Final[Path] = PROJECT_ROOT / "data" / "processed" / "balanced_training_sets.joblib"
BALANCING_METADATA_PATH: Final[Path] = PROJECT_ROOT / "data" / "processed" / "balancing_metadata.json"
FITTED_PREPROCESSING_PIPELINE_PATH: Final[Path] = PROJECT_ROOT / "models" / "fitted_preprocessing_pipeline.joblib"

TEST_SIZE: Final[float] = 0.2
RANDOM_STATE: Final[int] = 42
IMBALANCE_THRESHOLD: Final[float] = 0.2

REQUIRED_COLUMNS: Final[list[str]] = [
    "book_title",
    "book_category",
    "price_gbp",
    "rating",
    "availability_status",
    "product_url",
    "is_high_rating",
]

EXCLUDED_MODEL_COLUMNS: Final[list[str]] = [
    "rating",
    "product_url",
    TARGET_COLUMN,
]


@dataclass(frozen=True)
class ClassDistribution:
    class_0_count: int
    class_1_count: int
    class_0_percentage: float
    class_1_percentage: float
    minority_class: int
    minority_ratio: float
    imbalance_detected: bool


@dataclass(frozen=True)
class SamplingResult:
    strategy_name: str
    class_distribution: ClassDistribution
    sample_count: int


@dataclass(frozen=True)
class BalancingMetadata:
    input_path: str
    train_test_split_path: str
    balanced_training_sets_path: str
    fitted_preprocessing_pipeline_path: str
    test_size: float
    random_state: int
    original_dataset_distribution: ClassDistribution
    training_distribution_before_balancing: ClassDistribution
    test_distribution: ClassDistribution
    sampling_results: list[SamplingResult]
    chosen_strategy: str
    chosen_strategy_reason: str
    leakage_prevention_notes: list[str]


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def load_cleaned_data(file_path: Path) -> pd.DataFrame:
    if not file_path.exists():
        raise FileNotFoundError(f"Cleaned dataset not found: {file_path}")

    dataframe = pd.read_csv(file_path)

    missing_columns = set(REQUIRED_COLUMNS) - set(dataframe.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    return dataframe[REQUIRED_COLUMNS].copy()


def calculate_class_distribution(target: pd.Series | np.ndarray) -> ClassDistribution:
    target_series = pd.Series(target).astype(int)
    class_counts = target_series.value_counts().to_dict()

    class_0_count = int(class_counts.get(0, 0))
    class_1_count = int(class_counts.get(1, 0))
    total_count = class_0_count + class_1_count

    if total_count == 0:
        raise ValueError("Target contains no rows.")

    minority_class = 0 if class_0_count <= class_1_count else 1
    minority_count = min(class_0_count, class_1_count)
    minority_ratio = minority_count / total_count

    return ClassDistribution(
        class_0_count=class_0_count,
        class_1_count=class_1_count,
        class_0_percentage=round((class_0_count / total_count) * 100, 2),
        class_1_percentage=round((class_1_count / total_count) * 100, 2),
        minority_class=minority_class,
        minority_ratio=round(minority_ratio, 4),
        imbalance_detected=minority_ratio < IMBALANCE_THRESHOLD,
    )


def create_model_inputs(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    feature_dataframe = dataframe.drop(columns=EXCLUDED_MODEL_COLUMNS)
    target_series = dataframe[TARGET_COLUMN].astype("int64")

    return feature_dataframe, target_series


def split_dataset(
    features: pd.DataFrame,
    target: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    return train_test_split(
        features,
        target,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=target,
    )


def fit_and_transform_features(
    preprocessing_pipeline: Pipeline,
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, Pipeline]:
    x_train_processed = preprocessing_pipeline.fit_transform(x_train)
    x_test_processed = preprocessing_pipeline.transform(x_test)

    return x_train_processed, x_test_processed, preprocessing_pipeline


def apply_sampling_methods(
    x_train_processed: np.ndarray,
    y_train: pd.Series,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    samplers = {
        "smote": SMOTE(random_state=RANDOM_STATE),
        "adasyn": ADASYN(random_state=RANDOM_STATE),
        "random_oversampling": RandomOverSampler(random_state=RANDOM_STATE),
    }

    balanced_sets: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    for strategy_name, sampler in samplers.items():
        logging.info("Applying %s", strategy_name)
        balanced_features, balanced_target = sampler.fit_resample(x_train_processed, y_train)
        balanced_sets[strategy_name] = (
            np.asarray(balanced_features),
            np.asarray(balanced_target),
        )

    return balanced_sets


def create_sampling_results(
    balanced_sets: dict[str, tuple[np.ndarray, np.ndarray]],
) -> list[SamplingResult]:
    results: list[SamplingResult] = []

    for strategy_name, (features, target) in balanced_sets.items():
        results.append(
            SamplingResult(
                strategy_name=strategy_name,
                class_distribution=calculate_class_distribution(target),
                sample_count=int(features.shape[0]),
            )
        )

    return results


def save_train_test_split(
    x_train_processed: np.ndarray,
    x_test_processed: np.ndarray,
    y_train: pd.Series,
    y_test: pd.Series,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "x_train_processed": x_train_processed,
        "x_test_processed": x_test_processed,
        "y_train": y_train.to_numpy(),
        "y_test": y_test.to_numpy(),
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
    }

    joblib.dump(payload, output_path)


def save_balanced_training_sets(
    balanced_sets: dict[str, tuple[np.ndarray, np.ndarray]],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        strategy_name: {
            "x_train": features,
            "y_train": target,
        }
        for strategy_name, (features, target) in balanced_sets.items()
    }

    joblib.dump(payload, output_path)


def save_fitted_preprocessing_pipeline(pipeline: Pipeline, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, output_path)


def save_metadata(metadata: BalancingMetadata, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(asdict(metadata), file, indent=2)


def create_metadata(
    original_distribution: ClassDistribution,
    training_distribution: ClassDistribution,
    test_distribution: ClassDistribution,
    sampling_results: list[SamplingResult],
) -> BalancingMetadata:
    return BalancingMetadata(
        input_path=str(CLEANED_DATA_PATH),
        train_test_split_path=str(TRAIN_TEST_SPLIT_PATH),
        balanced_training_sets_path=str(BALANCED_TRAINING_SETS_PATH),
        fitted_preprocessing_pipeline_path=str(FITTED_PREPROCESSING_PIPELINE_PATH),
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        original_dataset_distribution=original_distribution,
        training_distribution_before_balancing=training_distribution,
        test_distribution=test_distribution,
        sampling_results=sampling_results,
        chosen_strategy="smote",
        chosen_strategy_reason=(
            "SMOTE is selected as the default training strategy because it balances "
            "the minority class by creating synthetic samples instead of only "
            "duplicating existing rows. ADASYN and random oversampling are saved "
            "for comparison in later modeling phases."
        ),
        leakage_prevention_notes=[
            "The train-test split is created before any balancing is applied.",
            "The preprocessing pipeline is fitted only on the training set.",
            "The test set is transformed but never balanced.",
            "SMOTE, ADASYN, and random oversampling are applied only to the processed training set.",
            "The rating column is excluded from input features because it directly defines the target.",
        ],
    )


def main() -> None:
    configure_logging()

    dataframe = load_cleaned_data(CLEANED_DATA_PATH)
    features, target = create_model_inputs(dataframe)

    x_train, x_test, y_train, y_test = split_dataset(features, target)

    preprocessing_pipeline = build_preprocessing_pipeline()
    x_train_processed, x_test_processed, fitted_pipeline = fit_and_transform_features(
        preprocessing_pipeline=preprocessing_pipeline,
        x_train=x_train,
        x_test=x_test,
    )

    balanced_sets = apply_sampling_methods(x_train_processed, y_train)
    sampling_results = create_sampling_results(balanced_sets)

    original_distribution = calculate_class_distribution(target)
    training_distribution = calculate_class_distribution(y_train)
    test_distribution = calculate_class_distribution(y_test)

    metadata = create_metadata(
        original_distribution=original_distribution,
        training_distribution=training_distribution,
        test_distribution=test_distribution,
        sampling_results=sampling_results,
    )

    save_train_test_split(
        x_train_processed=x_train_processed,
        x_test_processed=x_test_processed,
        y_train=y_train,
        y_test=y_test,
        output_path=TRAIN_TEST_SPLIT_PATH,
    )
    save_balanced_training_sets(balanced_sets, BALANCED_TRAINING_SETS_PATH)
    save_fitted_preprocessing_pipeline(fitted_pipeline, FITTED_PREPROCESSING_PIPELINE_PATH)
    save_metadata(metadata, BALANCING_METADATA_PATH)

    logging.info(
        "Original class distribution: class 0=%s, class 1=%s",
        original_distribution.class_0_count,
        original_distribution.class_1_count,
    )
    logging.info(
        "Training distribution before balancing: class 0=%s, class 1=%s",
        training_distribution.class_0_count,
        training_distribution.class_1_count,
    )

    for result in sampling_results:
        logging.info(
            "%s distribution after balancing: class 0=%s, class 1=%s, samples=%s",
            result.strategy_name,
            result.class_distribution.class_0_count,
            result.class_distribution.class_1_count,
            result.sample_count,
        )

    logging.info("Chosen strategy: %s", metadata.chosen_strategy)
    logging.info("Saved train-test split to %s", TRAIN_TEST_SPLIT_PATH)
    logging.info("Saved balanced training sets to %s", BALANCED_TRAINING_SETS_PATH)
    logging.info("Saved fitted preprocessing pipeline to %s", FITTED_PREPROCESSING_PIPELINE_PATH)
    logging.info("Saved balancing metadata to %s", BALANCING_METADATA_PATH)


if __name__ == "__main__":
    main()

    
