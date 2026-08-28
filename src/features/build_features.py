from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]

CLEANED_DATA_PATH: Final[Path] = PROJECT_ROOT / "data" / "processed" / "books_cleaned.csv"
FEATURED_DATA_PATH: Final[Path] = PROJECT_ROOT / "data" / "processed" / "books_features.csv"
FEATURE_METADATA_PATH: Final[Path] = PROJECT_ROOT / "data" / "processed" / "feature_metadata.json"
PREPROCESSING_PIPELINE_PATH: Final[Path] = PROJECT_ROOT / "models" / "preprocessing_pipeline.joblib"

TARGET_COLUMN: Final[str] = "is_high_rating"

REQUIRED_COLUMNS: Final[list[str]] = [
    "book_title",
    "book_category",
    "price_gbp",
    "rating",
    "availability_status",
    "product_url",
    "is_high_rating",
]

NUMERIC_FEATURES: Final[list[str]] = [
    "price_gbp",
    "price_log_gbp",
    "title_character_count",
    "title_word_count",
    "title_digit_count",
    "title_punctuation_count",
    "title_uppercase_ratio",
    "availability_quantity",
]

CATEGORICAL_FEATURES: Final[list[str]] = [
    "book_category",
    "availability_status",
    "price_band",
]

MODEL_INPUT_COLUMNS: Final[list[str]] = NUMERIC_FEATURES + CATEGORICAL_FEATURES


@dataclass(frozen=True)
class FeatureMetadata:
    input_path: str
    feature_data_path: str
    preprocessing_pipeline_path: str
    row_count: int
    target_column: str
    excluded_columns: list[str]
    numeric_features: list[str]
    categorical_features: list[str]
    feature_engineering_notes: list[str]


class BookFeatureCreator(BaseEstimator, TransformerMixin):
    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> BookFeatureCreator:
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        dataframe = X.copy()

        required_input_columns = [
            "book_title",
            "book_category",
            "price_gbp",
            "availability_status",
        ]
        missing_columns = set(required_input_columns) - set(dataframe.columns)

        if missing_columns:
            raise ValueError(f"Missing required feature columns: {sorted(missing_columns)}")

        dataframe["book_title"] = dataframe["book_title"].astype("string").fillna("")
        dataframe["book_category"] = dataframe["book_category"].astype("string").fillna("Unknown")
        dataframe["availability_status"] = dataframe["availability_status"].astype("string").fillna("Unknown")
        dataframe["price_gbp"] = pd.to_numeric(dataframe["price_gbp"], errors="coerce")

        dataframe["price_log_gbp"] = np.log1p(dataframe["price_gbp"])
        dataframe["price_band"] = dataframe["price_gbp"].apply(assign_price_band)

        dataframe["title_character_count"] = dataframe["book_title"].str.len()
        dataframe["title_word_count"] = dataframe["book_title"].apply(count_words)
        dataframe["title_digit_count"] = dataframe["book_title"].apply(count_digits)
        dataframe["title_punctuation_count"] = dataframe["book_title"].apply(count_punctuation)
        dataframe["title_uppercase_ratio"] = dataframe["book_title"].apply(calculate_uppercase_ratio)
        dataframe["availability_quantity"] = dataframe["availability_status"].apply(extract_availability_quantity)

        return dataframe[MODEL_INPUT_COLUMNS].copy()


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


def count_words(text: str) -> int:
    return len(re.findall(r"\b\w+\b", str(text)))


def count_digits(text: str) -> int:
    return len(re.findall(r"\d", str(text)))


def count_punctuation(text: str) -> int:
    return len(re.findall(r"[^\w\s]", str(text)))


def calculate_uppercase_ratio(text: str) -> float:
    characters = [character for character in str(text) if character.isalpha()]

    if not characters:
        return 0.0

    uppercase_count = sum(character.isupper() for character in characters)
    return round(uppercase_count / len(characters), 4)


def extract_availability_quantity(availability_text: str) -> int:
    match = re.search(r"\d+", str(availability_text))
    return int(match.group()) if match else 0


def assign_price_band(price: float) -> str:
    if pd.isna(price):
        return "Unknown"

    if price < 10:
        return "Under 10"
    if price < 20:
        return "10 to 19"
    if price < 30:
        return "20 to 29"
    if price < 40:
        return "30 to 39"
    if price < 50:
        return "40 to 49"

    return "50 and above"


def create_feature_dataset(dataframe: pd.DataFrame) -> pd.DataFrame:
    feature_creator = BookFeatureCreator()
    feature_dataframe = feature_creator.transform(dataframe)

    feature_dataframe[TARGET_COLUMN] = dataframe[TARGET_COLUMN].astype("int64").values

    return feature_dataframe


def build_preprocessing_pipeline() -> Pipeline:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    column_transformer = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )

    return Pipeline(
        steps=[
            ("feature_creator", BookFeatureCreator()),
            ("preprocessor", column_transformer),
            ("feature_selector", VarianceThreshold(threshold=0.0)),
        ]
    )


def create_feature_metadata(dataframe: pd.DataFrame) -> FeatureMetadata:
    return FeatureMetadata(
        input_path=str(CLEANED_DATA_PATH),
        feature_data_path=str(FEATURED_DATA_PATH),
        preprocessing_pipeline_path=str(PREPROCESSING_PIPELINE_PATH),
        row_count=len(dataframe),
        target_column=TARGET_COLUMN,
        excluded_columns=[
            "rating",
            "product_url",
        ],
        numeric_features=NUMERIC_FEATURES,
        categorical_features=CATEGORICAL_FEATURES,
        feature_engineering_notes=[
            "The raw rating column is excluded from model features because it defines the target.",
            "Product URL is excluded because it is an identifier-like field and is not suitable for generalizable prediction.",
            "Price is transformed using log1p to reduce skew.",
            "Book title is converted into length, word count, digit count, punctuation count, and uppercase ratio features.",
            "Categorical variables are handled through one-hot encoding inside the preprocessing pipeline.",
            "Numeric variables are median-imputed and standardized inside the preprocessing pipeline.",
            "VarianceThreshold is included for feature selection and will remove zero-variance features when fitted on training data.",
            "The preprocessing pipeline is intentionally saved unfitted to avoid train-test leakage.",
        ],
    )


def save_feature_dataset(dataframe: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(output_path, index=False)


def save_preprocessing_pipeline(pipeline: Pipeline, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, output_path)


def save_feature_metadata(metadata: FeatureMetadata, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(asdict(metadata), file, indent=2)


def main() -> None:
    configure_logging()

    cleaned_dataframe = load_cleaned_data(CLEANED_DATA_PATH)
    feature_dataframe = create_feature_dataset(cleaned_dataframe)
    preprocessing_pipeline = build_preprocessing_pipeline()
    metadata = create_feature_metadata(feature_dataframe)

    save_feature_dataset(feature_dataframe, FEATURED_DATA_PATH)
    save_preprocessing_pipeline(preprocessing_pipeline, PREPROCESSING_PIPELINE_PATH)
    save_feature_metadata(metadata, FEATURE_METADATA_PATH)

    logging.info("Rows processed: %s", len(feature_dataframe))
    logging.info("Numeric features: %s", len(NUMERIC_FEATURES))
    logging.info("Categorical features: %s", len(CATEGORICAL_FEATURES))
    logging.info("Saved feature dataset to %s", FEATURED_DATA_PATH)
    logging.info("Saved preprocessing pipeline to %s", PREPROCESSING_PIPELINE_PATH)
    logging.info("Saved feature metadata to %s", FEATURE_METADATA_PATH)


if __name__ == "__main__":
    main()
    
