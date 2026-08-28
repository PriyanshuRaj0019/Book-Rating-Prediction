from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final

import pandas as pd


PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]

RAW_DATA_PATH: Final[Path] = PROJECT_ROOT / "data" / "raw" / "books.csv"
PROCESSED_DATA_DIR: Final[Path] = PROJECT_ROOT / "data" / "processed"
CLEANED_DATA_PATH: Final[Path] = PROCESSED_DATA_DIR / "books_cleaned.csv"
CLEANING_METADATA_PATH: Final[Path] = PROCESSED_DATA_DIR / "cleaning_metadata.json"

EXPECTED_COLUMNS: Final[list[str]] = [
    "title",
    "category",
    "price_gbp",
    "rating",
    "availability",
    "product_url",
]

COLUMN_RENAME_MAP: Final[dict[str, str]] = {
    "title": "book_title",
    "category": "book_category",
    "price_gbp": "price_gbp",
    "rating": "rating",
    "availability": "availability_status",
    "product_url": "product_url",
}

MIN_RATING: Final[int] = 1
MAX_RATING: Final[int] = 5
HIGH_RATING_THRESHOLD: Final[int] = 4
OUTLIER_IQR_MULTIPLIER: Final[float] = 1.5


@dataclass(frozen=True)
class CleaningMetadata:
    input_path: str
    output_path: str
    rows_before_cleaning: int
    rows_after_cleaning: int
    duplicate_rows_removed: int
    rows_removed_missing_required_values: int
    rows_removed_invalid_values: int
    rows_removed_price_outliers: int
    target_column: str
    target_definition: str
    final_columns: list[str]


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def load_raw_data(file_path: Path) -> pd.DataFrame:
    if not file_path.exists():
        raise FileNotFoundError(f"Raw data file not found: {file_path}")

    dataframe = pd.read_csv(file_path)

    missing_columns = set(EXPECTED_COLUMNS) - set(dataframe.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    return dataframe[EXPECTED_COLUMNS].copy()


def standardize_column_names(dataframe: pd.DataFrame) -> pd.DataFrame:
    return dataframe.rename(columns=COLUMN_RENAME_MAP)


def normalize_text_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    text_columns = [
        "book_title",
        "book_category",
        "availability_status",
        "product_url",
    ]

    cleaned_dataframe = dataframe.copy()

    for column in text_columns:
        cleaned_dataframe[column] = (
            cleaned_dataframe[column]
            .astype("string")
            .str.strip()
            .str.replace(r"\s+", " ", regex=True)
        )

    return cleaned_dataframe


def enforce_data_types(dataframe: pd.DataFrame) -> pd.DataFrame:
    typed_dataframe = dataframe.copy()

    typed_dataframe["book_title"] = typed_dataframe["book_title"].astype("string")
    typed_dataframe["book_category"] = typed_dataframe["book_category"].astype("string")
    typed_dataframe["availability_status"] = typed_dataframe["availability_status"].astype("string")
    typed_dataframe["product_url"] = typed_dataframe["product_url"].astype("string")

    typed_dataframe["price_gbp"] = pd.to_numeric(
        typed_dataframe["price_gbp"],
        errors="coerce",
    )

    typed_dataframe["rating"] = pd.to_numeric(
        typed_dataframe["rating"],
        errors="coerce",
    ).astype("Int64")

    return typed_dataframe


def remove_missing_required_values(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    required_columns = [
        "book_title",
        "book_category",
        "price_gbp",
        "rating",
        "availability_status",
        "product_url",
    ]

    rows_before = len(dataframe)
    cleaned_dataframe = dataframe.dropna(subset=required_columns).copy()
    rows_removed = rows_before - len(cleaned_dataframe)

    return cleaned_dataframe, rows_removed


def remove_duplicates(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    rows_before = len(dataframe)

    cleaned_dataframe = dataframe.drop_duplicates(
        subset=["book_title", "product_url"],
        keep="first",
    ).copy()

    rows_removed = rows_before - len(cleaned_dataframe)

    return cleaned_dataframe, rows_removed


def remove_invalid_values(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    rows_before = len(dataframe)

    valid_price_mask = dataframe["price_gbp"] > 0
    valid_rating_mask = dataframe["rating"].between(MIN_RATING, MAX_RATING)
    valid_url_mask = dataframe["product_url"].str.startswith("https://books.toscrape.com/", na=False)
    valid_title_mask = dataframe["book_title"].str.len() > 0

    cleaned_dataframe = dataframe[
        valid_price_mask
        & valid_rating_mask
        & valid_url_mask
        & valid_title_mask
    ].copy()

    rows_removed = rows_before - len(cleaned_dataframe)

    return cleaned_dataframe, rows_removed


def remove_price_outliers(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    rows_before = len(dataframe)

    first_quartile = dataframe["price_gbp"].quantile(0.25)
    third_quartile = dataframe["price_gbp"].quantile(0.75)
    interquartile_range = third_quartile - first_quartile

    lower_bound = first_quartile - OUTLIER_IQR_MULTIPLIER * interquartile_range
    upper_bound = third_quartile + OUTLIER_IQR_MULTIPLIER * interquartile_range

    cleaned_dataframe = dataframe[
        dataframe["price_gbp"].between(lower_bound, upper_bound)
    ].copy()

    rows_removed = rows_before - len(cleaned_dataframe)

    return cleaned_dataframe, rows_removed


def create_target_column(dataframe: pd.DataFrame) -> pd.DataFrame:
    transformed_dataframe = dataframe.copy()

    transformed_dataframe["is_high_rating"] = (
        transformed_dataframe["rating"] >= HIGH_RATING_THRESHOLD
    ).astype("int64")

    return transformed_dataframe


def finalize_dataset(dataframe: pd.DataFrame) -> pd.DataFrame:
    final_columns = [
        "book_title",
        "book_category",
        "price_gbp",
        "rating",
        "availability_status",
        "product_url",
        "is_high_rating",
    ]

    finalized_dataframe = dataframe[final_columns].copy()
    finalized_dataframe = finalized_dataframe.sort_values(
        by=["book_title", "product_url"],
        ascending=True,
    ).reset_index(drop=True)

    finalized_dataframe["rating"] = finalized_dataframe["rating"].astype("int64")
    finalized_dataframe["price_gbp"] = finalized_dataframe["price_gbp"].round(2)

    return finalized_dataframe


def clean_books_data(raw_dataframe: pd.DataFrame) -> tuple[pd.DataFrame, CleaningMetadata]:
    rows_before_cleaning = len(raw_dataframe)

    cleaned_dataframe = standardize_column_names(raw_dataframe)
    cleaned_dataframe = normalize_text_columns(cleaned_dataframe)
    cleaned_dataframe = enforce_data_types(cleaned_dataframe)

    cleaned_dataframe, rows_removed_missing = remove_missing_required_values(cleaned_dataframe)
    cleaned_dataframe, duplicate_rows_removed = remove_duplicates(cleaned_dataframe)
    cleaned_dataframe, rows_removed_invalid = remove_invalid_values(cleaned_dataframe)
    cleaned_dataframe, rows_removed_outliers = remove_price_outliers(cleaned_dataframe)

    cleaned_dataframe = create_target_column(cleaned_dataframe)
    cleaned_dataframe = finalize_dataset(cleaned_dataframe)

    metadata = CleaningMetadata(
        input_path=str(RAW_DATA_PATH),
        output_path=str(CLEANED_DATA_PATH),
        rows_before_cleaning=rows_before_cleaning,
        rows_after_cleaning=len(cleaned_dataframe),
        duplicate_rows_removed=duplicate_rows_removed,
        rows_removed_missing_required_values=rows_removed_missing,
        rows_removed_invalid_values=rows_removed_invalid,
        rows_removed_price_outliers=rows_removed_outliers,
        target_column="is_high_rating",
        target_definition="1 if rating is greater than or equal to 4, otherwise 0",
        final_columns=list(cleaned_dataframe.columns),
    )

    return cleaned_dataframe, metadata


def save_cleaned_data(dataframe: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(output_path, index=False)


def save_cleaning_metadata(metadata: CleaningMetadata, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(asdict(metadata), file, indent=2)


def main() -> None:
    configure_logging()

    raw_dataframe = load_raw_data(RAW_DATA_PATH)
    cleaned_dataframe, metadata = clean_books_data(raw_dataframe)

    save_cleaned_data(cleaned_dataframe, CLEANED_DATA_PATH)
    save_cleaning_metadata(metadata, CLEANING_METADATA_PATH)

    logging.info("Rows before cleaning: %s", metadata.rows_before_cleaning)
    logging.info("Rows after cleaning: %s", metadata.rows_after_cleaning)
    logging.info("Duplicate rows removed: %s", metadata.duplicate_rows_removed)
    logging.info("Rows removed with missing values: %s", metadata.rows_removed_missing_required_values)
    logging.info("Rows removed with invalid values: %s", metadata.rows_removed_invalid_values)
    logging.info("Rows removed as price outliers: %s", metadata.rows_removed_price_outliers)
    logging.info("Saved cleaned dataset to %s", CLEANED_DATA_PATH)
    logging.info("Saved cleaning metadata to %s", CLEANING_METADATA_PATH)


if __name__ == "__main__":
    main()
