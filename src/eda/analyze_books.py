from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]

CLEANED_DATA_PATH: Final[Path] = PROJECT_ROOT / "data" / "processed" / "books_cleaned.csv"
REPORTS_DOCS_DIR: Final[Path] = PROJECT_ROOT / "reports" / "docs"
FIGURES_DIR: Final[Path] = PROJECT_ROOT / "reports" / "figures"

EDA_SUMMARY_PATH: Final[Path] = REPORTS_DOCS_DIR / "eda_summary.json"

PRICE_DISTRIBUTION_PATH: Final[Path] = FIGURES_DIR / "price_distribution.png"
RATING_DISTRIBUTION_PATH: Final[Path] = FIGURES_DIR / "rating_distribution.png"
TARGET_DISTRIBUTION_PATH: Final[Path] = FIGURES_DIR / "target_distribution.png"
PRICE_BY_RATING_PATH: Final[Path] = FIGURES_DIR / "price_by_rating.png"
CORRELATION_HEATMAP_PATH: Final[Path] = FIGURES_DIR / "correlation_heatmap.png"

REQUIRED_COLUMNS: Final[list[str]] = [
    "book_title",
    "book_category",
    "price_gbp",
    "rating",
    "availability_status",
    "product_url",
    "is_high_rating",
]

NUMERIC_COLUMNS: Final[list[str]] = [
    "price_gbp",
    "rating",
    "is_high_rating",
]


@dataclass(frozen=True)
class PriceSummary:
    minimum: float
    maximum: float
    mean: float
    median: float
    standard_deviation: float


@dataclass(frozen=True)
class TargetSummary:
    high_rating_count: int
    low_rating_count: int
    high_rating_percentage: float
    low_rating_percentage: float


@dataclass(frozen=True)
class EdaSummary:
    input_path: str
    row_count: int
    column_count: int
    duplicate_rows: int
    missing_values_by_column: dict[str, int]
    price_summary: PriceSummary
    rating_distribution: dict[str, int]
    target_summary: TargetSummary
    correlation_matrix: dict[str, dict[str, float]]
    saved_figures: list[str]
    business_observations: list[str]


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


def create_price_summary(dataframe: pd.DataFrame) -> PriceSummary:
    price_series = dataframe["price_gbp"]

    return PriceSummary(
        minimum=round(float(price_series.min()), 2),
        maximum=round(float(price_series.max()), 2),
        mean=round(float(price_series.mean()), 2),
        median=round(float(price_series.median()), 2),
        standard_deviation=round(float(price_series.std()), 2),
    )


def create_target_summary(dataframe: pd.DataFrame) -> TargetSummary:
    target_counts = dataframe["is_high_rating"].value_counts().to_dict()

    high_rating_count = int(target_counts.get(1, 0))
    low_rating_count = int(target_counts.get(0, 0))
    total_count = len(dataframe)

    return TargetSummary(
        high_rating_count=high_rating_count,
        low_rating_count=low_rating_count,
        high_rating_percentage=round((high_rating_count / total_count) * 100, 2),
        low_rating_percentage=round((low_rating_count / total_count) * 100, 2),
    )


def create_correlation_matrix(dataframe: pd.DataFrame) -> dict[str, dict[str, float]]:
    correlation_dataframe = dataframe[NUMERIC_COLUMNS].corr(numeric_only=True)
    rounded_correlation = correlation_dataframe.round(4)

    return {
        column: {
            index: float(value)
            for index, value in rounded_correlation[column].items()
        }
        for column in rounded_correlation.columns
    }


def save_price_distribution(dataframe: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(10, 6))
    sns.histplot(dataframe["price_gbp"], bins=30, kde=True)
    plt.title("Book Price Distribution")
    plt.xlabel("Price in GBP")
    plt.ylabel("Book Count")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def save_rating_distribution(dataframe: pd.DataFrame, output_path: Path) -> None:
    rating_counts = dataframe["rating"].value_counts().sort_index()

    plt.figure(figsize=(8, 6))
    sns.barplot(x=rating_counts.index, y=rating_counts.values)
    plt.title("Book Rating Distribution")
    plt.xlabel("Rating")
    plt.ylabel("Book Count")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def save_target_distribution(dataframe: pd.DataFrame, output_path: Path) -> None:
    target_counts = dataframe["is_high_rating"].value_counts().sort_index()
    target_labels = ["Low Rating", "High Rating"]

    plt.figure(figsize=(8, 6))
    sns.barplot(x=target_labels, y=target_counts.values)
    plt.title("Target Distribution")
    plt.xlabel("Target Class")
    plt.ylabel("Book Count")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def save_price_by_rating(dataframe: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=dataframe, x="rating", y="price_gbp")
    plt.title("Book Price by Rating")
    plt.xlabel("Rating")
    plt.ylabel("Price in GBP")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def save_correlation_heatmap(dataframe: pd.DataFrame, output_path: Path) -> None:
    correlation_dataframe = dataframe[NUMERIC_COLUMNS].corr(numeric_only=True)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        correlation_dataframe,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        square=True,
        linewidths=0.5,
    )
    plt.title("Numeric Feature Correlation")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def save_visualizations(dataframe: pd.DataFrame) -> list[Path]:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    save_price_distribution(dataframe, PRICE_DISTRIBUTION_PATH)
    save_rating_distribution(dataframe, RATING_DISTRIBUTION_PATH)
    save_target_distribution(dataframe, TARGET_DISTRIBUTION_PATH)
    save_price_by_rating(dataframe, PRICE_BY_RATING_PATH)
    save_correlation_heatmap(dataframe, CORRELATION_HEATMAP_PATH)

    return [
        PRICE_DISTRIBUTION_PATH,
        RATING_DISTRIBUTION_PATH,
        TARGET_DISTRIBUTION_PATH,
        PRICE_BY_RATING_PATH,
        CORRELATION_HEATMAP_PATH,
    ]


def create_business_observations(
    dataframe: pd.DataFrame,
    price_summary: PriceSummary,
    target_summary: TargetSummary,
) -> list[str]:
    most_common_rating = int(dataframe["rating"].mode().iloc[0])

    return [
        f"The dataset contains {len(dataframe)} scraped book records after cleaning.",
        f"Book prices range from £{price_summary.minimum} to £{price_summary.maximum}.",
        f"The average book price is £{price_summary.mean}, while the median price is £{price_summary.median}.",
        f"The most common rating in the dataset is {most_common_rating}.",
        f"High-rated books represent {target_summary.high_rating_percentage}% of the dataset.",
        "The dataset is suitable for binary classification using the target is_high_rating.",
        "Raw rating should not be used as a model feature later because it directly defines the target.",
        "The source is a scraping sandbox, so insights are educational and not real market conclusions.",
    ]


def create_eda_summary(dataframe: pd.DataFrame, saved_figures: list[Path]) -> EdaSummary:
    price_summary = create_price_summary(dataframe)
    target_summary = create_target_summary(dataframe)

    rating_distribution = {
        str(rating): int(count)
        for rating, count in dataframe["rating"].value_counts().sort_index().items()
    }

    missing_values_by_column = {
        column: int(count)
        for column, count in dataframe.isna().sum().items()
    }

    return EdaSummary(
        input_path=str(CLEANED_DATA_PATH),
        row_count=len(dataframe),
        column_count=len(dataframe.columns),
        duplicate_rows=int(dataframe.duplicated().sum()),
        missing_values_by_column=missing_values_by_column,
        price_summary=price_summary,
        rating_distribution=rating_distribution,
        target_summary=target_summary,
        correlation_matrix=create_correlation_matrix(dataframe),
        saved_figures=[str(path) for path in saved_figures],
        business_observations=create_business_observations(
            dataframe=dataframe,
            price_summary=price_summary,
            target_summary=target_summary,
        ),
    )


def save_eda_summary(summary: EdaSummary, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(asdict(summary), file, indent=2)


def main() -> None:
    configure_logging()

    dataframe = load_cleaned_data(CLEANED_DATA_PATH)
    saved_figures = save_visualizations(dataframe)
    eda_summary = create_eda_summary(dataframe, saved_figures)
    save_eda_summary(eda_summary, EDA_SUMMARY_PATH)

    logging.info("Rows analyzed: %s", eda_summary.row_count)
    logging.info("Columns analyzed: %s", eda_summary.column_count)
    logging.info("High rating percentage: %s%%", eda_summary.target_summary.high_rating_percentage)
    logging.info("Saved EDA summary to %s", EDA_SUMMARY_PATH)

    for figure_path in saved_figures:
        logging.info("Saved figure to %s", figure_path)


if __name__ == "__main__":
    main()
