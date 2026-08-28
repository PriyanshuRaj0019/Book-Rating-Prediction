from __future__ import annotations

import json
import logging
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.features.build_features import TARGET_COLUMN


PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]

CLEANED_DATA_PATH: Final[Path] = PROJECT_ROOT / "data" / "processed" / "books_cleaned.csv"
FINAL_MODEL_PATH: Final[Path] = PROJECT_ROOT / "models" / "final_model.joblib"
FITTED_PREPROCESSING_PIPELINE_PATH: Final[Path] = PROJECT_ROOT / "models" / "fitted_preprocessing_pipeline.joblib"

INFERENCE_BUNDLE_JOBLIB_PATH: Final[Path] = PROJECT_ROOT / "models" / "inference_bundle.joblib"
INFERENCE_BUNDLE_PICKLE_PATH: Final[Path] = PROJECT_ROOT / "models" / "inference_bundle.pkl"
PERSISTENCE_METADATA_PATH: Final[Path] = PROJECT_ROOT / "data" / "processed" / "persistence_metadata.json"

MODEL_INPUT_COLUMNS: Final[list[str]] = [
    "book_title",
    "book_category",
    "price_gbp",
    "availability_status",
]

REQUIRED_CLEANED_COLUMNS: Final[list[str]] = [
    "book_title",
    "book_category",
    "price_gbp",
    "rating",
    "availability_status",
    "product_url",
    TARGET_COLUMN,
]


@dataclass(frozen=True)
class PersistenceMetadata:
    model_path: str
    preprocessing_pipeline_path: str
    joblib_bundle_path: str
    pickle_bundle_path: str
    input_columns: list[str]
    scaler_saved_inside_pipeline: bool
    encoder_saved_inside_pipeline: bool
    joblib_load_test_passed: bool
    pickle_load_test_passed: bool
    sample_prediction: int
    sample_prediction_probability: float | None
    notes: list[str]


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def load_joblib_file(file_path: Path) -> Any:
    if not file_path.exists():
        raise FileNotFoundError(f"Required file not found: {file_path}")

    return joblib.load(file_path)


def load_cleaned_data(file_path: Path) -> pd.DataFrame:
    if not file_path.exists():
        raise FileNotFoundError(f"Cleaned dataset not found: {file_path}")

    dataframe = pd.read_csv(file_path)

    missing_columns = set(REQUIRED_CLEANED_COLUMNS) - set(dataframe.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    return dataframe[REQUIRED_CLEANED_COLUMNS].copy()


def validate_preprocessing_pipeline(preprocessing_pipeline: Pipeline) -> tuple[bool, bool]:
    if not isinstance(preprocessing_pipeline, Pipeline):
        raise TypeError("Preprocessing artifact must be a fitted sklearn Pipeline.")

    required_steps = {
        "feature_creator",
        "preprocessor",
        "feature_selector",
    }

    missing_steps = required_steps - set(preprocessing_pipeline.named_steps)
    if missing_steps:
        raise ValueError(f"Preprocessing pipeline is missing steps: {sorted(missing_steps)}")

    preprocessor = preprocessing_pipeline.named_steps["preprocessor"]

    if not hasattr(preprocessor, "named_transformers_"):
        raise ValueError("ColumnTransformer is not fitted. Missing named_transformers_.")

    numeric_pipeline = preprocessor.named_transformers_.get("numeric")
    categorical_pipeline = preprocessor.named_transformers_.get("categorical")

    if numeric_pipeline is None:
        raise ValueError("Fitted preprocessing pipeline is missing numeric transformer.")

    if categorical_pipeline is None:
        raise ValueError("Fitted preprocessing pipeline is missing categorical transformer.")

    scaler_exists = "scaler" in numeric_pipeline.named_steps
    encoder_exists = "encoder" in categorical_pipeline.named_steps

    if not scaler_exists:
        raise ValueError("Numeric preprocessing pipeline is missing fitted scaler.")

    if not encoder_exists:
        raise ValueError("Categorical preprocessing pipeline is missing fitted encoder.")

    return scaler_exists, encoder_exists


def build_inference_bundle(model: Any, preprocessing_pipeline: Pipeline) -> dict[str, Any]:
    return {
        "model": model,
        "preprocessing_pipeline": preprocessing_pipeline,
        "input_columns": MODEL_INPUT_COLUMNS,
        "target_column": TARGET_COLUMN,
        "prediction_labels": {
            0: "low_rating",
            1: "high_rating",
        },
        "bundle_version": "1.0.0",
    }


def save_joblib_bundle(bundle: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, output_path)


def save_pickle_bundle(bundle: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("wb") as file:
        pickle.dump(bundle, file, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle_bundle(file_path: Path) -> dict[str, Any]:
    if not file_path.exists():
        raise FileNotFoundError(f"Pickle bundle not found: {file_path}")

    with file_path.open("rb") as file:
        payload = pickle.load(file)

    if not isinstance(payload, dict):
        raise TypeError("Pickle bundle must contain a dictionary payload.")

    return payload


def create_sample_input(dataframe: pd.DataFrame) -> pd.DataFrame:
    sample = dataframe[MODEL_INPUT_COLUMNS].head(1).copy()

    if sample.empty:
        raise ValueError("Cannot create sample input from an empty dataset.")

    return sample


def predict_with_bundle(bundle: dict[str, Any], input_dataframe: pd.DataFrame) -> tuple[int, float | None]:
    missing_columns = set(bundle["input_columns"]) - set(input_dataframe.columns)
    if missing_columns:
        raise ValueError(f"Inference input is missing columns: {sorted(missing_columns)}")

    model = bundle["model"]
    preprocessing_pipeline = bundle["preprocessing_pipeline"]

    processed_features = preprocessing_pipeline.transform(input_dataframe[bundle["input_columns"]])
    prediction = int(model.predict(processed_features)[0])

    probability: float | None = None
    if hasattr(model, "predict_proba"):
        predicted_probabilities = model.predict_proba(processed_features)
        if predicted_probabilities.shape[1] >= 2:
            probability = round(float(predicted_probabilities[0][1]), 6)

    return prediction, probability


def validate_loaded_bundle(bundle: dict[str, Any], sample_input: pd.DataFrame) -> tuple[int, float | None]:
    required_keys = {
        "model",
        "preprocessing_pipeline",
        "input_columns",
        "target_column",
        "prediction_labels",
        "bundle_version",
    }

    missing_keys = required_keys - set(bundle)
    if missing_keys:
        raise ValueError(f"Inference bundle is missing keys: {sorted(missing_keys)}")

    return predict_with_bundle(bundle, sample_input)


def save_metadata(metadata: PersistenceMetadata, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(asdict(metadata), file, indent=2)


def main() -> None:
    configure_logging()

    cleaned_dataframe = load_cleaned_data(CLEANED_DATA_PATH)
    final_model = load_joblib_file(FINAL_MODEL_PATH)
    preprocessing_pipeline = load_joblib_file(FITTED_PREPROCESSING_PIPELINE_PATH)

    scaler_exists, encoder_exists = validate_preprocessing_pipeline(preprocessing_pipeline)

    inference_bundle = build_inference_bundle(
        model=final_model,
        preprocessing_pipeline=preprocessing_pipeline,
    )

    save_joblib_bundle(inference_bundle, INFERENCE_BUNDLE_JOBLIB_PATH)
    save_pickle_bundle(inference_bundle, INFERENCE_BUNDLE_PICKLE_PATH)

    sample_input = create_sample_input(cleaned_dataframe)

    loaded_joblib_bundle = load_joblib_file(INFERENCE_BUNDLE_JOBLIB_PATH)
    joblib_prediction, joblib_probability = validate_loaded_bundle(
        loaded_joblib_bundle,
        sample_input,
    )

    loaded_pickle_bundle = load_pickle_bundle(INFERENCE_BUNDLE_PICKLE_PATH)
    pickle_prediction, _ = validate_loaded_bundle(
        loaded_pickle_bundle,
        sample_input,
    )

    if joblib_prediction != pickle_prediction:
        raise ValueError("Joblib and pickle bundles produced different predictions.")

    metadata = PersistenceMetadata(
        model_path=str(FINAL_MODEL_PATH),
        preprocessing_pipeline_path=str(FITTED_PREPROCESSING_PIPELINE_PATH),
        joblib_bundle_path=str(INFERENCE_BUNDLE_JOBLIB_PATH),
        pickle_bundle_path=str(INFERENCE_BUNDLE_PICKLE_PATH),
        input_columns=MODEL_INPUT_COLUMNS,
        scaler_saved_inside_pipeline=scaler_exists,
        encoder_saved_inside_pipeline=encoder_exists,
        joblib_load_test_passed=True,
        pickle_load_test_passed=True,
        sample_prediction=joblib_prediction,
        sample_prediction_probability=joblib_probability,
        notes=[
            "The final estimator and fitted preprocessing pipeline are saved together for safe inference.",
            "The fitted scaler is stored inside the numeric preprocessing pipeline.",
            "The fitted encoder is stored inside the categorical preprocessing pipeline.",
            "Joblib is the preferred artifact for this project because it is standard for scikit-learn objects.",
            "Pickle is saved only to satisfy persistence-format coverage and should be used carefully.",
            "Both saved bundles were loaded back and tested with a real sample input.",
        ],
    )

    save_metadata(metadata, PERSISTENCE_METADATA_PATH)

    logging.info("Scaler saved inside pipeline: %s", scaler_exists)
    logging.info("Encoder saved inside pipeline: %s", encoder_exists)
    logging.info("Sample prediction: %s", metadata.sample_prediction)
    logging.info("Sample prediction probability: %s", metadata.sample_prediction_probability)
    logging.info("Saved joblib inference bundle to %s", INFERENCE_BUNDLE_JOBLIB_PATH)
    logging.info("Saved pickle inference bundle to %s", INFERENCE_BUNDLE_PICKLE_PATH)
    logging.info("Saved persistence metadata to %s", PERSISTENCE_METADATA_PATH)


if __name__ == "__main__":
    main()
