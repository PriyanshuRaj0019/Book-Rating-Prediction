from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Final

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, field_validator


PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
INFERENCE_BUNDLE_PATH: Final[Path] = PROJECT_ROOT / "models" / "inference_bundle.joblib"
STATIC_DIR: Final[Path] = PROJECT_ROOT / "static"
INDEX_HTML_PATH: Final[Path] = STATIC_DIR / "index.html"

PREDICTION_LABELS: Final[dict[int, str]] = {
    0: "low_rating",
    1: "high_rating",
}

app_state: dict[str, Any] = {}


class BookPredictionRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    book_title: str = Field(..., min_length=1, max_length=300)
    book_category: str = Field(..., min_length=1, max_length=100)
    price_gbp: float = Field(..., gt=0, le=1_000)
    availability_status: str = Field(..., min_length=1, max_length=150)

    @field_validator("book_title", "book_category", "availability_status")
    @classmethod
    def reject_blank_text(cls, value: str) -> str:
        cleaned_value = " ".join(value.split())

        if not cleaned_value:
            raise ValueError("Text fields cannot be blank.")

        return cleaned_value


class BookPredictionResponse(BaseModel):
    prediction: int
    prediction_label: str
    high_rating_probability: float | None
    model_version: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    bundle_path: str


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def load_inference_bundle(bundle_path: Path) -> dict[str, Any]:
    if not bundle_path.exists():
        raise FileNotFoundError(f"Inference bundle not found: {bundle_path}")

    bundle = joblib.load(bundle_path)

    if not isinstance(bundle, dict):
        raise TypeError("Inference bundle must be a dictionary.")

    required_keys = {
        "model",
        "preprocessing_pipeline",
        "input_columns",
        "prediction_labels",
        "bundle_version",
    }

    missing_keys = required_keys - set(bundle)
    if missing_keys:
        raise ValueError(f"Inference bundle is missing keys: {sorted(missing_keys)}")

    return bundle


def create_input_dataframe(
    request: BookPredictionRequest,
    input_columns: list[str],
) -> pd.DataFrame:
    input_payload = request.model_dump()
    dataframe = pd.DataFrame([input_payload])

    missing_columns = set(input_columns) - set(dataframe.columns)
    if missing_columns:
        raise ValueError(f"Prediction input is missing columns: {sorted(missing_columns)}")

    return dataframe[input_columns].copy()


def predict_high_rating(
    bundle: dict[str, Any],
    request: BookPredictionRequest,
) -> BookPredictionResponse:
    model = bundle["model"]
    preprocessing_pipeline = bundle["preprocessing_pipeline"]
    input_columns = bundle["input_columns"]

    input_dataframe = create_input_dataframe(request, input_columns)
    processed_features = preprocessing_pipeline.transform(input_dataframe)

    prediction = int(model.predict(processed_features)[0])

    high_rating_probability: float | None = None
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(processed_features)
        if probabilities.shape[1] >= 2:
            high_rating_probability = round(float(probabilities[0][1]), 6)

    return BookPredictionResponse(
        prediction=prediction,
        prediction_label=PREDICTION_LABELS.get(prediction, "unknown"),
        high_rating_probability=high_rating_probability,
        model_version=str(bundle["bundle_version"]),
    )


@asynccontextmanager
async def lifespan(_: FastAPI):
    configure_logging()

    try:
        app_state["inference_bundle"] = load_inference_bundle(INFERENCE_BUNDLE_PATH)
        logging.info("Inference bundle loaded from %s", INFERENCE_BUNDLE_PATH)
    except Exception as exc:
        app_state["inference_bundle"] = None
        logging.exception("Failed to load inference bundle: %s", exc)

    yield

    app_state.clear()


app = FastAPI(
    title="Book Rating Prediction API",
    description="FastAPI service for predicting whether a book is likely to be high-rated.",
    version="1.0.0",
    lifespan=lifespan,
)

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
def serve_frontend() -> FileResponse:
    if not INDEX_HTML_PATH.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Frontend file not found. Create static/index.html first.",
        )

    return FileResponse(INDEX_HTML_PATH)


@app.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    return HealthResponse(
        status="ok",
        model_loaded=app_state.get("inference_bundle") is not None,
        bundle_path=str(INFERENCE_BUNDLE_PATH),
    )


@app.post("/predict", response_model=BookPredictionResponse)
def predict(request: BookPredictionRequest) -> BookPredictionResponse:
    bundle = app_state.get("inference_bundle")

    if bundle is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Inference bundle is not loaded. Run Phase 11 first and restart the API.",
        )

    try:
        return predict_high_rating(bundle, request)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logging.exception("Prediction failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Prediction failed due to an internal server error.",
        ) from exc
        
