from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from src.api.app import app


VALID_PAYLOAD: dict[str, Any] = {
    "book_title": "A Light in the Attic",
    "book_category": "Books",
    "price_gbp": 51.77,
    "availability_status": "In stock",
}


@pytest.fixture()
def client() -> TestClient:
    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint_returns_model_status(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200

    payload = response.json()

    assert payload["status"] == "ok"
    assert payload["model_loaded"] is True
    assert "inference_bundle.joblib" in payload["bundle_path"]


def test_prediction_endpoint_accepts_valid_payload(client: TestClient) -> None:
    response = client.post("/predict", json=VALID_PAYLOAD)

    assert response.status_code == 200

    payload = response.json()

    assert payload["prediction"] in [0, 1]
    assert payload["prediction_label"] in ["low_rating", "high_rating"]
    assert payload["model_version"] == "1.0.0"

    if payload["high_rating_probability"] is not None:
        assert 0 <= payload["high_rating_probability"] <= 1


@pytest.mark.parametrize(
    "payload",
    [
        {
            "book_title": "The Silent Patient",
            "book_category": "Mystery",
            "price_gbp": 24.99,
            "availability_status": "In stock",
        },
        {
            "book_title": "Python Machine Learning",
            "book_category": "Technology",
            "price_gbp": 42.50,
            "availability_status": "Only 3 left in stock",
        },
        {
            "book_title": "Deep Work",
            "book_category": "Nonfiction",
            "price_gbp": 18.75,
            "availability_status": "In stock",
        },
    ],
)
def test_prediction_endpoint_handles_multiple_valid_inputs(
    client: TestClient,
    payload: dict[str, Any],
) -> None:
    response = client.post("/predict", json=payload)

    assert response.status_code == 200

    response_payload = response.json()

    assert response_payload["prediction"] in [0, 1]
    assert response_payload["prediction_label"] in ["low_rating", "high_rating"]


@pytest.mark.parametrize("invalid_price", [0, -5, 1000.01])
def test_prediction_endpoint_rejects_invalid_price(
    client: TestClient,
    invalid_price: float,
) -> None:
    payload = VALID_PAYLOAD | {"price_gbp": invalid_price}

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


@pytest.mark.parametrize(
    "field_name",
    [
        "book_title",
        "book_category",
        "availability_status",
    ],
)
def test_prediction_endpoint_rejects_blank_text_fields(
    client: TestClient,
    field_name: str,
) -> None:
    payload = VALID_PAYLOAD | {field_name: "   "}

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_prediction_endpoint_rejects_missing_required_field(client: TestClient) -> None:
    payload = VALID_PAYLOAD.copy()
    payload.pop("book_title")

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_prediction_endpoint_rejects_wrong_price_type(client: TestClient) -> None:
    payload = VALID_PAYLOAD | {"price_gbp": "not-a-number"}

    response = client.post("/predict", json=payload)

    assert response.status_code == 422
    
