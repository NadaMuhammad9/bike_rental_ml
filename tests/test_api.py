from __future__ import annotations

from typing import Any
from unittest.mock import Mock

from fastapi.testclient import TestClient
import pytest

from src.app.main import service


def test_health_endpoint(client: TestClient) -> None:
    """Test that the API health check endpoint returns status ok."""
    # Send a health check request to the API
    response = client.get("/health")

    # Verify the response status code and content
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_prediction_endpoint_success(client: TestClient) -> None:
    """Test that the prediction endpoint returns correct predictions for valid input."""
    # Mock the prediction service to return a fixed value
    service.predict = Mock(return_value=453)

    # Send a POST request with valid user input
    response = client.post(
        "/predict",
        json={
            "date": "2026-06-14",
            "hour": 14,
            "temperature": 25.4,
            "humidity": 72.0,
            "windspeed": 12.5,
            "visibility": 10.0,
            "dewpoint": 18.2,
            "seasons": "Summer",
            "holiday": "No Holiday",
            "functioning_day": "Yes",
            "previous_7_rented_bikes_per_hour": [
                320,
                305,
                290,
                310,
                325,
                300,
                315,
            ],
            "rented_bikes_same_hour_yesterday": 318,
        },
    )

    # Verify the response status code is successful
    assert response.status_code == 200

    # Verify the prediction value in the response
    assert response.json() == {"prediction": 453}

    # Verify the prediction service was called exactly once
    service.predict.assert_called_once()


def test_prediction_endpoint_invalid_hour(client: TestClient) -> None:
    """Test that the prediction endpoint rejects invalid hour values (must be 0-23)"""
    # Mock the prediction service
    service.predict = Mock(return_value=453)

    # Send a POST request with an invalid hour value (25 is out of range)
    response = client.post(
        "/predict",
        json={
            "date": "2026-06-14",
            "hour": 25,
            "temperature": 25.4,
            "humidity": 72.0,
            "windspeed": 12.5,
            "visibility": 10.0,
            "dewpoint": 18.2,
            "seasons": "Summer",
            "holiday": "No Holiday",
            "functioning_day": "Yes",
            "previous_7_rented_bikes_per_hour": [
                320,
                305,
                290,
                310,
                325,
                300,
                315,
            ],
            "rented_bikes_same_hour_yesterday": 318,
        },
    )

    # Verify that Pydantic validation rejects the invalid hour value
    assert response.status_code == 422


def test_prediction_endpoint_invalid_history_length(client: TestClient) -> None:
    """Test that the prediction endpoint rejects incorrect historical data lengths."""
    # Mock the prediction service
    service.predict = Mock(return_value=453)

    # Send a POST request with invalid history length (only 3 values instead of 7)
    response = client.post(
        "/predict",
        json={
            "date": "2026-06-14",
            "hour": 14,
            "temperature": 25.4,
            "humidity": 72.0,
            "windspeed": 12.5,
            "visibility": 10.0,
            "dewpoint": 18.2,
            "seasons": "Summer",
            "holiday": "No Holiday",
            "functioning_day": "Yes",
            "previous_7_rented_bikes_per_hour": [
                320,
                305,
                290,
            ],
            "rented_bikes_same_hour_yesterday": 318,
        },
    )

    # Verify that Pydantic validation rejects the request due to incorrect history length
    assert response.status_code == 422