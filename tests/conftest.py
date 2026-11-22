from __future__ import annotations

from fastapi.testclient import TestClient
import pytest

from src.app.main import app


# Shared pytest fixtures for test suite


@pytest.fixture
def client() -> TestClient:
    """Provide a FastAPI test client for API endpoint testing.
    
    This fixture creates a TestClient instance connected to the FastAPI app,
    allowing tests to make HTTP requests to endpoints without starting a server.
    
    Yields:
        TestClient: A test client connected to the FastAPI application.
    """
    return TestClient(app)
