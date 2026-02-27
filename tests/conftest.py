"""
Test Fixtures
==============
Shared pytest fixtures for testing the FastAPI backend.
Uses FastAPI's TestClient to make requests without starting a real server.
"""

import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient


@pytest.fixture
def mock_model():
    """
    Create a mock model and tokenizer so tests don't need a GPU.

    Returns a tuple of (mock_model, mock_tokenizer) that mimics
    the interface used by generate_code().
    """
    model = MagicMock()
    tokenizer = MagicMock()
    return model, tokenizer


@pytest.fixture
def client():
    """
    Create a FastAPI TestClient with mocked model loading.

    The model is replaced with a mock so tests run instantly
    without downloading or loading any model weights.
    """
    # Mock the model loader so the app starts without a real model
    with patch("backend.app.load_model") as mock_load:
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_load.return_value = (mock_model, mock_tokenizer)

        from backend.app import app
        with TestClient(app) as test_client:
            yield test_client


@pytest.fixture
def client_no_model():
    """
    Create a TestClient where the model fails to load.

    Used to test error handling when the model is unavailable.
    """
    with patch("backend.app.load_model") as mock_load:
        mock_load.side_effect = Exception("Model not found")

        from backend.app import app
        # Reset global model to None
        import backend.app as app_module
        app_module.model = None
        app_module.tokenizer = None

        with TestClient(app, raise_server_exceptions=False) as test_client:
            yield test_client
