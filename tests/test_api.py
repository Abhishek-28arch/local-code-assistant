"""
API Tests
==========
Tests for the FastAPI backend endpoints using pytest.

These tests use mocked models so they run instantly without
a GPU or model weights. They verify:
  - Health check returns correct status
  - Generate endpoint accepts valid requests
  - Generate endpoint rejects invalid requests
  - Error handling works correctly
  - Response schema matches expectations

Usage:
    pytest tests/ -v
"""

from unittest.mock import patch


# ── Health Check Tests ───────────────────────────────────────────────
class TestHealthEndpoint:
    """Tests for GET /api/health."""

    def test_health_returns_200(self, client):
        """Health endpoint should return 200 with status info."""
        response = client.get("/api/health")
        assert response.status_code == 200

    def test_health_response_fields(self, client):
        """Health response should contain required fields."""
        data = client.get("/api/health").json()
        assert "status" in data
        assert "model_loaded" in data
        assert data["status"] == "healthy"

    def test_health_shows_version(self, client):
        """Health response should include version info."""
        data = client.get("/api/health").json()
        assert "version" in data


# ── Generate Endpoint Tests ──────────────────────────────────────────
class TestGenerateEndpoint:
    """Tests for POST /api/generate."""

    def test_generate_accepts_valid_request(self, client):
        """Valid prompt should return 200."""
        with patch("backend.app.generate_code", return_value="def foo(): pass"):
            response = client.post("/api/generate", json={
                "prompt": "Write a hello world function",
            })
            assert response.status_code == 200

    def test_generate_response_schema(self, client):
        """Response should match GenerateResponse schema."""
        with patch("backend.app.generate_code", return_value="def foo(): pass"):
            data = client.post("/api/generate", json={
                "prompt": "Write a hello world function",
            }).json()

            assert "generated_code" in data
            assert "prompt" in data
            assert "model_name" in data
            assert "generation_time_ms" in data

    def test_generate_returns_prompt_echo(self, client):
        """Response should echo back the original prompt."""
        prompt = "Write a binary search function"
        with patch("backend.app.generate_code", return_value="def search(): pass"):
            data = client.post("/api/generate", json={
                "prompt": prompt,
            }).json()
            assert data["prompt"] == prompt

    def test_generate_custom_parameters(self, client):
        """Custom max_length and temperature should be accepted."""
        with patch("backend.app.generate_code", return_value="code"):
            response = client.post("/api/generate", json={
                "prompt": "Write code",
                "max_length": 512,
                "temperature": 0.5,
            })
            assert response.status_code == 200


# ── Validation Tests ─────────────────────────────────────────────────
class TestInputValidation:
    """Tests for request validation."""

    def test_empty_prompt_rejected(self, client):
        """Empty prompt should return 422 (validation error)."""
        response = client.post("/api/generate", json={
            "prompt": "",
        })
        assert response.status_code == 422

    def test_missing_prompt_rejected(self, client):
        """Missing prompt field should return 422."""
        response = client.post("/api/generate", json={})
        assert response.status_code == 422

    def test_temperature_too_high_rejected(self, client):
        """Temperature > 2.0 should be rejected."""
        response = client.post("/api/generate", json={
            "prompt": "Write code",
            "temperature": 5.0,
        })
        assert response.status_code == 422

    def test_temperature_too_low_rejected(self, client):
        """Temperature < 0.1 should be rejected."""
        response = client.post("/api/generate", json={
            "prompt": "Write code",
            "temperature": 0.0,
        })
        assert response.status_code == 422

    def test_max_length_too_small_rejected(self, client):
        """max_length < 32 should be rejected."""
        response = client.post("/api/generate", json={
            "prompt": "Write code",
            "max_length": 10,
        })
        assert response.status_code == 422

    def test_max_length_too_large_rejected(self, client):
        """max_length > 1024 should be rejected."""
        response = client.post("/api/generate", json={
            "prompt": "Write code",
            "max_length": 5000,
        })
        assert response.status_code == 422


# ── UI Serving Tests ─────────────────────────────────────────────────
class TestUIServing:
    """Tests for the frontend UI served by FastAPI."""

    def test_root_returns_html(self, client):
        """Root / should return the chat UI HTML page."""
        response = client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]

    def test_root_contains_title(self, client):
        """Root page should contain the app title."""
        response = client.get("/")
        assert "Local AI Coding Assistant" in response.text


# ── Error Handling Tests ─────────────────────────────────────────────
class TestErrorHandling:
    """Tests for error handling scenarios."""

    def test_generate_with_model_failure(self, client):
        """If generate_code raises, should return 500."""
        with patch("backend.app.generate_code", side_effect=RuntimeError("OOM")):
            response = client.post("/api/generate", json={
                "prompt": "Write code",
            })
            assert response.status_code == 500

    def test_invalid_endpoint_returns_404(self, client):
        """Non-existent endpoints should return 404."""
        response = client.get("/api/nonexistent")
        assert response.status_code == 404
