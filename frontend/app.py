"""
Flask Frontend Server
======================
A simple Flask web server that serves the chat UI and proxies
requests to the FastAPI backend.

The frontend is intentionally lightweight — all the AI logic
lives in the FastAPI backend. This separation makes deployment
and scaling easier.

Usage:
    python frontend/app.py
"""

import requests
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# URL of the FastAPI backend
BACKEND_URL = "http://localhost:8000"


@app.route("/")
def index():
    """Serve the main chat interface page."""
    return render_template("index.html")


@app.route("/api/generate", methods=["POST"])
def generate():
    """
    Proxy endpoint that forwards requests to the FastAPI backend.

    Accepts JSON with:
        - prompt (str): The coding question or instruction.
        - max_length (int, optional): Max tokens to generate.
        - temperature (float, optional): Sampling temperature.

    Returns:
        JSON response from the backend with generated code.
    """
    data = request.get_json()

    try:
        # Forward the request to the FastAPI backend
        response = requests.post(
            f"{BACKEND_URL}/generate",
            json=data,
            timeout=120,  # Model generation can take a while
        )
        response.raise_for_status()
        return jsonify(response.json())

    except requests.ConnectionError:
        return jsonify({
            "error": "Cannot connect to the AI backend. "
                     "Make sure the FastAPI server is running on port 8000."
        }), 503

    except requests.Timeout:
        return jsonify({
            "error": "Request timed out. The model may be loading or "
                     "generating a long response."
        }), 504

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Entry Point ──────────────────────────────────────────────────────
if __name__ == "__main__":
    print("🌐 Starting Flask frontend on http://localhost:5000")
    print("   Make sure the FastAPI backend is running on port 8000")
    app.run(debug=True, port=5000)
