"""
FastAPI Backend Server
=======================
Serves the fine-tuned code generation model via a REST API and
serves the web UI directly — no separate Flask server needed.

Architecture:
    Single server on port 8000 handles both:
    - Static files (HTML/CSS/JS) at /
    - API endpoints at /api/generate and /api/health

This unified approach eliminates the Flask proxy layer, reducing
latency, moving parts, and deployment complexity.

Usage:
    uvicorn backend.app:app --host 0.0.0.0 --port 8000
"""

import time
import logging
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from backend.model_loader import load_model, generate_code

# ── Structured Logging ──────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("backend")

# ── Global model references (set during startup) ────────────────────
model = None
tokenizer = None

# ── Paths ────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
TEMPLATES_DIR = BASE_DIR / "frontend" / "templates"
STATIC_DIR = BASE_DIR / "frontend" / "static"


# ── Lifespan: load model once at startup ─────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model when the server starts, clean up on shutdown."""
    global model, tokenizer
    logger.info("Starting server — loading model...")
    try:
        model, tokenizer = load_model()
        logger.info("Model loaded and ready!")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        logger.warning("Server will start but /api/generate will return errors.")
    yield
    logger.info("Server shutting down.")


# ── FastAPI App ──────────────────────────────────────────────────────
app = FastAPI(
    title="Local AI Coding Assistant",
    description="Generate Python code using a fine-tuned CodeGemma model.",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/api/docs",       # Move Swagger docs to /api/docs
    redoc_url="/api/redoc",
)

# CORS — allow any origin during development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request Logging Middleware ───────────────────────────────────────
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log every incoming request with method, path, and response time."""
    start = time.time()
    response = await call_next(request)
    duration_ms = (time.time() - start) * 1000
    # Skip logging static file requests to reduce noise
    if not request.url.path.startswith("/static"):
        logger.info(
            f"{request.method} {request.url.path} → {response.status_code} "
            f"({duration_ms:.0f}ms)"
        )
    return response


# ── Request / Response Schemas ───────────────────────────────────────
class GenerateRequest(BaseModel):
    """Schema for the /api/generate endpoint request body."""
    prompt: str = Field(
        ...,
        description="Natural language description of the code to generate.",
        min_length=1,
        examples=["Write a Python function to reverse a linked list"],
    )
    max_length: int = Field(
        default=256,
        ge=32,
        le=1024,
        description="Maximum number of new tokens to generate.",
    )
    temperature: float = Field(
        default=0.7,
        ge=0.1,
        le=2.0,
        description="Sampling temperature (higher = more creative).",
    )


class GenerateResponse(BaseModel):
    """Schema for the /api/generate endpoint response body."""
    generated_code: str
    prompt: str
    model_name: str = "codegen-finetuned"
    generation_time_ms: float = 0.0


# ── API Endpoints ────────────────────────────────────────────────────
@app.get("/api/health")
async def health_check():
    """
    Health check endpoint.

    Returns server status, model readiness, and basic system info.
    """
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "version": "1.0.0",
    }


@app.post("/api/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    """
    Generate code from a natural language prompt.

    Takes a prompt describing what code to write, and returns
    the AI-generated Python code.

    Raises:
        HTTPException 503: If the model is not loaded yet.
        HTTPException 500: If code generation fails.
    """
    # Guard: ensure model is loaded
    if model is None or tokenizer is None:
        logger.error("Generate called but model is not loaded.")
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded. Please check server logs.",
        )

    try:
        start = time.time()
        generated = generate_code(
            model=model,
            tokenizer=tokenizer,
            prompt=request.prompt,
            max_new_tokens=request.max_length,
            temperature=request.temperature,
        )
        duration_ms = (time.time() - start) * 1000
        logger.info(
            f"Generated {len(generated)} chars for prompt "
            f"'{request.prompt[:50]}...' in {duration_ms:.0f}ms"
        )

        return GenerateResponse(
            generated_code=generated,
            prompt=request.prompt,
            generation_time_ms=round(duration_ms, 1),
        )

    except Exception as e:
        logger.exception(f"Generation failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Code generation failed: {str(e)}",
        )


# ── Serve Frontend ──────────────────────────────────────────────────
# Mount static files (CSS, JS)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    """Serve the chat UI directly from FastAPI — no Flask needed."""
    index_path = TEMPLATES_DIR / "index.html"
    return HTMLResponse(content=index_path.read_text(), status_code=200)
