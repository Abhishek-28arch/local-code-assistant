# ============================================================
#  Dockerfile — HuggingFace Spaces Deployment
#  Runs the FastAPI backend + Flask frontend together.
#  Uses CPU inference with 8-bit quantization for Spaces.
# ============================================================

FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the full project
COPY . .

# Create models directory
RUN mkdir -p models/codegen-finetuned

# Expose the port HuggingFace Spaces expects
EXPOSE 7860

# Run the FastAPI app on port 7860 (Spaces default)
CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "7860"]
