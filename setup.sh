#!/bin/bash
# ============================================================
#  Setup Script — Local AI Coding Assistant
#  Creates a virtual environment, installs dependencies, and
#  verifies the GPU is available for training/inference.
# ============================================================

set -e  # Exit on any error

echo "============================================="
echo "  🚀 Local AI Coding Assistant — Setup"
echo "============================================="

# ── 1. Create virtual environment ────────────────────────────
echo ""
echo "📦 Creating virtual environment..."
python3 -m venv .venv
source .venv/bin/activate
echo "   ✅ Virtual environment created at .venv/"

# ── 2. Upgrade pip ───────────────────────────────────────────
echo ""
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# ── 3. Install dependencies ──────────────────────────────────
echo ""
echo "📥 Installing dependencies (this may take a few minutes)..."
pip install -r requirements.txt

# ── 4. GPU Check ─────────────────────────────────────────────
echo ""
echo "🔍 Checking GPU availability..."
python3 -c "
import torch
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f'   ✅ GPU detected: {gpu_name} ({vram:.1f} GB VRAM)')
else:
    print('   ⚠️  No CUDA GPU detected. Training will be very slow on CPU.')
    print('      Make sure NVIDIA drivers and CUDA toolkit are installed.')
"

# ── 5. Create model output directory ─────────────────────────
mkdir -p models/codegen-finetuned
echo ""
echo "📁 Created models/ directory for saving fine-tuned weights."

# ── Done ─────────────────────────────────────────────────────
echo ""
echo "============================================="
echo "  ✅ Setup complete!"
echo ""
echo "  Next steps:"
echo "    1. Activate the env:  source .venv/bin/activate"
echo "    2. Fine-tune:         python -m training.fine_tune"
echo "    3. Start backend:     uvicorn backend.app:app --port 8000"
echo "    4. Start frontend:    python frontend/app.py"
echo "============================================="
