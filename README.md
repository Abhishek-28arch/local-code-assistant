# ⚡ Local AI Coding Assistant

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/🤗_HuggingFace-Transformers-FFD21E?style=for-the-badge)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**Fine-tune a code model on a 4 GB GPU. Serve it locally. Chat with it.**

An end-to-end ML project that demonstrates real engineering under hardware constraints — not just `model.generate()`.

[Features](#-features) · [Architecture](#-architecture) · [Engineering Decisions](#-engineering-decisions) · [Quick Start](#-quick-start) · [Benchmarks](#-benchmarks) · [Limitations](#-limitations)

</div>

---

## ✨ Features

- 🧠 **QLoRA Fine-Tuning** — Train a 2B-parameter model on 18K Python problems using 4-bit quantization, fitting entirely in 4 GB VRAM
- 🚀 **Unified FastAPI Server** — Single server serves both the REST API and the web UI (no proxy layers)
- 💬 **Chat Interface** — Dark-themed UI with syntax highlighting, code copy buttons, and interactive suggestions
- 📊 **Benchmarking Suite** — Automated VRAM, latency, and throughput measurement scripts
- ✅ **Tested** — 15+ pytest API tests with mocked models (runs without GPU)
- 📝 **Structured Logging** — Request-level logging with timing, structured format, and error tracking

---

## 🏗 Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        User's Browser                          │
│                     (Chat Interface)                            │
└──────────────────────────┬──────────────────────────────────────┘
                           │ HTTP
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│  FastAPI Server (port 8000)                                     │
│                                                                  │
│  ┌────────────────────┐  ┌────────────────────────────────────┐ │
│  │  Static Files       │  │  API Endpoints                     │ │
│  │  GET /              │  │  POST /api/generate                │ │
│  │  GET /static/*      │  │  GET  /api/health                  │ │
│  └────────────────────┘  │  GET  /api/docs (Swagger)           │ │
│                           └────────────────────┬───────────────┘ │
│                                                │                 │
│                           ┌────────────────────▼───────────────┐ │
│                           │  Middleware                         │ │
│                           │  • Request logging (method, path,  │ │
│                           │    status, duration_ms)             │ │
│                           │  • CORS                            │ │
│                           └────────────────────┬───────────────┘ │
└────────────────────────────────────────────────┼─────────────────┘
                                                 │
                           ┌─────────────────────▼───────────────┐
                           │  Quantized Model                    │
                           │  • 4-bit NF4 (bitsandbytes)         │
                           │  • LoRA adapter (rank 16, α=32)     │
                           │  • ~800 MB VRAM at inference        │
                           └─────────────────────────────────────┘
```

> **Why no Flask proxy?** The initial version used Flask → FastAPI. This added latency, an extra process, and deployment complexity. FastAPI natively serves static files and HTML, so we consolidated into a **single server** — fewer moving parts, simpler deployment, lower latency.

---

## 📁 Project Structure

```
local-code-assistant/
├── training/
│   ├── config.py           # All hyperparameters (dataclass)
│   ├── dataset_prep.py     # Download & format dataset
│   └── fine_tune.py        # QLoRA fine-tuning entry point
├── backend/
│   ├── model_loader.py     # Load quantized model + LoRA adapter
│   └── app.py              # Unified FastAPI server (API + UI)
├── frontend/
│   ├── templates/
│   │   └── index.html      # Chat UI
│   └── static/
│       ├── css/style.css   # Dark theme
│       └── js/chat.js      # Client-side logic
├── benchmarks/
│   └── benchmark.py        # VRAM, latency, throughput tests
├── tests/
│   ├── conftest.py         # Pytest fixtures (mocked models)
│   └── test_api.py         # 15+ API tests
├── requirements.txt
├── setup.sh                # One-command setup
├── Dockerfile              # HuggingFace Spaces deployment
├── BENCHMARKS.md           # Performance results
├── RESUME_BULLETS.md       # Resume-ready descriptions
└── README.md
```

---

## 🧪 Engineering Decisions

Every technical choice in this project was driven by a single constraint: **4 GB VRAM** on an RTX 3050. Here's why each decision was made:

### Why 4-bit NF4 (not 8-bit or FP16)?

| Format | Memory for 2B params | Fits in 4 GB? |
|--------|---------------------|---------------|
| FP16 | ~4,000 MB | ❌ No headroom |
| INT8 | ~2,000 MB | ⚠️ Tight with KV cache |
| **NF4** | **~800 MB** | **✅ Comfortable** |

NF4 (Normalized Float 4-bit) from the [QLoRA paper](https://arxiv.org/abs/2305.14314) maps quantization levels to a normal distribution, which matches how neural network weights are typically distributed. This gives better precision per bit compared to uniform INT4.

### Why Double Quantization?

Standard quantization stores a scaling factor per block of weights. Double quantization **quantizes the scaling factors themselves**, saving an additional ~0.4 bits per parameter. On a 2B model, this saves approximately **100 MB** of VRAM — a meaningful margin on a 4 GB card.

### Why LoRA Rank 16?

| Rank | Trainable Params | Quality | Memory |
|------|-----------------|---------|--------|
| 4 | ~4M | Lower | Minimal |
| 8 | ~8M | Good | Low |
| **16** | **~30M (1.5%)** | **Strong** | **Moderate** |
| 32 | ~60M | Slightly better | Higher |
| 64 | ~120M | Diminishing gains | Much higher |

Rank 16 hits the sweet spot: enough capacity to learn task-specific patterns without inflating memory usage. Research shows diminishing returns beyond rank 16 for most fine-tuning tasks.

### Why Target `q_proj` and `v_proj`?

In transformer attention layers, there are four projection matrices: Q, K, V, and O. We target **Q (query)** and **V (value)** because:

- **Q** controls what the model "asks" at each position — critical for understanding code structure
- **V** controls what information gets passed forward — important for generating correct code
- K and O adapters add memory with marginal quality improvement for code generation tasks

This follows the original LoRA paper's findings on optimal target selection.

### Why `max_seq_length = 512`?

Each token in the sequence consumes KV cache memory that scales as `O(seq_len × hidden_dim)`. At 512 tokens:

- KV cache fits comfortably alongside the model
- Covers 80%+ of Python functions in the training dataset
- Longer sequences (1024+) would risk OOM during training with batch size > 1

### Why Paged AdamW 8-bit?

Standard AdamW stores **two state tensors** per parameter (momentum + variance), effectively tripling memory. Paged AdamW 8-bit:

1. Stores optimizer states in 8-bit (halves optimizer memory)
2. Uses CPU paging for overflow — if GPU memory fills up, states automatically page to RAM
3. Saves approximately **30% total VRAM** compared to standard FP32 AdamW

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- NVIDIA GPU with ≥4 GB VRAM
- CUDA toolkit installed
- Git

### 1. Clone & Setup

```bash
git clone https://github.com/YOUR_USERNAME/local-code-assistant.git
cd local-code-assistant
chmod +x setup.sh && ./setup.sh
```

### 2. Fine-Tune the Model

```bash
source .venv/bin/activate
python -m training.fine_tune
```

> ⏱ Takes ~2-3 hours on an RTX 3050. Adapter saved to `models/codegen-finetuned/`.

### 3. Start the Server

```bash
uvicorn backend.app:app --host 0.0.0.0 --port 8000
```

### 4. Open the UI

Visit **http://localhost:8000** and start generating code! 🎉

### 5. Run Tests

```bash
pytest tests/ -v
```

---

## 📊 Benchmarks

| Metric | Value | Device |
|--------|-------|--------|
| Model VRAM (idle) | ~800 MB | RTX 3050 |
| Peak VRAM (inference) | ~1,400 MB | RTX 3050 |
| Peak VRAM (training) | ~3,200 MB | RTX 3050 |
| Inference latency | 3-5s | RTX 3050 |
| Throughput | ~25-40 tok/s | RTX 3050 |
| CPU throughput | ~2-5 tok/s | i5/i7 |

Run benchmarks: `python -m benchmarks.benchmark`

Full results and methodology: **[BENCHMARKS.md](BENCHMARKS.md)**

---

## ⚠️ Limitations

This project intentionally operates under significant constraints. Being transparent about limitations demonstrates engineering maturity.

### Context Window (512 tokens)

The model processes a maximum of 512 tokens per request. This means:
- Can generate single functions and short classes effectively
- Cannot reason about multi-file projects or long code contexts
- Complex problems that require >30 lines of context may produce incomplete results

### Small Model Capacity (2B parameters)

A 2B model is orders of magnitude smaller than GPT-4 (estimated ~1.8T) or CodeGemma-7B:
- Struggles with nuanced requirements or ambiguous prompts
- May produce syntactically correct but logically flawed code
- Limited ability to follow complex multi-step instructions

### Quantization Tradeoffs

4-bit quantization reduces model footprint by ~5× but introduces precision loss:
- Slight degradation in output quality compared to FP16 inference
- Occasional numerical instability with very long generation sequences
- Not suitable for tasks requiring high numerical precision (scientific computing)

### Not Production-Ready

This is a portfolio project demonstrating ML/systems skills, not a production system:
- Single-user concurrency (one inference at a time, no batching)
- No authentication, rate limiting, or input sanitization beyond Pydantic
- CORS allows all origins (would need restriction in production)
- No model versioning or A/B testing infrastructure

### Dataset Limitations

The training dataset (18K Python problems) is relatively small and narrow:
- Fine-tuned specifically for Python — no multi-language support
- Biased toward algorithmic problems — weaker on web dev, data science, etc.
- No deduplication or quality filtering applied to the training data

---

## 🛠 API Reference

### `POST /api/generate`

```json
// Request
{ "prompt": "Write a function to reverse a linked list", "max_length": 256, "temperature": 0.7 }

// Response
{ "generated_code": "def reverse(head): ...", "prompt": "...", "model_name": "codegen-finetuned", "generation_time_ms": 3842.1 }
```

### `GET /api/health`

```json
{ "status": "healthy", "model_loaded": true, "version": "1.0.0" }
```

### `GET /api/docs`

Interactive Swagger documentation (auto-generated by FastAPI).

---

## 🐳 Deploy to HuggingFace Spaces

1. Create a new Space at [huggingface.co/spaces](https://huggingface.co/spaces) (select **Docker** SDK)
2. Push:
   ```bash
   git remote add space https://huggingface.co/spaces/YOUR_USERNAME/local-code-assistant
   git push space main
   ```

---

## 🙏 Acknowledgments

- [QLoRA Paper](https://arxiv.org/abs/2305.14314) — Efficient fine-tuning technique
- [HuggingFace](https://huggingface.co/) — Transformers, PEFT, TRL, Datasets
- [bitsandbytes](https://github.com/TimDettmers/bitsandbytes) — 4-bit quantization

---

## 📄 License

MIT License. See [LICENSE](LICENSE) for details.
