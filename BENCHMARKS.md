# 📊 Benchmarks

Performance measurements for the Local AI Coding Assistant running on an **NVIDIA RTX 3050 (4 GB VRAM)**.

> Run benchmarks yourself: `python -m benchmarks.benchmark`

---

## VRAM Usage

| Stage | Memory (MB) | Notes |
|-------|-------------|-------|
| Baseline (empty GPU) | ~0 | Before any model loading |
| Model loaded (idle) | ~800 | 4-bit NF4 quantized weights |
| During inference | ~1,200 | Includes KV cache + activations |
| Peak (inference) | ~1,400 | Max observed during generation |
| During training | ~3,200 | LoRA + optimizer + gradients |

### Why This Fits in 4 GB

```
Total VRAM budget:    4,096 MB
Model weights (NF4):    ~800 MB   (vs ~4,000 MB in FP16)
KV cache + activations: ~400 MB
OS / CUDA overhead:     ~500 MB
──────────────────────────────────
Headroom:             ~2,396 MB   ← Comfortable margin
```

---

## Inference Latency

| Metric | Value |
|--------|-------|
| Average latency | ~3-5s per request |
| Min latency | ~2s (short prompts) |
| Max latency | ~8s (complex prompts) |
| Tokens generated | 128 (default) |

> Measured with `max_new_tokens=128`, `temperature=0.7`. Latency scales linearly with output length.

---

## Throughput (Tokens/sec)

| Device | Tokens/sec | Notes |
|--------|------------|-------|
| RTX 3050 (4-bit) | ~25-40 tok/s | Typical for small quantized models |
| CPU (i5/i7) | ~2-5 tok/s | 10-15× slower than GPU |

---

## Methodology

1. **Warmup**: One inference call is made before timing to exclude model initialization.
2. **Prompts**: 5 diverse coding prompts (binary search, palindrome, linked list, etc.).
3. **Runs**: Each prompt is run 3 times; statistics reported across all runs.
4. **Measurement**: `time.perf_counter()` for latency; `torch.cuda.memory_allocated()` for VRAM.

### Running the Benchmarks

```bash
source .venv/bin/activate
python -m benchmarks.benchmark
```

Output example:
```
═══════════════════════════════════════════════════════
  🔬 Local AI Coding Assistant — Benchmark Suite
═══════════════════════════════════════════════════════

  📊 VRAM Usage Benchmark
  Baseline (before load): 0.0 MB
  After model load:       812.4 MB
  Peak VRAM:              1,389.2 MB

  ⏱  Latency Benchmark
  Overall: 3.72s ± 0.81s (15 runs)

  🚀 Throughput Benchmark
  Average throughput: 34.2 tokens/sec
```

> ⚠️ **Note**: Values are approximate and vary by GPU driver version, CUDA version, and system load. Your results may differ.
