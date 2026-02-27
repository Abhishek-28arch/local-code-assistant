"""
Benchmarking Suite
===================
Measures key performance metrics for the code generation model:
  - VRAM usage (idle, peak during inference)
  - Inference latency (time per request)
  - Tokens per second (throughput)
  - CPU vs GPU comparison

Usage:
    python -m benchmarks.benchmark

Results are printed to stdout and can be redirected to a file.
"""

import time
import statistics
import torch
from backend.model_loader import load_model, generate_code


# ── Test Prompts ─────────────────────────────────────────────────────
TEST_PROMPTS = [
    "Write a Python function to reverse a linked list",
    "Implement binary search in Python",
    "Write a function to check if a string is a palindrome",
    "Create a Python class for a stack with push, pop, and peek methods",
    "Write a function to find the nth Fibonacci number using dynamic programming",
]


def get_gpu_memory_mb() -> float:
    """
    Get current GPU memory usage in MB.

    Returns:
        GPU memory allocated in megabytes, or 0.0 if no GPU.
    """
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024 ** 2)
    return 0.0


def get_gpu_memory_reserved_mb() -> float:
    """
    Get GPU memory reserved (cached) by PyTorch in MB.

    Returns:
        GPU memory reserved in megabytes, or 0.0 if no GPU.
    """
    if torch.cuda.is_available():
        return torch.cuda.memory_reserved() / (1024 ** 2)
    return 0.0


def benchmark_vram():
    """
    Measure VRAM usage at different stages:
      1. Before model load (baseline)
      2. After model load (idle)
      3. During inference (peak)
    """
    print("=" * 60)
    print("  📊 VRAM Usage Benchmark")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("  ⚠️  No CUDA GPU detected. Skipping VRAM benchmark.")
        return

    # Baseline
    torch.cuda.reset_peak_memory_stats()
    baseline_mb = get_gpu_memory_mb()
    print(f"  Baseline (before load): {baseline_mb:.1f} MB")

    # Load model
    model, tokenizer = load_model()
    idle_mb = get_gpu_memory_mb()
    print(f"  After model load:       {idle_mb:.1f} MB")

    # Run inference to measure peak
    _ = generate_code(model, tokenizer, TEST_PROMPTS[0], max_new_tokens=128)
    peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    inference_mb = get_gpu_memory_mb()

    print(f"  During inference:       {inference_mb:.1f} MB")
    print(f"  Peak VRAM:              {peak_mb:.1f} MB")
    print(f"  Model size (approx):    {idle_mb - baseline_mb:.1f} MB")
    print()

    return model, tokenizer


def benchmark_latency(model, tokenizer, num_runs: int = 5):
    """
    Measure inference latency across multiple prompts.

    Args:
        model: Loaded language model.
        tokenizer: Corresponding tokenizer.
        num_runs: Number of times to run each prompt.

    Returns:
        Dict with latency statistics.
    """
    print("=" * 60)
    print("  ⏱  Latency Benchmark")
    print("=" * 60)

    all_times = []

    for prompt in TEST_PROMPTS:
        prompt_times = []
        for _ in range(num_runs):
            start = time.perf_counter()
            _ = generate_code(model, tokenizer, prompt, max_new_tokens=128)
            elapsed = time.perf_counter() - start
            prompt_times.append(elapsed)
        
        avg = statistics.mean(prompt_times)
        all_times.extend(prompt_times)
        print(f"  '{prompt[:45]}...'")
        print(f"    Avg: {avg:.2f}s | Min: {min(prompt_times):.2f}s | "
              f"Max: {max(prompt_times):.2f}s")

    overall_avg = statistics.mean(all_times)
    overall_std = statistics.stdev(all_times) if len(all_times) > 1 else 0
    print(f"\n  Overall: {overall_avg:.2f}s ± {overall_std:.2f}s "
          f"({len(all_times)} runs)")
    print()

    return {"avg": overall_avg, "std": overall_std, "all_times": all_times}


def benchmark_throughput(model, tokenizer, num_runs: int = 5):
    """
    Measure tokens-per-second throughput.

    Args:
        model: Loaded language model.
        tokenizer: Corresponding tokenizer.
        num_runs: Number of runs to average.

    Returns:
        Average tokens per second.
    """
    print("=" * 60)
    print("  🚀 Throughput Benchmark (Tokens/sec)")
    print("=" * 60)

    token_rates = []
    max_new_tokens = 128

    for i, prompt in enumerate(TEST_PROMPTS[:num_runs]):
        start = time.perf_counter()
        output = generate_code(
            model, tokenizer, prompt, max_new_tokens=max_new_tokens
        )
        elapsed = time.perf_counter() - start

        # Count output tokens
        output_tokens = len(tokenizer.encode(output))
        tps = output_tokens / elapsed if elapsed > 0 else 0
        token_rates.append(tps)

        print(f"  Run {i+1}: {output_tokens} tokens in {elapsed:.2f}s "
              f"→ {tps:.1f} tok/s")

    avg_tps = statistics.mean(token_rates)
    print(f"\n  Average throughput: {avg_tps:.1f} tokens/sec")
    print()

    return avg_tps


def run_all_benchmarks():
    """Run the full benchmark suite and print a summary."""
    print("\n" + "═" * 60)
    print("  🔬 Local AI Coding Assistant — Benchmark Suite")
    print("═" * 60 + "\n")

    result = benchmark_vram()
    if result is None:
        print("Cannot continue benchmarks without GPU.")
        return
    
    model, tokenizer = result
    latency = benchmark_latency(model, tokenizer, num_runs=3)
    throughput = benchmark_throughput(model, tokenizer, num_runs=3)

    # Summary
    print("=" * 60)
    print("  📋 Summary")
    print("=" * 60)
    peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    print(f"  Peak VRAM:         {peak_mb:.0f} MB")
    print(f"  Avg Latency:       {latency['avg']:.2f}s")
    print(f"  Avg Throughput:    {throughput:.1f} tok/s")
    print(f"  Device:            {torch.cuda.get_device_name(0)}")
    print()


if __name__ == "__main__":
    run_all_benchmarks()
