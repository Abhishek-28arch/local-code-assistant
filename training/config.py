"""
Training Configuration
======================
Centralizes all hyperparameters, model paths, and training settings
in a single dataclass. Modify values here instead of scattered across scripts.

Defaults are tuned for a 4 GB VRAM GPU (e.g., NVIDIA RTX 3050) using
4-bit quantization and gradient checkpointing to stay within memory limits.
"""

from dataclasses import dataclass, field


@dataclass
class TrainingConfig:
    """All training hyperparameters in one place."""

    # ── Model ────────────────────────────────────────────────────────
    # Change to "google/codegemma-2b" after running `huggingface-cli login`
    # CodeGemma is gated and requires access approval + authentication.
    model_name: str = "Qwen/Qwen2.5-Coder-0.5B"
    output_dir: str = "./models/codegen-finetuned"
    
    # ── Dataset ──────────────────────────────────────────────────────
    dataset_name: str = "iamtarun/python_code_instructions_18k_alpaca"
    max_seq_length: int = 512          # Max tokens per sample (keep low for 4GB)
    
    # ── QLoRA / BitsAndBytes ─────────────────────────────────────────
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"   # Normalized Float 4-bit
    use_double_quant: bool = True      # Nested quantization for extra savings
    
    # ── LoRA ─────────────────────────────────────────────────────────
    lora_rank: int = 16                # Rank of the low-rank matrices
    lora_alpha: int = 32               # Scaling factor (alpha / rank)
    lora_dropout: float = 0.05
    target_modules: list = field(
        default_factory=lambda: ["q_proj", "v_proj"]
    )
    
    # ── Training ─────────────────────────────────────────────────────
    num_epochs: int = 3
    per_device_batch_size: int = 1     # Keep at 1 for 4 GB VRAM
    gradient_accumulation_steps: int = 4  # Effective batch size = 4
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    
    # ── Memory Optimization ──────────────────────────────────────────
    fp16: bool = True                  # Mixed precision training
    gradient_checkpointing: bool = True
    optim: str = "paged_adamw_8bit"    # 8-bit optimizer to save VRAM
    
    # ── Logging ──────────────────────────────────────────────────────
    logging_steps: int = 25
    save_strategy: str = "epoch"
    
    # ── Prompt Template ──────────────────────────────────────────────
    prompt_template: str = (
        "### Instruction:\n{instruction}\n\n### Response:\n"
    )
