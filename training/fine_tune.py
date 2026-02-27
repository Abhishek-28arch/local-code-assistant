"""
QLoRA Fine-Tuning Script
=========================
Fine-tunes CodeGemma-2B (or any compatible model) on Python coding problems
using QLoRA (Quantized Low-Rank Adaptation).

Key techniques used to fit on a 4 GB GPU:
  - 4-bit NF4 quantization via bitsandbytes
  - LoRA adapters (only ~1-2% of params are trainable)
  - Gradient checkpointing (trades compute for memory)
  - Paged AdamW 8-bit optimizer
  - FP16 mixed-precision training

Usage:
    python -m training.fine_tune
"""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

from training.config import TrainingConfig
from training.dataset_prep import load_and_prepare_dataset


def create_bnb_config(config: TrainingConfig) -> BitsAndBytesConfig:
    """
    Create a BitsAndBytes quantization config for 4-bit loading.

    This dramatically reduces VRAM usage by storing weights in 4-bit
    NF4 format while computing in FP16.

    Args:
        config: TrainingConfig with quantization settings.

    Returns:
        A BitsAndBytesConfig instance.
    """
    return BitsAndBytesConfig(
        load_in_4bit=config.load_in_4bit,
        bnb_4bit_quant_type=config.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quantization=config.use_double_quant,
    )


def create_lora_config(config: TrainingConfig) -> LoraConfig:
    """
    Create a LoRA adapter configuration.

    LoRA injects small trainable matrices into the model's attention layers,
    so we only train ~1-2% of the total parameters.

    Args:
        config: TrainingConfig with LoRA hyperparameters.

    Returns:
        A LoraConfig instance.
    """
    return LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )


def load_model_and_tokenizer(config: TrainingConfig):
    """
    Load the base model in 4-bit quantization and its tokenizer.

    Args:
        config: TrainingConfig with model name and quantization settings.

    Returns:
        Tuple of (model, tokenizer).
    """
    print(f"🔧 Loading model: {config.model_name}")
    
    bnb_config = create_bnb_config(config)
    
    # Load base model with 4-bit quantization
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        quantization_config=bnb_config,
        device_map="auto",           # Automatically place layers on GPU/CPU
        trust_remote_code=True,
    )
    
    # Prepare the quantized model for training
    model = prepare_model_for_kbit_training(model)
    
    # Apply LoRA adapters
    lora_config = create_lora_config(config)
    model = get_peft_model(model, lora_config)
    
    # Print trainable parameter count
    model.print_trainable_parameters()
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    return model, tokenizer


def create_training_args(config: TrainingConfig) -> TrainingArguments:
    """
    Build HuggingFace TrainingArguments from our config.

    Args:
        config: TrainingConfig with training hyperparameters.

    Returns:
        A TrainingArguments instance.
    """
    return TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.per_device_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        lr_scheduler_type=config.lr_scheduler_type,
        fp16=config.fp16,
        gradient_checkpointing=config.gradient_checkpointing,
        optim=config.optim,
        logging_steps=config.logging_steps,
        save_strategy=config.save_strategy,
        report_to="none",             # Disable W&B / MLflow logging
        max_grad_norm=0.3,            # Gradient clipping
    )


def fine_tune():
    """
    Main fine-tuning pipeline.

    Steps:
        1. Load configuration.
        2. Prepare the dataset.
        3. Load & quantize the model; attach LoRA adapters.
        4. Train with SFTTrainer.
        5. Save the fine-tuned adapter weights.
    """
    # ── 1. Configuration ─────────────────────────────────────────────
    config = TrainingConfig()
    print("=" * 60)
    print("  🚀 Local AI Coding Assistant — Fine-Tuning")
    print("=" * 60)
    print(f"  Model        : {config.model_name}")
    print(f"  Dataset      : {config.dataset_name}")
    print(f"  LoRA Rank    : {config.lora_rank}")
    print(f"  Batch Size   : {config.per_device_batch_size} "
          f"(effective: {config.per_device_batch_size * config.gradient_accumulation_steps})")
    print(f"  Epochs       : {config.num_epochs}")
    print(f"  Output Dir   : {config.output_dir}")
    print("=" * 60)

    # ── 2. Dataset ───────────────────────────────────────────────────
    dataset = load_and_prepare_dataset(config)

    # ── 3. Model + Tokenizer ─────────────────────────────────────────
    model, tokenizer = load_model_and_tokenizer(config)

    # ── 4. Train ─────────────────────────────────────────────────────
    training_args = create_training_args(config)

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        max_seq_length=config.max_seq_length,
        dataset_text_field="text",     # Column name in the dataset
        packing=False,                 # Disable packing for simplicity
    )

    print("\n🏋️ Starting training...")
    trainer.train()

    # ── 5. Save ──────────────────────────────────────────────────────
    print(f"\n💾 Saving fine-tuned adapter to {config.output_dir}")
    trainer.model.save_pretrained(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)

    print("✅ Fine-tuning complete!")


# ── Entry Point ──────────────────────────────────────────────────────
if __name__ == "__main__":
    fine_tune()
