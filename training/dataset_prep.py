"""
Dataset Preparation
===================
Downloads the Python code-instructions dataset from HuggingFace Hub,
formats each sample into a prompt-completion pair using the instruction
template, and returns a tokenized dataset ready for the SFT Trainer.

Dataset: iamtarun/python_code_instructions_18k_alpaca
  - ~18 000 Python coding problems with instructions and solutions.
"""

from datasets import load_dataset
from transformers import AutoTokenizer

from training.config import TrainingConfig


def format_prompt(sample: dict, template: str) -> dict:
    """
    Convert a raw dataset sample into the instruction-response format.

    Args:
        sample: A dict with 'prompt' (the instruction) and 'output' (the code).
        template: The prompt template string with an {instruction} placeholder.

    Returns:
        Dict with a single 'text' key containing the formatted string.
    """
    instruction = sample.get("prompt", sample.get("instruction", ""))
    response = sample.get("output", "")
    
    text = template.format(instruction=instruction) + response
    return {"text": text}


def load_and_prepare_dataset(config: TrainingConfig = None):
    """
    Download, format, and tokenize the training dataset.

    Steps:
        1. Load dataset from HuggingFace Hub.
        2. Apply the instruction prompt template to every sample.
        3. Shuffle and return the formatted dataset.

    Args:
        config: TrainingConfig instance (uses defaults if None).

    Returns:
        A HuggingFace Dataset with a 'text' column ready for SFTTrainer.
    """
    if config is None:
        config = TrainingConfig()

    # ── Step 1: Download ─────────────────────────────────────────────
    print(f"📥 Loading dataset: {config.dataset_name}")
    dataset = load_dataset(config.dataset_name, split="train")
    print(f"   Total samples: {len(dataset)}")

    # ── Step 2: Format into prompt-completion pairs ──────────────────
    print("📝 Formatting samples with instruction template...")
    dataset = dataset.map(
        lambda sample: format_prompt(sample, config.prompt_template),
        remove_columns=dataset.column_names,  # Keep only the 'text' column
    )

    # ── Step 3: Shuffle ───────────────────────────────────────────────
    dataset = dataset.shuffle(seed=42)
    print(f"✅ Dataset ready — {len(dataset)} samples")

    return dataset


# ── Quick test ───────────────────────────────────────────────────────
if __name__ == "__main__":
    config = TrainingConfig()
    ds = load_and_prepare_dataset(config)
    # Print a sample to verify formatting
    print("\n── Sample ──")
    print(ds[0]["text"][:500])
