"""
Model Loader
=============
Handles loading the fine-tuned (or base) code generation model with
4-bit quantization for inference. Designed to minimize VRAM usage
so the model can run on a 4 GB GPU.

This module is used by the FastAPI backend at startup to load the model
once and keep it in memory for fast inference.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# ── Default Paths ────────────────────────────────────────────────────
# Change to "google/codegemma-2b" after running `huggingface-cli login`
BASE_MODEL_NAME = "Qwen/Qwen2.5-Coder-0.5B"
FINETUNED_ADAPTER_PATH = "./models/codegen-finetuned"


def get_quantization_config() -> BitsAndBytesConfig:
    """
    Create a 4-bit quantization config for inference.

    Uses NF4 quantization with double quantization for
    maximum memory savings during inference.

    Returns:
        BitsAndBytesConfig for 4-bit inference.
    """
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quantization=True,
    )


def load_model(
    base_model_name: str = BASE_MODEL_NAME,
    adapter_path: str = FINETUNED_ADAPTER_PATH,
):
    """
    Load the code generation model for inference.

    Tries to load the fine-tuned LoRA adapter on top of the base model.
    If no adapter is found, falls back to the base model alone.

    Args:
        base_model_name: HuggingFace model ID for the base model.
        adapter_path: Local path to the saved LoRA adapter weights.

    Returns:
        Tuple of (model, tokenizer) ready for text generation.
    """
    print(f"🔧 Loading base model: {base_model_name}")

    bnb_config = get_quantization_config()

    # Load the base model in 4-bit
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    # Try loading the fine-tuned LoRA adapter
    try:
        print(f"🔗 Loading fine-tuned adapter from: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
        print("✅ Fine-tuned adapter loaded successfully!")
    except Exception as e:
        print(f"⚠️  No fine-tuned adapter found at '{adapter_path}'.")
        print(f"   Reason: {e}")
        print("   Falling back to base model.")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name,
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.eos_token

    model.eval()  # Set to evaluation mode
    return model, tokenizer


def generate_code(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> str:
    """
    Generate code from a prompt using the loaded model.

    Uses the model's built-in chat template if available (works with
    Qwen, CodeGemma, Phi, etc.). Falls back to a simple prompt format
    if no chat template is found.

    Args:
        model: The loaded language model.
        tokenizer: The corresponding tokenizer.
        prompt: The instruction/prompt to generate code for.
        max_new_tokens: Maximum number of tokens to generate.
        temperature: Sampling temperature (higher = more creative).
        top_p: Nucleus sampling probability threshold.

    Returns:
        The generated code as a string.
    """
    # ── Format the prompt ────────────────────────────────────────────
    # Try using the model's native chat template (Qwen, Phi, etc.)
    if hasattr(tokenizer, "chat_template") and tokenizer.chat_template:
        messages = [
            {"role": "system", "content": "You are a helpful coding assistant. Write clean, well-commented Python code."},
            {"role": "user", "content": prompt},
        ]
        formatted_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        # Fallback for models without a chat template (e.g., after fine-tuning)
        formatted_prompt = f"### Instruction:\n{prompt}\n\n### Response:\n"

    # ── Tokenize ─────────────────────────────────────────────────────
    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    ).to(model.device)

    # ── Generate ─────────────────────────────────────────────────────
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.2,   # Prevents the repeating-prompt issue
        )

    # ── Decode only the new tokens (skip the prompt) ─────────────────
    generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    return generated_text.strip()

