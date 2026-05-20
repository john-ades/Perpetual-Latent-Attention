import os
import re

# ==========================================
# 0. PREVENT MEMORY FRAGMENTATION
# 🚨 MUST be set before importing PyTorch
# ==========================================
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import argparse
import psutil
import torch
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
)
from peft import LoraConfig
import tptt

# ==========================================
# Helper: Memory Auto-Sizing
# ==========================================
DTYPE_SIZE = {torch.float32: 4, torch.float16: 2, torch.bfloat16: 2}


def extract_model_size(base_model: str) -> int:
    """Estimates the parameter count of a model based on its HF name."""
    try:
        if "B" not in base_model.upper(): return 1_000_000_000
        before_b = base_model.upper().rsplit("B", 1)[0]
        parts = re.split(r"[^0-9._]", before_b)
        numbers = [p for p in parts if p.strip()]
        last_num_str = numbers[-1].replace("_", ".").replace("-", ".")
        return int(float(last_num_str) * 1_000_000_000)
    except:
        return 1_000_000_000


def estimate_optimal_batch_size(args, max_batch=64) -> int:
    """Calculates safe batch size to occupy exactly ~75% of available VRAM."""
    model_size = extract_model_size(args.model_name)
    dtype = torch.float16 if args.fp16 else torch.bfloat16

    param_mem_bytes = model_size * DTYPE_SIZE.get(dtype, 2)
    activation_mem_bytes = 1 * args.max_length * 4 * DTYPE_SIZE.get(dtype, 2) * 20

    if args.use_4bit:
        param_mem_bytes *= 0.25  # Quantization memory reduction

    optimizer_overhead = param_mem_bytes * 1.2
    total_mem_bytes = param_mem_bytes + activation_mem_bytes + optimizer_overhead

    # 🚨 Delta Product mathematically requires 2x memory during backward pass
    if "delta_product" in args.operator_mode:
        total_mem_bytes *= 2

    total_mem_gb = total_mem_bytes / (1024 ** 3)

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        available_mem_gb = props.total_memory / (1024 ** 3)
    else:
        available_mem_gb = psutil.virtual_memory().available / (1024 ** 3)

    batch_size = min(max(1, int(0.75 * available_mem_gb // total_mem_gb)), max_batch)
    print(
        f"🔥 Auto batch sizing: {batch_size} (Available VRAM: {available_mem_gb:.1f}GB, Est. Base Mem: {total_mem_gb:.1f}GB)")
    return batch_size


# ==========================================
# Helper: Zero-Padding Sequence Packing
# ==========================================
def packed_stream_generator(dataset_stream, tokenizer, max_length, text_column="text"):
    """Yields densely packed sequences of max_length with ZERO padding."""
    buffer = []
    for row in dataset_stream:
        # Fallback dictionary matching
        if text_column and text_column in row:
            raw_text = row[text_column]
        elif "text" in row:
            raw_text = row["text"]
        elif "content" in row:
            raw_text = row["content"]
        else:
            raw_text = str(row)

        if not raw_text: continue

        text = raw_text + tokenizer.eos_token
        tokens = tokenizer(text, add_special_tokens=False)["input_ids"]
        buffer.extend(tokens)

        while len(buffer) >= max_length:
            chunk = buffer[:max_length]
            buffer = buffer[max_length:]
            yield {
                "input_ids": chunk,
                "labels": chunk.copy(),
                "attention_mask": [1] * max_length
            }


# ==========================================
# Helper: Safe Inference Testing
# ==========================================
def dummy_generate(model, tokenizer, prompt="The history of artificial intelligence is"):
    """Tests the model post-training, ensuring the memory cache is flushed first."""
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"\n🤖 Testing generation for prompt: '{prompt}'")

    # 🚨 CRITICAL: Flush recurrent memory states before generating to prevent hallucinations
    if hasattr(model, "linear_cache"):
        model.linear_cache.reset()
    elif hasattr(model, "tptt_model") and hasattr(model.tptt_model, "linear_cache"):
        model.tptt_model.linear_cache.reset()

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    model.to(device).eval()

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=50, do_sample=False)
    print("✨ OUTPUT:\n", tokenizer.decode(outputs[0], skip_special_tokens=True))
    print("=" * 50)


# ==========================================
# CLI Arguments
# ==========================================
def parse_args():
    parser = argparse.ArgumentParser(description="Production TPTT Training Script")

    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--dataset_config", type=str, default=None)
    parser.add_argument("--dataset_split", type=str, default="train")
    parser.add_argument("--train_samples", type=int, default=100_000)
    parser.add_argument("--text_column", type=str, default="text")
    parser.add_argument("--output_dir", type=str, default="./tptt-trained-model")

    parser.add_argument("--max_length", type=int, default=8192)
    parser.add_argument("--chunk_size", type=int, default=256, help="Kept small to prevent Delta Product OOM.")
    parser.add_argument("--operator_mode", type=str, default="delta_rule")
    parser.add_argument("--cross_gate", action="store_true")

    parser.add_argument("--liza_mode", type=str, default="linear", choices=["constant", "linear", "cyclic"])
    parser.add_argument("--liza_weight", type=float, default=0.5)

    parser.add_argument("--attn_impl", type=str, default="sdpa")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--use_4bit", action="store_true")

    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--lora_target_modules", nargs="+", default=["q_proj", "k_proj", "v_proj", "o_proj"])

    parser.add_argument("--batch_size", type=str, default="auto", help="'auto' for automatic VRAM-based sizing")
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=-1)  # -1 defers to epochs
    parser.add_argument("--lr", type=float, default=2e-4)

    return parser.parse_args()


def main():
    args = parse_args()
    print(f"🚀 Initializing production TPTT training for: {args.model_name}")

    dtype = torch.float16 if args.fp16 else torch.bfloat16

    if args.batch_size.lower() == "auto":
        actual_batch_size = estimate_optimal_batch_size(args)
    else:
        actual_batch_size = int(args.batch_size)

    # ==========================================
    # 1. Tokenizer Setup
    # ==========================================
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token or "[PAD]"
    tokenizer.padding_side = "right"

    # ==========================================
    # 2. Packed Dataset Stream
    # ==========================================
    print(f"🌊 Streaming dataset: {args.dataset}")
    dataset_kwargs = {"split": args.dataset_split, "streaming": True}
    if args.dataset_config:
        dataset_kwargs["name"] = args.dataset_config

    dataset = load_dataset(args.dataset, **dataset_kwargs)
    eval_steps = 500

    print(f"✂️ Extracting {args.train_samples} samples and packing perfectly dense sequences (Zero Padding)...")
    eval_chunk = list(dataset.take(eval_steps))
    train_chunk = list(dataset.skip(eval_steps).take(args.train_samples))

    eval_dataset = Dataset.from_generator(
        lambda: packed_stream_generator(eval_chunk, tokenizer, args.max_length, args.text_column)
    )
    train_dataset = Dataset.from_generator(
        lambda: packed_stream_generator(train_chunk, tokenizer, args.max_length, args.text_column)
    )

    # ==========================================
    # 3. LoRA & Quantization Configuration
    # ==========================================
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=dtype,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    ) if args.use_4bit else None

    lora_config_dict = None
    if args.use_lora:
        # 🚨 FIX: Target the base modules with LoRA, but fully unfreeze and SAVE the injected memory gates!
        modules_to_save = ["memory_gate", "mapping_func"]

        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=list(set(args.lora_target_modules)),
            modules_to_save=modules_to_save,
        )
        lora_config_dict = lora_config.to_dict()

    # ==========================================
    # 4. TPTT Model Construction
    # ==========================================
    print(f"🧠 Injecting TPTT architecture into {args.model_name}...")
    model_config = tptt.TpttConfig(
        base_model_name=args.model_name,
        operator_mode=args.operator_mode,
        model_task="causal_lm",
        max_chunk_size=args.chunk_size,
        lora_config=lora_config_dict,
        mag_weight=args.liza_weight,
        cross_gate=args.cross_gate,
        linear_precision="float16" if args.fp16 else "bfloat16",
        use_linear_checkpoint=True,  # 🚨 FIX: Saves massive VRAM on chunks
        padding_side=tokenizer.padding_side,
        trust_remote_code=True,
    )

    model = tptt.TpttModel(
        model_config,
        trust_remote_code=True,
        attn_implementation=args.attn_impl,
        torch_dtype=dtype,
        quantization_config=bnb_config,
    )

    if len(tokenizer) > model.config.vocab_size:
        model.resize_token_embeddings(len(tokenizer))

    # 🚨 FIX: Monkey-patch Gradient Checkpointing for Trainer compatibility
    if hasattr(model, "tptt_model"):
        if hasattr(model.tptt_model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable = model.tptt_model.gradient_checkpointing_enable
        if hasattr(model.tptt_model, "gradient_checkpointing_disable"):
            model.gradient_checkpointing_disable = model.tptt_model.gradient_checkpointing_disable
        if hasattr(model.tptt_model, "enable_input_require_grads"):
            model.enable_input_require_grads = model.tptt_model.enable_input_require_grads
        model.supports_gradient_checkpointing = True

    # ==========================================
    # 5. Training Arguments
    # ==========================================
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=actual_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        weight_decay=0.01,
        bf16=not args.fp16,
        fp16=args.fp16,

        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        ddp_find_unused_parameters=False,

        logging_steps=10,
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=1000,
        report_to="none",
        remove_unused_columns=False,
    )

    liza_callback = tptt.LiZACallback(
        model=model,
        mode=args.liza_mode,
        initial_weight=0.0 if args.liza_mode == "linear" else args.liza_weight,
        final_weight=args.liza_weight,
        transition_step=500,
        weight_list=[0.0, 0.5, 1.0],
        switch_period=1,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        callbacks=[liza_callback],
    )

    # ==========================================
    # 6. Execute & Post-Processing
    # ==========================================
    print("🔥 Starting Training...")
    trainer.train()

    print(f"✅ Saving model to {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # 🚨 INTEGRATED from prototype: Model Card Generation
    print("📝 Generating Model Card...")
    try:
        last_log = trainer.state.log_history[-1] if len(trainer.state.log_history) > 0 else {}
        train_vars = {
            "batch_size": actual_batch_size,
            "dataset": args.dataset,
            "loss": last_log.get("loss", last_log.get("train_loss", "N/A")),
            "learning_rate": args.lr,
            "epochs": args.epochs,
        }
        tptt.generate_model_card(
            output_path=args.output_dir,
            config=model.config,
            template="model_card_template",
            extra_variables=train_vars,
        )
    except Exception as e:
        print(f"⚠️ Could not generate model card: {e}")

    # 🚨 INTEGRATED from prototype: Prove inference works post-training!
    dummy_generate(model, tokenizer)


if __name__ == "__main__":
    main()