import os
import argparse
# Prevent memory fragmentation in PyTorch during long context training (must be set before torch import)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
from datasets import load_dataset, IterableDataset, Dataset
from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
)
from peft import LoraConfig
import tptt




def packed_stream_generator(dataset_stream, tokenizer, max_length, text_column="text"):
    """
    Consumes the text stream, tokenizes on the fly,
    and yields perfectly packed sequences of max_length without padding.
    """
    buffer = []

    for row in dataset_stream:
        # Extract text and append the EOS token to act as a document boundary
        text = row[text_column] + tokenizer.eos_token

        # Tokenize raw text without padding or truncation
        # add_special_tokens=False because we manually appended eos_token
        tokens = tokenizer(text, add_special_tokens=False)["input_ids"]
        buffer.extend(tokens)

        # Once our buffer is larger than max_length, yield a perfect chunk
        while len(buffer) >= max_length:
            chunk = buffer[:max_length]
            buffer = buffer[max_length:]  # Keep the remainder for the next chunk

            yield {
                "input_ids": chunk,
                "labels": chunk.copy(),
                "attention_mask": [1] * max_length  # All 1s because there is zero padding
            }

def parse_args():
    parser = argparse.ArgumentParser(description="TPTT Training Script")

    # Model & Data Configuration
    parser.add_argument("--model_name", type=str, required=True,
                        help="Hugging Face model ID (e.g., Qwen/Qwen2.5-3B, mistralai/Mistral-7B-v0.3)")
    parser.add_argument("--dataset", type=str, required=True,
                        help="HuggingFace dataset name (e.g., HuggingFaceH4/ultrachat_200k)")
    parser.add_argument("--dataset_config", type=str, default=None,
                        help="Dataset config/subset name (e.g., sample-10BT for fineweb-edu)")
    parser.add_argument("--dataset_split", type=str, default="train", help="Dataset split to use for training")
    parser.add_argument("--train_samples", type=int, default=10000,
                        help="Number of samples to extract from the stream for training")
    parser.add_argument("--text_column", type=str, default=None,
                        help="Explicitly specify the column name containing the text data")
    parser.add_argument("--output_dir", type=str, default="./tptt-trained-model")

    # TPTT Specific Arguments
    parser.add_argument("--max_length", type=int, default=8192, help="Max sequence length for tokenization")
    parser.add_argument("--chunk_size", type=int, default=2048,
                        help="TPTT chunk size (controls VRAM usage, typically 2048 or 4096)")
    parser.add_argument("--operator_mode", type=str, default="delta_rule",
                        help="Operator mode for the recurrent memory (e.g., linear, delta_rule)")
    parser.add_argument("--cross_gate", action="store_true", help="Enable cross gating for memory")
    parser.add_argument("--liza_weight", type=float, default=0.5, help="Final weight for LiZA curriculum callback")

    # Model Agnostic Hardware/Backend Arguments
    parser.add_argument("--attn_impl", type=str, default="sdpa", choices=["sdpa", "flash_attention_2", "eager"],
                        help="Attention backend (sdpa is the safest default)")
    parser.add_argument("--fp16", action="store_true",
                        help="Use float16 precision instead of bfloat16 (necessary for pre-Ampere GPUs)")
    parser.add_argument("--use_4bit", action="store_true", help="Use 4-bit quantization (QLoRA)")

    # LoRA Arguments
    parser.add_argument("--use_lora", action="store_true", help="Use Parameter Efficient Fine-Tuning (LoRA)")
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument(
        "--lora_target_modules",
        nargs="+",
        default=["q_proj", "k_proj", "v_proj", "o_proj"],
        help="Space-separated list of base model modules to target. 'memory_gate' and 'mapping_func' are added automatically."
    )

    # Training Arguments
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)

    return parser.parse_args()


def format_and_tokenize(examples, tokenizer, max_length, text_column=None):
    """Agnostically formats data based on user specification or dataset structure."""

    # 1. User explicitly specified a column
    if text_column and text_column in examples:
        texts = examples[text_column]

    # 2. Conversational format (ChatML)
    elif "messages" in examples:
        try:
            texts = [tokenizer.apply_chat_template(msg, tokenize=False) for msg in examples["messages"]]
        except Exception:
            # Fallback if tokenizer lacks a chat template
            texts = ["\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in msg]) for msg in
                     examples["messages"]]

    # 3. Standard text format
    elif "text" in examples:
        texts = examples["text"]

    # 4. Alpaca-style Instruction format
    elif "instruction" in examples:
        texts = [
            f"User: {inst}\n{inp}\nAssistant: {out}" if inp else f"User: {inst}\nAssistant: {out}"
            for inst, inp, out in
            zip(examples.get("instruction", []), examples.get("input", [""] * len(examples["instruction"])),
                examples.get("output", []))
        ]

    else:
        raise ValueError(
            f"Dataset format not recognized. Available columns: {list(examples.keys())}. Please specify --text_column.")

    tokens = tokenizer(
        texts,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_attention_mask=True,
    )
    # Causal LM expects labels to mirror input_ids
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens


def main():
    args = parse_args()
    print(f"🚀 Initializing model-agnostic TPTT training for: {args.model_name}")

    dtype = torch.float16 if args.fp16 else torch.bfloat16

    # ==========================================
    # 1. Load Tokenizer
    # ==========================================
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)

    # Agnostic padding fallback logic
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        elif tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token
        else:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    # TPTT relies on causal autoregression, right padding is optimal
    tokenizer.padding_side = "right"

    # ==========================================
    # 2. Load and Prepare Dataset (CHUNK EXTRACTION)
    # ==========================================
    print(f"🌊 Streaming dataset: {args.dataset}")

    dataset_kwargs = {"split": args.dataset_split, "streaming": True}
    if args.dataset_config:
        dataset_kwargs["name"] = args.dataset_config

    dataset = load_dataset(args.dataset, **dataset_kwargs)

    eval_steps = 500
    
    print(f"📦 Extracting a chunk of {args.train_samples} training samples and {eval_steps} eval samples from the stream...")
    # Take a chunk for eval and a chunk for train
    eval_chunk = list(dataset.take(eval_steps))
    train_chunk = list(dataset.skip(eval_steps).take(args.train_samples))

    eval_dataset = Dataset.from_list(eval_chunk)
    train_dataset = Dataset.from_list(train_chunk)

    print("✂️ Tokenizing the extracted chunks...")
    column_names = train_dataset.column_names

    eval_dataset = eval_dataset.map(
        lambda x: format_and_tokenize(x, tokenizer, args.max_length, args.text_column),
        batched=True,
        remove_columns=column_names
    )

    train_dataset = train_dataset.map(
        lambda x: format_and_tokenize(x, tokenizer, args.max_length, args.text_column),
        batched=True,
        remove_columns=column_names
    )

    # ==========================================
    # 3. Configure Quantization & LoRA (PEFT)
    # ==========================================
    bnb_config = None
    if args.use_4bit:
        print("🗜️ Configuring 4-bit quantization...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    lora_config_dict = None
    if args.use_lora:
        # TPTT injects recurrent memory gates and mapping layers.
        # These MUST be targeted by LoRA to learn the cache updates.
        lora_targets = set(args.lora_target_modules)
        lora_targets.update(["memory_gate", "mapping_func"])

        print(f"🔧 Configuring LoRA targeting modules: {list(lora_targets)}")
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=list(lora_targets),
        )
        lora_config_dict = lora_config.to_dict()

    # ==========================================
    # 4. Configure and Initialize TPTT Model
    # ==========================================
    print(f"🧠 Injecting TPTT Titanesque architecture into {args.model_name}...")
    model_config = tptt.TpttConfig(
        base_model_name=args.model_name,
        operator_mode=args.operator_mode,
        model_task="causal_lm",
        max_chunk_size=args.chunk_size,
        lora_config=lora_config_dict,
        mag_weight=args.liza_weight,
        cross_gate=args.cross_gate,
        linear_precision="float16" if args.fp16 else "bfloat16",
        use_linear_checkpoint=True,
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

    # Resize embeddings in case a new [PAD] token had to be added
    if len(tokenizer) > model.config.vocab_size:
        model.resize_token_embeddings(len(tokenizer))

    if hasattr(model, "tptt_model"):
        # 1. Forward the enable/disable commands to the inner base model
        if hasattr(model.tptt_model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable = model.tptt_model.gradient_checkpointing_enable
        if hasattr(model.tptt_model, "gradient_checkpointing_disable"):
            model.gradient_checkpointing_disable = model.tptt_model.gradient_checkpointing_disable

        # 2. CRITICAL FOR LORA: Ensure frozen base inputs require gradients.
        # If the Trainer can't find this, PyTorch drops the gradients before they reach your adapters!
        if hasattr(model.tptt_model, "enable_input_require_grads"):
            model.enable_input_require_grads = model.tptt_model.enable_input_require_grads

        # 3. Bypass any older Trainer safety checks
        model.supports_gradient_checkpointing = True

    # ==========================================
    # 5. Setup Training Arguments & Callbacks
    # ==========================================
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,

        # REMOVE num_train_epochs. ADD max_steps.
        max_steps=10_000,  # Set this to however many update steps you want to run

        learning_rate=args.lr,
        weight_decay=0.01,
        bf16=not args.fp16,
        fp16=args.fp16,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=1000,
        report_to="none",
        remove_unused_columns=False,
    )

    # Setup LiZA Callback (Memory Gate Curriculum)
    liza_callback = tptt.LiZACallback(
        model=model,
        mode="linear",
        initial_weight=0.0,
        final_weight=args.liza_weight,
        transition_step=500,
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=[liza_callback],
    )

    # ==========================================
    # 6. Execute Training
    # ==========================================
    print("🔥 Starting Training...")
    trainer.train()

    print(f"✅ Training complete. Saving model to {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()