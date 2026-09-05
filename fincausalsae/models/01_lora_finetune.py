import sys
import math
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.cli import parse_mode

args = parse_mode()

import config
from config import get_logger

log = get_logger("phase1")
config.mode_banner()

import torch
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType


def load_tokenizer():
    tok = AutoTokenizer.from_pretrained(config.BASE_MODEL, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    return tok


def get_lora_config():
    # gpt2 and Llama use different attention module names.
    target_modules = (
        ["c_attn", "c_proj", "c_fc"] if config.BASE_MODEL == "gpt2"
        else ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    return LoraConfig(
        r=config.LORA_R,
        lora_alpha=config.LORA_ALPHA,
        target_modules=target_modules,
        lora_dropout=config.LORA_DROPOUT,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )


def prepare_dataset(tokenizer):
    """Loads the Phase 0 output and tokenizes for continued pretraining."""
    path = config.PROC_DIR / "train.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python data/00_collect_data.py "
            f"--mode {args.mode}` first."
        )
    df = pd.read_parquet(path)

    fomc_path = config.RAW_DIR / "fomc_minutes.parquet"
    if fomc_path.exists():
        df_fomc = pd.read_parquet(fomc_path)
        df = pd.concat([df[["text"]], df_fomc[["text"]]], ignore_index=True)

    char_limit = config.MAX_SEQ_LEN * 4
    df = df[["text"]].copy()
    df["text"] = df["text"].str.slice(0, char_limit)
    dataset = Dataset.from_pandas(df, preserve_index=False)
    log.info(f"Dataset size: {len(dataset)} documents")
    return dataset


def tokenize_dataset(dataset, tokenizer):
    def _tok(batch):
        return tokenizer(
            batch["text"], truncation=True, max_length=config.MAX_SEQ_LEN,
            padding="max_length",
        )
    tokenized = dataset.map(_tok, batched=True, remove_columns=["text"])
    tokenized = tokenized.map(lambda ex: {"labels": ex["input_ids"]})
    return tokenized


def compute_perplexity(model, tokenizer, texts, device="cpu", n_samples=20):
    model.eval()
    total_loss, n_tokens = 0.0, 0
    for text in texts[:n_samples]:
        enc = tokenizer(text[:1024], return_tensors="pt", truncation=True)
        enc = {k: v.to(device) for k, v in enc.items()}
        if enc["input_ids"].shape[1] < 2:
            continue
        with torch.no_grad():
            out = model(**enc, labels=enc["input_ids"])
        total_loss += out.loss.item() * enc["input_ids"].numel()
        n_tokens += enc["input_ids"].numel()
    model.train()
    return math.exp(total_loss / max(n_tokens, 1))


def train():
    tokenizer = load_tokenizer()
    dataset = prepare_dataset(tokenizer)
    tokenized = tokenize_dataset(dataset, tokenizer)

    log.info(f"Loading base model {config.BASE_MODEL}...")
    model_kwargs = dict(torch_dtype=config.DTYPE, trust_remote_code=True)

    if not config.DEMO_MODE:
        from transformers import BitsAndBytesConfig
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
        )
        model_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(config.BASE_MODEL, **model_kwargs)
    model.config.use_cache = False
    if hasattr(model.config, "pretraining_tp"):
        model.config.pretraining_tp = 1
    if config.DEMO_MODE:
        model = model.to(config.DEVICE)

    model = get_peft_model(model, get_lora_config())
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=str(config.MODEL_OUT),
        num_train_epochs=config.EPOCHS,
        per_device_train_batch_size=config.BATCH_SIZE,
        gradient_accumulation_steps=config.GRAD_ACCUM,
        learning_rate=config.LR_LORA,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        bf16=(config.DTYPE == torch.bfloat16 and config.DEVICE == "cuda"),
        logging_steps=5 if config.DEMO_MODE else 50,
        save_strategy="no" if config.DEMO_MODE else "steps",
        save_steps=500,
        report_to=[],
        dataloader_num_workers=0 if config.DEMO_MODE else 4,
        group_by_length=True,
    )

    from transformers import Trainer, DataCollatorForLanguageModeling
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model, args=training_args, train_dataset=tokenized,
        data_collator=collator,
    )

    val_path = config.PROC_DIR / "val.parquet"
    val_texts = pd.read_parquet(val_path)["text"].tolist() if val_path.exists() else dataset["text"][:10]

    baseline_ppl = compute_perplexity(model, tokenizer, val_texts, device=config.DEVICE)
    log.info(f"Baseline perplexity: {baseline_ppl:.2f}")

    trainer.train()

    final_ppl = compute_perplexity(model, tokenizer, val_texts, device=config.DEVICE)
    ppl_reduction = (baseline_ppl - final_ppl) / baseline_ppl * 100
    log.info(f"Final perplexity: {final_ppl:.2f} ({ppl_reduction:.1f}% reduction)")

    if ppl_reduction < 5 and config.DEMO_MODE:
        log.info("Small perplexity change is expected in demo mode (tiny data, 1 epoch).")
    elif ppl_reduction < 15:
        log.warning("Less than 15% perplexity reduction — consider more epochs/data.")
    else:
        log.info("Domain adaptation looks solid.")

    merged_dir = config.MODEL_OUT / "merged"
    model = model.merge_and_unload()
    model.save_pretrained(merged_dir)
    tokenizer.save_pretrained(merged_dir)
    log.info(f"Merged model saved to {merged_dir}")
    log.info(f"Next: python sae/02_train_sae.py --mode {args.mode}")


if __name__ == "__main__":
    train()
