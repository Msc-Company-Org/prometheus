#!/usr/bin/env python
# Prometheus — QLoRA fine-tune for customer-support reply generation.
# Base: meta-llama/Llama-3.1-8B-Instruct. TRL SFTTrainer + PEFT + bitsandbytes.
#
# Two-stage protocol (see training/config.yaml):
#   stage 1  smoke test  -> python train.py --config training/config.yaml --stage smoke_test
#   stage 2  full run     -> python train.py --config training/config.yaml --stage full_run
#
# Illustrative reference script from the MSC Labs eval harness. License: Apache-2.0.

import argparse
import json
import logging

import torch
import yaml
from datasets import load_dataset
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTConfig, SFTTrainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("prometheus.train")

SYSTEM_PROMPT = (
    "You are a senior support agent. Write one concise, on-brand reply. "
    "Use only the provided context. If the answer is not in context, say so and offer next steps."
)


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def build_messages(row: dict) -> dict:
    """Map a raw {customer_message, context, reply} row into chat messages."""
    user = f"Context:\n{row['context']}\n\nCustomer message:\n{row['customer_message']}"
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
            {"role": "assistant", "content": row["reply"]},
        ]
    }


def load_split(path: str, tokenizer, max_samples: int | None):
    ds = load_dataset("json", data_files=path, split="train")
    ds = ds.map(build_messages, remove_columns=ds.column_names)
    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))
    ds = ds.map(
        lambda b: {"text": tokenizer.apply_chat_template(b["messages"], tokenize=False)},
        remove_columns=["messages"],
    )
    return ds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="training/config.yaml")
    parser.add_argument("--stage", choices=["smoke_test", "full_run"], default="full_run")
    args = parser.parse_args()

    cfg = load_config(args.config)
    stage = cfg["stages"][args.stage]
    log.info("Starting stage=%s base=%s", args.stage, cfg["model"]["base_model"])

    # --- tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["base_model"])
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    tokenizer.padding_side = "right"

    # --- 4-bit base model (QLoRA) ---
    q = cfg["quantization"]
    bnb = BitsAndBytesConfig(
        load_in_4bit=q["load_in_4bit"],
        bnb_4bit_quant_type=q["bnb_4bit_quant_type"],
        bnb_4bit_compute_dtype=getattr(torch, q["bnb_4bit_compute_dtype"]),
        bnb_4bit_use_double_quant=q["bnb_4bit_use_double_quant"],
    )
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model"]["base_model"],
        quantization_config=bnb,
        attn_implementation=cfg["model"]["attn_implementation"],
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.config.use_cache = False

    # --- LoRA ---
    lcfg = cfg["lora"]
    peft_config = LoraConfig(
        r=lcfg["r"],
        lora_alpha=lcfg["alpha"],
        lora_dropout=lcfg["dropout"],
        bias=lcfg["bias"],
        task_type=lcfg["task_type"],
        target_modules=lcfg["target_modules"],
    )

    # --- data ---
    max_train = stage.get("max_train_samples")
    train_ds = load_split(cfg["data"]["train_file"], tokenizer, max_train)
    eval_ds = load_split(cfg["data"]["eval_file"], tokenizer, None)
    log.info("train rows=%d  eval rows=%d", len(train_ds), len(eval_ds))

    # --- trainer ---
    t = cfg["train"]
    sft_config = SFTConfig(
        output_dir=t["output_dir"],
        num_train_epochs=stage["num_train_epochs"],
        per_device_train_batch_size=t["per_device_train_batch_size"],
        gradient_accumulation_steps=t["gradient_accumulation_steps"],
        learning_rate=t["learning_rate"],
        lr_scheduler_type=t["lr_scheduler_type"],
        warmup_ratio=t["warmup_ratio"],
        max_grad_norm=t["max_grad_norm"],
        optim=t["optim"],
        bf16=t["bf16"],
        gradient_checkpointing=t["gradient_checkpointing"],
        logging_steps=t["logging_steps"],
        eval_strategy=t["eval_strategy"],
        eval_steps=t["eval_steps"],
        save_steps=t["save_steps"],
        save_total_limit=t["save_total_limit"],
        load_best_model_at_end=t["load_best_model_at_end"],
        metric_for_best_model=t["metric_for_best_model"],
        greater_is_better=t["greater_is_better"],
        max_seq_length=cfg["model"]["max_seq_len"],
        completion_only_loss=cfg["data"]["completion_only_loss"],
        packing=cfg["data"]["packing"],
        seed=t["seed"],
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        peft_config=peft_config,
        processing_class=tokenizer,
    )

    trainer.train()
    metrics = trainer.evaluate()
    log.info("final eval metrics: %s", json.dumps(metrics))

    # Smoke-test gate: refuse to "pass" if eval loss is above the configured threshold.
    if args.stage == "smoke_test":
        gate = stage["eval_loss_gate"]
        loss = metrics.get("eval_loss", float("inf"))
        if loss > gate:
            raise SystemExit(f"Smoke test FAILED: eval_loss {loss:.3f} > gate {gate}. Fix before full run.")
        log.info("Smoke test PASSED (eval_loss %.3f <= %.3f). Cleared for full run.", loss, gate)

    trainer.save_model(t["output_dir"])
    tokenizer.save_pretrained(t["output_dir"])
    log.info("saved adapter to %s", t["output_dir"])


if __name__ == "__main__":
    main()
