# Prometheus — customer-support reply generation

A fine-tuned **Llama-3.1-8B-Instruct** that writes on-brand, accurate B2B SaaS / fintech
support replies. In blind side-by-side review by support leads, it wins **94%** of
head-to-head comparisons against **GPT-4o** — at **22× lower** inference cost.

## What it is

Prometheus drafts the reply a human agent would send: grounded in the customer's message
and the supplied account/KB context, in the company's tone, without hallucinating policy.
It is a single-task model, not a chatbot. Give it `{customer_message, context}`, get back a
ready-to-send `reply`.

## Why a tuned 8B beats a frontier API here

Support reply quality is narrow and stylistic. A frontier model is general; it tends to be
verbose, over-hedge, and drift off house tone. Fine-tuning a small open model on ~38k
reviewed agent replies pins down tone, length, escalation rules, and refusal behavior — the
parts the baseline gets wrong — while staying cheap enough to run on every ticket.

## Results

Illustrative reference results from the MSC Labs eval harness. Baseline: GPT-4o (`gpt-4o-2024-08-06`).
Test set: 1,200 held-out tickets, blind side-by-side review by 4 support leads.

| Metric | GPT-4o (baseline) | Prometheus (8B, tuned) |
|---|---|---|
| Human-preference win rate | — | **94%** |
| Factual-grounding error rate | 6.1% | **1.4%** |
| Tone / brand-fit (1–5, lead-rated) | 3.6 | **4.6** |
| $ / 1k requests | $9.20 | **$0.41** |
| p50 latency | 1,910 ms | **430 ms** |

Win rate counts the share of tickets where leads preferred the Prometheus reply over GPT-4o
(ties split). Cost is **22.4× cheaper** per 1k requests. Full methodology, automatic metrics,
and the 2-stage training protocol are in [`eval/results.md`](eval/results.md).

## Quickstart

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

BASE = "meta-llama/Llama-3.1-8B-Instruct"
ADAPTER = "msc-labs/prometheus-support-8b"  # illustrative reference adapter

tok = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16, device_map="auto")
model = PeftModel.from_pretrained(model, ADAPTER)

SYSTEM = (
    "You are a senior support agent. Write one concise, on-brand reply. "
    "Use only the provided context. If the answer is not in context, say so and offer next steps."
)

def reply(customer_message: str, context: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"Context:\n{context}\n\nCustomer message:\n{customer_message}"},
    ]
    inputs = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
    out = model.generate(inputs, max_new_tokens=256, temperature=0.3, top_p=0.9)
    return tok.decode(out[0][inputs.shape[-1]:], skip_special_tokens=True)

print(reply(
    "I was charged twice for my May invoice. Can you refund the duplicate?",
    "Account: Pro plan. Billing log shows two charges of $49 on 2025-05-01 (txn_8841, txn_8842). "
    "Policy: duplicate charges are refundable within 60 days to the original card.",
))
```

## License

Apache-2.0 for code (see repo headers). Reference weights and datasets are illustrative and
not distributed.

---

> Reference model by **MSC Labs** — done-for-you custom model training.
> Want this for your task? → Book a free model audit: https://labs.msccompany.com.br/assessment
> Numbers are illustrative reference results from our standard eval harness.
