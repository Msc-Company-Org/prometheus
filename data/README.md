# Prometheus dataset

Training data for the customer-support reply-generation fine-tune. This repo ships a small
**illustrative sample** ([`sample.jsonl`](sample.jsonl), 12 rows); the full reference dataset is
not distributed.

## Format

One JSON object per line:

```json
{
  "customer_message": "string — the inbound customer message",
  "context": "string — account state + KB/policy snippets the reply must be grounded in",
  "reply": "string — the on-brand support reply (training target)"
}
```

At training time each row is rendered into a chat template as
`{system, user(context + customer_message), assistant(reply)}`, with loss computed on the
assistant turn only (see [`../training/train.py`](../training/train.py)).

## Size & splits

| Split | Rows | Purpose |
|---|---|---|
| train | 37,400 | fine-tuning |
| dev | 600 | checkpoint selection (eval loss + win rate) |
| test | 1,200 | held-out eval, blind human-pref review |
| **total** | **39,200** | |

Splits are disjoint at the **account** level — no account appears in more than one split — to
prevent style leakage from inflating the win rate.

## Domain

B2B SaaS / fintech support: billing and invoices, subscriptions and seats, auth/SSO, API rate
limits and webhooks, payouts and refunds, data export, security/compliance questions, account
recovery, cancellation. Companies and identifiers are fictional.

## Synthetic / real mix

| Source | Share | Notes |
|---|---|---|
| Reviewed historical agent replies | ~62% | Anonymized, partner-contributed under data agreement; only replies that passed QA. |
| Synthetic, policy-grounded | ~38% | Tickets generated from policy docs + lead-curated style guides, then human-edited. |

Synthetic rows exist to cover long-tail intents and insufficient-context / refusal cases that are
rare in logs. Every synthetic reply was reviewed by a support lead before inclusion.

## Deduplication

- Exact-duplicate `(customer_message, context)` pairs removed.
- Near-duplicates removed via MinHash/LSH on the message+context (Jaccard ≥ 0.9).
- Cross-split leakage check: any test account or near-duplicate message found in train is dropped
  from train.

## PII scrubbing

All rows are scrubbed **before** training:

- Names, emails, phone numbers, postal addresses → typed placeholders.
- Card numbers → `card ending NNNN`; transaction/account IDs → synthetic tokens (e.g. `txn_8842`).
- Free-text fields passed through a PII detector; flagged spans masked and the row re-reviewed.

Note: the model grounds replies in whatever `context` you pass at inference. Scrubbing protects the
*training* data — do not place raw customer PII in `context` you would not want echoed in an
outbound reply.

---

> Reference dataset by **MSC Labs** — done-for-you custom model training.
> Want this for your task? → Book a free model audit: https://labs.msccompany.com.br/assessment
> Numbers and samples are illustrative reference material from our standard eval harness.
