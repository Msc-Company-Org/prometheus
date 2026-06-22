# Model Card — Prometheus (support-reply 8B)

Prometheus is a QLoRA fine-tune of `meta-llama/Llama-3.1-8B-Instruct` for **customer-support
reply generation** in B2B SaaS / fintech. It produces a single, on-brand reply grounded in the
customer message and a supplied account/KB context.

- **Developed by:** MSC Labs
- **Model type:** Decoder-only LLM, LoRA adapter over Llama-3.1-8B-Instruct
- **Language:** English
- **License:** Apache-2.0 (code/adapter); base model under the Llama 3.1 Community License
- **Finetuned from:** `meta-llama/Llama-3.1-8B-Instruct`

> All quantitative claims below are **illustrative reference results from the MSC Labs eval
> harness**, included to document the pipeline MSC Labs delivers. Reference weights and data
> are not distributed.

## Intended use

- **Primary use:** Draft-generation for human support agents — suggest the reply, agent reviews
  and sends. Channels: email, chat, WhatsApp / in-app messaging.
- **Users:** Support teams at SaaS / fintech companies with a defined tone of voice and a
  knowledge base or account-context retrieval layer.
- **Input contract:** `{"customer_message": str, "context": str}` where `context` carries the
  relevant account state and KB/policy snippets. The model is trained to ground replies in
  `context` and to defer when context is insufficient.

## Out-of-scope use

- **Autonomous send without human review** for billing, refunds, account changes, or any action
  with financial or compliance impact.
- **Source of truth for policy or facts.** The model writes the reply; it does not retrieve.
  Wrong/empty `context` yields wrong replies.
- **Open-domain chat, multi-turn negotiation, or tool execution.** Single-task, single-turn.
- **Regulated advice** (legal, tax, individualized financial advice) beyond restating provided
  policy text.
- **Languages other than English** — not trained or evaluated.

## Training data

- **Size:** ~38,000 training pairs (37,400 train / 1,200 held-out test / 600 dev).
- **Domain:** B2B SaaS and fintech support — billing, subscriptions, auth/SSO, API limits, KYC,
  payouts, refunds, onboarding.
- **Composition:** ~62% reviewed historical agent replies (anonymized, partner-contributed under
  agreement), ~38% synthetic tickets generated from policy docs and lead-curated style guides,
  then human-edited. See [`data/README.md`](data/README.md) for the full breakdown.
- **Format:** chat-templated `{system, user(context+message), assistant(reply)}`. Loss computed on
  the assistant turn only.
- **PII:** scrubbed before training — names, emails, card/account numbers, addresses replaced with
  typed placeholders (e.g. `txn_8842`, `card ending 4242`). See data README for the scrub policy.

## Training procedure

**Method:** QLoRA (4-bit NF4 base, LoRA rank 16) with TRL `SFTTrainer`. **2-stage protocol:**

1. **Stage 1 — smoke test.** 1k-row subset, 1 epoch, on a single GPU. Validates the data schema,
   chat template, loss masking, and that train loss decreases and a held-out sample renders. Gate:
   no crash, eval loss below the stage-1 threshold, manual spot-check of 20 generations.
2. **Stage 2 — full run.** Full 37.4k set, 3 epochs, on the configured GPU. Promoted only after
   Stage 1 passes. Checkpoint selection by dev-set reward-model score + win rate vs the baseline.

Key hyperparameters (full set in [`training/config.yaml`](training/config.yaml)): LoRA r=16,
α=32, dropout 0.05; targets q/k/v/o + gate/up/down proj; lr 2e-4 cosine, warmup 3%; effective
batch 64 (bs 8 × grad-accum 8); max seq 4096; bf16; ~6 GPU-hours on one 80 GB GPU for the full run.

## Evaluation

- **Primary metric:** human-preference win rate vs GPT-4o — blind side-by-side review by 4 support
  leads on 1,200 held-out tickets. Reply source order randomized; ties split.
- **Secondary:** factual-grounding error rate (claims not supported by `context`), tone/brand-fit
  (1–5), automatic ROUGE-L / BERTScore vs gold reply, refusal correctness on insufficient-context
  cases.
- **Cost & latency:** $ / 1k requests and p50 latency measured on the harness reference deployment.

| Metric | GPT-4o | Prometheus (8B) |
|---|---|---|
| Human-pref win rate | — | 94% |
| Grounding error rate | 6.1% | 1.4% |
| Tone / brand-fit (1–5) | 3.6 | 4.6 |
| $ / 1k requests | $9.20 | $0.41 |
| p50 latency | 1,910 ms | 430 ms |

Full methodology, automatic metrics, and the cost model are in [`eval/results.md`](eval/results.md).

## Limitations

- **Grounding-bound.** Quality is capped by `context`. The reduced (not zero) grounding-error rate
  still means ~1 in 70 replies asserts something unsupported — human review remains required.
- **Tuned to one house style.** A model tuned for one brand's tone will not match another's;
  re-tuning is part of the per-customer pipeline.
- **English / in-domain only.** Off-domain tickets (HR, hardware RMA, healthcare) are untested.
- **Single-turn.** No conversation memory; multi-turn threads must be flattened into `context`.
- **Static knowledge.** The adapter does not learn new policies after training; policy lives in
  retrieval, not weights.

## Bias, risks, and mitigations

- **Tone bias toward the training brand** — replies may sound overly accommodating or apply the
  source company's refund/escalation posture. Mitigated by per-customer tuning and tone review.
- **Overconfident phrasing.** The model can state defers/uncertainty fluently enough to read as
  fact. Mitigation: grounding checks in the harness, mandatory human-in-the-loop for actionable
  replies.
- **PII leakage.** Training data is scrubbed, but generated replies echo whatever is in `context`;
  do not place raw PII in `context` you would not want in an outbound message.
- **Automation pressure.** Recommended deployment is draft-assist, not auto-send. Financial/account
  actions must route through existing authorization controls.

---

> Reference model by **MSC Labs** — done-for-you custom model training.
> Want this for your task? → Book a free model audit: https://labs.msccompany.com.br/assessment
> Numbers are illustrative reference results from our standard eval harness.
