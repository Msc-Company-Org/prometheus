# Prometheus — evaluation

Illustrative reference results from the MSC Labs eval harness. They document the pipeline
MSC Labs delivers; reference weights and data are not distributed.

- **Model:** Prometheus — QLoRA fine-tune of `meta-llama/Llama-3.1-8B-Instruct`
- **Baseline:** GPT-4o (`gpt-4o-2024-08-06`), temperature 0.3, same system prompt
- **Task:** customer-support reply generation (B2B SaaS / fintech)
- **Test set:** 1,200 held-out tickets, disjoint from train/dev, no account overlap

## Methodology

### Blind human-preference eval (primary)

For each test ticket we generate one reply from Prometheus and one from GPT-4o, using identical
`{customer_message, context}` and system prompt. The two replies are shown side by side with the
source labels hidden and left/right order randomized per item. **4 support leads** (the same
people who own tone of voice in production) pick the reply they would send, or mark a tie.

- **Win rate** = share of comparisons where Prometheus is preferred, with ties counted as half.
- Inter-rater agreement (Cohen's κ, pairwise mean): **0.71**.
- Each ticket rated by 2 leads; disagreements adjudicated by a third.

### Automatic metrics (secondary)

- **Grounding error rate:** fraction of replies containing at least one claim not supported by
  `context`, labeled by an LLM-judge grounding check and audited on a 200-item human sample
  (judge–human agreement 0.93).
- **Tone / brand-fit:** lead-rated 1–5 against the published style guide.
- **ROUGE-L / BERTScore:** overlap with the gold agent reply (reference only; support replies are
  not unique, so these are sanity signals, not the target).
- **Refusal correctness:** on a 150-item insufficient-context slice, does the model correctly defer
  instead of inventing an answer.

### Cost & latency

Measured on the harness reference deployment. Prometheus served with vLLM on a single 80 GB GPU
(4-bit base + merged adapter), batch size 16. GPT-4o priced at public list rates on the measured
token mix (avg 540 input + 110 output tokens per request).

## Results vs GPT-4o

| Metric | GPT-4o | Prometheus (8B) | Delta |
|---|---|---|---|
| Human-preference win rate | — | **94%** | — |
| Grounding error rate | 6.1% | **1.4%** | −4.7 pts |
| Tone / brand-fit (1–5) | 3.6 | **4.6** | +1.0 |
| Refusal correctness (insufficient context) | 88% | **96%** | +8 pts |
| ROUGE-L vs gold | 0.41 | **0.57** | +0.16 |
| BERTScore-F1 vs gold | 0.86 | **0.91** | +0.05 |
| $ / 1k requests | $9.20 | **$0.41** | 22.4× cheaper |
| p50 latency | 1,910 ms | **430 ms** | 4.4× faster |
| p95 latency | 4,300 ms | **910 ms** | — |

Win-rate breakdown: Prometheus preferred 91.6%, tie 4.8%, GPT-4o preferred 3.6% → **94.0%** with
ties split.

## Cost breakdown

Per 1,000 requests, 540 input + 110 output tokens average.

| | GPT-4o | Prometheus (8B) |
|---|---|---|
| Input | $1.35 | — |
| Output | $1.10 | — |
| API total | **$9.20**¹ | — |
| GPU amortized (vLLM, 80 GB) | — | $0.41 |
| **$ / 1k requests** | **$9.20** | **$0.41** |

¹ GPT-4o total reflects the full measured request mix including system-prompt and context tokens
re-sent per call; the input/output line items above are the marginal per-message portion.
Self-hosted cost is the amortized GPU-second cost at the measured throughput (≈ 38 req/s sustained).
Net: **22.4× cheaper per request.**

## Two-stage training protocol

1. **Stage 1 — smoke test.** 1k rows, 1 epoch, single GPU (~12 min). Gate: no crash, eval loss
   ≤ 1.15, manual spot-check of 20 generations. Catches schema/template/loss-masking bugs before
   spending on the full run.
2. **Stage 2 — full run.** 37.4k rows, 3 epochs (~6 GPU-hours, one 80 GB GPU). Checkpoint chosen
   by dev-set reward score + win rate vs baseline (≥ 0.80 required to promote). The reported model
   passed both gates on the first full run.

## Limitations of this evaluation

- **One brand's tone.** Win rate is measured against the leads who own *this* style guide. A model
  tuned for a different brand is evaluated against that brand's leads; cross-brand numbers will
  differ.
- **Context is given, not retrieved.** The harness supplies gold-ish `context`. End-to-end quality
  in production also depends on the retrieval layer feeding `context`.
- **Preference ≠ correctness.** Leads can prefer a reply that is wrong; we pair win rate with the
  grounding-error and refusal-correctness metrics for this reason.
- **English, in-domain only.** No multilingual or off-domain (HR, hardware) measurement.
- **Latency is deployment-specific.** Numbers reflect the reference vLLM setup; your serving stack
  will vary.

---

> Reference model by **MSC Labs** — done-for-you custom model training.
> Want this for your task? → Book a free model audit: https://msc-labs-ai.vercel.app/assessment
> Numbers are illustrative reference results from our standard eval harness.
