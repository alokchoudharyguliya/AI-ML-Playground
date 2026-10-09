## One objective to rule them all

A **GPT-style** model is a Transformer *decoder* (causal mask, no encoder) trained on a single, brutally simple objective: **predict the next token**. Factorise the probability of a text with the chain rule,

$$p(x_1,\dots,x_n)=\prod_{t=1}^{n}p(x_t\mid x_{<t}),$$

and minimise the cross-entropy at *every* position of *every* sequence. No labels, no task engineering — just trillions of tokens of text and code. Everything else (translation, summarisation, arithmetic, reasoning, tool use) must be learned *in service of* predicting the next token well.

| Generation | Year | Params | Notes |
|---|---|---|---|
| GPT-1 | 2018 | 117 M | pre-train + fine-tune |
| GPT-2 | 2019 | 1.5 B | zero-shot task transfer emerges |
| GPT-3 | 2020 | 175 B | **in-context learning**: examples in the prompt, no gradient updates |
| InstructGPT / ChatGPT | 2022 | — | supervised fine-tuning + RLHF → follows instructions |
| GPT-4, Llama 2/3, Mistral, Claude, Gemini… | 2023– | 7 B – 1 T+ | scaling, data quality, MoE, multimodality, long context |

## From base model to assistant

1. **Pre-training** (≈ 99% of compute): next-token prediction on web, books, code.
2. **Supervised fine-tuning (SFT)** on curated instruction–response pairs.
3. **Preference optimisation**: RLHF (reward model + PPO), or DPO / IPO / KTO directly on preference pairs; RL against verifiable rewards for reasoning.
4. **Safety and tool training**: refusal behaviour, function calling, retrieval.

## Decoding: turning probabilities into text

The model outputs logits over the vocabulary. *How you pick the next token* changes the text dramatically (first lab):

- **Greedy** ($T\to0$): always the argmax — deterministic, repetitive, can loop.
- **Beam search**: tracks the best $k$ sequences; good for translation/ASR, dull for open-ended text.
- **Temperature** $T$: divide logits by $T$ before the softmax.
- **Top-k**: sample only from the $k$ most likely tokens.
- **Top-p (nucleus)**: sample from the smallest set whose cumulative probability $\ge p$ — adapts to confidence.
- **Min-p**: drop tokens with probability $<\text{min\_p}\times p_{max}$.
- **Penalties**: repetition / frequency / presence penalties discourage loops.
- **Constrained decoding**: mask tokens that would violate a JSON schema or grammar (guaranteed valid output).
- **Speculative decoding**: a small draft model proposes $\gamma$ tokens, the big model verifies them in one parallel pass — 2–3× faster with *identical* output distribution.

## Inference has two very different phases

| Phase | What happens | Bound by |
|---|---|---|
| **Prefill** | process the whole prompt in parallel; build the KV cache | **compute** (big GEMMs, Tensor Cores) |
| **Decode** | one token at a time, reading all weights + KV cache | **memory bandwidth** |

**KV cache.** At step $t$ the new token's query must attend to the keys/values of all previous tokens. Those never change (causal mask), so we cache them: each step computes only one new $(k,v)$ per layer instead of re-encoding the entire prefix — $O(N)$ instead of $O(N^2)$ passes. The price is memory: $2\cdot L\cdot n_{kv}\cdot d_{head}\cdot\text{bytes}$ per token. For Llama-3-8B at 128k context that is 16 GB *per sequence* — often more than the weights. Second lab tab.

How production systems cope:

- **GQA / MQA** — share K/V heads across query heads (4–8× smaller cache).
- **PagedAttention** (vLLM) — page-table the cache to eliminate fragmentation.
- **Continuous batching** — admit/evict requests every step so the GPU stays full.
- **KV quantisation** (fp8 / int4), **prefix caching** (reuse the cache of a shared system prompt), **sliding-window / sparse** attention, **FlashAttention / FlashDecoding** kernels.
- **Tensor / pipeline parallelism** for models larger than one GPU.

Metrics that matter: **TTFT** (time to first token, prefill-bound), **TPOT** (time per output token, decode-bound), throughput (tokens/s/GPU) and cost per million tokens.

## In-context learning and prompting

Large models perform **in-context learning**: put a few examples in the prompt and the model infers the task without weight updates. Mechanistically, *induction heads* (attend to the token after a previous occurrence of the current token) and larger circuits implement pattern completion. Prompt techniques — few-shot examples, system prompts, **chain-of-thought** ("think step by step"), self-consistency, tool use and retrieval-augmented generation — all work by shaping the context the model conditions on.

## Known failure modes

- **Hallucination**: fluent but false content; mitigated by retrieval, tool use, calibration training and verification.
- **Exposure bias & repetition** under greedy decoding.
- **Context-window limits** and "lost in the middle" effects.
- **Tokenisation artefacts**: counting letters, arithmetic on long numbers, non-English cost.
- **Prompt injection** and jailbreaks in tool-using agents.
- **Evaluation contamination**: benchmark leakage in training data.
