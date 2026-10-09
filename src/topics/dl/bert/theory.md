## Pre-train once, fine-tune everywhere

Before 2018 every NLP task trained its own model from scratch (or from static word vectors like word2vec/GloVe, where *bank* has one vector regardless of context). **BERT** (Devlin et al., 2018 — *Bidirectional Encoder Representations from Transformers*) established the recipe that still defines NLP:

1. **Pre-train** a big Transformer encoder on unlabeled text with a self-supervised objective.
2. **Fine-tune** the whole network (plus a tiny task head) on a small labeled dataset in minutes.

It set new state of the art on 11 benchmarks at once. Representations became **contextual**: “river **bank**” and “bank **loan**” get different vectors because every token is computed from attention over the whole sentence.

## The encoder: bidirectional on purpose

A left-to-right language model (GPT) cannot see the right context, which hurts understanding tasks. A naïve bidirectional LM would let each word "see itself" through the stack. BERT's trick is **masked language modelling (MLM)**: hide a random 15% of tokens and predict them from *both* sides.

The MLM selection rule (first lab tab) — of the chosen tokens

- **80%** → replaced with `[MASK]`,
- **10%** → replaced with a *random* token,
- **10%** → left unchanged,

so the model cannot rely on seeing `[MASK]` (it never appears at fine-tuning time) and must keep a good representation for **every** input token.

The original BERT also used **Next Sentence Prediction** (is B the sentence after A?). Later work (RoBERTa) showed it adds little and dropped it; ALBERT replaced it with sentence-order prediction.

## Architecture details

- **Input** (third lab tab): `[CLS] sentence A [SEP] sentence B [SEP]`; each position = token + segment + learned position embedding, LayerNorm'd.
- **Tokenizer**: WordPiece, 30k sub-words; unknown words split into `##` pieces.
- **Model**: 12 (base, 110M) or 24 (large, 340M) post-LN Transformer encoder layers, full bidirectional attention, GELU FFN, context ≤ 512.
- **Outputs**: one contextual vector per token. `[CLS]`'s final vector (through a small pooler) represents the whole sequence.

## Fine-tuning recipes

| Task | Head on top of BERT | Example |
|---|---|---|
| Sentence classification | linear on `[CLS]` | sentiment, intent |
| Sentence-pair classification | linear on `[CLS]` of `A [SEP] B` | NLI, paraphrase |
| Token classification | linear on each token | NER, POS tagging |
| Span extraction | start/end logits per token | SQuAD QA |
| Embeddings / retrieval | mean-pool (+ contrastive training) | semantic search (Sentence-BERT) |

Typical hyper-parameters: 2–4 epochs, LR $2\text{–}5\times10^{-5}$ with linear warm-up/decay, batch 16–32, AdamW weight decay 0.01. **Layer-wise LR decay** (lower LR for lower layers) and re-initialising the top layers help on small data.

## The family tree

| Model | Idea |
|---|---|
| **RoBERTa** | Same architecture, 10× more data, longer, dynamic masking, no NSP — a far stronger baseline |
| **ALBERT** | Factorised embeddings + cross-layer parameter sharing → 10× fewer params |
| **DistilBERT / TinyBERT** | Knowledge distillation: smaller, 2–6× faster, ~97% quality |
| **ELECTRA** | Replaced-token *detection* on every position — much more sample-efficient |
| **DeBERTa** | Disentangled content/position attention; top of GLUE for years |
| **SpanBERT / T5** | Mask contiguous spans (T5: encoder–decoder with span corruption) |
| **Sentence-BERT, E5, BGE** | Contrastive fine-tuning for dense retrieval embeddings |
| **ModernBERT (2024)** | RoPE, GeGLU, FlashAttention, 8k context, modern data — BERT with a decade of lessons |

## Where encoders still beat LLMs

Large decoder LLMs get the headlines, but **encoders remain the right tool** when you need: *low latency/cost* classification at scale, **embeddings and rerankers for retrieval** (RAG pipelines use bi-encoders for recall and cross-encoders for precision), NER/PII tagging, content moderation, and on-device NLP. A 100M-parameter encoder is 100–1000× cheaper per query than prompting a frontier LLM.

## Limitations

- Context length (512 in the original) and quadratic attention.
- Cannot generate coherent text natively (mask-filling is not autoregression).
- MLM trains on only 15% of positions (sample-inefficient).
- Pre-train/fine-tune mismatch from `[MASK]`; static position embeddings.
- Inherits dataset biases — audit before deployment.
