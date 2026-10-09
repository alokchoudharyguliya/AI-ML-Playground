## Exercises

**Q1.** A sentence has 40 tokens. With the BERT recipe, how many tokens are selected on average, and in expectation how many become `[MASK]`, random, and unchanged?

<details>
<summary>Show answer</summary>

Selected: $0.15\cdot40=6$. Of those: $4.8$ `[MASK]`, $0.6$ random, $0.6$ unchanged. All 6 are in the loss.

</details>

**Q2.** Why can't you simply read $P(w\mid\text{context})$ from BERT to score the likelihood of a sentence like a language model?

<details>
<summary>Show answer</summary>

MLM gives conditionals $p(x_i\mid x_{\neq i})$, which are not consistent with a single joint distribution. A pseudo-log-likelihood $\sum_i\log p(x_i\mid x_{\neq i})$ (masking each token in turn, $n$ forward passes) is used as an approximation, but it is not a normalised probability.

</details>

**Q3.** You fine-tune BERT on 2,000 labelled examples and accuracy fluctuates wildly across seeds. List four remedies.

<details>
<summary>Show answer</summary>

(1) Lower LR ($2\times10^{-5}$) with warm-up and bias correction; (2) more epochs with early stopping and several seeds averaged; (3) layer-wise LR decay / re-initialise the top layer(s); (4) use a stronger base (RoBERTa/DeBERTa) or intermediate-task / domain-adaptive pre-training; also consider SetFit-style contrastive few-shot training.

</details>

**Q4.** Compute the parameter count of an encoder with $V=32000$, $P=512$, $H=512$, $F=2048$, $L=6$.

<details>
<summary>Show answer</summary>

Embeddings: $32000\cdot512+512\cdot512+2\cdot512+2\cdot512=16{,}384{,}000+262{,}144+2{,}048=16{,}648{,}192$. Per layer: $4(512^2+512)=1{,}050{,}624$; FFN $2\cdot512\cdot2048+2048+512=2{,}099{,}712$; LN $4\cdot512=2{,}048$ → $3{,}152{,}384$; ×6 = $18{,}914{,}304$. Pooler $262{,}656$. Total $\approx35.8$ M.

</details>

**Q5.** When would you pick a *bi-encoder* vs a *cross-encoder* for search?

<details>
<summary>Show answer</summary>

**Bi-encoder**: encode query and documents independently → pre-compute and index document vectors (ANN search) → fast recall over millions. **Cross-encoder**: concatenate query and document into one sequence → far more accurate but must run per pair → use it to *rerank* the top 20–100 candidates. Production RAG uses both.

</details>

**Q6 (code).** Implement the 80/10/10 masking vectorised and verify the proportions.

<details>
<summary>Show answer</summary>

`mlm_masking.py` does this in NumPy (and mirrors Hugging Face's `DataCollatorForLanguageModeling`): draw a Bernoulli mask at 15% (excluding special tokens), then split the selected set with a second uniform draw; labels are `-100` elsewhere so the loss ignores them.

</details>

## In practice

- **Classification at scale**: distilled/quantised encoders (DistilBERT, MiniLM, DeBERTa-small) in ONNX or TensorRT serve thousands of requests/sec per GPU.
- **RAG & search**: E5/BGE/GTE bi-encoders for embeddings; `cross-encoder/ms-marco-*` rerankers; store vectors in FAISS, pgvector or a vector DB.
- **Long documents**: chunk with overlap, use ModernBERT/Longformer (sliding-window attention), or hierarchical pooling.
- **Domain adaptation**: continue MLM pre-training on in-domain text (BioBERT, FinBERT, CodeBERT) before fine-tuning — often worth several points.
- **Evaluation**: hold-out set per slice, calibration (ECE), and robustness checks; fine-tuned encoders can be over-confident.

## Common pitfalls

- Truncating long inputs silently at 512 tokens (answer span / key evidence lost).
- Using the raw `[CLS]` vector of a *non*-contrastively trained BERT as a sentence embedding — it is poor; use SBERT-style models or mean pooling.
- Forgetting the attention mask for padded batches.
- Tokenizer/model mismatch (cased vs uncased).
- Fine-tuning with a learning rate that is too high (catastrophic forgetting).
