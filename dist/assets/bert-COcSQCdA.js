import{r as e}from"./rolldown-runtime-hePW80VL.js";import{b as t,v as n}from"./r3f-x1z21uF6.js";import{a as r,c as i,g as a,h as o,m as s,n as c,s as l,t as u,y as d}from"./viz-CPys2405.js";import{c as f,i as p,l as m,o as h,r as g,s as _,t as v,u as y}from"./hooks-Dw7oo1m3.js";var b=e(t(),1),x=`## Pre-train once, fine-tune everywhere

Before 2018 every NLP task trained its own model from scratch (or from static word vectors like word2vec/GloVe, where *bank* has one vector regardless of context). **BERT** (Devlin et al., 2018 — *Bidirectional Encoder Representations from Transformers*) established the recipe that still defines NLP:

1. **Pre-train** a big Transformer encoder on unlabeled text with a self-supervised objective.
2. **Fine-tune** the whole network (plus a tiny task head) on a small labeled dataset in minutes.

It set new state of the art on 11 benchmarks at once. Representations became **contextual**: “river **bank**” and “bank **loan**” get different vectors because every token is computed from attention over the whole sentence.

## The encoder: bidirectional on purpose

A left-to-right language model (GPT) cannot see the right context, which hurts understanding tasks. A naïve bidirectional LM would let each word "see itself" through the stack. BERT's trick is **masked language modelling (MLM)**: hide a random 15% of tokens and predict them from *both* sides.

The MLM selection rule (first lab tab) — of the chosen tokens

- **80%** → replaced with \`[MASK]\`,
- **10%** → replaced with a *random* token,
- **10%** → left unchanged,

so the model cannot rely on seeing \`[MASK]\` (it never appears at fine-tuning time) and must keep a good representation for **every** input token.

The original BERT also used **Next Sentence Prediction** (is B the sentence after A?). Later work (RoBERTa) showed it adds little and dropped it; ALBERT replaced it with sentence-order prediction.

## Architecture details

- **Input** (third lab tab): \`[CLS] sentence A [SEP] sentence B [SEP]\`; each position = token + segment + learned position embedding, LayerNorm'd.
- **Tokenizer**: WordPiece, 30k sub-words; unknown words split into \`##\` pieces.
- **Model**: 12 (base, 110M) or 24 (large, 340M) post-LN Transformer encoder layers, full bidirectional attention, GELU FFN, context ≤ 512.
- **Outputs**: one contextual vector per token. \`[CLS]\`'s final vector (through a small pooler) represents the whole sequence.

## Fine-tuning recipes

| Task | Head on top of BERT | Example |
|---|---|---|
| Sentence classification | linear on \`[CLS]\` | sentiment, intent |
| Sentence-pair classification | linear on \`[CLS]\` of \`A [SEP] B\` | NLI, paraphrase |
| Token classification | linear on each token | NER, POS tagging |
| Span extraction | start/end logits per token | SQuAD QA |
| Embeddings / retrieval | mean-pool (+ contrastive training) | semantic search (Sentence-BERT) |

Typical hyper-parameters: 2–4 epochs, LR $2\\text{–}5\\times10^{-5}$ with linear warm-up/decay, batch 16–32, AdamW weight decay 0.01. **Layer-wise LR decay** (lower LR for lower layers) and re-initialising the top layers help on small data.

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
- Pre-train/fine-tune mismatch from \`[MASK]\`; static position embeddings.
- Inherits dataset biases — audit before deployment.
`,S=`## Masked language modelling objective

Let $\\mathcal M$ be the set of selected positions ($|\\mathcal M|\\approx0.15\\,n$) and $\\tilde x$ the corrupted input. The loss is cross-entropy **only over $\\mathcal M$**:

$$
\\mathcal L_{MLM}=-\\sum_{i\\in\\mathcal M}\\log p_\\theta\\big(x_i\\mid\\tilde x\\big),\\qquad
p_\\theta(\\cdot\\mid\\tilde x)=\\mathrm{softmax}\\big(E\\,\\mathrm{LN}(\\mathrm{GELU}(W h_i))+b\\big),
$$

with the decoder matrix tied to the input embedding $E$.

**Why 80/10/10?** For a selected position the input is \`[MASK]\` w.p. $0.8$, a uniformly random token w.p. $0.1$, the original w.p. $0.1$. Per token, the expected corruption is $0.15\\cdot0.9=13.5\\%$ non-original, and the model sees an unchanged-but-supervised token in $1.5\\%$ of positions, so it must keep contextual information for every token instead of copying.

Expected number of tokens carrying loss per sequence $=0.15\\,n$; for $n=512$ that's $\\approx77$ targets — compared with $512$ for a causal LM.

## Input representation

$$
h^{(0)}_i=\\mathrm{LN}\\big(E_{tok}[x_i]+E_{seg}[s_i]+E_{pos}[i]\\big),\\qquad
h^{(\\ell)}=\\mathrm{EncoderLayer}(h^{(\\ell-1)}) .
$$

Post-LN encoder layer (original BERT):

$$
u=\\mathrm{LN}\\big(h+\\mathrm{MHA}(h)\\big),\\qquad h'=\\mathrm{LN}\\big(u+\\mathrm{FFN}(u)\\big),\\quad
\\mathrm{FFN}(u)=W_2\\,\\mathrm{GELU}(W_1u).
$$

## Parameter count (exact)

For vocabulary $V$, positions $P$, hidden $H$, FFN size $F=rH$, layers $L$:

$$
\\begin{aligned}
\\text{embeddings}&=VH+PH+2H+2H\\quad(\\text{token, position, 2 segments, LN})\\\\
\\text{per layer}&=\\underbrace{4(H^2+H)}_{QKVO}+\\underbrace{2rH^2+rH+H}_{\\text{FFN}}+\\underbrace{4H}_{2\\text{ LNs}}\\\\
\\text{pooler}&=H^2+H
\\end{aligned}
$$

BERT-base ($V=30522,P=512,H=768,F=3072,L=12$): embeddings $23{,}837{,}184$; layers $12\\times7{,}087{,}872=85{,}054{,}464$; pooler $590{,}592$ → **109,482,240** ✓.

FLOPs per token (forward) $\\approx2\\cdot N_{non\\text{-}emb}+4LnH$, where the second term is attention over a context of $n$ tokens.

## Fine-tuning heads

**Classification** with $K$ classes:

$$
p(y\\mid x)=\\mathrm{softmax}\\big(W_c\\tanh(W_ph_{[CLS]}+b_p)+b_c\\big),\\qquad \\mathcal L=-\\log p(y^*\\mid x).
$$

**Span extraction** (QA): learn start and end vectors $S,T\\in\\mathbb R^H$:

$$
P_i^{start}=\\frac{e^{S\\cdot h_i}}{\\sum_je^{S\\cdot h_j}},\\quad P_i^{end}=\\frac{e^{T\\cdot h_i}}{\\sum_je^{T\\cdot h_j}},\\quad
\\mathcal L=-\\log P^{start}_{a}-\\log P^{end}_{b},
$$

predict the span $(i,j)$, $i\\le j$, maximising $S\\cdot h_i+T\\cdot h_j$.

## ELECTRA: replaced-token detection

A small generator proposes replacements; the discriminator labels each token as original/replaced:

$$
\\mathcal L_{disc}=-\\sum_{i=1}^{n}\\Big[\\mathbb 1[\\hat x_i=x_i]\\log D_i+\\mathbb 1[\\hat x_i\\neq x_i]\\log(1-D_i)\\Big],
$$

giving a gradient from **all $n$** positions rather than 15%.

## Sentence embeddings (contrastive)

For a batch of positive pairs $(a_i,b_i)$ with embeddings $u_i,v_i$ and temperature $\\tau$ (InfoNCE):

$$
\\mathcal L=-\\frac1B\\sum_{i}\\log\\frac{\\exp\\big(\\cos(u_i,v_i)/\\tau\\big)}{\\sum_{j}\\exp\\big(\\cos(u_i,v_j)/\\tau\\big)} .
$$

## Distillation

$$
\\mathcal L=\\alpha\\,\\mathrm{CE}(y,\\sigma(z_s))+(1-\\alpha)\\,\\tau^2\\,\\mathrm{KL}\\big(\\sigma(z_t/\\tau)\\,\\|\\,\\sigma(z_s/\\tau)\\big),
$$

with teacher logits $z_t$, student logits $z_s$ and temperature $\\tau$ (softened targets carry "dark knowledge" about class similarity).

## Layer-wise learning-rate decay

$$
\\eta_\\ell=\\eta_{top}\\cdot\\gamma^{\\,L-\\ell},\\qquad\\gamma\\in[0.8,0.95],
$$

so lower layers (general features) change less than upper layers (task-specific).
`,C=`## Exercises

**Q1.** A sentence has 40 tokens. With the BERT recipe, how many tokens are selected on average, and in expectation how many become \`[MASK]\`, random, and unchanged?

<details>
<summary>Show answer</summary>

Selected: $0.15\\cdot40=6$. Of those: $4.8$ \`[MASK]\`, $0.6$ random, $0.6$ unchanged. All 6 are in the loss.

</details>

**Q2.** Why can't you simply read $P(w\\mid\\text{context})$ from BERT to score the likelihood of a sentence like a language model?

<details>
<summary>Show answer</summary>

MLM gives conditionals $p(x_i\\mid x_{\\neq i})$, which are not consistent with a single joint distribution. A pseudo-log-likelihood $\\sum_i\\log p(x_i\\mid x_{\\neq i})$ (masking each token in turn, $n$ forward passes) is used as an approximation, but it is not a normalised probability.

</details>

**Q3.** You fine-tune BERT on 2,000 labelled examples and accuracy fluctuates wildly across seeds. List four remedies.

<details>
<summary>Show answer</summary>

(1) Lower LR ($2\\times10^{-5}$) with warm-up and bias correction; (2) more epochs with early stopping and several seeds averaged; (3) layer-wise LR decay / re-initialise the top layer(s); (4) use a stronger base (RoBERTa/DeBERTa) or intermediate-task / domain-adaptive pre-training; also consider SetFit-style contrastive few-shot training.

</details>

**Q4.** Compute the parameter count of an encoder with $V=32000$, $P=512$, $H=512$, $F=2048$, $L=6$.

<details>
<summary>Show answer</summary>

Embeddings: $32000\\cdot512+512\\cdot512+2\\cdot512+2\\cdot512=16{,}384{,}000+262{,}144+2{,}048=16{,}648{,}192$. Per layer: $4(512^2+512)=1{,}050{,}624$; FFN $2\\cdot512\\cdot2048+2048+512=2{,}099{,}712$; LN $4\\cdot512=2{,}048$ → $3{,}152{,}384$; ×6 = $18{,}914{,}304$. Pooler $262{,}656$. Total $\\approx35.8$ M.

</details>

**Q5.** When would you pick a *bi-encoder* vs a *cross-encoder* for search?

<details>
<summary>Show answer</summary>

**Bi-encoder**: encode query and documents independently → pre-compute and index document vectors (ANN search) → fast recall over millions. **Cross-encoder**: concatenate query and document into one sequence → far more accurate but must run per pair → use it to *rerank* the top 20–100 candidates. Production RAG uses both.

</details>

**Q6 (code).** Implement the 80/10/10 masking vectorised and verify the proportions.

<details>
<summary>Show answer</summary>

\`mlm_masking.py\` does this in NumPy (and mirrors Hugging Face's \`DataCollatorForLanguageModeling\`): draw a Bernoulli mask at 15% (excluding special tokens), then split the selected set with a second uniform draw; labels are \`-100\` elsewhere so the loss ignores them.

</details>

## In practice

- **Classification at scale**: distilled/quantised encoders (DistilBERT, MiniLM, DeBERTa-small) in ONNX or TensorRT serve thousands of requests/sec per GPU.
- **RAG & search**: E5/BGE/GTE bi-encoders for embeddings; \`cross-encoder/ms-marco-*\` rerankers; store vectors in FAISS, pgvector or a vector DB.
- **Long documents**: chunk with overlap, use ModernBERT/Longformer (sliding-window attention), or hierarchical pooling.
- **Domain adaptation**: continue MLM pre-training on in-domain text (BioBERT, FinBERT, CodeBERT) before fine-tuning — often worth several points.
- **Evaluation**: hold-out set per slice, calibration (ECE), and robustness checks; fine-tuned encoders can be over-confident.

## Common pitfalls

- Truncating long inputs silently at 512 tokens (answer span / key evidence lost).
- Using the raw \`[CLS]\` vector of a *non*-contrastively trained BERT as a sentence embedding — it is poor; use SBERT-style models or mean pooling.
- Forgetting the attention mask for padded batches.
- Tokenizer/model mismatch (cased vs uncased).
- Fine-tuning with a learning rate that is too high (catastrophic forgetting).
`,w=`import numpy as np

MASK_ID, VOCAB = 103, 30522          # BERT-base-uncased ids
SPECIAL = {0, 101, 102, 103}         # [PAD] [CLS] [SEP] [MASK]


def mlm_mask(ids, rng, p=0.15, special=SPECIAL):
    """Return (inputs, labels). labels = -100 where no loss is applied."""
    ids = np.asarray(ids)
    can_mask = ~np.isin(ids, list(special))
    chosen = (rng.random(ids.shape) < p) & can_mask            # which tokens are predicted

    labels = np.where(chosen, ids, -100)
    inputs = ids.copy()

    r = rng.random(ids.shape)
    to_mask = chosen & (r < 0.8)                                # 80%  -> [MASK]
    to_rand = chosen & (r >= 0.8) & (r < 0.9)                   # 10%  -> random token
    # remaining 10% of chosen: unchanged (but still in the loss!)
    inputs[to_mask] = MASK_ID
    inputs[to_rand] = rng.integers(1000, VOCAB, size=to_rand.sum())
    return inputs, labels


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    batch = rng.integers(1000, VOCAB, size=(512, 128))          # 65,536 ordinary tokens
    inputs, labels = mlm_mask(batch, rng)

    chosen = labels != -100
    print(f"selected: {chosen.mean():.3%}  (target 15%)")
    print(f"[MASK]   : {(inputs[chosen] == MASK_ID).mean():.3%}  (target 80%)")
    rand = (inputs[chosen] != MASK_ID) & (inputs[chosen] != batch[chosen])
    print(f"random   : {rand.mean():.3%}  (target 10%)")
    print(f"unchanged: {(inputs[chosen] == batch[chosen]).mean():.3%}  (target 10%)")
`,T=`import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

device = "cuda" if torch.cuda.is_available() else "cpu"
name = "bert-base-uncased"
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForSequenceClassification.from_pretrained(name, num_labels=2).to(device)

ds = load_dataset("glue", "sst2")
def encode(batch):
    return tok(batch["sentence"], truncation=True, max_length=128)
ds = ds.map(encode, batched=True)
ds = ds.rename_column("label", "labels")
ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

def collate(items):                       # dynamic padding: pad to the longest in the batch
    batch = tok.pad({k: [it[k] for it in items] for k in ("input_ids", "attention_mask")}, return_tensors="pt")
    batch["labels"] = torch.stack([it["labels"] for it in items])
    return batch

train = DataLoader(ds["train"], batch_size=32, shuffle=True, collate_fn=collate)
val = DataLoader(ds["validation"], batch_size=128, collate_fn=collate)

# layer-wise learning-rate decay: lower layers move less
def param_groups(model, lr=3e-5, decay=0.9, wd=0.01):
    layers = model.bert.encoder.layer
    groups = [{"params": list(model.classifier.parameters()) + list(model.bert.pooler.parameters()), "lr": lr}]
    for i, layer in enumerate(reversed(layers)):
        groups.append({"params": layer.parameters(), "lr": lr * decay ** (i + 1)})
    groups.append({"params": model.bert.embeddings.parameters(), "lr": lr * decay ** (len(layers) + 1)})
    for g in groups:
        g["weight_decay"] = wd
    return groups

epochs = 3
opt = torch.optim.AdamW(param_groups(model))
steps = epochs * len(train)
sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
scaler = torch.amp.GradScaler(enabled=device == "cuda")

for epoch in range(epochs):
    model.train()
    for batch in train:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.autocast(device_type=device, dtype=torch.float16, enabled=device == "cuda"):
            loss = model(**batch).loss
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update(); sched.step()

    model.eval(); correct = total = 0
    with torch.no_grad():
        for batch in val:
            batch = {k: v.to(device) for k, v in batch.items()}
            pred = model(**{k: v for k, v in batch.items() if k != "labels"}).logits.argmax(-1)
            correct += (pred == batch["labels"]).sum().item(); total += len(pred)
    print(f"epoch {epoch + 1}: val accuracy {correct / total:.3%}")
`,E=n(),D=`the quick brown fox jumps over the lazy dog near the river bank`.split(` `),O=[`apple`,`seven`,`cloud`,`blue`,`runs`,`piano`,`window`,`green`,`under`,`table`,`cold`,`paper`,`while`,`engine`];function k(e,t){let n=D.length,r=Math.max(1,Math.round(e*n)),i=[...Array(n).keys()].sort(()=>t()-.5).slice(0,r),a={},o={};return i.forEach(e=>{let n=t();a[e]=n<.8?`mask`:n<.9?`rand`:`keep`,a[e]===`rand`&&(o[e]=O[Math.floor(t()*O.length)])}),{kind:a,rnd:o}}function A(){let[e,t]=(0,b.useState)(.15),n=(0,b.useRef)(o(Date.now()&65535)),[r,a]=(0,b.useState)(()=>k(.15,n.current)),[s,c]=(0,b.useState)({mask:0,rand:0,keep:0}),l=()=>{let t=k(e,n.current);a(t),c(e=>{let n={...e};return Object.values(t.kind).forEach(e=>{n[e]++}),n})},d=()=>{let t={...s};for(let r=0;r<500;r++)Object.values(k(e,n.current).kind).forEach(e=>{t[e]++});c(t)},m=s.mask+s.rand+s.keep,_=e=>m?(100*s[e]/m).toFixed(1):{mask:`80`,rand:`10`,keep:`10`}[e],v=D.map((e,t)=>r.kind[t]===`mask`?`[MASK]`:r.kind[t]===`rand`?r.rnd[t]:e);return(0,E.jsxs)(E.Fragment,{children:[(0,E.jsxs)(`div`,{className:`lab-card`,children:[(0,E.jsxs)(`div`,{className:`tokrow`,children:[(0,E.jsx)(`span`,{className:`rl`,children:`Original`}),D.map((e,t)=>(0,E.jsx)(`span`,{className:`tk`+(r.kind[t]?` sel`:``),children:e},t))]}),(0,E.jsxs)(`div`,{className:`tokrow`,children:[(0,E.jsx)(`span`,{className:`rl`,children:`Model input`}),v.map((e,t)=>(0,E.jsx)(`span`,{className:`tk `+(r.kind[t]||``),children:e},t))]}),(0,E.jsxs)(`div`,{className:`tokrow`,children:[(0,E.jsx)(`span`,{className:`rl`,children:`Predict (loss)`}),D.map((e,t)=>(0,E.jsx)(`span`,{className:`tk `+(r.kind[t]?`keep`:`dim`),children:r.kind[t]?e:`·`},t))]}),(0,E.jsxs)(`div`,{className:`tokrow`,style:{fontSize:12,color:u.mute},children:[(0,E.jsx)(`span`,{className:`rl`}),(0,E.jsxs)(`span`,{children:[(0,E.jsx)(`span`,{className:`tk mask`,children:`[MASK]`}),` 80% replaced by the mask token`]}),(0,E.jsxs)(`span`,{children:[(0,E.jsx)(`span`,{className:`tk rand`,children:`random`}),` 10% replaced by a random token`]}),(0,E.jsxs)(`span`,{children:[(0,E.jsx)(`span`,{className:`tk keep`,children:`same`}),` 10% left unchanged`]})]}),(0,E.jsx)(`div`,{children:(0,E.jsxs)(`div`,{className:`statbar`,children:[(0,E.jsxs)(`i`,{style:{width:(m?100*s.mask/m:80)+`%`,background:u.c},children:[_(`mask`),`%`]}),(0,E.jsxs)(`i`,{style:{width:(m?100*s.rand/m:10)+`%`,background:u.d},children:[_(`rand`),`%`]}),(0,E.jsxs)(`i`,{style:{width:(m?100*s.keep/m:10)+`%`,background:u.b},children:[_(`keep`),`%`]})]})})]}),(0,E.jsxs)(p,{children:[(0,E.jsx)(f,{label:`Mask rate`,min:.05,max:.5,step:.01,value:e,onChange:t,fmt:e=>Math.round(e*100)+`%`}),(0,E.jsx)(g,{primary:!0,onClick:l,children:`Resample`}),(0,E.jsx)(g,{onClick:d,children:`+500 samples`}),(0,E.jsx)(g,{onClick:()=>c({mask:0,rand:0,keep:0}),children:`Reset tally`})]}),(0,E.jsxs)(h,{children:[`Only the highlighted `,(0,E.jsx)(`b`,{children:Object.keys(r.kind).length}),` of `,D.length,` positions (`,Math.round(100*Object.keys(r.kind).length/D.length),`%) contribute to the loss — MLM is `,(0,E.jsx)(`b`,{children:`sample-inefficient`}),` (ELECTRA fixes this). The 10% random / 10% unchanged cases stop the model learning “only predict when I see [MASK]”, since [MASK] never appears at fine-tuning time. Tally over `,i(m),` selected tokens converges to 80 / 10 / 10.`]})]})}var j=12;function M(e,t,n,r){return e===`bi`?!0:e===`causal`?n<=t:n<r||n<=t}function N(){let[e,t]=(0,b.useState)(`bi`),[n,r]=(0,b.useState)(5),[i,o]=(0,b.useState)(8),s={bi:[`Bidirectional (encoder)`,`BERT, RoBERTa, DeBERTa, ViT encoders: every token sees every token → rich representations, but cannot generate left-to-right.`],causal:[`Causal (decoder)`,`GPT, Llama: token i sees only ≤ i → can be trained on every position as next-token prediction and generate autoregressively.`],prefix:[`Prefix-LM`,`T5-style decoders / UL2 / PaLM-prefix: bidirectional over the prompt (prefix), causal over the continuation — the best of both for conditional generation.`]}[e],[l]=v(360,(t,r,o)=>{let l=Math.min(26,(o-80)/j,(r*.5-70)/j);d(t,`who can attend to whom  (row = query, column = key)`,70,20,{size:12,color:u.mute,weight:600});for(let r=0;r<j;r++)for(let a=0;a<j;a++)t.fillStyle=M(e,r,a,n)?r===i?u.c:c(u.a,.8):`#10131f`,t.fillRect(70+a*l,56+r*l,l-1.5,l-1.5);for(let e=0;e<j;e++)d(t,e,70+e*l+l/2,46,{size:10,align:`center`,color:u.dim,mono:!0}),d(t,`tok `+e,62,56+e*l+l/2,{size:10,align:`right`,color:e===i?u.c:u.dim,mono:!0});e===`prefix`&&(t.strokeStyle=u.d,t.lineWidth=2,t.setLineDash([4,3]),t.strokeRect(69,55,n*l,j*l+1),t.setLineDash([]));let f=70+j*l+60,p=r-f-20;d(t,`Token ${i} can see:`,f,20,{size:12,color:u.mute,weight:600});let m=Math.min(40,p/j);for(let r=0;r<j;r++){let o=M(e,i,r,n);t.fillStyle=o?c(u.c,.85):`#141830`,a(t,f+r*m,40,m-4,34,6),t.fill(),d(t,r,f+r*m+m/2-2,57,{size:11,align:`center`,color:o?`#fff`:u.dim,mono:!0})}let h=Array.from({length:j},(t,r)=>M(e,i,r,n)).filter(Boolean).length;d(t,`${h} of ${j} positions visible`,f,100,{size:12,color:u.ink,mono:!0}),d(t,s[0],f,140,{size:14,color:`#fff`,weight:700});let g=s[1].split(` `),_=``,v=168;g.forEach(e=>{(_+e).length*6.6>p&&(d(t,_,f,v,{size:12,color:u.mute}),_=``,v+=18),_+=e+` `}),d(t,_,f,v,{size:12,color:u.mute})});return(0,E.jsxs)(E.Fragment,{children:[(0,E.jsx)(`canvas`,{...l}),(0,E.jsxs)(p,{children:[(0,E.jsx)(_,{label:`Mask`,value:e,onChange:t,options:[[`bi`,`Bidirectional (BERT)`],[`causal`,`Causal (GPT)`],[`prefix`,`Prefix-LM (T5/UL2)`]]}),(0,E.jsx)(f,{label:`Prefix length`,min:1,max:11,value:n,onChange:r}),(0,E.jsx)(f,{label:`Inspect token`,min:0,max:11,value:i,onChange:o})]})]})}var P=[`[CLS]`,`my`,`dog`,`is`,`cute`,`[SEP]`,`he`,`likes`,`play`,`##ing`,`[SEP]`],F=[0,0,0,0,0,0,1,1,1,1,1],I=10,L=e=>{let t=o(e);return Array.from({length:40},()=>Array.from({length:I},()=>s(t)*.6))},R=L(1),z=L(2).slice(0,2),B=L(3);function V(){let[e,t]=(0,b.useState)(!1),[n]=v(480,(t,n,i)=>{let a=Math.min(70,(n-150)/P.length),o=[[`Token emb`,e=>R[e%40],u.c],[`Segment emb`,e=>z[F[e]],u.d],[`Position emb`,e=>B[e],u.e]],s=54;P.forEach((e,n)=>{d(t,e,130+n*a+a/2-2,26,{size:11,align:`center`,color:`#fff`,weight:600})});let c=P.map((e,t)=>Array.from({length:I},(e,n)=>o.reduce((e,[,r])=>e+r(t)[n],0)));o.forEach(([e,n,i],o)=>{d(t,e,118,s+40,{size:12,align:`right`,color:i,weight:600}),P.forEach((e,i)=>n(i).forEach((e,n)=>{t.fillStyle=r(e/1.5),t.fillRect(130+i*a+4,s+n*8,a-8,7)})),s+=96,d(t,o<2?`+`:`=`,70,s-14,{size:18,align:`center`,color:u.mute,weight:700})}),d(t,e?`LayerNorm(sum)`:`Sum → Transformer`,118,s+40,{size:12,align:`right`,color:`#fff`,weight:700}),c.forEach((n,i)=>{let o=n.reduce((e,t)=>e+t,0)/n.length,c=Math.sqrt(n.reduce((e,t)=>e+(t-o)**2,0)/n.length);n.forEach((n,l)=>{t.fillStyle=r((e?(n-o)/c:n)/(e?2.2:2.5)),t.fillRect(130+i*a+4,s+l*8,a-8,7)})});let l=130+6*a;t.strokeStyle=u.d,t.lineWidth=2,t.beginPath(),t.moveTo(134,i-14),t.lineTo(l-4,i-14),t.stroke(),t.strokeStyle=u.b,t.beginPath(),t.moveTo(l+4,i-14),t.lineTo(130+P.length*a-4,i-14),t.stroke(),d(t,`sentence A (segment 0)`,(134+l)/2,i-28,{size:11,align:`center`,color:u.d}),d(t,`sentence B (segment 1)`,(l+130+P.length*a)/2,i-28,{size:11,align:`center`,color:u.b})});return(0,E.jsxs)(E.Fragment,{children:[(0,E.jsx)(`canvas`,{...n}),(0,E.jsx)(p,{children:(0,E.jsx)(y,{label:`Apply LayerNorm to the sum`,value:e,onChange:t})}),(0,E.jsxs)(h,{children:[`BERT's input to layer 1 is the `,(0,E.jsx)(`b`,{children:`element-wise sum`}),` of three learned lookups — token (WordPiece vocabulary of 30,522), segment (A/B) and absolute position (≤512). Each column is one token's vector (10 of 768 dims shown). Sub-word pieces like `,(0,E.jsx)(`b`,{children:`##ing`}),` let a fixed vocabulary cover any word.`]})]})}var H={base:{name:`BERT-base`,L:12,H:768,A:12,V:30522,P:512,F:4},large:{name:`BERT-large`,L:24,H:1024,A:16,V:30522,P:512,F:4},distil:{name:`DistilBERT`,L:6,H:768,A:12,V:30522,P:512,F:4},tiny:{name:`TinyBERT-4`,L:4,H:312,A:12,V:30522,P:512,F:4}};function U({L:e,H:t,V:n,P:r,F:i}){let a=n*t+r*t+2*t+2*t,o=e*4*(t*t+t),s=e*(2*t*i*t+i*t+t),c=e*4*t,l=t*t+t;return{emb:a,attn:o,ffn:s,ln:c,pool:l,total:a+o+s+c+l}}function W(){let[e,t]=(0,b.useState)(`base`),[n,r]=(0,b.useState)(H.base),[a,o]=(0,b.useState)(512),s=(e,t)=>r(n=>({...n,[e]:t,name:`custom`})),c=U(n),m=2*(c.total-c.emb)+4*n.L*a*n.H,[g]=v(210,(e,t,r)=>{let a=[[`Embeddings`,c.emb,u.c],[`Attention (QKVO)`,c.attn,u.a],[`FFN`,c.ffn,u.b],[`LayerNorm`,c.ln,u.d],[`Pooler`,c.pool,u.e]],o=t-48;d(e,`${n.name} — ${i(c.total)} parameters`,24,22,{size:14,color:`#fff`,weight:700});let s=24;a.forEach(([t,n,r])=>{let i=n/c.total*o;e.fillStyle=r,e.fillRect(s,44,Math.max(1,i-1),40),s+=i}),a.forEach(([t,n,r],i)=>{let a=24+i%3*(o/3),s=112+Math.floor(i/3)*34;e.fillStyle=r,e.fillRect(a,s-6,12,12),d(e,`${t}`,a+20,s,{size:12,color:u.ink,weight:600}),d(e,`${(n/1e6).toFixed(2)}M · ${(100*n/c.total).toFixed(1)}%`,a+20,s+15,{size:11,mono:!0,color:u.mute})})});return(0,E.jsxs)(E.Fragment,{children:[(0,E.jsx)(`canvas`,{...g}),(0,E.jsxs)(p,{children:[(0,E.jsx)(_,{label:`Preset`,value:e,onChange:e=>{t(e),r(H[e])},options:Object.entries(H).map(([e,t])=>[e,t.name])}),(0,E.jsx)(f,{label:`Layers L`,min:2,max:48,value:n.L,onChange:e=>s(`L`,e)}),(0,E.jsx)(f,{label:`Hidden H`,min:128,max:2048,step:64,value:n.H,onChange:e=>s(`H`,e)}),(0,E.jsx)(f,{label:`FFN ratio`,min:2,max:8,step:1,value:n.F,onChange:e=>s(`F`,e)}),(0,E.jsx)(f,{label:`Seq length n`,min:64,max:2048,step:64,value:a,onChange:o})]}),(0,E.jsxs)(h,{children:[`fp32 weights: `,(0,E.jsx)(`b`,{children:l(c.total*4)}),` · fp16: `,(0,E.jsx)(`b`,{children:l(c.total*2)}),` · ≈ `,(0,E.jsxs)(`b`,{children:[(m/1e9).toFixed(2),` GFLOPs`]}),`/token forward at n=`,a,`. BERT-base check: the formula gives `,(0,E.jsx)(`b`,{children:i(U(H.base).total)}),` (published: 109,482,240). Note how `,(0,E.jsx)(`b`,{children:`FFN ≈ 2× attention`}),` and the embedding table is a fixed cost that dominates tiny models.`]})]})}function G(){return(0,E.jsx)(m,{views:[{id:`mask`,label:`Masked language modeling`,render:()=>(0,E.jsx)(A,{})},{id:`attn`,label:`Attention masks`,render:()=>(0,E.jsx)(N,{})},{id:`input`,label:`Input representation`,render:()=>(0,E.jsx)(V,{})},{id:`size`,label:`Model size calculator`,render:()=>(0,E.jsx)(W,{})}]})}var K={Lab:G,vizTitle:`How BERT sees text: masking, attention patterns, inputs and size`,tryIt:[`Hit **+500 samples** and watch the tally converge to the 80 / 10 / 10 split.`,`Raise the mask rate to 40% — the model must predict many tokens with little context; 15% was a tuned compromise.`,`Compare **bidirectional vs causal vs prefix** masks and count how many positions token 8 can see.`,`In the size calculator pick **BERT-large**, then shrink the hidden size and notice FFN dominating compute.`],theory:x,math:S,practice:C,code:[{title:`BERT-style 80/10/10 masking (vectorised, with tests)`,lang:`python`,src:w},{title:`Fine-tune BERT for text classification (Hugging Face)`,lang:`python`,note:`Plain PyTorch loop: tokenizer, AdamW with layer-wise LR decay, linear warmup, mixed precision.`,src:T}],quiz:[{q:`Why does BERT sometimes replace selected tokens with a random token or leave them unchanged, rather than always using [MASK]?`,options:[`To speed up training`,`Because [MASK] never appears at fine-tuning time; the mismatch is reduced`,`To regularize embeddings`,`To increase vocabulary`],answer:1,why:`The 80/10/10 recipe forces the model to maintain a good representation of every token, not just masked ones.`},{q:`What fraction of tokens contribute to the MLM loss?`,options:[`100%`,`≈50%`,`≈15%`,`≈1%`],answer:2,why:`Only the selected 15% are predicted — one reason MLM is less sample-efficient than next-token prediction.`},{q:`Which statement distinguishes BERT from GPT?`,options:[`BERT uses recurrence`,`BERT is bidirectional and cannot generate text autoregressively without adaptation`,`GPT uses an encoder`,`BERT has no attention`],answer:1,why:`BERT's unmasked attention sees both sides; GPT's causal mask enables left-to-right generation.`},{q:`ELECTRA improves on MLM by:`,options:[`Using more layers`,`Training a discriminator to detect replaced tokens at every position`,`Adding recurrence`,`Removing attention`],answer:1,why:`Replaced-token detection gives a learning signal at all positions, ≈ 4× more sample-efficient.`},{q:`In fine-tuning for sentence classification, the standard BERT head sits on:`,options:[`The first token's ([CLS]) final hidden state`,`The embedding layer`,`The attention weights`,`The positional table`],answer:0,why:`[CLS] aggregates sequence information through attention; a linear layer maps it to class logits.`}]};export{K as default};