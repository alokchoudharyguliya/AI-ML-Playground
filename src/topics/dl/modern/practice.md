## Exercises

**Q1.** A GQA model has 32 query heads, 8 KV heads, $d_{head}=128$, 32 layers. By what factor does GQA shrink the KV cache versus MHA, and what is the per-token cache in fp16?

<details>
<summary>Show answer</summary>

Factor $32/8=4$. Per token: $2\cdot32\cdot8\cdot128\cdot2=131{,}072$ B $=128$ KiB (MHA would be 512 KiB).

</details>

**Q2.** An MoE layer has 64 experts each with 100M parameters, top-2 routing, plus 2B shared (attention/embedding) parameters. Compute total and active parameters.

<details>
<summary>Show answer</summary>

Total $=2\text{B}+64\cdot0.1\text{B}=8.4$ B. Active $=2\text{B}+2\cdot0.1\text{B}=2.2$ B. Per-token FLOPs resemble a 2.2B dense model while holding 8.4B of knowledge — but you pay memory for all 8.4B and all-to-all communication.

</details>

**Q3.** For a $4096\times4096$ matrix, LoRA rank 16: trainable parameters and compression ratio?

<details>
<summary>Show answer</summary>

$2\cdot4096\cdot16=131{,}072$ vs $16{,}777{,}216$ → **0.78%** (128× fewer). Doing this on $q,v$ in all 32 layers: $32\cdot2\cdot131{,}072\approx8.4$ M parameters, a 33 MB fp32 adapter.

</details>

**Q4.** You have $10^{24}$ FLOPs. Using $C=6ND$ and a 20-tokens/parameter rule, what model and dataset size would you train? How would you change this if the model will be served to millions of users?

<details>
<summary>Show answer</summary>

$D=20N$, so $C=120N^2\Rightarrow N=\sqrt{10^{24}/120}\approx9.1\times10^{10}$ (91B), $D\approx1.8$ T tokens. For heavy inference, train a **smaller model on far more tokens** (e.g. 30B on 6–8T): training loss is slightly worse than optimal for the compute, but every future query is cheaper.

</details>

**Q5.** Why does DPO not need a reward model, and what is a typical failure mode?

<details>
<summary>Show answer</summary>

The reward is implicit in the log-ratio $\beta\log(\pi_\theta/\pi_{ref})$, and the Bradley–Terry preference likelihood becomes a simple logistic loss. Failure modes: both chosen and rejected likelihoods can *decrease* (probability mass drifts to unseen text), length exploitation, and sensitivity to $\beta$ and to the quality/off-policyness of preference data.

</details>

**Q6 (code).** Verify the RoPE relative-position property numerically.

<details>
<summary>Show answer</summary>

`modern_blocks.py` rotates random $q,k$ at $(m,n)$ and $(m+s,n+s)$ and asserts equal dot products to $10^{-10}$, then checks that $(m,n)$ vs $(m,n+1)$ differ.

</details>

## In practice

**Fine-tuning recipe.** Start from an instruct model; LoRA $r=16$–64 on all linear layers, $\alpha=2r$ (or $\alpha=16$), LR $10^{-4}$–$2\times10^{-4}$, 1–3 epochs, 1–10k *high-quality* examples beat 100k noisy ones. Mask the loss to the response tokens only. Evaluate on held-out tasks + regression tests to detect forgetting.

**Serving tips.** Merge LoRA into weights for single-tenant, or use multi-LoRA serving (S-LoRA, vLLM) for many tenants on one base. Quantise to AWQ/GPTQ int4 for 2–3× cheaper serving.

**MoE in practice.** Great for throughput at scale, painful on small clusters (memory, all-to-all). For fine-tuning, LoRA on attention + shared experts is typical; watch router collapse.

**Choosing a base model.** Match context length, license, multilingual coverage and tokenizer efficiency to your use case; always benchmark on your own data.

**Reading a model card.** Parameters (total/active), tokens trained, context, tokenizer size, attention type (GQA/MLA), norm/activation, RoPE base — you can now estimate memory, FLOPs and KV cost from these alone.

## Common pitfalls

- Forgetting the chat template during fine-tuning (train/serve mismatch).
- High LoRA rank with high LR → instability; low rank on tasks requiring new knowledge → underfit.
- Training on the prompt tokens (loss not masked).
- Merging LoRA into a *quantised* base without dequantising correctly.
- Assuming MoE "active parameters" equals memory footprint.
- Extending context with RoPE scaling but not fine-tuning on long sequences.
- Evaluating alignment only with a reward model that was also used for training (reward hacking).
