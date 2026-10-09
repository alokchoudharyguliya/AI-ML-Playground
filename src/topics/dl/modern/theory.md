## The recipe converged — here are the ingredients

Open any 2024–25 open-weights LLM (Llama 3, Mistral, Qwen, DeepSeek, Gemma) and the architecture looks like the 2017 Transformer *after a decade of ablations*. The differences are individually small and collectively decisive:

| Component | Original (2017) | Modern default | Why |
|---|---|---|---|
| Norm | post-LayerNorm | **pre-RMSNorm** | stable at depth, cheaper (no mean/bias) |
| Activation / FFN | ReLU, $4d$ | **SwiGLU**, $\tfrac83d$ | gated units train better at equal params |
| Positions | sinusoidal / learned | **RoPE** | relative, extrapolates with scaling tricks |
| Attention | MHA | **GQA** (or MLA) | much smaller KV cache |
| Biases | everywhere | **none** | simpler, no quality loss |
| Embeddings | tied | tied (small) / untied (large) | capacity vs params |
| Attention kernel | naive | **FlashAttention** | IO-aware, exact, long context |
| Optimiser | Adam | **AdamW**, β₂≈0.95, cosine/WSD | stability at scale |
| Precision | fp32 | **bf16** (fp8 emerging) | 2–4× throughput |

## RoPE and long context

**Rotary Position Embedding** rotates each 2-D slice of $q$ and $k$ by an angle proportional to position, with geometrically spaced frequencies. The dot product then depends only on the *offset* — verified in the first lab tab. To extend context beyond training length, rescale positions or frequencies: **Position Interpolation**, **NTK-aware** scaling, **YaRN**, or train with a larger RoPE base. Further long-context tools: sliding-window / local–global attention, attention sinks, ring attention for sequence parallelism, and KV-cache compression.

## Mixture of Experts (MoE)

Replace the dense FFN with $E$ expert FFNs and a **router**; each token runs through only its top-$k$ experts, so **parameters ≫ active parameters**. Mixtral-8×7B has 47B parameters but ~13B active per token; DeepSeek-V3 has 671B total / 37B active.

Engineering realities (second lab tab):

- **Load balancing**: unconstrained routers collapse onto a few experts. An auxiliary loss ($E\sum_i f_iP_i$) or bias-based balancing keeps experts used equally.
- **Capacity & dropped tokens**: each expert has a fixed buffer; overflow tokens skip the layer (or are rerouted).
- **Expert parallelism**: experts live on different GPUs; tokens are exchanged with all-to-all communication — often the bottleneck.
- **Fine-grained + shared experts** (DeepSeek): many small experts plus always-on shared experts improve specialisation.
- MoE trades **memory for compute**: cheaper per token, but you must store every expert.

## Parameter-efficient fine-tuning

Full fine-tuning a 7B model needs ~16 bytes/parameter ≈ 112 GB for weights + optimiser state. **LoRA** freezes $W$ and trains a low-rank update $\Delta W=BA$ ($r\ll d$): 0.1–1% of the parameters, near-identical quality on many tasks, tiny adapter files that can be hot-swapped per customer. **QLoRA** also stores the frozen base in 4-bit NF4, fitting a 65B model on one 48 GB GPU. Related: adapters, prefix/prompt tuning, DoRA, LoRA+, GaLore.

## Quantisation and efficient inference

Post-training weight quantisation (GPTQ, AWQ, SmoothQuant) to int8/int4, FP8 training/inference on Hopper+, KV-cache quantisation, speculative decoding, continuous batching and paged KV memory. Rule of thumb: **int4 weights ≈ 3–4× less memory, <1–2% quality loss** for ≥7B models.

## Scaling laws

Loss falls as a power law in parameters $N$, data $D$ and compute $C\approx6ND$. **Kaplan et al. (2020)** suggested parameters matter most; **Chinchilla (Hoffmann et al., 2022)** corrected this: for compute-optimal training, scale $N$ and $D$ *together* (~20 tokens/parameter in the original fits). Modern labs **over-train** smaller models on 100–1000+ tokens/parameter because inference cost, not training cost, dominates a product's lifetime. Scaling laws also guide hyper-parameters (μP/μTransfer), data mixtures and when *not* to scale (data walls, synthetic data).

## Alignment and post-training

- **SFT** on demonstrations, then **preference optimisation**: **RLHF** (reward model + PPO) or the simpler offline **DPO** family.
- **RL with verifiable rewards** (GRPO and relatives) for math/code reasoning; **test-time compute** (long chain-of-thought, self-verification, best-of-$n$) became a second scaling axis.
- **Distillation** from strong models; **constitutional / RLAIF** methods; **safety training** and red-teaming.

## Training at scale

Thousands of GPUs combine **data parallel** (FSDP/ZeRO shards weights, grads, optimiser state), **tensor parallel** (split matmuls within a node), **pipeline parallel** (split layers) and **sequence/context parallel**, plus activation checkpointing, fused kernels and overlapping communication with compute. Model FLOPs utilisation (**MFU**) of 40–55% is considered excellent. This is where the CUDA track becomes essential.

## Where this is heading

Multimodal tokens (vision/audio/video), hybrid attention–SSM stacks (Jamba, Mamba-2), longer context, reasoning models, agentic tool use and retrieval — all built from the same ingredients you have now seen: attention, gating, residual streams, careful optimisation, and hardware-aware kernels.
