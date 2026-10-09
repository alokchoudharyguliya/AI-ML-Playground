import math
import torch
import torch.nn.functional as F

torch.manual_seed(0)
B, H, T, D = 2, 4, 128, 64
q, k, v = (torch.randn(B, H, T, D) for _ in range(3))


def manual(q, k, v, causal=True):
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))
    if causal:
        mask = torch.triu(torch.ones(q.size(-2), k.size(-2), dtype=torch.bool), 1)
        scores = scores.masked_fill(mask, float("-inf"))
    return scores.softmax(-1) @ v


ref = manual(q, k, v)
fused = F.scaled_dot_product_attention(q, k, v, is_causal=True)   # FlashAttention / mem-efficient on GPU
print("max |manual - fused| =", (ref - fused).abs().max().item())

# why the sqrt(d) matters: score std with and without scaling
raw = (q @ k.transpose(-2, -1))
print("score std unscaled:", raw.std().item(), "  scaled:", (raw / math.sqrt(D)).std().item())

# select a backend explicitly (PyTorch >= 2.3)
if torch.cuda.is_available():
    from torch.nn.attention import SDPBackend, sdpa_kernel
    qc, kc, vc = (t.half().cuda() for t in (q, k, v))
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        out = F.scaled_dot_product_attention(qc, kc, vc, is_causal=True)
    print("flash backend ok:", out.shape)
