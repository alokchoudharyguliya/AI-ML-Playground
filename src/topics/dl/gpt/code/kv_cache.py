import math
import torch
import torch.nn as nn


class CachedAttention(nn.Module):
    """Single causal attention layer that supports both full-prefix and cached decoding."""

    def __init__(self, d_model=64, n_heads=4, n_kv_heads=2):
        super().__init__()
        self.h, self.kvh, self.dh = n_heads, n_kv_heads, d_model // n_heads
        self.wq = nn.Linear(d_model, n_heads * self.dh, bias=False)
        self.wk = nn.Linear(d_model, n_kv_heads * self.dh, bias=False)   # GQA: fewer K/V heads
        self.wv = nn.Linear(d_model, n_kv_heads * self.dh, bias=False)
        self.wo = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, cache=None):
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.h, self.dh).transpose(1, 2)        # (B, H, T, dh)
        k = self.wk(x).view(B, T, self.kvh, self.dh).transpose(1, 2)      # (B, KVH, T, dh)
        v = self.wv(x).view(B, T, self.kvh, self.dh).transpose(1, 2)

        if cache is not None:                                             # append new K/V to the cache
            k = torch.cat([cache[0], k], dim=2)
            v = torch.cat([cache[1], v], dim=2)
        new_cache = (k, v)

        rep = self.h // self.kvh                                          # share each KV head across `rep` query heads
        kk, vv = k.repeat_interleave(rep, 1), v.repeat_interleave(rep, 1)

        S = kk.size(2)
        scores = q @ kk.transpose(-2, -1) / math.sqrt(self.dh)            # (B, H, T, S)
        # query i (absolute position S-T+i) may see keys <= its own position
        mask = torch.ones(T, S, dtype=torch.bool).tril(S - T)
        scores = scores.masked_fill(~mask, float("-inf"))
        out = (scores.softmax(-1) @ vv).transpose(1, 2).reshape(B, T, -1)
        return self.wo(out), new_cache


if __name__ == "__main__":
    torch.manual_seed(0)
    attn = CachedAttention().eval()
    x = torch.randn(1, 12, 64)

    with torch.no_grad():
        full, _ = attn(x)                                   # recompute the whole prefix once

        cache, outs = None, []
        for t in range(x.size(1)):                          # one token per step, reusing the cache
            y, cache = attn(x[:, t:t + 1], cache)
            outs.append(y)
        cached = torch.cat(outs, 1)

    print("max |full - cached| =", (full - cached).abs().max().item())
    kv_bytes = sum(t.numel() * t.element_size() for t in cache)
    print("cache after 12 tokens:", tuple(cache[0].shape), f"-> {kv_bytes} bytes (GQA: 2 KV heads vs 4 query heads)")
