import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.h, self.dh = n_heads, d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)   # fused W_Q, W_K, W_V
        self.out = nn.Linear(d_model, d_model, bias=False)       # W_O
        self.drop = dropout

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(x).split(D, dim=-1)
        # (B, T, D) -> (B, H, T, d_head)
        q, k, v = (t.view(B, T, self.h, self.dh).transpose(1, 2) for t in (q, k, v))

        # scores = q k^T / sqrt(d_head), causal mask, softmax, weighted sum of v
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.dh)
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), 1)
        scores = scores.masked_fill(mask, float("-inf"))
        att = F.dropout(scores.softmax(-1), self.drop, self.training)
        y = (att @ v).transpose(1, 2).reshape(B, T, D)             # concat heads
        return self.out(y)


class Block(nn.Module):
    """Pre-LN residual block: x + MHA(LN(x)),  x + FFN(LN(x))."""

    def __init__(self, d_model, n_heads, dropout=0.0):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.ffn = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Linear(4 * d_model, d_model), nn.Dropout(dropout))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.ffn(self.ln2(x))


class GPT(nn.Module):
    def __init__(self, vocab, d_model=256, n_heads=4, n_layers=4, max_len=512, dropout=0.1):
        super().__init__()
        self.tok = nn.Embedding(vocab, d_model)
        self.pos = nn.Embedding(max_len, d_model)                  # learned absolute positions
        self.blocks = nn.ModuleList(Block(d_model, n_heads, dropout) for _ in range(n_layers))
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab, bias=False)
        self.head.weight = self.tok.weight                         # weight tying
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.tok(idx) + self.pos(torch.arange(T, device=idx.device))
        for blk in self.blocks:
            x = blk(x)
        logits = self.head(self.ln_f(x))
        loss = None if targets is None else F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new=50, temperature=1.0, top_k=None):
        for _ in range(max_new):
            logits, _ = self(idx[:, -self.pos.num_embeddings:])
            logits = logits[:, -1] / temperature
            if top_k:
                v, _ = logits.topk(top_k)
                logits[logits < v[:, [-1]]] = float("-inf")
            idx = torch.cat([idx, torch.multinomial(logits.softmax(-1), 1)], 1)
        return idx


if __name__ == "__main__":
    model = GPT(vocab=1000)
    x = torch.randint(0, 1000, (2, 16))
    logits, loss = model(x, x)
    print(logits.shape, float(loss), "(expected ~ln(1000) =", round(math.log(1000), 2), ")")
    print("parameters:", sum(p.numel() for p in model.parameters()) / 1e6, "M")
