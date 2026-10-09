import torch
import torch.nn as nn
import torch.nn.functional as F


class Expert(nn.Module):
    def __init__(self, d, d_ff):
        super().__init__()
        self.gate, self.up, self.down = nn.Linear(d, d_ff, bias=False), nn.Linear(d, d_ff, bias=False), nn.Linear(d_ff, d, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))        # SwiGLU expert


class MoE(nn.Module):
    def __init__(self, d, d_ff, n_experts=8, top_k=2, aux_weight=0.01):
        super().__init__()
        self.router = nn.Linear(d, n_experts, bias=False)
        self.experts = nn.ModuleList(Expert(d, d_ff) for _ in range(n_experts))
        self.E, self.k, self.aux_weight = n_experts, top_k, aux_weight

    def forward(self, x):                                  # x: (B, T, d)
        B, T, d = x.shape
        x = x.reshape(-1, d)                               # (N, d) flatten tokens
        logits = self.router(x)                            # (N, E)
        probs = logits.softmax(-1)
        top_p, top_i = logits.topk(self.k, dim=-1)         # choose k experts per token
        gates = top_p.softmax(-1)                          # renormalise over the chosen experts

        out = torch.zeros_like(x)
        for e, expert in enumerate(self.experts):          # (production: grouped GEMM / all-to-all)
            tok, slot = (top_i == e).nonzero(as_tuple=True)
            if tok.numel():
                out.index_add_(0, tok, gates[tok, slot, None] * expert(x[tok]))

        # Switch-Transformer load-balancing loss: E * sum_e f_e * P_e
        f = F.one_hot(top_i[:, 0], self.E).float().mean(0)     # fraction of tokens whose top-1 is e
        P = probs.mean(0)                                      # mean router probability
        aux = self.aux_weight * self.E * (f * P).sum()
        return out.view(B, T, d), aux


if __name__ == "__main__":
    torch.manual_seed(0)
    moe = MoE(d=64, d_ff=128)
    y, aux = moe(torch.randn(4, 16, 64))
    total = sum(p.numel() for p in moe.parameters())
    active = sum(p.numel() for p in moe.router.parameters()) + moe.k * sum(p.numel() for p in moe.experts[0].parameters())
    print(y.shape, "aux loss:", float(aux), f"(min {moe.aux_weight})", f"| total params {total:,}, active per token {active:,}")
