import torch
import torch.nn as nn


class LSTMCellScratch(nn.Module):
    """One fused matmul computes all four gates (this is how cuDNN does it)."""

    def __init__(self, d, n):
        super().__init__()
        self.n = n
        self.W = nn.Linear(d, 4 * n)                 # x -> [f, i, g, o]
        self.U = nn.Linear(n, 4 * n, bias=False)     # h -> [f, i, g, o]
        with torch.no_grad():
            self.W.bias.zero_()
            self.W.bias[:n] = 1.0                    # forget-gate bias = 1

    def forward(self, x, state):
        h, c = state
        z = self.W(x) + self.U(h)
        f, i, g, o = z.chunk(4, dim=-1)
        f, i, o = torch.sigmoid(f), torch.sigmoid(i), torch.sigmoid(o)
        g = torch.tanh(g)
        c = f * c + i * g                            # additive memory update
        h = o * torch.tanh(c)
        return h, (h, c)


def run(cell, xs):
    B, T, _ = xs.shape
    h = c = xs.new_zeros(B, cell.n)
    outs = []
    for t in range(T):
        y, (h, c) = cell(xs[:, t], (h, c))
        outs.append(y)
    return torch.stack(outs, 1), (h, c)


if __name__ == "__main__":
    torch.manual_seed(0)
    d, n, B, T = 8, 16, 4, 12
    ours = LSTMCellScratch(d, n)
    ref = nn.LSTM(d, n, batch_first=True)

    # copy weights: PyTorch orders the gates (i, f, g, o); ours is (f, i, g, o)
    def reorder(w):
        i, f, g, o = w.chunk(4, 0)
        return torch.cat([f, i, g, o], 0)

    with torch.no_grad():
        ref.weight_ih_l0.copy_(reorder(ours.W.weight))
        ref.weight_hh_l0.copy_(reorder(ours.U.weight))
        ref.bias_ih_l0.copy_(reorder(ours.W.bias))
        ref.bias_hh_l0.zero_()

    x = torch.randn(B, T, d)
    y1, _ = run(ours, x)
    y2, _ = ref(x)
    print("max |scratch - nn.LSTM| =", (y1 - y2).abs().max().item())
