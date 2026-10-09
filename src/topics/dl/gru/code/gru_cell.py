import torch
import torch.nn as nn


class GRUCellScratch(nn.Module):
    """PyTorch-compatible GRU: gate order (r, z, n), reset applied AFTER U_h h."""

    def __init__(self, d, n):
        super().__init__()
        self.n = n
        self.W = nn.Linear(d, 3 * n)     # x -> [r, z, n]
        self.U = nn.Linear(n, 3 * n)     # h -> [r, z, n]

    def forward(self, x, h):
        xr, xz, xn = self.W(x).chunk(3, -1)
        hr, hz, hn = self.U(h).chunk(3, -1)
        r = torch.sigmoid(xr + hr)               # reset gate
        z = torch.sigmoid(xz + hz)               # update gate
        n = torch.tanh(xn + r * hn)              # candidate
        return (1 - z) * n + z * h               # NOTE: PyTorch's z weights the OLD state


def run(cell, xs):
    B, T, _ = xs.shape
    h = xs.new_zeros(B, cell.n)
    ys = []
    for t in range(T):
        h = cell(xs[:, t], h)
        ys.append(h)
    return torch.stack(ys, 1)


if __name__ == "__main__":
    torch.manual_seed(0)
    d, n = 6, 10
    ours, ref = GRUCellScratch(d, n), nn.GRU(d, n, batch_first=True)
    with torch.no_grad():
        ref.weight_ih_l0.copy_(ours.W.weight); ref.bias_ih_l0.copy_(ours.W.bias)
        ref.weight_hh_l0.copy_(ours.U.weight); ref.bias_hh_l0.copy_(ours.U.bias)
    x = torch.randn(3, 9, d)
    print("max |scratch - nn.GRU| =", (run(ours, x) - ref(x)[0]).abs().max().item())
