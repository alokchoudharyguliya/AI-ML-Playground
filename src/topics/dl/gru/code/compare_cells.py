import time
import torch
import torch.nn as nn
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"


def make_batch(B=128, T=60, d=8):
    """Copy-first-token task: output the sign of x[:,0] after T steps of distractors."""
    x = torch.randn(B, T, d, device=device)
    y = (x[:, 0, 0] > 0).long()
    return x, y


class Net(nn.Module):
    def __init__(self, kind, d=8, n=64):
        super().__init__()
        self.rnn = {"rnn": nn.RNN, "gru": nn.GRU, "lstm": nn.LSTM}[kind](d, n, batch_first=True)
        self.fc = nn.Linear(n, 2)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out[:, -1])


for kind in ["rnn", "gru", "lstm"]:
    torch.manual_seed(0)
    net = Net(kind).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    n_params = sum(p.numel() for p in net.parameters())
    t0 = time.time()
    for step in range(400):
        x, y = make_batch()
        loss = F.cross_entropy(net(x), y)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
    x, y = make_batch(2048)
    acc = (net(x).argmax(1) == y).float().mean().item()
    print(f"{kind:5s} params={n_params:6d}  time={time.time() - t0:5.1f}s  accuracy={acc:.2%}")
# Expect: the vanilla RNN stays near 50% (it cannot carry the bit across 60 steps);
# GRU and LSTM learn the task.
