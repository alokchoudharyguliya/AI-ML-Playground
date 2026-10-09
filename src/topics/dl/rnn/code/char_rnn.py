import torch
import torch.nn as nn
import torch.nn.functional as F


class CharModel(nn.Module):
    def __init__(self, vocab, hidden=256, layers=2, kind="lstm", dropout=0.2):
        super().__init__()
        self.embed = nn.Embedding(vocab, hidden)
        rnn = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}[kind]
        self.rnn = rnn(hidden, hidden, layers, batch_first=True, dropout=dropout)  # dropout between layers only
        self.head = nn.Linear(hidden, vocab)

    def forward(self, x, state=None):
        out, state = self.rnn(self.embed(x), state)
        return self.head(out), state


def detach(state):
    """Cut the graph between truncated-BPTT windows but keep the values."""
    return tuple(s.detach() for s in state) if isinstance(state, tuple) else state.detach()


def train(text, kind="lstm", bptt=128, batch=64, epochs=5, lr=2e-3, device="cuda" if torch.cuda.is_available() else "cpu"):
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    data = torch.tensor([stoi[c] for c in text])
    # arrange text into `batch` parallel streams so hidden state can carry across windows
    n = (len(data) - 1) // batch * batch
    X = data[:n].view(batch, -1)
    Y = data[1:n + 1].view(batch, -1)

    model = CharModel(len(chars), kind=kind).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    for epoch in range(epochs):
        state = None
        for i in range(0, X.size(1) - bptt, bptt):
            x, y = X[:, i:i + bptt].to(device), Y[:, i:i + bptt].to(device)
            logits, state = model(x, state)
            state = detach(state)                       # truncated BPTT
            loss = F.cross_entropy(logits.reshape(-1, len(chars)), y.reshape(-1))
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)   # essential for RNNs
            opt.step()
        print(f"epoch {epoch + 1}  loss {loss.item():.3f}")
    return model, chars, stoi


@torch.no_grad()
def sample(model, chars, stoi, prompt="The ", n=300, temperature=0.8, device="cpu"):
    model.eval()
    x = torch.tensor([[stoi[c] for c in prompt]], device=device)
    logits, state = model(x)
    out = prompt
    for _ in range(n):
        probs = F.softmax(logits[0, -1] / temperature, -1)
        idx = torch.multinomial(probs, 1)
        out += chars[idx.item()]
        logits, state = model(idx[None], state)
    return out


if __name__ == "__main__":
    text = open("input.txt").read()      # e.g. tiny-shakespeare
    model, chars, stoi = train(text, kind="gru")
    print(sample(model, chars, stoi))
