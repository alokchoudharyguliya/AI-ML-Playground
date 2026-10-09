import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class LSTMClassifier(nn.Module):
    def __init__(self, vocab, emb=128, hidden=256, classes=2, layers=2, bidirectional=True, pad_idx=0):
        super().__init__()
        self.embed = nn.Embedding(vocab, emb, padding_idx=pad_idx)
        self.lstm = nn.LSTM(emb, hidden, layers, batch_first=True, dropout=0.3, bidirectional=bidirectional)
        out = hidden * (2 if bidirectional else 1)
        self.drop = nn.Dropout(0.3)
        self.fc = nn.Linear(out, classes)

    def forward(self, tokens, lengths):
        x = self.drop(self.embed(tokens))
        # pack so padding never touches the recurrent state
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(out, batch_first=True)          # (B, T, H)

        # masked mean + max pooling over valid time steps
        mask = (torch.arange(out.size(1), device=out.device)[None] < lengths[:, None]).unsqueeze(-1)
        mean = (out * mask).sum(1) / lengths[:, None]
        mx = out.masked_fill(~mask, float("-inf")).max(1).values
        return self.fc(self.drop(mean + mx))


if __name__ == "__main__":
    model = LSTMClassifier(vocab=5000)
    tokens = torch.randint(1, 5000, (4, 20))
    lengths = torch.tensor([20, 15, 9, 5])
    for i, L in enumerate(lengths):
        tokens[i, L:] = 0                                           # padding
    print(model(tokens, lengths).shape)                              # (4, 2)
