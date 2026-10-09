import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, vocab, emb, hid, pad=0):
        super().__init__()
        self.embed = nn.Embedding(vocab, emb, padding_idx=pad)
        self.rnn = nn.GRU(emb, hid, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(2 * hid, hid)

    def forward(self, src):
        out, h = self.rnn(self.embed(src))               # out: (B, S, 2H)
        h0 = torch.tanh(self.fc(torch.cat([h[0], h[1]], -1)))   # initial decoder state
        return out, h0


class BahdanauAttention(nn.Module):
    """e_ti = v^T tanh(W s + U h_i)"""

    def __init__(self, hid):
        super().__init__()
        self.W = nn.Linear(hid, hid, bias=False)
        self.U = nn.Linear(2 * hid, hid, bias=False)
        self.v = nn.Linear(hid, 1, bias=False)

    def forward(self, s, enc, mask):
        # s: (B, H)  enc: (B, S, 2H)  mask: (B, S) True for real tokens
        e = self.v(torch.tanh(self.W(s).unsqueeze(1) + self.U(enc))).squeeze(-1)   # (B, S)
        e = e.masked_fill(~mask, float("-inf"))                                     # ignore padding
        alpha = F.softmax(e, dim=-1)
        context = torch.bmm(alpha.unsqueeze(1), enc).squeeze(1)                     # (B, 2H)
        return context, alpha


class Decoder(nn.Module):
    def __init__(self, vocab, emb, hid, pad=0):
        super().__init__()
        self.embed = nn.Embedding(vocab, emb, padding_idx=pad)
        self.attn = BahdanauAttention(hid)
        self.cell = nn.GRUCell(emb + 2 * hid, hid)
        self.out = nn.Linear(hid + 2 * hid + emb, vocab)

    def step(self, y_prev, s, enc, mask):
        e = self.embed(y_prev)
        context, alpha = self.attn(s, enc, mask)
        s = self.cell(torch.cat([e, context], -1), s)
        logits = self.out(torch.cat([s, context, e], -1))
        return logits, s, alpha


class Seq2Seq(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, emb=256, hid=512, pad=0):
        super().__init__()
        self.pad = pad
        self.enc, self.dec = Encoder(src_vocab, emb, hid, pad), Decoder(tgt_vocab, emb, hid, pad)

    def forward(self, src, tgt):                  # teacher forcing
        mask = src != self.pad
        enc, s = self.enc(src)
        logits = []
        for t in range(tgt.size(1) - 1):          # predict tgt[t+1] from tgt[t]
            lg, s, _ = self.dec.step(tgt[:, t], s, enc, mask)
            logits.append(lg)
        return torch.stack(logits, 1)             # (B, T-1, V)

    @torch.no_grad()
    def greedy(self, src, bos, eos, max_len=50):
        mask = src != self.pad
        enc, s = self.enc(src)
        y = torch.full((src.size(0),), bos, device=src.device)
        out, attn = [], []
        for _ in range(max_len):
            lg, s, a = self.dec.step(y, s, enc, mask)
            y = lg.argmax(-1)
            out.append(y); attn.append(a)
            if (y == eos).all():
                break
        return torch.stack(out, 1), torch.stack(attn, 1)     # tokens, alignments (B, T, S)


if __name__ == "__main__":
    model = Seq2Seq(1000, 1000, emb=64, hid=128)
    src = torch.randint(1, 1000, (4, 9)); src[:, 7:] = 0     # padded
    tgt = torch.randint(1, 1000, (4, 11))
    logits = model(src, tgt)
    loss = F.cross_entropy(logits.reshape(-1, 1000), tgt[:, 1:].reshape(-1), ignore_index=0, label_smoothing=0.1)
    print(logits.shape, float(loss))
