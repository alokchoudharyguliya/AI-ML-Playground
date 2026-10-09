import numpy as np


def rmsnorm(x, gamma, eps=1e-6):
    return gamma * x / np.sqrt((x ** 2).mean(-1, keepdims=True) + eps)


def silu(z):
    return z / (1 + np.exp(-z))


def swiglu(x, w_gate, w_up, w_down):
    return (silu(x @ w_gate) * (x @ w_up)) @ w_down


def rope_angles(positions, d_head, base=10000.0):
    inv_freq = base ** (-np.arange(0, d_head, 2) / d_head)       # theta_i
    return np.outer(positions, inv_freq)                          # (T, d/2)


def apply_rope(x, positions, base=10000.0):
    """x: (T, d_head). Rotate each (x[2i], x[2i+1]) pair by position * theta_i."""
    ang = rope_angles(positions, x.shape[-1], base)
    cos, sin = np.cos(ang), np.sin(ang)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    out = np.empty_like(x)
    out[..., 0::2] = x1 * cos - x2 * sin
    out[..., 1::2] = x1 * sin + x2 * cos
    return out


def gqa_attention(q, k, v, n_heads, n_kv_heads):
    """q: (T, H*d)  k,v: (T, G*d). Causal GQA, returns (T, H*d)."""
    T = q.shape[0]
    d = q.shape[1] // n_heads
    q = q.reshape(T, n_heads, d)
    k = np.repeat(k.reshape(T, n_kv_heads, d), n_heads // n_kv_heads, axis=1)   # share KV heads
    v = np.repeat(v.reshape(T, n_kv_heads, d), n_heads // n_kv_heads, axis=1)
    scores = np.einsum("thd,shd->hts", q, k) / np.sqrt(d)
    scores = np.where(np.tril(np.ones((T, T), bool)), scores, -1e9)
    w = np.exp(scores - scores.max(-1, keepdims=True)); w /= w.sum(-1, keepdims=True)
    return np.einsum("hts,shd->thd", w, v).reshape(T, -1)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    d = 64
    q, k = rng.standard_normal(d), rng.standard_normal(d)

    def score(m, n):
        return apply_rope(q[None], [m])[0] @ apply_rope(k[None], [n])[0]

    s1, s2 = score(5, 12), score(5 + 100, 12 + 100)
    print("score(5,12) =", round(s1, 10), " score(105,112) =", round(s2, 10), " |diff| =", abs(s1 - s2))
    assert abs(s1 - s2) < 1e-9, "RoPE logit must depend only on n - m"
    assert abs(score(5, 12) - score(5, 13)) > 1e-6

    x = rng.standard_normal((4, 16))
    print("RMSNorm rms per row:", np.sqrt((rmsnorm(x, np.ones(16)) ** 2).mean(-1)).round(3))

    T, H, G = 6, 8, 2
    out = gqa_attention(rng.standard_normal((T, H * 16)), rng.standard_normal((T, G * 16)), rng.standard_normal((T, G * 16)), H, G)
    print("GQA output:", out.shape, "-> KV cache is", H // G, "x smaller than MHA")
