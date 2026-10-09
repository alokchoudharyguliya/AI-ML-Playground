import numpy as np


def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)          # numerical stability
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def attention(Q, K, V, mask=None, scale=True):
    """Q: (T, d)  K: (S, d)  V: (S, dv)  mask: (T, S) True = may attend."""
    scores = Q @ K.T
    if scale:
        scores = scores / np.sqrt(Q.shape[-1])
    if mask is not None:
        scores = np.where(mask, scores, -1e9)
    weights = softmax(scores)
    return weights @ V, weights


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    T = S = 6
    Q, K = rng.standard_normal((T, 8)), rng.standard_normal((S, 8))
    V = rng.standard_normal((S, 4))

    out, w = attention(Q, K, V)
    print("rows sum to 1:", np.allclose(w.sum(-1), 1), " output shape:", out.shape)

    causal = np.tril(np.ones((T, S), bool))          # position t sees only <= t
    out_c, w_c = attention(Q, K, V, mask=causal)
    print("causal: weights above the diagonal are zero:", np.allclose(w_c[~causal], 0))

    # attention is a soft dictionary lookup: a very sharp query returns (almost) one value
    q = K[2:3] * 20
    out_q, w_q = attention(q, K, V, scale=False)
    print("sharp query selects key 2:", w_q.argmax(), " weight:", round(float(w_q.max()), 4))
