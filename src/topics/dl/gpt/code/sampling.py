import numpy as np


def softmax(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def sample_next(logits, temperature=1.0, top_k=None, top_p=None, min_p=None, rng=None):
    """Temperature -> top-k -> top-p -> min-p, then sample. Returns (token_id, final_probs)."""
    rng = rng or np.random.default_rng()
    logits = np.asarray(logits, dtype=np.float64)
    if temperature == 0:                                   # greedy
        probs = np.zeros_like(logits); probs[logits.argmax()] = 1.0
        return int(logits.argmax()), probs

    probs = softmax(logits / temperature)
    order = np.argsort(-probs)                             # indices by descending probability
    keep = np.zeros_like(probs, dtype=bool)

    k = len(probs) if top_k is None else top_k
    keep[order[:k]] = True                                 # top-k

    if top_p is not None:                                  # nucleus on the renormalised top-k set
        p = np.where(keep, probs, 0.0); p /= p.sum()
        cum = np.cumsum(p[order])
        cutoff = np.searchsorted(cum, top_p) + 1           # smallest prefix with cum >= top_p
        nucleus = np.zeros_like(keep); nucleus[order[:cutoff]] = True
        keep &= nucleus

    if min_p is not None:
        keep &= probs >= min_p * probs[keep].max()

    final = np.where(keep, probs, 0.0)
    final /= final.sum()
    return int(rng.choice(len(final), p=final)), final


if __name__ == "__main__":
    logits = np.log([0.5, 0.3, 0.1, 0.05, 0.03, 0.02])

    _, p = sample_next(logits, top_k=3);    print("top-k=3      :", p.round(3))
    _, p = sample_next(logits, top_p=0.85); print("top-p=0.85   :", p.round(3))
    _, p = sample_next(logits, top_p=0.75); print("top-p=0.75   :", p.round(3))
    _, p = sample_next(logits, min_p=0.2);  print("min-p=0.2    :", p.round(3))
    _, p = sample_next(logits, temperature=0.5); print("T=0.5        :", p.round(3))

    rng = np.random.default_rng(0)
    draws = [sample_next(logits, temperature=1.0, top_p=0.9, rng=rng)[0] for _ in range(20000)]
    print("empirical    :", (np.bincount(draws, minlength=6) / 20000).round(3))
    assert sample_next(logits, temperature=0)[0] == 0
