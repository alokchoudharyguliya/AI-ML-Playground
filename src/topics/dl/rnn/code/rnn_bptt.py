import numpy as np

rng = np.random.default_rng(0)
n, d, T = 5, 3, 7                      # hidden, input, sequence length

params = {
    "Wxh": rng.standard_normal((n, d)) * 0.5,
    "Whh": rng.standard_normal((n, n)) * 0.5,
    "Why": rng.standard_normal((1, n)) * 0.5,
    "b": np.zeros(n),
}
xs = rng.standard_normal((T, d))
target = 0.7


def forward(p):
    h = np.zeros(n)
    cache = [(None, h)]
    for t in range(T):
        h = np.tanh(p["Whh"] @ h + p["Wxh"] @ xs[t] + p["b"])
        cache.append((xs[t], h))
    y = (p["Why"] @ h)[0]
    loss = 0.5 * (y - target) ** 2
    return loss, y, cache


def backward(p, y, cache):
    g = {k: np.zeros_like(v) for k, v in p.items()}
    h_T = cache[-1][1]
    dy = y - target
    g["Why"] += dy * h_T[None, :]
    dh = dy * p["Why"][0]                       # dL/dh_T
    for t in range(T, 0, -1):                   # walk back through time
        x, h = cache[t]
        h_prev = cache[t - 1][1]
        da = dh * (1 - h ** 2)                  # through tanh
        g["Wxh"] += np.outer(da, x)             # shared weights: gradients SUM over time
        g["Whh"] += np.outer(da, h_prev)
        g["b"] += da
        dh = p["Whh"].T @ da                    # to h_{t-1}
    return g


loss, y, cache = forward(params)
grads = backward(params, y, cache)

# finite-difference check
worst = 0.0
for k, v in params.items():
    for idx in np.ndindex(*v.shape):
        old = v[idx]
        v[idx] = old + 1e-5; lp = forward(params)[0]
        v[idx] = old - 1e-5; lm = forward(params)[0]
        v[idx] = old
        num = (lp - lm) / 2e-5
        worst = max(worst, abs(num - grads[k][idx]) / (abs(num) + abs(grads[k][idx]) + 1e-12))
print(f"loss {loss:.5f}   worst relative gradient error: {worst:.2e}")

# vanishing gradient demo: norm of dh_t as we go back
for rho in (0.5, 0.9, 1.1):
    W = rho * np.linalg.qr(rng.standard_normal((n, n)))[0]
    v = np.ones(n)
    norms = []
    for _ in range(30):
        v = W.T @ v
        norms.append(np.linalg.norm(v))
    print(f"rho={rho}: gradient norm after 30 steps back = {norms[-1]:.3e}")
