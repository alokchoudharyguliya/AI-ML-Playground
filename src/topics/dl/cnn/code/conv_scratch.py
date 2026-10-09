import numpy as np


def conv2d_naive(x, w, b, stride=1, pad=0):
    """x: (C_in, H, W)   w: (C_out, C_in, k, k)   b: (C_out,)  ->  (C_out, H', W')"""
    c_out, c_in, k, _ = w.shape
    xp = np.pad(x, ((0, 0), (pad, pad), (pad, pad)))
    h_out = (xp.shape[1] - k) // stride + 1
    w_out = (xp.shape[2] - k) // stride + 1
    y = np.zeros((c_out, h_out, w_out))
    for o in range(c_out):
        for i in range(h_out):
            for j in range(w_out):
                patch = xp[:, i * stride:i * stride + k, j * stride:j * stride + k]
                y[o, i, j] = np.sum(patch * w[o]) + b[o]
    return y


def im2col(x, k, stride=1, pad=0):
    """Unfold every k*k*C_in patch into a row -> (H'*W', C_in*k*k)."""
    c, h, w = x.shape
    xp = np.pad(x, ((0, 0), (pad, pad), (pad, pad)))
    h_out = (h + 2 * pad - k) // stride + 1
    w_out = (w + 2 * pad - k) // stride + 1
    cols = np.empty((h_out * w_out, c * k * k))
    for i in range(h_out):
        for j in range(w_out):
            cols[i * w_out + j] = xp[:, i * stride:i * stride + k, j * stride:j * stride + k].ravel()
    return cols, h_out, w_out


def conv2d_gemm(x, w, b, stride=1, pad=0):
    """The whole layer is ONE matrix multiply (this is what GPUs do)."""
    c_out, c_in, k, _ = w.shape
    cols, h_out, w_out = im2col(x, k, stride, pad)
    y = cols @ w.reshape(c_out, -1).T + b   # (H'W', C_in*k*k) @ (C_in*k*k, C_out)
    return y.T.reshape(c_out, h_out, w_out)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    x = rng.standard_normal((3, 12, 12))
    w = rng.standard_normal((8, 3, 3, 3))
    b = rng.standard_normal(8)

    y1 = conv2d_naive(x, w, b, stride=2, pad=1)
    y2 = conv2d_gemm(x, w, b, stride=2, pad=1)
    print("shape:", y1.shape, " max |naive - gemm| =", np.abs(y1 - y2).max())

    try:  # optional cross-check against PyTorch
        import torch
        import torch.nn.functional as F
        y3 = F.conv2d(torch.tensor(x)[None], torch.tensor(w), torch.tensor(b), stride=2, padding=1)[0].numpy()
        print("max |gemm - torch|  =", np.abs(y2 - y3).max())
    except ImportError:
        pass
