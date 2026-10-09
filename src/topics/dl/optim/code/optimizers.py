import numpy as np


def ravine(p):
    x, y = p
    return 0.05 * x * x + y * y, np.array([0.1 * x, 2 * y])


class SGD:
    def __init__(self, lr): self.lr = lr
    def step(self, p, g): return p - self.lr * g


class Momentum:
    def __init__(self, lr, beta=0.9): self.lr, self.beta, self.v = lr, beta, 0
    def step(self, p, g):
        self.v = self.beta * self.v + g
        return p - self.lr * self.v


class RMSProp:
    def __init__(self, lr, beta=0.9, eps=1e-8): self.lr, self.beta, self.eps, self.s = lr, beta, eps, 0
    def step(self, p, g):
        self.s = self.beta * self.s + (1 - self.beta) * g * g
        return p - self.lr * g / (np.sqrt(self.s) + self.eps)


class Adam:
    def __init__(self, lr, b1=0.9, b2=0.999, eps=1e-8, weight_decay=0.0):
        self.lr, self.b1, self.b2, self.eps, self.wd = lr, b1, b2, eps, weight_decay
        self.m = self.v = 0
        self.t = 0

    def step(self, p, g):
        self.t += 1
        self.m = self.b1 * self.m + (1 - self.b1) * g
        self.v = self.b2 * self.v + (1 - self.b2) * g * g
        m_hat = self.m / (1 - self.b1 ** self.t)           # bias correction
        v_hat = self.v / (1 - self.b2 ** self.t)
        update = m_hat / (np.sqrt(v_hat) + self.eps)
        return p - self.lr * (update + self.wd * p)         # wd != 0  ->  AdamW (decoupled)


def run(opt, steps=300, start=(-7.0, 3.2)):
    p = np.array(start)
    for _ in range(steps):
        _, g = ravine(p)
        p = opt.step(p, g)
    return ravine(p)[0]


if __name__ == "__main__":
    for name, opt in [("SGD", SGD(0.4)), ("Momentum", Momentum(0.05)), ("RMSProp", RMSProp(0.1)), ("Adam", Adam(0.3))]:
        print(f"{name:9s} final loss after 300 steps: {run(opt):.3e}")
