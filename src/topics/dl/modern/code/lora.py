import copy
import math
import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """y = W0 x + (alpha / r) * B (A x)   with W0 frozen."""

    def __init__(self, base: nn.Linear, r=16, alpha=32, dropout=0.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)                                   # freeze pretrained weights
        self.r, self.scale = r, alpha / r
        self.A = nn.Parameter(torch.empty(r, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))      # zero init -> starts as the base model
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.base(x) + self.scale * (self.drop(x) @ self.A.T @ self.B.T)

    @torch.no_grad()
    def merge(self):
        """Fold the adapter into W0 for zero-overhead inference."""
        self.base.weight += self.scale * (self.B @ self.A)
        return self.base


def add_lora(model, target=("q_proj", "v_proj"), r=16, alpha=32):
    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            if child_name in target and isinstance(child, nn.Linear):
                setattr(module, child_name, LoRALinear(child, r, alpha))
    return model


if __name__ == "__main__":
    torch.manual_seed(0)
    base = nn.Linear(256, 256)
    lora = LoRALinear(base, r=8, alpha=16)
    x = torch.randn(4, 256)
    assert torch.allclose(lora(x), base(x), atol=1e-6), "B=0 => identical to base at init"

    nn.init.normal_(lora.B, std=0.02)                                 # pretend we trained
    y = lora(x)
    merged = copy.deepcopy(lora).merge()                              # fold BA into W0
    print("max |adapter - merged| =", (y - merged(x)).abs().max().item())

    trainable = sum(p.numel() for p in lora.parameters() if p.requires_grad)
    print(f"trainable {trainable:,} vs full {256 * 256 + 256:,}  ({trainable / (256 * 256):.1%})")
