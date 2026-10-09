import math
import torch
import torch.nn.functional as F


def build_optimizer(model, lr=3e-4, weight_decay=0.1, betas=(0.9, 0.95)):
    """AdamW with weight decay only on matrices (not biases / norm gains)."""
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (decay if p.ndim >= 2 else no_decay).append(p)
    groups = [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]
    return torch.optim.AdamW(groups, lr=lr, betas=betas, fused=torch.cuda.is_available())


def warmup_cosine(step, total, warmup, min_ratio=0.1):
    """Multiplier in [min_ratio, 1]; use with LambdaLR."""
    if step < warmup:
        return (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))


def train(model, loader, total_steps, accum=4, lr=3e-4, warmup=200, clip=1.0, device="cuda"):
    opt = build_optimizer(model, lr)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: warmup_cosine(s, total_steps, warmup))
    use_amp = device == "cuda"
    step, micro = 0, 0
    model.train()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=use_amp):
            loss = F.cross_entropy(model(x), y) / accum          # average over micro-batches
        loss.backward()
        micro += 1
        if micro % accum:
            continue

        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip)   # returns pre-clip norm
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        if step % 50 == 0:
            print(f"step {step:6d}  loss {loss.item() * accum:.4f}  grad-norm {gnorm:.2f}  lr {sched.get_last_lr()[0]:.2e}")
        if step >= total_steps:
            break
