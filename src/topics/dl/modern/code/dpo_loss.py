import torch
import torch.nn.functional as F


def sequence_logprob(logits, labels, mask):
    """Sum of token log-probs of `labels` under `logits`. mask=1 on response tokens only."""
    logp = logits.log_softmax(-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    return (logp * mask).sum(-1)


def dpo_loss(pol_chosen, pol_rejected, ref_chosen, ref_rejected, beta=0.1):
    """All inputs are (B,) sequence log-probs. Returns loss and implicit-reward accuracy."""
    chosen_reward = beta * (pol_chosen - ref_chosen)          # implicit reward  beta * log(pi / pi_ref)
    rejected_reward = beta * (pol_rejected - ref_rejected)
    margin = chosen_reward - rejected_reward
    loss = -F.logsigmoid(margin).mean()
    return loss, (margin > 0).float().mean()


if __name__ == "__main__":
    torch.manual_seed(0)
    B = 8
    ref_c, ref_r = torch.randn(B) - 20, torch.randn(B) - 20
    # a policy equal to the reference gives margin 0 -> loss = log 2
    loss, acc = dpo_loss(ref_c, ref_r, ref_c, ref_r)
    print("loss at init:", loss.item(), "(= ln 2 =", torch.log(torch.tensor(2.0)).item(), ")")
    # a policy that raises chosen / lowers rejected reduces the loss
    loss, acc = dpo_loss(ref_c + 3, ref_r - 3, ref_c, ref_r)
    print("improved policy loss:", loss.item(), " reward accuracy:", acc.item())
