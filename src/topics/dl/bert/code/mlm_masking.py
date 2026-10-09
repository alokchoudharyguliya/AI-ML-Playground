import numpy as np

MASK_ID, VOCAB = 103, 30522          # BERT-base-uncased ids
SPECIAL = {0, 101, 102, 103}         # [PAD] [CLS] [SEP] [MASK]


def mlm_mask(ids, rng, p=0.15, special=SPECIAL):
    """Return (inputs, labels). labels = -100 where no loss is applied."""
    ids = np.asarray(ids)
    can_mask = ~np.isin(ids, list(special))
    chosen = (rng.random(ids.shape) < p) & can_mask            # which tokens are predicted

    labels = np.where(chosen, ids, -100)
    inputs = ids.copy()

    r = rng.random(ids.shape)
    to_mask = chosen & (r < 0.8)                                # 80%  -> [MASK]
    to_rand = chosen & (r >= 0.8) & (r < 0.9)                   # 10%  -> random token
    # remaining 10% of chosen: unchanged (but still in the loss!)
    inputs[to_mask] = MASK_ID
    inputs[to_rand] = rng.integers(1000, VOCAB, size=to_rand.sum())
    return inputs, labels


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    batch = rng.integers(1000, VOCAB, size=(512, 128))          # 65,536 ordinary tokens
    inputs, labels = mlm_mask(batch, rng)

    chosen = labels != -100
    print(f"selected: {chosen.mean():.3%}  (target 15%)")
    print(f"[MASK]   : {(inputs[chosen] == MASK_ID).mean():.3%}  (target 80%)")
    rand = (inputs[chosen] != MASK_ID) & (inputs[chosen] != batch[chosen])
    print(f"random   : {rand.mean():.3%}  (target 10%)")
    print(f"unchanged: {(inputs[chosen] == batch[chosen]).mean():.3%}  (target 10%)")
