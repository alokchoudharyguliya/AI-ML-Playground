import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

device = "cuda" if torch.cuda.is_available() else "cpu"
name = "bert-base-uncased"
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForSequenceClassification.from_pretrained(name, num_labels=2).to(device)

ds = load_dataset("glue", "sst2")
def encode(batch):
    return tok(batch["sentence"], truncation=True, max_length=128)
ds = ds.map(encode, batched=True)
ds = ds.rename_column("label", "labels")
ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

def collate(items):                       # dynamic padding: pad to the longest in the batch
    batch = tok.pad({k: [it[k] for it in items] for k in ("input_ids", "attention_mask")}, return_tensors="pt")
    batch["labels"] = torch.stack([it["labels"] for it in items])
    return batch

train = DataLoader(ds["train"], batch_size=32, shuffle=True, collate_fn=collate)
val = DataLoader(ds["validation"], batch_size=128, collate_fn=collate)

# layer-wise learning-rate decay: lower layers move less
def param_groups(model, lr=3e-5, decay=0.9, wd=0.01):
    layers = model.bert.encoder.layer
    groups = [{"params": list(model.classifier.parameters()) + list(model.bert.pooler.parameters()), "lr": lr}]
    for i, layer in enumerate(reversed(layers)):
        groups.append({"params": layer.parameters(), "lr": lr * decay ** (i + 1)})
    groups.append({"params": model.bert.embeddings.parameters(), "lr": lr * decay ** (len(layers) + 1)})
    for g in groups:
        g["weight_decay"] = wd
    return groups

epochs = 3
opt = torch.optim.AdamW(param_groups(model))
steps = epochs * len(train)
sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
scaler = torch.amp.GradScaler(enabled=device == "cuda")

for epoch in range(epochs):
    model.train()
    for batch in train:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.autocast(device_type=device, dtype=torch.float16, enabled=device == "cuda"):
            loss = model(**batch).loss
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update(); sched.step()

    model.eval(); correct = total = 0
    with torch.no_grad():
        for batch in val:
            batch = {k: v.to(device) for k, v in batch.items()}
            pred = model(**{k: v for k, v in batch.items() if k != "labels"}).logits.argmax(-1)
            correct += (pred == batch["labels"]).sum().item(); total += len(pred)
    print(f"epoch {epoch + 1}: val accuracy {correct / total:.3%}")
