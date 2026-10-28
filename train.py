"""Train tinyPIP-0 on TinyShakespeare (character level).

Local:   python train.py
Colab:   !pip install -q torch && !python train.py
Output:  out/tinypip0.pt (weights, config, and character vocabulary)
"""
import os
import urllib.request
from dataclasses import asdict

import torch

from model import Config, TinyPIP, count_params

DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_PATH = "input.txt"
OUT_DIR = "out"

BATCH_SIZE = 32
MAX_STEPS = 3000
EVAL_EVERY = 500
EVAL_ITERS = 50
LR = 3e-3
SEED = 1337


def load_text():
    if not os.path.exists(DATA_PATH):
        print("Downloading TinyShakespeare...")
        urllib.request.urlretrieve(DATA_URL, DATA_PATH)
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        return f.read()


def main():
    torch.manual_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    text = load_text()

    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)

    n = int(0.9 * len(data))
    train_data, val_data = data[:n], data[n:]

    cfg = Config(vocab_size=len(chars))
    model = TinyPIP(cfg).to(device)
    print(f"device={device} vocab={len(chars)} params={count_params(model):,}")

    def get_batch(split):
        src = train_data if split == "train" else val_data
        ix = torch.randint(len(src) - cfg.block_size, (BATCH_SIZE,))
        x = torch.stack([src[i:i + cfg.block_size] for i in ix])
        y = torch.stack([src[i + 1:i + cfg.block_size + 1] for i in ix])
        return x.to(device), y.to(device)

    @torch.no_grad()
    def estimate_loss():
        model.eval()
        out = {}
        for split in ("train", "val"):
            losses = torch.zeros(EVAL_ITERS)
            for k in range(EVAL_ITERS):
                x, y = get_batch(split)
                _, loss = model(x, y)
                losses[k] = loss.item()
            out[split] = losses.mean().item()
        model.train()
        return out

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.1)
    for step in range(MAX_STEPS + 1):
        if step % EVAL_EVERY == 0:
            l = estimate_loss()
            print(f"step {step:5d} | train {l['train']:.4f} | val {l['val']:.4f}")
        if step == MAX_STEPS:
            break
        x, y = get_batch("train")
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, "tinypip0.pt")
    torch.save({
        "model": model.state_dict(),
        "config": asdict(cfg),
        "itos": itos,
    }, path)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
