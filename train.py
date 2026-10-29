"""Train tinyPIP-0 on TinyShakespeare (character level).

Local:    python train.py
Colab:    !pip install -q torch && !python train.py
Resume:   python train.py                          (auto-resumes if out/tinypip0.pt exists)
          python train.py --resume                 (require and resume from --ckpt)
          python train.py --resume path/to/x.pt    (resume from a specific file)
Scratch:  python train.py --scratch                (ignore any existing checkpoint)
Output:   out/tinypip0.pt (weights, config, vocabulary, optimizer state, step)

If data_phase2.txt exists next to this file, it is appended to input.txt in memory
(input.txt is only read, never written). Delete or rename data_phase2.txt to train
on input.txt alone.
"""
import argparse
import os
import urllib.request
from dataclasses import asdict

import torch

from model import Config, TinyPIP, count_params

DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_PATH = "input.txt"
PHASE2_PATH = "data_phase2.txt"
OUT_DIR = "out"
DEFAULT_CKPT = os.path.join(OUT_DIR, "tinypip0.pt")

BATCH_SIZE = 32
MAX_STEPS = 3000
EVAL_EVERY = 500
EVAL_ITERS = 50
LR = 3e-3
SEED = 1337


def parse_args():
    p = argparse.ArgumentParser(description="Train tinyPIP-0 (resumes from a checkpoint when one exists).")
    p.add_argument("--ckpt", default=DEFAULT_CKPT,
                   help="checkpoint path to save to and to auto-resume from (default: %(default)s)")
    p.add_argument("--resume", nargs="?", const=DEFAULT_CKPT, default=None, metavar="PATH",
                   help="resume from PATH (or --ckpt if PATH is omitted); errors if the file is missing")
    p.add_argument("--scratch", action="store_true",
                   help="ignore any existing checkpoint and train from random initialization")
    p.add_argument("--steps", type=int, default=MAX_STEPS,
                   help="total training step to reach, counting from 0 (default: %(default)s)")
    p.add_argument("--lr", type=float, default=LR, help="learning rate (default: %(default)s)")
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="(default: %(default)s)")
    p.add_argument("--eval-every", type=int, default=EVAL_EVERY, help="(default: %(default)s)")
    p.add_argument("--eval-iters", type=int, default=EVAL_ITERS, help="(default: %(default)s)")
    p.add_argument("--seed", type=int, default=SEED, help="(default: %(default)s)")
    args = p.parse_args()

    if args.scratch and args.resume is not None:
        p.error("--scratch and --resume are mutually exclusive")
    if args.steps < 0:
        p.error("--steps must be >= 0")
    if args.lr <= 0:
        p.error("--lr must be > 0")
    if args.batch_size <= 0:
        p.error("--batch-size must be > 0")
    if args.eval_every <= 0:
        p.error("--eval-every must be > 0")
    if args.eval_iters <= 0:
        p.error("--eval-iters must be > 0")
    return args


def pick_device():
    """CUDA if available, then Apple MPS, then CPU."""
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_text():
    """Return input.txt, with data_phase2.txt appended when that file exists."""
    if not os.path.exists(DATA_PATH):
        print("Downloading TinyShakespeare...")
        urllib.request.urlretrieve(DATA_URL, DATA_PATH)
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        text = f.read()
    if os.path.exists(PHASE2_PATH):
        with open(PHASE2_PATH, "r", encoding="utf-8") as f:
            extra = f.read()
        if text and not text.endswith("\n"):
            text += "\n"
        text += extra
        print(f"loaded {PHASE2_PATH} ({len(extra):,} chars) after {DATA_PATH}")
    return text


def find_resume_path(args):
    """Return the checkpoint to resume from, or None to train from scratch."""
    if args.scratch:
        return None
    if args.resume is not None:
        if not os.path.exists(args.resume):
            raise SystemExit(f"--resume: checkpoint not found: {args.resume}")
        return args.resume
    if os.path.exists(args.ckpt):
        return args.ckpt
    return None


def save_checkpoint(path, model, cfg, itos, opt, step):
    """Write atomically so an interrupted save cannot corrupt an existing checkpoint."""
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    tmp = path + ".tmp"
    torch.save({
        "model": model.state_dict(),
        "config": asdict(cfg),
        "itos": itos,
        "optimizer": opt.state_dict(),
        "step": step,
    }, tmp)
    os.replace(tmp, path)
    print(f"saved {path} (step {step})")


def main():
    args = parse_args()
    device = pick_device()
    text = load_text()

    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)

    n = int(0.9 * len(data))
    train_data, val_data = data[:n], data[n:]

    resume_path = find_resume_path(args)
    ckpt = None
    start_step = 0
    if resume_path is not None:
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        if ckpt["itos"] != itos:
            raise SystemExit(
                "checkpoint vocabulary does not match the current training text; "
                "use --scratch to train a new model")
        cfg = Config(**ckpt["config"])
        model = TinyPIP(cfg).to(device)
        model.load_state_dict(ckpt["model"])
        start_step = int(ckpt.get("step", 0))
        print(f"resuming from {resume_path} at step {start_step}")
    else:
        cfg = Config(vocab_size=len(chars))
        model = TinyPIP(cfg).to(device)
        print("starting from scratch")
    print(f"device={device} vocab={len(chars)} params={count_params(model):,}")

    if start_step >= args.steps:
        print(f"checkpoint is already at step {start_step} (>= --steps {args.steps}); nothing to do")
        return

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)
    if ckpt is not None and "optimizer" in ckpt:
        opt.load_state_dict(ckpt["optimizer"])
        for group in opt.param_groups:
            group["lr"] = args.lr  # honor --lr instead of the value saved in the checkpoint
    elif ckpt is not None:
        print("checkpoint has no optimizer state; AdamW moments start fresh")

    # Offset the seed by the resume step so a resumed run does not replay the same batches.
    torch.manual_seed(args.seed + start_step)

    def get_batch(split):
        src = train_data if split == "train" else val_data
        ix = torch.randint(len(src) - cfg.block_size, (args.batch_size,))
        x = torch.stack([src[i:i + cfg.block_size] for i in ix])
        y = torch.stack([src[i + 1:i + cfg.block_size + 1] for i in ix])
        return x.to(device), y.to(device)

    @torch.no_grad()
    def estimate_loss():
        model.eval()
        out = {}
        for split in ("train", "val"):
            losses = torch.zeros(args.eval_iters)
            for k in range(args.eval_iters):
                x, y = get_batch(split)
                _, loss = model(x, y)
                losses[k] = loss.item()
            out[split] = losses.mean().item()
        model.train()
        return out

    for step in range(start_step, args.steps + 1):
        if step % args.eval_every == 0:
            l = estimate_loss()
            print(f"step {step:5d} | train {l['train']:.4f} | val {l['val']:.4f}")
        if step == args.steps:
            break
        x, y = get_batch("train")
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    save_checkpoint(args.ckpt, model, cfg, itos, opt, args.steps)


if __name__ == "__main__":
    main()
