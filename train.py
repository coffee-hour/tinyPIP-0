"""Train tinyPIP-0 on TinyShakespeare (character level).

Local:    python train.py
Colab:    !pip install -q torch && !python train.py
Resume:   python train.py                          (auto-resumes if out/tinypip0.pt exists)
          python train.py --resume                 (require and resume from --ckpt)
          python train.py --resume path/to/x.pt    (resume from a specific file)
Scratch:  python train.py --scratch                (ignore any existing checkpoint)
Output:   out/tinypip0.pt (weights, config, vocabulary, optimizer state, step)

In-memory training text is built in this order:
  1. input.txt (the base file, only read, never written)
  2. every data/curated_*.txt file, in sorted filename order
  3. data_phase2.txt, if it exists (optional SmolTalk data from fetch_data.py)
Missing files are skipped. Delete or rename the extra files to train on input.txt alone.

Vocabulary growth: when resuming, characters already in the checkpoint keep their
ids. Any new characters in the training text are appended with new ids. The token
embedding (shared with lm_head) is expanded to the new size: old rows are copied
unchanged, new rows are initialized with N(0, 0.02), and the AdamW moments for the
new rows start at zero. Resuming therefore never requires --scratch because of
vocabulary changes. The vocabulary never shrinks.
"""
import argparse
import glob
import os
import urllib.request
from dataclasses import asdict, replace

import torch

from model import Config, TinyPIP, count_params

DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_PATH = "input.txt"
CURATED_GLOB = os.path.join("data", "curated_*.txt")
PHASE2_PATH = "data_phase2.txt"
OUT_DIR = "out"
DEFAULT_CKPT = os.path.join(OUT_DIR, "tinypip0.pt")

BATCH_SIZE = 32
MAX_STEPS = 3000
EVAL_EVERY = 500
EVAL_ITERS = 50
LR = 3e-3
SEED = 1337
EMBED_INIT_STD = 0.02


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


def extra_data_paths():
    """Curated files in sorted order, then the optional phase-2 file."""
    paths = sorted(glob.glob(CURATED_GLOB))
    if os.path.exists(PHASE2_PATH):
        paths.append(PHASE2_PATH)
    return paths


def load_text():
    """Return input.txt with the curated and optional extra files appended."""
    if not os.path.exists(DATA_PATH):
        print("Downloading TinyShakespeare...")
        urllib.request.urlretrieve(DATA_URL, DATA_PATH)
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        text = f.read()
    for extra_path in extra_data_paths():
        with open(extra_path, "r", encoding="utf-8") as f:
            extra = f.read()
        if text and not text.endswith("\n"):
            text += "\n"
        text += extra
        print(f"loaded {extra_path} ({len(extra):,} chars)")
    print(f"total training text: {len(text):,} chars")
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


def build_vocab(text, old_itos=None):
    """Return (itos, new_chars).

    Without a checkpoint, ids are assigned to the sorted characters of text.
    With a checkpoint, every existing id is kept and characters that are not yet
    in the vocabulary are appended in sorted order.
    """
    if old_itos is None:
        return dict(enumerate(sorted(set(text)))), []
    itos = dict(old_itos)
    known = set(itos.values())
    next_id = max(itos) + 1 if itos else 0
    new_chars = sorted(set(text) - known)
    for ch in new_chars:
        itos[next_id] = ch
        next_id += 1
    return itos, new_chars


def resize_embedding_state(state, new_vocab):
    """Grow tok_emb.weight (and the tied lm_head.weight) to new_vocab rows.

    Existing rows are copied unchanged. New rows are drawn from N(0, EMBED_INIT_STD).
    """
    old = state["tok_emb.weight"]
    n_old, dim = old.shape
    if new_vocab < n_old:
        raise SystemExit(f"vocabulary cannot shrink ({n_old} -> {new_vocab})")
    new = torch.empty(new_vocab, dim, dtype=old.dtype, device=old.device)
    torch.nn.init.normal_(new, mean=0.0, std=EMBED_INIT_STD)
    new[:n_old] = old
    state = dict(state)
    state["tok_emb.weight"] = new
    if "lm_head.weight" in state:
        state["lm_head.weight"] = new  # keep the weight tying
    return state


def pad_optimizer_state(opt_state, params):
    """Zero-pad AdamW moment tensors whose first dimension grew with the vocabulary."""
    for idx, st in opt_state["state"].items():
        p = params[int(idx)]
        for k, v in list(st.items()):
            if (torch.is_tensor(v) and v.dim() > 0 and v.shape != p.shape
                    and v.shape[1:] == p.shape[1:] and v.shape[0] < p.shape[0]):
                pad = torch.zeros(p.shape[0] - v.shape[0], *v.shape[1:], dtype=v.dtype, device=v.device)
                st[k] = torch.cat([v, pad], dim=0)
    return opt_state


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

    resume_path = find_resume_path(args)
    ckpt = None
    old_itos = None
    start_step = 0
    if resume_path is not None:
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        old_itos = {int(k): v for k, v in ckpt["itos"].items()}

    itos, new_chars = build_vocab(text, old_itos)
    stoi = {ch: i for i, ch in itos.items()}
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)

    n = int(0.9 * len(data))
    train_data, val_data = data[:n], data[n:]

    vocab_grew = False
    if ckpt is not None:
        cfg = Config(**ckpt["config"])
        old_vocab = cfg.vocab_size
        state = ckpt["model"]
        if len(itos) != old_vocab:
            if len(itos) < old_vocab:
                raise SystemExit("vocabulary shrank below the checkpoint size; this should not happen")
            vocab_grew = True
            cfg = replace(cfg, vocab_size=len(itos))
            state = resize_embedding_state(state, len(itos))
            added = [ch for ch in new_chars]
            print(f"vocabulary grew from {old_vocab} to {len(itos)} characters; "
                  f"added {added!r}")
        model = TinyPIP(cfg).to(device)
        model.load_state_dict(state)
        start_step = int(ckpt.get("step", 0))
        print(f"resuming from {resume_path} at step {start_step}")
    else:
        cfg = Config(vocab_size=len(itos))
        model = TinyPIP(cfg).to(device)
        print("starting from scratch")
    print(f"device={device} vocab={len(itos)} params={count_params(model):,}")

    if start_step >= args.steps:
        print(f"checkpoint is already at step {start_step} (>= --steps {args.steps}); nothing to do")
        return

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)
    if ckpt is not None and "optimizer" in ckpt:
        saved_opt = ckpt["optimizer"]
        if vocab_grew:
            saved_opt = pad_optimizer_state(saved_opt, list(model.parameters()))
        opt.load_state_dict(saved_opt)
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
