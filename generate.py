"""Generate text from a trained tinyPIP-0 checkpoint.

Usage:
  python generate.py --ckpt out/tinypip0.pt --prompt "ROMEO:" --tokens 300
"""
import argparse

import torch

from model import Config, TinyPIP


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="out/tinypip0.pt")
    p.add_argument("--prompt", default="\n")
    p.add_argument("--tokens", type=int, default=300)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    cfg = Config(**ckpt["config"])
    model = TinyPIP(cfg)
    model.load_state_dict(ckpt["model"])
    model.eval()

    itos = {int(k): v for k, v in ckpt["itos"].items()}
    stoi = {v: k for k, v in itos.items()}
    unknown = sorted({c for c in args.prompt if c not in stoi})
    if unknown:
        raise SystemExit(f"prompt has characters not in vocabulary: {unknown}")

    if args.seed is not None:
        torch.manual_seed(args.seed)
    start = torch.tensor([[stoi[c] for c in args.prompt]], dtype=torch.long)
    out = model.generate(start, args.tokens, temperature=args.temperature, top_k=args.top_k)
    print("".join(itos[i] for i in out[0].tolist()))


if __name__ == "__main__":
    main()
