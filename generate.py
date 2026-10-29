"""Generate text from a trained tinyPIP-0 checkpoint.

Usage:
  python generate.py --ckpt out/tinypip0.pt --prompt "ROMEO:" --tokens 300
  python generate.py --ckpt out/tinypip0.pt --chat

Chat mode keeps a running User:/Pip: transcript (matching input.txt) and
prints one Pip reply per turn. Commands: /reset clears history, /quit exits.
"""
import argparse

import torch

from model import Config, TinyPIP


def chat_loop(model, stoi, itos, args):
    history = ""
    print("tinyPIP-0 chat. Type /reset to clear history, /quit to exit.")
    while True:
        try:
            user = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue
        if user in ("/quit", "/exit"):
            break
        if user == "/reset":
            history = ""
            print("(history cleared)")
            continue

        turn = f"User: {user}\nPip:"
        unknown = sorted({c for c in turn if c not in stoi})
        if unknown:
            print(f"(skipped: message has characters not in vocabulary: {unknown})")
            continue

        history += turn
        start = torch.tensor([[stoi[c] for c in history]], dtype=torch.long)
        out = model.generate(start, args.tokens, temperature=args.temperature, top_k=args.top_k)
        new = "".join(itos[i] for i in out[0, start.size(1):].tolist())
        # Pip replies are one line in the training data; stop at the newline.
        reply = new.lstrip(" ").split("\n")[0].strip()
        history += " " + reply + "\n"
        print(f"Pip: {reply}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="out/tinypip0.pt")
    p.add_argument("--prompt", default="\n")
    p.add_argument("--chat", action="store_true", help="interactive back-and-forth mode")
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

    if args.seed is not None:
        torch.manual_seed(args.seed)

    if args.chat:
        chat_loop(model, stoi, itos, args)
        return

    unknown = sorted({c for c in args.prompt if c not in stoi})
    if unknown:
        raise SystemExit(f"prompt has characters not in vocabulary: {unknown}")

    start = torch.tensor([[stoi[c] for c in args.prompt]], dtype=torch.long)
    out = model.generate(start, args.tokens, temperature=args.temperature, top_k=args.top_k)
    print("".join(itos[i] for i in out[0].tolist()))


if __name__ == "__main__":
    main()
