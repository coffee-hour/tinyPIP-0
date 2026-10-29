"""Export out/tinypip0.pt to browser-readable files for the web chat page.

Usage:
  python export_web.py
  python export_web.py --ckpt out/tinypip0.pt --out web/model

Writes:
  web/model/weights.bin     all tensors as little-endian float32, concatenated
  web/model/manifest.json   config, character vocabulary, and each tensor's name, shape,
                            offset (in floats), and length

tok_emb.weight is also the output head (weight tying), so lm_head.weight is skipped.
"""
import argparse
import json
import os

import torch


def main():
    p = argparse.ArgumentParser(description="Export a tinyPIP-0 checkpoint for the browser.")
    p.add_argument("--ckpt", default=os.path.join("out", "tinypip0.pt"))
    p.add_argument("--out", default=os.path.join("web", "model"))
    args = p.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    state = ckpt["model"]
    names = sorted(k for k in state.keys() if k != "lm_head.weight")

    os.makedirs(args.out, exist_ok=True)
    manifest = {
        "config": ckpt["config"],
        "itos": ckpt["itos"],
        "step": int(ckpt.get("step", 0)),
        "tensors": {},
    }

    offset = 0
    weights_path = os.path.join(args.out, "weights.bin")
    with open(weights_path, "wb") as f:
        for name in names:
            t = state[name].detach().to(torch.float32).contiguous().cpu()
            f.write(t.numpy().astype("<f4").tobytes())
            manifest["tensors"][name] = {
                "shape": list(t.shape),
                "offset": offset,
                "length": int(t.numel()),
            }
            offset += int(t.numel())
    manifest["total_floats"] = offset

    manifest_path = os.path.join(args.out, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1)

    print(f"wrote {weights_path} ({offset:,} floats, {offset * 4:,} bytes)")
    print(f"wrote {manifest_path} ({len(names)} tensors, step {manifest['step']})")


if __name__ == "__main__":
    main()
