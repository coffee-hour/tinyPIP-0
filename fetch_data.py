"""Download a conversational corpus (HuggingFaceTB/smol-smoltalk) into data_phase2.txt.

Usage: python fetch_data.py [--max-mb 15] [--out data_phase2.txt] [--split train]

Rows are pulled from the Hugging Face datasets-server API (100 rows per request), converted
to the same plain-text format as input.txt ("User: ..." / "Pip: ..."), restricted to ASCII,
and written until --max-mb is reached. System messages are dropped. Existing output is
replaced. train.py loads data_phase2.txt automatically when it exists.
"""
import argparse
import json
import os
import sys
import time
import urllib.parse
import urllib.request

DATASET = "HuggingFaceTB/smol-smoltalk"
API = "https://datasets-server.huggingface.co/rows"
PAGE = 100
ROLE_LABEL = {"user": "User", "assistant": "Pip"}
MAX_TURN_CHARS = 2000


def fetch_page(split, offset, retries=5):
    q = urllib.parse.urlencode({"dataset": DATASET, "config": "default", "split": split,
                                "offset": offset, "length": PAGE})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(f"{API}?{q}", timeout=60) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception as e:  # network hiccup: back off and retry
            if attempt == retries - 1:
                raise
            print(f"retry {attempt + 1} at offset {offset}: {e}", file=sys.stderr)
            time.sleep(2 ** attempt)


def to_ascii(s):
    return s.encode("ascii", "ignore").decode("ascii")


def conversation_to_text(messages):
    lines = []
    for m in messages:
        label = ROLE_LABEL.get(m.get("role"))
        if label is None:
            continue  # skip system and anything unknown
        content = to_ascii(m.get("content") or "").strip()
        if not content:
            continue
        content = " ".join(content.split())[:MAX_TURN_CHARS]
        lines.append(f"{label}: {content}")
    if len(lines) < 2:
        return ""
    return "\n".join(lines) + "\n\n"


def main():
    p = argparse.ArgumentParser(description="Build data_phase2.txt from SmolTalk conversations.")
    p.add_argument("--max-mb", type=float, default=15.0, help="stop after roughly this many MB (default: %(default)s)")
    p.add_argument("--out", default="data_phase2.txt")
    p.add_argument("--split", default="train")
    args = p.parse_args()

    limit = int(args.max_mb * 1024 * 1024)
    written = 0
    convs = 0
    offset = 0
    tmp = args.out + ".tmp"
    with open(tmp, "w", encoding="ascii") as f:
        while written < limit:
            page = fetch_page(args.split, offset)
            rows = page.get("rows", [])
            if not rows:
                break
            for item in rows:
                text = conversation_to_text(item["row"].get("messages", []))
                if not text:
                    continue
                f.write(text)
                written += len(text)
                convs += 1
                if written >= limit:
                    break
            offset += len(rows)
            print(f"offset {offset}: {convs} conversations, {written / 1048576:.2f} MB")
    if convs == 0:
        os.remove(tmp)
        raise SystemExit("no conversations were written; check network access to datasets-server.huggingface.co")
    os.replace(tmp, args.out)
    print(f"wrote {args.out}: {convs} conversations, {written / 1048576:.2f} MB")


if __name__ == "__main__":
    main()
