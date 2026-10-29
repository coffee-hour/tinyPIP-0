"""Build data_phase2.txt from SmolTalk conversations (optional, best effort).

Usage: python fetch_data.py [--max-mb 15] [--out data_phase2.txt] [--split train]

This script never leaves a broken file behind and always exits 0 on network or
API failure, so train.py can continue on input.txt alone. Progress is saved to a
temporary file after every page; on timeout, rate limit, or interruption the
partial data is kept if it has enough conversations, otherwise it is discarded
and a warning is printed. Exit status is 0 in all cases; the final line says
which path was taken (wrote, partial, or skipped).
"""
import argparse
import json
import os
import signal
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

DATASET = "HuggingFaceTB/smol-smoltalk"
API = "https://datasets-server.huggingface.co/rows"
PAGE = 100
ROLE_LABEL = {"user": "User", "assistant": "Pip"}
MAX_TURN_CHARS = 2000
MIN_PARTIAL_CONVS = 500      # keep a partial file only if it has at least this many conversations
REQUEST_TIMEOUT = 30
MAX_RETRIES = 4
MAX_CONSECUTIVE_FAILURES = 6  # stop after this many pages in a row fail
PAUSE_BETWEEN_PAGES = 0.25    # be polite to the public API


class Stop(Exception):
    """Raised to stop cleanly (signal or too many failures)."""


def _on_signal(signum, frame):
    raise Stop(f"received signal {signum}")


def fetch_page(split, offset):
    """Return the JSON page, or None if all retries fail or the API signals we should give up."""
    q = urllib.parse.urlencode({"dataset": DATASET, "config": "default", "split": split,
                                "offset": offset, "length": PAGE})
    url = f"{API}?{q}"
    for attempt in range(MAX_RETRIES):
        try:
            with urllib.request.urlopen(url, timeout=REQUEST_TIMEOUT) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            # 429 and 5xx are transient; other client errors will not fix themselves.
            if e.code == 429 or e.code >= 500:
                wait = int(e.headers.get("Retry-After", 0) or 0) or (2 ** attempt)
                print(f"warning: HTTP {e.code} at offset {offset}, waiting {wait}s", file=sys.stderr)
                time.sleep(min(wait, 60))
                continue
            print(f"warning: HTTP {e.code} at offset {offset}; not retrying", file=sys.stderr)
            return None
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError, ValueError) as e:
            print(f"warning: attempt {attempt + 1} at offset {offset} failed: {e}", file=sys.stderr)
            time.sleep(2 ** attempt)
    return None


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


def build(args, tmp):
    limit = int(args.max_mb * 1024 * 1024)
    written = 0
    convs = 0
    offset = 0
    failures = 0
    stop_reason = "completed"
    with open(tmp, "w", encoding="ascii") as f:
        try:
            while written < limit:
                page = fetch_page(args.split, offset)
                if page is None:
                    failures += 1
                    if failures >= MAX_CONSECUTIVE_FAILURES:
                        stop_reason = f"gave up after {failures} consecutive failed pages"
                        break
                    offset += PAGE  # skip the bad page and keep going
                    continue
                failures = 0
                rows = page.get("rows", [])
                if not rows:
                    break
                for item in rows:
                    text = conversation_to_text(item.get("row", {}).get("messages", []))
                    if not text:
                        continue
                    f.write(text)
                    written += len(text)
                    convs += 1
                    if written >= limit:
                        break
                f.flush()
                os.fsync(f.fileno())
                offset += len(rows)
                print(f"offset {offset}: {convs} conversations, {written / 1048576:.2f} MB", flush=True)
                time.sleep(PAUSE_BETWEEN_PAGES)
        except Stop as e:
            stop_reason = f"interrupted ({e})"
    return convs, written, stop_reason


def main():
    p = argparse.ArgumentParser(description="Build data_phase2.txt from SmolTalk (best effort).")
    p.add_argument("--max-mb", type=float, default=15.0, help="stop after roughly this many MB (default: %(default)s)")
    p.add_argument("--out", default="data_phase2.txt")
    p.add_argument("--split", default="train")
    args = p.parse_args()

    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)

    tmp = args.out + ".tmp"
    if os.path.exists(tmp):
        os.remove(tmp)
    try:
        convs, written, reason = build(args, tmp)
    except Exception as e:  # never let this script crash the CI job
        print(f"warning: unexpected error: {e}", file=sys.stderr)
        convs, written, reason = 0, 0, f"error ({e})"

    if convs == 0:
        if os.path.exists(tmp):
            os.remove(tmp)
        # Never delete an existing args.out: it may be curated data that train.py loads.
        print(f"warning: no SmolTalk data written ({reason}); existing {args.out} (if any) left untouched")
        print("skipped: no new data file written")
        return 0

    if reason == "completed" or convs >= MIN_PARTIAL_CONVS:
        os.replace(tmp, args.out)
        kind = "wrote" if reason == "completed" else "partial"
        print(f"{kind} {args.out}: {convs} conversations, {written / 1048576:.2f} MB ({reason})")
        if reason != "completed":
            print(f"warning: stopped early: {reason}", file=sys.stderr)
        return 0

    os.remove(tmp)
    print(f"warning: only {convs} conversations before {reason}; discarded (need {MIN_PARTIAL_CONVS})")
    print(f"skipped: no new data file written; existing {args.out} (if any) left untouched")
    return 0


if __name__ == "__main__":
    sys.exit(main())
