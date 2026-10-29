#!/usr/bin/env bash
# tinyPIP-0 one-command launcher. Installs requirements.txt if torch or numpy is missing, reports the device,
# and runs train.py (which downloads input.txt if missing and auto-resumes from out/tinypip0.pt).
# Usage: bash train.sh [train.py flags, e.g. --steps 6000 or --scratch]
#        bash train.sh --huge-data [--max-mb 15] [train.py flags]
#   --huge-data  first runs fetch_data.py to build data_phase2.txt from SmolTalk (default 15 MB),
#                then trains on input.txt + data_phase2.txt. --max-mb is passed to fetch_data.py.
set -euo pipefail
cd "$(dirname "$0")"

HUGE=0
FETCH_ARGS=()
TRAIN_ARGS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --huge-data) HUGE=1; shift ;;
    --max-mb) FETCH_ARGS+=("--max-mb" "$2"); shift 2 ;;
    *) TRAIN_ARGS+=("$1"); shift ;;
  esac
done

if [ -n "${PYTHON:-}" ]; then PY="$PYTHON"
elif command -v python3 >/dev/null 2>&1; then PY=python3
elif command -v python >/dev/null 2>&1; then PY=python
else echo "ERROR: Python 3 not found." >&2; exit 1
fi

if ! "$PY" -c "import torch, numpy" >/dev/null 2>&1; then
  echo "Installing dependencies from requirements.txt..."
  "$PY" -m pip install -q -r requirements.txt
fi

"$PY" -c "import torch; m = getattr(torch.backends, 'mps', None); d = 'cuda' if torch.cuda.is_available() else ('mps' if m and m.is_available() else 'cpu'); print('device:', d)"

if [ "$HUGE" -eq 1 ]; then
  echo "Fetching SmolTalk conversations into data_phase2.txt..."
  "$PY" fetch_data.py "${FETCH_ARGS[@]+"${FETCH_ARGS[@]}"}"
fi

exec "$PY" train.py "${TRAIN_ARGS[@]+"${TRAIN_ARGS[@]}"}"
