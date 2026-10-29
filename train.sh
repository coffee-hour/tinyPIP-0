#!/usr/bin/env bash
# tinyPIP-0 one-command launcher. Installs deps if torch is missing, reports the device,
# and runs train.py (which downloads input.txt if missing and auto-resumes from out/tinypip0.pt).
# Usage: bash train.sh [train.py flags, e.g. --steps 6000 or --scratch]
set -euo pipefail
cd "$(dirname "$0")"

if [ -n "${PYTHON:-}" ]; then PY="$PYTHON"
elif command -v python3 >/dev/null 2>&1; then PY=python3
elif command -v python >/dev/null 2>&1; then PY=python
else echo "ERROR: Python 3 not found." >&2; exit 1
fi

if ! "$PY" -c "import torch" >/dev/null 2>&1; then
  echo "Installing dependencies from requirements.txt..."
  "$PY" -m pip install -q -r requirements.txt
fi

"$PY" -c "import torch; m = getattr(torch.backends, 'mps', None); d = 'cuda' if torch.cuda.is_available() else ('mps' if m and m.is_available() else 'cpu'); print('device:', d)"

exec "$PY" train.py "$@"
