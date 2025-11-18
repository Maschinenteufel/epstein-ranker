#!/usr/bin/env bash
set -euo pipefail

PORT="${1:-9000}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VIEWER_DIR="${SCRIPT_DIR}/viewer"
DATA_DIR="${SCRIPT_DIR}/data"

if [ ! -d "$VIEWER_DIR" ]; then
  echo "viewer/ directory not found next to viewer.sh. Are you running from the repo root?" >&2
  exit 1
fi

echo "Serving viewer on http://localhost:${PORT}"
echo "Press Ctrl+C to stop."
cd "$VIEWER_DIR"
if [ ! -e "data" ] && [ -d "$DATA_DIR" ]; then
  ln -s ../data data
fi
python -m http.server "$PORT"
