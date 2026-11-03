#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}${PROJECT_ROOT}/src"

MODE=${MODE:-persona}
CONFIG=${CONFIG:-configs/bigfive_${MODE}.yaml}

case "$MODE" in
  direct)
    python -m popalign generate bigfive-direct --config "$CONFIG"
    ;;
  persona)
    python -m popalign generate bigfive-persona --config "$CONFIG"
    ;;
  *)
    echo "Unknown MODE=$MODE. Use MODE=direct or MODE=persona." >&2
    exit 1
    ;;
esac
