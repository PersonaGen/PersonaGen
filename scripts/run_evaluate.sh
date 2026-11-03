#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}${PROJECT_ROOT}/src"

TARGET=${TARGET:-bigfive}
CONFIG=${CONFIG:-configs/eval_${TARGET}_example.yaml}

case "$TARGET" in
  bigfive|ood|wvs|yrbss)
    python -m popalign evaluate "$TARGET" --config "$CONFIG"
    ;;
  *)
    echo "Unknown TARGET=$TARGET. Use bigfive, ood, wvs, or yrbss." >&2
    exit 1
    ;;
esac
