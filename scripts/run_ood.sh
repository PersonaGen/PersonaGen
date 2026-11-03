#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}${PROJECT_ROOT}/src"

CONFIG=${CONFIG:-configs/ood_persona.yaml}

python -m popalign generate ood-persona --config "$CONFIG"
