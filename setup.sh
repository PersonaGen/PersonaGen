#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip wheel
pip install -e .

printf "\nSetup complete. Activate with: source .venv/bin/activate\n"
