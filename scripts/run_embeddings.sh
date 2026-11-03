#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}${PROJECT_ROOT}/src"

STAGE=${STAGE:-all}

run_stage() {
  case "$1" in
    training-data)
      python -m popalign embeddings training-data --config "${TRAINING_CONFIG:-configs/embedding_training.yaml}"
      ;;
    filter)
      python -m popalign embeddings filter --config "${FILTER_CONFIG:-configs/embedding_filter.yaml}"
      ;;
    match)
      python -m popalign embeddings match --config "${MATCH_CONFIG:-configs/embedding_match_wvs.yaml}"
      ;;
    generate-personas)
      python -m popalign embeddings generate-personas --config "${PERSONA_CONFIG:-configs/embedding_generate_personas.yaml}"
      ;;
    generate-responses)
      python -m popalign embeddings generate-responses --config "${RESPONSE_CONFIG:-configs/embedding_generate_responses.yaml}"
      ;;
    *)
      echo "Unknown embedding STAGE=$1" >&2
      exit 1
      ;;
  esac
}

if [[ "$STAGE" == "all" ]]; then
  run_stage training-data
  run_stage filter
  run_stage match
  run_stage generate-personas
  run_stage generate-responses
else
  run_stage "$STAGE"
fi
