# PersonaGen — Public Research Release

Selected code and demo data accompanying **Population-Aligned Persona Generation for LLM-based Social Simulation** (NeurIPS 2026).

- **Paper:** https://arxiv.org/abs/2509.10127v3
- **Dataset:** https://huggingface.co/datasets/PersonaGen/PersonaGen

## Included

- Persona demo data: `demo.jsonl`, containing 4,990 records with `persona_id`, `persona`, and `id` fields.
- Persona-conditioned survey generation and adaptation utilities.
- A small-scale CPU reference implementation of importance-based selection and entropic optimal transport.
- IPIP-50 scoring and population-level evaluation utilities.
- Runnable examples, workflow documentation, and CPU tests.

## Availability

Due to company policy, we can currently release only the code and examples in this repository and the demo dataset on Hugging Face. The full internal pipeline and remaining research artifacts are not included.

This is a partial research release. The [reproducibility guide](docs/reproducibility.md) documents the available implementations and additional inputs needed for research runs; the released examples do not establish full reproduction of the paper's results.
