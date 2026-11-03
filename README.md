<div align="center">

# PersonaGen

### Population-Aligned Persona Generation for LLM-based Social Simulation

**NeurIPS 2026**

[![Python](https://img.shields.io/badge/Python-3.10%2B-4B2E83)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/Code-MIT-4B2E83)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2509.10127-B7A57A)](https://arxiv.org/abs/2509.10127)
[Paper](https://arxiv.org/abs/2509.10127) · [Quick start](#quick-start) · [Examples](examples/) · [Workflows](docs/usage.md) · [Reproducibility](docs/reproducibility.md)

</div>

**Better social simulation starts with a representative population.** PersonaGen studies how to construct persona sets that reflect population diversity, then adapt them to the population needed by a simulation.

The paper combines narrative persona construction and quality assessment, global alignment to human Big Five response distributions through importance-based selection and entropic optimal-transport refinement, and retrieval-based adaptation to target populations.

## What is included

| Component | Included |
| :-- | :-- |
| Global alignment reference | New CPU implementation of KDE top-70% selection + entropic OT; [scope and example](docs/alignment.md) |
| Survey simulation | Direct Big Five prompting; persona-conditioned Big Five and OOD responses |
| Persona adaptation | Query-pair generation, negative filtering, embedding retrieval, persona refinement |
| Scoring and evaluation | Strict IPIP-50 item → trait scoring; Big Five and OOD linear-kernel MMD; WVS and YRBSS metrics |
| Examples | [Sample personas, IPIP questions, and a runnable offline demo](examples/) |
| Local validation | An offline metric demo, regression tests, and CPU CI |

The repository provides code and runnable examples. Start with the examples below, then use the [input schemas](docs/usage.md#data-format) to connect your own persona and survey files. The [paper-to-code map](docs/reproducibility.md) describes the implementation and inputs for each stage.

The implementation retains the original `popalign` Python namespace and CLI for compatibility. `personagen` is an equivalent CLI entry point.

## Quick start

```bash
git clone git@github.com:PersonaGen/PersonaGen.git
cd PersonaGen
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
personagen --help
```

If the system Python does not include `venv`/`ensurepip`, use an existing Python environment or `uv venv --python 3.12 .venv` followed by `uv pip install --python .venv/bin/python -e .`.

## Examples

The [examples directory](examples/) contains small inputs and commands for trying the workflows:

| Example | Contents |
| --- | --- |
| [Persona inputs](examples/fictional_personas.jsonl) | Two fictional profiles showing the persona schema |
| [Short questionnaire](examples/bigfive_questions.jsonl) | Three Big Five items for a six-call API example |
| [Full IPIP questionnaire](examples/ipip50_questions.jsonl) | 50 public-domain items for the scoring workflow |
| [Offline evaluation](examples/offline_demo.py) | Synthetic response vectors evaluated locally |

### Try it without a GPU or API key

```bash
python examples/offline_demo.py
```

This runs the Big Five evaluator on small synthetic populations and writes `outputs/demo/baseline_metrics.csv`.

### Explore the global alignment method

```bash
python -m popalign.alignment --help
```

The [alignment walkthrough](docs/alignment.md) supplies a synthetic CSV example for importance-based selection and entropic OT. The CPU reference solver takes scored responses and writes selected source rows, input hashes, and convergence diagnostics. See the walkthrough for its input format and size limits.

### Generate responses using an existing endpoint

```bash
cp configs/bigfive_api_example.yaml configs/my_bigfive.yaml
# Edit model.name and model.api_base to match your endpoint.
export OPENAI_API_KEY="your-key"  # Use the environment, not a committed config.
personagen generate bigfive-persona --config configs/my_bigfive.yaml
```

The small API example uses two explicitly fictional personas and three Big Five items. API generation may incur charges. All output stays under `outputs/`. The full Big Five config uses the two fictional personas and 50 public-domain IPIP items. OOD configs require user-supplied inputs under `data/local/`. The larger configs in `configs/` use local Qwen2.5-72B serving by default; install `pip install -e '.[serving]'` and provision suitable GPUs before using them. Embedding matching additionally requires `pip install -e '.[embeddings]'`.

For complete 50-item outputs, [score responses into trait means](docs/scoring.md) before evaluating:

```bash
python -m popalign.scoring --input path/to/responses.jsonl --output outputs/scored/personas.jsonl
```

[Full workflow commands and schemas →](docs/usage.md)

## Repository layout

```text
src/popalign/       Generation, adaptation, evaluation, and CLI
configs/           YAML examples for individual workflows
examples/          Runnable examples, sample personas, and IPIP questions
docs/              Workflow commands, input formats, and reproducibility
tests/             CPU-only regression and smoke tests
```

## Development

```bash
python -m unittest discover -s tests -v
python -m compileall -q src tests examples
```

CI installs the core package and runs the tests and offline demo. It does not call paid APIs, download model weights, or reproduce GPU experiments. Please include your config (with credentials removed), command, and error when opening an issue. See [CONTRIBUTING.md](CONTRIBUTING.md).

## Citation

If you use PersonaGen in your research, please cite:

```bibtex
@inproceedings{hu2026population,
  title = {Population-Aligned Persona Generation for LLM-based Social Simulation},
  author = {Hu, Zhengyu and Lian, Jianxun and Xiao, Zheyuan and Xiong, Max and
            Wang, Tianfu and Xiao, Teng and Lei, Yuxuan and Jiang, Fengqing and
            Ding, Kaize and Xiao, Ziang and Yuan, Nicholas Jing and Xie, Xing and
            Poovendran, Radha},
  booktitle = {Advances in Neural Information Processing Systems},
  year = {2026}
}
```

Machine-readable metadata is in [CITATION.cff](CITATION.cff).

## License

The original software and code documentation are released under the [MIT License](LICENSE). IPIP questionnaire items are public domain. Third-party datasets, papers, and model weights retain their own terms. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
