<div align="center">

# PersonaGen

**Population-Aligned Persona Generation for LLM-based Social Simulation**

NeurIPS 2026

[![Paper](https://img.shields.io/badge/arXiv-2509.10127-b31b1b?style=flat-square)](https://arxiv.org/abs/2509.10127v3)
[![Dataset](https://img.shields.io/badge/🤗_Hugging_Face-Dataset-FFD21E?style=flat-square)](https://huggingface.co/datasets/PersonaGen/PersonaGen)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square)](https://www.python.org/)
[![License](https://img.shields.io/badge/Code-MIT-2563EB?style=flat-square)](LICENSE)

[Paper](https://arxiv.org/abs/2509.10127v3) · [Dataset](https://huggingface.co/datasets/PersonaGen/PersonaGen) · [Quick start](#quick-start) · [Documentation](docs/usage.md) · [Release notes](RELEASE_NOTES.md)

</div>

---

PersonaGen constructs persona populations for LLM-based social simulation, aligning persona responses with human reference distributions and adapting them to target groups.

The paper introduces three stages:

1. **Persona construction.** Generate narrative personas from long-term social media histories and assess their quality.
2. **Population alignment.** Combine importance-based selection with optimal transport to align persona responses with human Big Five distributions.
3. **Targeted adaptation.** Retrieve and refine personas for specific subpopulations and simulation contexts.

## Public release

This release provides selected research code, runnable examples, and a persona demo dataset. See the [reproducibility guide](docs/reproducibility.md) for the correspondence between the paper and the released implementation.

| Resource | Description |
| :--- | :--- |
| [Persona dataset](https://huggingface.co/datasets/PersonaGen/PersonaGen) | `demo.jsonl`: 4,990 records with `persona_id`, `persona`, and `id` fields |
| [Global alignment](docs/alignment.md) | Small-scale CPU reference for KDE-based selection and entropic optimal transport |
| [Simulation workflows](docs/usage.md) | Direct and persona-conditioned survey responses, retrieval, and refinement |
| [Scoring and evaluation](docs/scoring.md) | IPIP-50 trait scoring; Big Five, OOD, WVS, and YRBSS evaluation utilities |
| [Runnable examples](examples/) | Fictional persona inputs, questionnaire examples, and an offline evaluation demo |

### Dataset

The released data is hosted on [Hugging Face](https://huggingface.co/datasets/PersonaGen/PersonaGen). Download [demo.jsonl](https://huggingface.co/datasets/PersonaGen/PersonaGen/resolve/main/demo.jsonl) or browse the dataset files there. Research workflows may require additional survey inputs, human references, and model configuration; see the [input schemas](docs/usage.md#data-format).

## Quick start

Requires **Python 3.10 or later**.

```bash
git clone https://github.com/PersonaGen/PersonaGen.git
cd PersonaGen
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
personagen --help
```

The package retains the `popalign` Python namespace for compatibility. The `personagen` and `popalign` commands are equivalent entry points.

### 1. Run the offline demo

No GPU or API key is required.

```bash
python examples/offline_demo.py
```

This evaluates small synthetic populations and writes `outputs/demo/baseline_metrics.csv`.

### 2. Explore population alignment

```bash
python -m popalign.alignment --help
```

Follow the [alignment walkthrough](docs/alignment.md) for a runnable synthetic CSV example, input formats, and solver limits. This is a newly written CPU reference implementation; full experimental reproduction requires additional inputs and settings.

### 3. Generate persona-conditioned responses

```bash
cp configs/bigfive_api_example.yaml configs/my_bigfive.yaml
# Set model.name and model.api_base in your config.
export OPENAI_API_KEY="your-key"
personagen generate bigfive-persona --config configs/my_bigfive.yaml
```

The API example uses two fictional personas and three questionnaire items. API calls may incur charges. Outputs are written under `outputs/`.

For full IPIP-50 responses, convert item responses into trait means before evaluation:

```bash
python -m popalign.scoring \
  --input path/to/responses.jsonl \
  --output outputs/scored/personas.jsonl
```

Local model serving requires `pip install -e '.[serving]'` and suitable GPUs. Embedding retrieval requires `pip install -e '.[embeddings]'`. Larger configs default to local Qwen2.5-72B serving; review their settings and data paths before running.

## Documentation

| Guide | Contents |
| :--- | :--- |
| [Usage](docs/usage.md) | Workflow commands, configuration, and input schemas |
| [Alignment](docs/alignment.md) | Selection and optimal-transport reference implementation |
| [Scoring](docs/scoring.md) | IPIP-50 reverse coding and trait means |
| [Reproducibility](docs/reproducibility.md) | Paper-to-code map, required inputs, and implementation limits |
| [Examples](examples/README.md) | Sample inputs and runnable demonstrations |

## Repository structure

```text
PersonaGen/
├── configs/          Workflow configuration examples
├── docs/             Usage, scoring, alignment, and reproducibility
├── examples/         Sample inputs and offline demo
├── scripts/          Workflow helper scripts
├── src/popalign/     Generation, adaptation, evaluation, and CLI
└── tests/            CPU regression and smoke tests
```

## Development

```bash
python -m unittest discover -s tests -v
python -m compileall -q src tests examples
```

CI runs CPU tests and the offline demo. See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidance.

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

Original code and code documentation are released under the [MIT License](LICENSE). IPIP questionnaire items are public domain. Third-party datasets, papers, and model weights retain their own terms; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
