# Workflow reference

Run from the repository root after an editable install. See [reproducibility](reproducibility.md) for what each stage can and cannot reproduce.

## Global alignment reference

The new release reference runs separately from the legacy generation/adaptation CLI:

```bash
python -m popalign.alignment --help
```

See [alignment.md](alignment.md) for a complete synthetic CSV example and camera-ready Eq. 3 top-70% selection and the available OT cost conventions. This is a small-scale CPU implementation, not the historical experiment code; it requires already scored numeric response vectors and cannot replace the missing full data/preprocessing pipeline.

## Configuration

All workflows are launched from YAML configs in `configs/`. A config usually has four sections:

| Section | Purpose |
| --- | --- |
| `model` | Model name, local model path, vLLM port, API base, and optional API key |
| `runtime` | Parallelism, max tokens, temperature, and related execution settings |
| `data` | Survey files, persona files, generated-response paths, or human-reference paths |
| `output` | Output directory or file for the workflow |

PopAlign supports two execution backends:

- Local vLLM: set `model.path`; PopAlign starts and stops a local vLLM OpenAI-compatible server.
- Existing endpoint: omit `model.path` and set `model.api_base` to an OpenAI-compatible endpoint.

Example local vLLM config:

```yaml
model:
  name: qwen2.5-72b-instruct
  path: Qwen/Qwen2.5-72B-Instruct
  port: 8000
  gpu: 2
runtime:
  threads: 16
  max_tokens: 512
  temperature: 0.7
```

Example API config:

```yaml
model:
  name: qwen2.5-72b-instruct
  api_base: http://localhost:8000/v1
  api_key: EMPTY
runtime:
  threads: 16
  max_tokens: 512
  temperature: 0.7
```

## Workflow 1: Big Five Population Alignment

This workflow compares a model's default response distribution with persona-conditioned response distributions on Big Five personality items.

Direct prompting baseline:

```bash
python -m popalign generate bigfive-direct --config configs/bigfive_direct.yaml
```

Persona-conditioned generation:

```bash
python -m popalign generate bigfive-persona --config configs/bigfive_persona.yaml
```

Convenience runner:

```bash
MODE=direct scripts/run_bigfive.sh
MODE=persona scripts/run_bigfive.sh
```

Default outputs:

```text
outputs/bigfive/<model_name>/bigfive_direct_<model_name>_temperature_<temperature>.jsonl
outputs/bigfive/<model_name>/<persona_file>_bigfive_<model_name>.jsonl
```

## Workflow 2: OOD Survey Simulation

This workflow evaluates whether persona-conditioned behavior transfers to out-of-distribution social and psychological surveys. Supply your own CFCS, Duckworth, FBPS or other questionnaire prompts and persona files under `data/local/`; these research inputs are not bundled.

```bash
python -m popalign generate ood-persona --config configs/ood_persona.yaml
```

Convenience runner:

```bash
scripts/run_ood.sh
```

Default output:

```text
outputs/ood/<model_name>/<survey_file>_<persona_file>_<model_name>.jsonl
```

## Workflow 3: Embedding-Based Persona Adaptation

The embedding utilities retrieve from your supplied persona pool (the config defaults to two fictional examples) and adapt it to target prompts or subpopulations. Stages are intentionally separated so expensive retrieval, filtering, generation, and response simulation can be resumed independently.

| Stage | Command | Purpose |
| --- | --- | --- |
| `training-data` | `python -m popalign embeddings training-data --config configs/embedding_training.yaml` | Generate long and short retrieval queries from seed personas |
| `filter` | `python -m popalign embeddings filter --config configs/embedding_filter.yaml` | Filter candidate negatives with an LLM judge |
| `match` | `python -m popalign embeddings match --config configs/embedding_match_wvs.yaml` | Retrieve top-k personas for target WVS/YRBSS prompts |
| `generate-personas` | `python -m popalign embeddings generate-personas --config configs/embedding_generate_personas.yaml` | Generate target-group personas from retrieved seeds |
| `generate-responses` | `python -m popalign embeddings generate-responses --config configs/embedding_generate_responses.yaml` | Generate survey responses with adapted personas |

These stages are separate utilities, **not a complete retriever-training pipeline**. Filtering does not feed the matcher automatically; fine-tuning code and the trained retriever checkpoint are not included. The example response stage requires survey prompts at `data/local/surveys`; no WVS/YRBSS or OOD research questionnaire is bundled.

After supplying all required data, runtime dependencies, and intermediate inputs, run the available stages sequentially:

```bash
STAGE=all scripts/run_embeddings.sh
```

Run one stage:

```bash
STAGE=match scripts/run_embeddings.sh
```

Supported `data.survey_type` values:

```text
wvs
wvs_east
wvs_west
wvs_north_america
yrbss
```

Embedding caches are written under `embedding.cache_dir` and are keyed by model, survey type, and persona input metadata to avoid accidental reuse across experiments.

## Score raw Big Five responses

```bash
python -m popalign.scoring --input path/to/full_responses.jsonl --output outputs/scored/personas.jsonl
```

This new utility requires all 50 numeric items and applies the [documented IPIP key](scoring.md). The three-question API smoke test cannot produce full trait scores. Score human and persona responses with consistent item order, polarity and scale before evaluation; EST means emotional stability.

## Evaluation

Evaluation configs in `configs/eval_*_example.yaml` contain placeholder paths for local human-reference data and generated outputs. Copy or edit these configs before running metrics.

```bash
python -m popalign evaluate bigfive --config configs/eval_bigfive_example.yaml
python -m popalign evaluate ood --config configs/eval_ood_example.yaml
python -m popalign evaluate wvs --config configs/eval_wvs_example.yaml
python -m popalign evaluate yrbss --config configs/eval_yrbss_example.yaml
```

Convenience runner:

```bash
TARGET=bigfive scripts/run_evaluate.sh
TARGET=ood scripts/run_evaluate.sh
TARGET=wvs scripts/run_evaluate.sh
TARGET=yrbss scripts/run_evaluate.sh
```

Required local references:

| Evaluation | Fields to replace |
| --- | --- |
| Big Five | `data.human_path`, optional `data.resampled_jsonl`, optional `data.baselines` |
| OOD | `data.human_dir`, optional `data.baselines` |
| WVS | `data.human_root`, `data.llm_root` |
| YRBSS | `data.human_file`, `data.method_dir` |

Default result directories:

```text
results/bigfive
results/ood
results/wvs/evaluation
results/yrbss/evaluation
```

## Data Format

The [examples directory](../examples/) contains sample personas, a three-question API example, and the full 50-item IPIP inventory. Put your research inputs under `data/local/` and use the following schemas.

Persona row:

```json
{"persona": "Persona description...", "unique_id": "source_id", "source": "source_name"}
```

Big Five survey row:

```json
{"survey_name": "BIG5_1_million", "original_id": "EXT1", "question": "I am the life of the party."}
```

OOD survey row:

```json
{
  "survey_name": "CFCS",
  "original_id": "Q1",
  "prompt_with_persona_system": "... {ROLE_PROFILE} ...",
  "prompt_with_persona_user": "..."
}
```

## Development

```bash
PYTHONPATH=src python -m unittest discover -s tests
PYTHONPATH=src python -m compileall src/popalign tests
find scripts -name '*.sh' -exec bash -n {} \;
PYTHONPATH=src python -m popalign --help
```
