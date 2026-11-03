# Examples

These examples show the input formats and how to run the workflows. The persona profiles and generated response vectors are fictional or synthetic; the questionnaire items come from IPIP.

| File | Contents | Use |
| --- | --- | --- |
| [fictional_personas.jsonl](fictional_personas.jsonl) | Two fictional persona profiles | Persona-conditioned generation and retrieval inputs |
| [bigfive_questions.jsonl](bigfive_questions.jsonl) | Three Big Five questions | Small API example: 2 personas × 3 questions |
| [ipip50_questions.jsonl](ipip50_questions.jsonl) | All 50 IPIP factor-marker items | Complete questionnaire generation and trait scoring |
| [offline_demo.py](offline_demo.py) | Generates two synthetic response populations | Run the evaluator without a GPU or API |

Run commands from the repository root after following the [installation instructions](../README.md#quick-start).

## Offline evaluation

```bash
python examples/offline_demo.py
```

The demo generates 64 five-trait response vectors for each of two synthetic populations and compares them with the Big Five evaluator. Outputs are saved under `outputs/demo/`:

- `synthetic_reference.jsonl`
- `synthetic_shifted.jsonl`
- `baseline_metrics.csv`
- `baseline_metrics.txt`

These numbers demonstrate the software workflow; they are not the paper's experimental results.

## Persona-conditioned responses

Use the [small API config](../configs/bigfive_api_example.yaml) with an existing OpenAI-compatible endpoint:

```bash
cp configs/bigfive_api_example.yaml configs/my_bigfive.yaml
# Set model.name and model.api_base in configs/my_bigfive.yaml.
export OPENAI_API_KEY="your-key"
personagen generate bigfive-persona --config configs/my_bigfive.yaml
```

This makes six requests and writes responses under `outputs/api_example/`. Use `examples/ipip50_questions.jsonl` as `data.question_file` for a complete 50-item run before [scoring trait means](../docs/scoring.md).

## Population alignment

Follow the [CPU alignment example](../docs/alignment.md) to create synthetic response CSVs, run importance sampling and optimal transport, and inspect the selected rows.

## Use your own inputs

Place local research inputs under `data/local/` and set their paths in the workflow config. That directory is ignored by Git. See the [input schemas](../docs/usage.md#data-format) and [evaluation configuration](../docs/usage.md#evaluation).

IPIP items are public domain; source links and notices are in [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).
