# From IPIP-50 responses to trait scores

The generation commands return raw answers under `EXT1`…`OPN10`; the Big Five evaluator expects five numeric trait columns. This **new release utility** provides that bridge with an explicit key. It is not claimed to reconstruct the paper's unavailable historical cleaning/preprocessing script.

```bash
python -m popalign.scoring \
  --input outputs/bigfive/model/personas_bigfive_model.jsonl \
  --output outputs/scored/personas.jsonl
```

All 50 answers must be present and integer-valued from 1 through 5. Numeric strings such as `"4"` are accepted; explanations such as `"Answer: 4"`, missing items, boolean values, fractions, and out-of-range values are rejected with their line/item. The utility never imputes, extracts a convenient digit from prose, or silently filters records. A bad later row leaves no partial output file. Existing output files are not overwritten.

The three-question API smoke example is intentionally insufficient for this command. Use `examples/ipip50_questions.jsonl` for the full inventory, then score the complete generated responses. The two fictional personas are software examples, not meaningful population estimates.

## Scoring convention

For each trait, reverse the negatively keyed responses as `6 - answer`, then take the mean of its ten items. Output scores stay on the 1–5 scale. The polarity follows the [official IPIP factor-marker key](https://ipip.ori.org/newBigFive5broadKey.htm), with identifiers mapped to the questions in `examples/ipip50_questions.jsonl`.

| Output | Meaning | Reverse-keyed item numbers |
| --- | --- | --- |
| EXT | Extraversion | 2, 4, 6, 8, 10 |
| AGR | Agreeableness | 1, 3, 5, 7 |
| CSN | Conscientiousness | 2, 4, 6, 8 |
| EST | **Emotional stability**, higher means more stable | 1, 3, 5, 6, 7, 8, 9, 10 |
| OPN | Intellect / imagination, labeled OPN by this code | 2, 4, 6 |

EST is not a neuroticism score: for instance, “get stressed out easily” is reversed while “relaxed most of the time” is positive. Use the same direction and key for synthetic and human references. Other inventories/item orders require their own verified mapping; renaming columns alone is insufficient.

The output preserves available ID fields (`persona_unique_id`, `persona_id`, `unique_id`, `id`, `iteration_id`) and writes only those IDs plus the five scores. A `.metadata.json` sidecar records the input SHA-256, scoring key, direction, row count, and invalid-response policy. It does not perform respondent quality filtering, survey weighting, demographic balancing, or human-data access checks.

## CSV/TSV inputs and alignment

To read a provider's tabular file, first confirm that its item identifiers and wording match this exact key. Do not treat placeholder zeros or skipped questions as valid responses.

```bash
python -m popalign.scoring \
  --input data/local/reference.tsv --input-format tsv \
  --output outputs/scored/reference.csv --output-format csv
```

CSV output includes ID columns, so pass explicit traits when using the alignment reference:

```bash
python -m popalign.alignment \
  --persona-csv outputs/scored/personas.csv \
  --human-csv outputs/scored/reference.csv \
  --columns EXT AGR CSN EST OPN \
  --retain-fraction 0.7 --n-output 128 \
  --output outputs/alignment-demo
```

This is **trait-space** alignment. Main-text equations also describe prompt/item vectors; averaging into five traits changes the space and must be reported. To align item vectors, use a correctly keyed/scaled item matrix directly rather than the five-trait scorer. Dense OT is bounded to small inputs by default; see [alignment.md](alignment.md).
