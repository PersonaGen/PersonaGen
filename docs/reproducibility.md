# Reproducibility and scope

This repository provides code and runnable examples for the paper's workflows. The table below maps each component to its implementation and the additional inputs needed for research runs.

## Paper-to-code map

| Paper component | Code or data in this release | Remaining inputs or implementation |
| --- | --- | --- |
| Narrative personas from social-media histories | Fictional [persona input examples](../examples/fictional_personas.jsonl) | Collection, cleaning, de-identification, narrative generation, and quality-assessment pipeline |
| Global population alignment | New small-scale [KDE + entropic OT reference](alignment.md), Big Five evaluators | Original experiment scripts/settings, full input pool, human reference preprocessing, and scalable full-population execution |
| Persona-conditioned simulation | `src/popalign/generation/` | Served LLM and model-specific runtime configuration |
| Task-specific retrieval | `src/popalign/embeddings/matching.py` | Full aligned pool; paper-trained retriever weights |
| Query generation and filtering | `training_data.py`, `filtering.py` | Full negative-pool construction and contrastive fine-tuning/training loop |
| Targeted persona refinement | `persona_generation.py` | LLM endpoint and full retrieved seeds |
| Big Five results | `evaluation/bigfive_metrics.py` | Human reference trait scores, full experiment responses, verified historical experiment preprocessing |
| OOD / WVS / YRBSS results | `evaluation/*_metrics.py` | Licensed human reference files, WVS/YRBSS questionnaires and converted schemas, experiment responses |

The default matcher uses the public Qwen3 embedding model named in the config. It does not automatically train a model from filtered pairs. `scripts/run_embeddings.sh` executes available utilities sequentially; it is not an end-to-end reproduction of the trained retriever. Its response stage requires user-supplied survey prompts; OOD, WVS and YRBSS research questionnaires are not bundled.

The runnable [examples](../examples/) use fictional personas and synthetic response vectors. Research runs require the corresponding persona pool, human references, and experiment configuration. Big Five provides the alignment reference; the other questionnaires and behavioral tasks evaluate transfer.

## New alignment reference

`python -m popalign.alignment --help` exposes a separate CPU reference implementation of camera-ready Section 3.2. It takes already scored numeric CSVs, retains the highest-weight 70% of persona rows (Eq. 3), performs log-domain Sinkhorn OT and draws final outputs with replacement, and saves source indices/IDs, input hashes, and convergence diagnostics. It is newly written release code, not the original experiment script. The default dense-matrix limit is 2,000,000 candidate × human entries. See [the runnable synthetic example](alignment.md) for inputs, commands, equation correspondence, and limitations.

## Input contract

Generation outputs raw item responses (for example `EXT1` through `EXT10`). The Big Five evaluator expects numeric trait columns `EXT`, `AGR`, `CSN`, `EST`, and `OPN`, each on a 1–5 scale. The new [IPIP-50 scorer](scoring.md) provides strict reverse-coding and five trait means for the bundled public-domain question mapping. It does not replace historical respondent filtering, survey weighting, or item-mapping decisions, and it is not validated as the original experiment preprocessing. The example evaluation config uses placeholder human paths and must be edited.

The existing `quantile_warp_csn` option transforms the resampled CSN scores using the evaluation reference. It is disabled by default in this release, including the example config. The reviewer snapshot enabled it by default; set `quantile_warp_csn: true` only for an explicitly reference-calibrated comparison and disclose that choice. It is not applied to the `origin` or baseline rows. The synthetic demo explicitly disables it.

For embedding-stage API authentication, set `OPENAI_API_KEY` in the environment. The generation runners also support `model.api_key`; avoid committing it. For CPU or single-device embedding use, set `embedding.multi_gpu: false` and `embedding.no_flash_attention: true`. The default multi-GPU route requires CUDA; embedding extras do not automatically install FlashAttention.

## Local validation

The release includes smoke tests, numerical comparisons of the sparse histogram metrics against dense reference histograms, generation/resume tests with a mocked chat response, config validation checks, and vLLM process cleanup tests. `examples/offline_demo.py` invokes the actual evaluation runner on synthetic populations. These checks do not validate paper-level scientific reproducibility.

The five-dimensional 40-bin histogram metrics now store occupied bins only. The histogram boundaries and mathematical definitions are preserved; numerical regression tests compare with NumPy's dense histogram implementation. The release also fixes pandas 3 compatibility in result formatting and cleans up failed local server launches.

No LLM requests, model downloads, GPU runs, or human-data evaluations are required by CI. Record exact model revisions, input hashes, seeds, software versions, generation parameters, and preprocessing choices for new experimental runs.

## Resource limits

The legacy Energy Distance estimator uses pairwise distance arrays. Its memory use is quadratic in the sampled population size (the Big Five Energy Distance routine allows up to 50,000 samples per population). Start with small inputs, inspect available memory, and report any sampling changes for large evaluations. The sparse histogram improvement does not remove these separate pairwise costs.

## Camera-ready alignment conventions

IS follows the current camera-ready Eq. 3: rank KDE density ratios and retain the top 70% of distinct input rows, with input-order tie breaking. The retained count is `floor(0.7 * N)`; only the final OT sampling step uses replacement. `--retain-fraction` supports the Top-k ablations. The OT solver retains its explicit legacy cost options and demonstration settings; see [alignment.md](alignment.md) for their relation to the current camera-ready.

## Maximum Mean Discrepancy

Both Big Five and OOD evaluators report **linear-kernel MMD** in the `MMD` output column. With `k(x, y) = xᵀy`, the feature mapping in the paper's general Eq. 13 is `phi(x) = x`. The empirical metric is:

```python
mmd = np.linalg.norm(X.mean(axis=0) - Y.mean(axis=0))
```

The shared implementation is `evaluation/mmd.py:mmd_linear`. It uses every supplied row, with no bandwidth, subsampling, kernel matrix, or extra normalization. Computing the means costs O((n + m) d). Inputs must be nonempty, finite 2D arrays with the same feature dimensions; a single row per population is valid.

Linear-kernel MMD measures differences in population means. Populations with the same mean have zero MMD even if their variances or distribution shapes differ. Its magnitude depends on feature units and is not bounded by sqrt(2). For example, a mean difference of `[2, 1, 0, 0, 0]` gives sqrt(5), approximately 2.2361.

This release replaces the previous RBF estimator with an empirical linear-kernel distance. The implementation follows the paper's general MMD definition with this explicit kernel choice; the change does not establish numerical reproduction of the paper's existing tables. The old internal `mmd_rbf` function and its bandwidth/sampling arguments are replaced by `mmd_linear(X, Y)`.
