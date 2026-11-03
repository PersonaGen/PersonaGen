# Small-scale alignment reference implementation

**This is a new reference implementation for the public release, not the original
script used to produce the paper's results.** IS follows the camera-ready
paper's Section 3.2, Eq. 3: retain the highest-weight 70% of input personas.
OT cost conventions are described below; the remaining small defaults are
demonstration settings. Running it does not reproduce the paper's reported tables
without the corresponding inputs, simulator outputs, and experiment settings.

The implementation runs on CPU with the repository's core NumPy, SciPy, and
scikit-learn dependencies. It makes no network calls and requires no model/API
credentials. Its entry point is separate from the legacy pipeline CLI:

```bash
python -m popalign.alignment --help
```

## Input contract

Provide two UTF-8 CSV files with a header and the **same ordered numeric features
on the same scale**: one row per candidate persona response vector, and one row
per human response vector. These are already scored responses, not narrative
persona text. Missing, nonnumeric, or non-finite features are rejected.

For an item-level reproduction, use the item response vectors described in the
paper. Pre-aggregated trait scores are also accepted, but change the alignment
space and must be documented as a separate setting. The implementation does not
reverse-code, average, normalize, standardize, or impute responses.

An optional `--id-column` is excluded from features and must exist in both inputs.
Without it, source IDs default to zero-based input data-row numbers. Duplicate
IDs are allowed: the row indices remain the unambiguous provenance keys.
`--columns` specifies a common feature order explicitly and allows unrelated
columns to be ignored. Otherwise all non-ID column names and their order must
match exactly.

Minimal synthetic example (for checking the software, not an experiment):

```bash
mkdir -p demo
cat > demo/persona.csv <<'CSV'
id,trait_a,trait_b
p0,1,2
p1,2,1
p2,2,2
p3,3,2
CSV
cat > demo/human.csv <<'CSV'
id,trait_a,trait_b
h0,1,1
h1,2,2
h2,2,3
CSV
python -m popalign.alignment \
  --persona-csv demo/persona.csv --human-csv demo/human.csv \
  --id-column id --columns trait_a trait_b \
  --retain-fraction 0.7 --n-output 16 --seed 7 --epsilon 0.5 \
  --output demo/aligned
```

The output directory must not already exist. Each run writes:

- `importance.csv`: all original persona row indices, IDs, and IS probabilities.
- `candidates.csv`: each retained row's candidate index, original persona row/ID,
  IS probability, mass-weighted transport cost, and OT probability.
- `selection.csv`: each final draw, its source persona row/ID and candidate
  index, and the candidate's OT probability. Repeated rows are intentional.
- `metadata.json`: input SHA-256 hashes, paths, feature order, effective weights,
  all sampler parameters, RNG seed, convergence diagnostics, and software version.
- With `--save-plan`, `transport.npz`: dense `cost` and `plan` arrays. Plan rows
  follow `candidates.csv`; columns follow original human data-row order.

All indices are zero-based, excluding the CSV header. Source files are not
modified, and outputs do not silently deduplicate repeated draws. Keep the exact
input files alongside their hashes when archiving a run.

## Equation correspondence

1. **Eq. 2:** fit Gaussian KDEs to human and persona vectors using bandwidth
   `--bandwidth`. Evaluate their log-density ratio at every persona vector.
2. **Eq. 3:** rank the human/persona KDE density ratios in descending order
   and retain `floor(retain_fraction * N)` distinct input rows. The default
   `--retain-fraction 0.7` keeps the top 70%. Ranking uses log ratios to avoid
   underflow; exact ties preserve input row order. IS selection is independent
   of the random seed. A fraction in `(0, 1]` supports Top-k ablations.
   If rounding leaves zero candidates, the command fails with an input error.
   `--retain-fraction` replaces the previous `--n-candidates` argument.
3. **Eq. 4:** build the dense weighted squared Euclidean cost matrix. Optional
   `--feature-weights` supplies one positive weight per feature; defaults are one.
4. **Eq. 5:** solve entropic OT with uniform candidate and human marginals,
   using log-domain Sinkhorn updates and `--epsilon`.
5. Normalize `exp(-c_i / temperature)` and draw `--n-output` personas with
   replacement. This final OT draw can repeat input rows.

### OT cost conventions

The existing `--transport-cost-convention` options are retained. The default
`main-text` is a legacy option computing `c_i = sum_j Gamma_ij * C_ij`.
The `appendix` option divides by `a_i = 1 / n_candidates`, matching the expected
cost in the current camera-ready Eq. 6 and Appendix O Eq. 28. The option names
come from the earlier manuscript; the current manuscript uses the conditional
cost in both places. Metadata records the selected convention.

The current paper's Appendix A.1 specifies KDE bandwidth 0.20, median-cost-scaled
OT regularization 0.08, 250 Sinkhorn iterations and batched execution. The CPU
reference still uses demonstration defaults and a dense OT solver; changing IS
to top-70% selection does not reproduce those full experimental settings.

## Resource and numerical limits

This implementation is intentionally a dense reference, suitable for small
inputs. The default limit is **2,000,000 candidate × human cost entries**. Several
float64 matrices and temporary arrays coexist, so budget substantially more than
8 bytes per entry. Exceeding the limit raises an error **before KDE or dense OT**;
it does not automatically subsample humans or approximate transport. Increase
`--max-cost-entries` only with a deliberate memory budget. Exact KDE can also be
slow on large persona/reference pools. This code is not a scalable implementation
for the full million-respondent paper reference.

The default maximum absolute marginal error is `1e-9`, checked during Sinkhorn
iteration; the default iteration limit is 5,000. Failure to converge raises an
error. Very small epsilon or poorly scaled features may require more iterations
or a different explicitly recorded regularization choice. Extreme magnitudes
that overflow costs or log weights are rejected.

To verify the implementation locally:

```bash
python -m unittest discover -s tests -p 'test_alignment.py' -v
```

The tests compare against an analytic two-point OT solution and explicit Gaussian
KDE ratios, check weighted costs and the explicit OT cost conventions, verify uniform
marginals, top-fraction ranking, deterministic ties and final replacement draws, and exercise input/memory failures
and CSV provenance. They are software checks, not empirical validation of the
paper's research claims.
