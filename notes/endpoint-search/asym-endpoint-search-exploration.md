# Asymmetric Endpoint Search Exploration

Date: 2026-06-24

Branch: `codex-cloud-asym-avg-local-experiments`

Baseline commit noted by the user: `c95642a docs: validate simplified fit profile sweep`.

## Summary

I kept one small endpoint-search change:

- `binary_search_error()` now keeps a per-search cache for successful profile
  evaluations and records endpoint-search evaluation counters.
- The final endpoint verification reuses the profile value just computed at
  the final bisection point instead of solving the same profile point a second
  time. The scientific check is unchanged because the returned endpoint is
  exactly that final bisection coordinate.

The constrained profile solver was not changed.

On the focused hard set, this saved exactly two profile calls per target/node
with no row failures and no meaningful residual movement.

## Benchmark Harness

Added notes-only harness:

- `notes/endpoint-search/run_asym_endpoint_search_benchmark.py`

Machine-readable outputs from this pass:

- `notes/endpoint-search/asym_endpoint_search_fit_baseline.csv`
- `notes/endpoint-search/asym_endpoint_search_fit_baseline.jsonl`
- `notes/endpoint-search/asym_endpoint_search_avg_baseline.csv`
- `notes/endpoint-search/asym_endpoint_search_avg_baseline.jsonl`
- `notes/endpoint-search/asym_endpoint_search_fit_cache_reuse.csv`
- `notes/endpoint-search/asym_endpoint_search_fit_cache_reuse.jsonl`
- `notes/endpoint-search/asym_endpoint_search_avg_cache_reuse.csv`
- `notes/endpoint-search/asym_endpoint_search_avg_cache_reuse.jsonl`
- `notes/logs/asym_endpoint_search_baseline_stdout.log`
- `notes/logs/asym_endpoint_search_cache_reuse_stdout.log`

The logs are small, about 12 KiB each.

## Focused Hard Set

Fit targets:

- `chi_c012 psi(2S)`: `M056.6`, `M055B1`, `M056.11`, `M055B11`, `M055B10`
- `Lam-b-0`: `S040.29`, `S040R29`, `S040.10`, `S040R10`
- Previous failure clusters represented by all staged rows for `B0`,
  `B0S-BR`, and `eta_c J/psi psi(2S)`
- `eta_c(2S) / M059.4`
- `K_3^*(1780) / M060.6`

Average targets:

- `M026R08`
- Top slowest, highest-residual, and nuisance-heavy nodes selected from
  `notes/asym_avg_sweep_results.csv`: `M002W`, `M056R50`, `M057B18`,
  `M070R84`, `S032B94`, `S041B46`, `S042CKS`, `S042R2`, `S051R05`,
  `S051R06`, `S086R46`

## Results

Residuals are `profile_chi2(endpoint) - (chi2_min + 1)` after profiling.
For averages, the table uses the fresh fixed-coordinate verification residuals.

| Scope | Variant | Rows ok | Failures | Runtime sum s | Profile calls | Max abs residual | Notes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| fits | baseline current code | 37 | 0 | 1257.900 | 627 | 0.0049359 | Current bisection with duplicate endpoint re-evaluation |
| fits | cache + endpoint reuse | 37 | 0 | 1125.275 | 553 | 0.0049359 | Exactly -2 profile calls per row |
| averages | baseline current code | 12 | 0 | 72.896 | 322 | 0.0049997 | Current bisection with duplicate endpoint re-evaluation |
| averages | cache + endpoint reuse | 12 | 0 | 71.007 | 298 | 0.0049997 | Exactly -2 profile calls per node |

The patched diagnostics reported:

| Scope | Endpoint function evals | Cache hits | Bracket evals | Bisection evals |
| --- | ---: | ---: | ---: | ---: |
| fits | 516 | 0 | 76 | 440 |
| averages | 286 | 0 | 24 | 262 |

Interpretation:

- The per-search cache did not hit on this hard set. Plain bisection rarely
  asks for the same coordinate twice.
- The saved calls came from reusing the final bisection evaluation for endpoint
  verification.
- Timing improved in this run, especially on slow fit rows, but runtime is
  noisy. The hard evidence is the deterministic two-call reduction per
  endpoint pair and unchanged verification residuals.

Slowest patched fit rows:

| Label | Target | Baseline calls | Patched calls | Patched runtime s | Residuals |
| --- | --- | ---: | ---: | ---: | --- |
| `chi_c012 psi(2S)` | `M056.6` | 21 | 19 | 303.718 | `(-0.000500, 0.001096)` |
| `chi_c012 psi(2S)` | `M055B1` | 21 | 19 | 173.831 | `(0.001973, 0.000897)` |
| `chi_c012 psi(2S)` | `M056.11` | 23 | 21 | 142.749 | `(-0.001550, 0.001785)` |
| `Lam-b-0` | `S040.29` | 22 | 20 | 78.594 | `(0.000568, -0.004936)` |
| `Lam-b-0` | `S040R29` | 22 | 20 | 78.000 | `(0.000568, -0.004936)` |

## Variants Tried Or Considered

| Variant | Kept? | Evidence / reason |
| --- | --- | --- |
| Successful-value cache inside `binary_search_error()` | Kept, but not credited for speed | Simple bookkeeping; zero cache hits on the focused hard set, so it is not the measured runtime driver. |
| Reuse final bisection value for endpoint verification | Kept | Saves exactly two profile calls per target/node; all hard fit and average rows passed; residuals unchanged to benchmark precision except tiny optimizer-noise differences up to `4.7e-6` on fits. |
| Brent-style `brentq` replacement | Not kept | I did not replace bisection. A `brentq` wrapper would require an x-tolerance in addition to the scientific residual tolerance and an extra verification layer. The current bisection stops directly on `abs(profile_chi2 - target) <= residual_tol`, and the simple reuse change already removes the obvious duplicate solves. |
| Bracket expansion/contraction changes | Not kept | No bracket failures appeared in the hard benchmark. Existing contraction behavior stayed untouched. |
| Warm-start/profile solver changes | Not kept | The task explicitly asked not to complicate the profile solver unless necessary; no evidence from this pass required touching it. |

## Commands

Common environment:

```bash
env PYTHONPATH=/root/pdgfits-private/src \
  PDGFITS_DATA_BACKEND=snapshot \
  PDGFITS_SNAPSHOT_DIR=/root/pdgfits-private/data/pdg-snapshot \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  prlimit --as=7800000000 --rss=7800000000 \
  /root/micromamba/micromamba run -r /root/micromamba-root -n pdg <command>
```

Baseline benchmark:

```bash
python notes/endpoint-search/run_asym_endpoint_search_benchmark.py \
  --variant baseline-current \
  --fit-csv notes/endpoint-search/asym_endpoint_search_fit_baseline.csv \
  --fit-jsonl notes/endpoint-search/asym_endpoint_search_fit_baseline.jsonl \
  --avg-csv notes/endpoint-search/asym_endpoint_search_avg_baseline.csv \
  --avg-jsonl notes/endpoint-search/asym_endpoint_search_avg_baseline.jsonl \
  --stdout-log notes/logs/asym_endpoint_search_baseline_stdout.log \
  --target-timeout-sec 900
```

Patched benchmark:

```bash
python notes/endpoint-search/run_asym_endpoint_search_benchmark.py \
  --variant cache-reuse \
  --fit-csv notes/endpoint-search/asym_endpoint_search_fit_cache_reuse.csv \
  --fit-jsonl notes/endpoint-search/asym_endpoint_search_fit_cache_reuse.jsonl \
  --avg-csv notes/endpoint-search/asym_endpoint_search_avg_cache_reuse.csv \
  --avg-jsonl notes/endpoint-search/asym_endpoint_search_avg_cache_reuse.jsonl \
  --stdout-log notes/logs/asym_endpoint_search_cache_reuse_stdout.log \
  --target-timeout-sec 900
```

Validation:

```bash
python -m pytest tests/test_asym_errors.py -q
python -m pytest tests/ -q
python -m pdgfits.run_fits --fit_label 'Upsilon(2S)' --calc_asym_errors
```

Validation results:

- `tests/test_asym_errors.py`: `4 passed in 4.26s`
- `tests/`: `160 passed, 2 skipped in 15.09s`
- `Upsilon(2S)`: completed; six printed endpoints verified, max printed
  residual about `0.0035`

The focused hard benchmark itself reran the requested `Lam-b-0` and
`eta_c J/psi psi(2S)` rows after the endpoint-search change.

## Scientific Caveats

- This is not a new broad sweep. It is a focused hard-set endpoint-search
  benchmark.
- The endpoint acceptance invariant is unchanged: endpoints are accepted only
  through `profile_chi2(endpoint) = chi2_min + 1` within tolerance after
  profiling.
- The final verification is no longer an independent duplicate optimizer call
  when the endpoint is exactly the just-evaluated bisection point. That is
  intentional; it avoids redundant work while preserving the same mathematical
  check.
- The measured runtime improvement should not be overinterpreted because JAX
  and optimizer runtimes vary. The robust finding is the exact profile-call
  reduction with unchanged residuals.
