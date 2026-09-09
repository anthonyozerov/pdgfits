# Asymmetric Error Efficiency Round 2

Date: 2026-06-24

## Summary

Kept three small changes:

- Average-side nuisance profiling now runs Nelder-Mead only when BFGS did not
  already produce a successful or existing-gradient-certified profile point.
- Fit-side `calc_asym_errors()` no longer solves a constrained profile at the
  MLE target value. The unconstrained optimum already satisfies that target, so
  it now checks the direct MLE target/chi2 consistency and leaves endpoint
  correctness to the existing profiled endpoint residual verification.
- Fit-side constrained-profile acceptance checks cache the boolean decision on
  each immutable `OptimizeResult` candidate, avoiding repeated projected
  gradient/descent checks for the same candidate.

No endpoint tolerance, asymmetric interpolation formula, bracketed bisection
logic, fit fallback stack, or acceptance criterion was loosened.

## Changed Code Paths

- `src/pdgfits/avg.py`
  - Inside fixed-primary average nuisance profiling, BFGS remains first.
  - Nelder-Mead remains available, but is skipped when BFGS succeeds or when
    its nuisance-gradient norm satisfies the existing certificate:
    `norm <= 1e-5 * max(abs(fun), 1.0)`.

- `src/pdgfits/asym_errors.py`
  - `calc_asym_errors()` replaces the base constrained solve
    `profile_chi2(target_value)` with direct checks at `fitted_values`.
  - `build_constrained_profile_chi2().result_ok()` stores
    `profile_result_ok` on the candidate result after the first certification.

## Benchmark Artifacts

- `notes/asym_efficiency_round2_avg_baseline.csv`
- `notes/asym_efficiency_round2_avg_bfgs_gated.csv`
- `notes/asym_efficiency_round2_avg_bfgs_central_profile.csv`
- `notes/asym_efficiency_round2_avg_focused_final.csv`
- `notes/asym_efficiency_round2_fit_baseline.csv`
- `notes/asym_efficiency_round2_fit_cache.csv`
- Matching JSONL files with the same stems.
- Logs under `notes/logs/asym_efficiency_round2_*`.

All commands used the snapshot backend with resource caps:

```bash
env PYTHONPATH=/root/pdgfits-private/src \
  PDGFITS_DATA_BACKEND=snapshot \
  PDGFITS_SNAPSHOT_DIR=/root/pdgfits-private/data/pdg-snapshot \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  prlimit --as=7800000000 --rss=7800000000 \
  /root/micromamba/micromamba run -r /root/micromamba-root -n pdg <command>
```

## Average Results

Focused four-node apples-to-apples benchmark:

```bash
python notes/endpoint-search/run_asym_endpoint_search_benchmark.py \
  --fit-target 'Upsilon(2S)::M052.4' \
  --fit-target 'Upsilon(2S)::M052.6' \
  --fit-target 'Upsilon(2S)::M052R22' \
  --fit-target 'Upsilon(2S)::M052R4' \
  --fit-target 'Upsilon(2S)::M052R6' \
  --fit-target 'Upsilon(2S)::nuisance_M048.8' \
  --avg-node M002W --avg-node S042CKS --avg-node M026R08 --avg-node S086R46 \
  --target-timeout-sec 300
```

| Variant | Avg rows ok | Total avg runtime s | Max abs fresh residual |
| --- | ---: | ---: | ---: |
| Current uncommitted baseline | 4 / 4 | 17.743 | 0.004999656 |
| BFGS-gated nuisance profile | 4 / 4 | 12.468 | 0.004999656 |

Per-node:

| Node | Baseline s | Gated s | Endpoint values/residuals |
| --- | ---: | ---: | --- |
| `M002W` | 2.147 | 2.089 | unchanged |
| `M026R08` | 4.704 | 3.548 | unchanged |
| `S042CKS` | 1.411 | 1.258 | unchanged |
| `S086R46` | 9.481 | 5.572 | unchanged |

The final 12-node focused average validation passed `12/12` rows, total runtime
`33.654 s`, max abs fresh residual `0.004999656`.

Nodes:
`M002W`, `M026R08`, `M056R50`, `M057B18`, `M070R84`, `S032B94`,
`S041B46`, `S042CKS`, `S042R2`, `S051R05`, `S051R06`, `S086R46`.

## Fit Results

For `calc_asym_errors()` on `Upsilon(2S)` six targets, replacing the base
constrained MLE solve with direct MLE checks changed no errors or endpoint
residuals and reduced the measured asymmetric-error phase:

| Variant | Runtime s | Max abs residual |
| --- | ---: | ---: |
| Baseline base constrained solve | 6.926 | 0.003450197 |
| Direct MLE base check | 6.647 | 0.003450197 |

The focused fit harness after the result-certification cache passed
`7/7` target rows:

- `B0::S042B09`
- all six `Upsilon(2S)` targets from the average benchmark command above

Max abs fit residual was `0.003450197`. `B0::S042B09` remained consistent with
prior optimized artifacts: `8.809 s` here versus `8.804 s` in
`notes/asym_jax_efficiency_fit_optimized.csv`, so the cache should be treated
as a small cleanup rather than a proven hard-case speedup.

## Rejected Ideas

### Average Central BFGS First

Tested BFGS for the main average MLE with Nelder-Mead fallback. It passed
endpoint residual checks, but it changed central values and chi2 at small but
real levels on simple nodes and was not consistently faster.

Examples versus the current uncommitted baseline:

| Node | Delta value | Delta chi2 | Runtime s |
| --- | ---: | ---: | ---: |
| `M002W` | `8.23e-05` | `-1.48e-05` | 2.321 vs 2.147 baseline |
| `S042CKS` | `-6.42e-04` | `-5.21e-06` | 1.369 vs 1.411 baseline |
| `M026R08` | `-3.55e-05` | `-1.17e-05` | 3.602 vs 4.704 baseline |

Rejected because central average reproducibility is more important than this
uncertain speedup.

### Endpoint Root Search Alternatives

Not changed. Previous endpoint-search evidence already showed the obvious win
was avoiding duplicate final profile solves. A Brent/root-scalar variant would
add another stopping tolerance and verification layer, while current bisection
stops directly on the scientific residual criterion.

### Direct-Coordinate Fit Profiling

Not shipped. Some fit targets equal direct parameters/nodes, but BR/BRU mapped
fit space and nonlinear node functions make a safe special case less obvious
than the average path. The current generic constrained fit profiler has broad
hard-case evidence; replacing it needs a dedicated benchmark, not introspection
logic.

## Validation

Targeted tests:

```bash
python -m pytest tests/test_asym_errors.py tests/test_build_chi2.py tests/test_build_funcs.py -q
```

Result:

```text
21 passed, 1 skipped in 6.76s
```

Full non-DB suite:

```bash
python -m pytest tests/ -q
```

Result:

```text
160 passed, 2 skipped in 13.57s
```

Representative endpoint validations:

- Four-node average benchmark: `4/4 ok`, max abs fresh residual
  `0.004999656`.
- Twelve-node focused average benchmark: `12/12 ok`, max abs fresh residual
  `0.004999656`.
- Focused fit benchmark: `7/7 ok`, max abs residual `0.003450197`.

## Caveats

- This is a focused timing pass, not a full snapshot rerun.
- The average-side speedup is strongest for nuisance-heavy nodes; no-nuisance
  nodes mostly show noise-level changes.
- The fit-side result cache is intentionally small. It removes redundant
  certification work but did not produce a clear hard-case runtime gain on
  `B0::S042B09`.
