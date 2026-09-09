# Asymmetric Error JAX Efficiency

Date: 2026-06-24

## Summary

Kept two small JAX usage optimizations that preserve the profile-likelihood
algorithm and endpoint acceptance criteria:

- Average-side nuisance profiling now uses a compiled `jax.value_and_grad`
  callable for BFGS, so each BFGS evaluation returns chi2 and gradient together.
- Fit-side `calc_asym_errors()` now creates the objective gradient and Hessian
  JIT wrappers once per fit and reuses them across target profiles.

No statistical model, interpolation formula, bracketed bisection logic,
endpoint residual tolerance, or constrained-profile acceptance check was
changed.

## Changed Code Paths

- `src/pdgfits/avg.py`
  - `chi2_val_and_grad = jax.jit(jax.value_and_grad(chi2))`
  - average nuisance-profile BFGS now calls `scipy_minimize(..., jac=True)`
    with a callable returning `(value, nuisance_grad)`.

- `src/pdgfits/asym_errors.py`
  - `build_constrained_profile_chi2()` accepts optional prebuilt
    `chi2_grad_jax` and `chi2_hess_jax`.
  - `calc_asym_errors()` reuses those objective derivative callables across
    all selected targets for a fit.

## Benchmark Artifacts

- Average baseline: `notes/asym_jax_efficiency_avg_baseline.csv`
- Average optimized: `notes/asym_jax_efficiency_avg_optimized.csv`
- Single-target fit baseline: `notes/asym_jax_efficiency_fit_baseline.csv`
- Single-target fit optimized: `notes/asym_jax_efficiency_fit_optimized.csv`
- Fit shared-derivative probe: `notes/asym_jax_efficiency_fit_share_probe.csv`
- Logs:
  - `notes/logs/asym_jax_efficiency_baseline_stdout.log`
  - `notes/logs/asym_jax_efficiency_optimized_stdout.log`
  - `notes/logs/asym_jax_efficiency_fit_no_shared_derivatives.log`
  - `notes/logs/asym_jax_efficiency_fit_shared_derivatives.log`
  - `notes/logs/asym_jax_efficiency_run_fits_upsilon.log`

All runs used the snapshot backend with:

```bash
env PYTHONPATH=/root/pdgfits-private/src \
  PDGFITS_DATA_BACKEND=snapshot \
  PDGFITS_SNAPSHOT_DIR=/root/pdgfits-private/data/pdg-snapshot \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  prlimit --as=7800000000 --rss=7800000000 \
  /root/micromamba/micromamba run -r /root/micromamba-root -n pdg <command>
```

## Average Results

Cases:

- `M002W`: simple no-nuisance average.
- `S042CKS`: simple no-nuisance average with high endpoint residual near the
  bisection tolerance.
- `M026R08`: BR-adjusted average with 2 nuisance parameters.
- `S086R46`: BR-adjusted average with 4 nuisance parameters.

Benchmark command:

```bash
python notes/endpoint-search/run_asym_endpoint_search_benchmark.py \
  --variant jax-baseline \
  --fit-csv notes/asym_jax_efficiency_fit_baseline.csv \
  --fit-jsonl notes/asym_jax_efficiency_fit_baseline.jsonl \
  --avg-csv notes/asym_jax_efficiency_avg_baseline.csv \
  --avg-jsonl notes/asym_jax_efficiency_avg_baseline.jsonl \
  --stdout-log notes/logs/asym_jax_efficiency_baseline_stdout.log \
  --fit-target 'Upsilon(2S)::M052.6' \
  --fit-target 'B0::S042B09' \
  --avg-node M026R08 --avg-node M002W --avg-node S042CKS --avg-node S086R46 \
  --target-timeout-sec 300
```

The optimized run used the same command with `--variant jax-optimized` and
the optimized output paths.

| Variant | Rows ok | Total avg runtime s | Max abs fresh residual |
| --- | ---: | ---: | ---: |
| baseline | 4 / 4 | 18.526 | 0.004999656 |
| optimized | 4 / 4 | 17.686 | 0.004999656 |

Per-node runtime:

| Node | Baseline s | Optimized s | Residuals changed? |
| --- | ---: | ---: | --- |
| `M002W` | 2.420 | 2.129 | no |
| `M026R08` | 5.043 | 4.576 | no |
| `S042CKS` | 1.334 | 1.232 | no |
| `S086R46` | 9.729 | 9.748 | no |

Interpretation: the average-side value-and-grad change is a modest win on this
probe and does not affect endpoint residuals. The slowest 4-nuisance node was
runtime-neutral in this run.

## Fit Results

Single-target benchmark cases:

- `B0 / S042B09`: harder BRU fit target using exact-Hessian/KKT paths.
- `Upsilon(2S) / M052.6`: small BRU fit target.

| Variant | Rows ok | Total target runtime s | Max abs residual |
| --- | ---: | ---: | ---: |
| baseline | 2 / 2 | 12.594 | 0.002115812 |
| optimized | 2 / 2 | 11.822 | 0.002115812 |

This single-target benchmark is only a smoke check for the fit-side change,
because derivative reuse matters most when `calc_asym_errors()` evaluates
multiple targets for one already-built fit.

Explicit multi-target derivative-reuse probe:

- Fit: `Upsilon(2S)`
- Targets: `M052.4`, `M052.6`, `M052R22`, `M052R4`, `M052R6`,
  `nuisance_M048.8`
- Endpoint function evaluations: 45 in both variants.

| Variant | Asym runtime s | Endpoint function evals | Max abs residual |
| --- | ---: | ---: | ---: |
| no shared objective derivatives | 8.836 | 45 | 0.003450197 |
| shared objective derivatives | 6.881 | 45 | 0.003450197 |

Interpretation: sharing the objective derivative JIT wrappers saved about
1.96 s, a 22% reduction in asymmetric-error runtime for this six-target fit,
with the same endpoint calls and residuals.

## Rejected JAX Variants

- `build_chi2(..., use_jit=False)` for averages: rejected. A monkeypatch probe
  on the same four average nodes slowed total runtime from 18.283 s to
  142.495 s. The nuisance-heavy `S086R46` case alone went from 9.897 s to
  106.718 s. Endpoint residuals were unchanged, but performance was much worse.

- Forcing average `get_mu_vectorized(..., jit=True)` / node funcs JIT: not
  kept. The same probe moved total runtime from 18.283 s to 17.704 s with
  unchanged residuals, but this only JITs fallback/node callables and the gain
  was small enough to treat as noise without broader evidence.

- Combining `chi2_np` and `chi2_grad_np` inside the fit constrained solver:
  not kept. SciPy's SLSQP/trust-constr callback structure calls objective,
  constraint, Jacobian, and Hessian through separate APIs. Changing that path
  would touch the high-risk constrained-profile solver for unclear gain.

## Validation

Targeted tests:

```bash
python -m pytest tests/test_asym_errors.py tests/test_build_chi2.py tests/test_build_funcs.py -q
```

Result:

```text
21 passed, 1 skipped in 6.97s
```

Full non-DB test suite:

```bash
python -m pytest tests/ -q
```

Result:

```text
160 passed, 2 skipped in 13.86s
```

Representative fit CLI validation:

```bash
python -m pdgfits.run_fits --fit_label 'Upsilon(2S)' --calc_asym_errors
```

Result:

- Completed successfully.
- Verified 12 printed endpoint residuals.
- Max absolute printed residual: `0.0035`, within the existing endpoint
  tolerance.

## Caveats

- This was a focused timing pass, not a broad full-snapshot sweep.
- The average speedup is modest and may be noisy on no-nuisance nodes.
- The fit-side speedup applies most directly to `calc_asym_errors()` calls that
  evaluate multiple targets for the same fit. Single-target harnesses should
  not be expected to show the full benefit.
