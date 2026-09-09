# Asym Refactor Round 08 MINOS Covariance Starts

Date: 2026-06-25

Production code changed: no. `src/pdgfits/build_chi2.py` was not edited.

## Question

Round 07 showed that `iminuit`/MINOS is a useful oracle only when the profiled
quantity is a direct coordinate. This round tested the most plausible MINOS
idea to borrow for general pdgfits targets: covariance/error-matrix predicted
conditional starts for fixed-target profile minimization.

The local predictor used in the notes-only harness is the quadratic constrained
minimum implied by the MLE covariance:

```text
dx = C grad(t) (v - t0) / (grad(t)^T C grad(t))
```

where `C` is the fitted-coordinate covariance, `t` is the scalar target, and
`v` is the fixed target value. For direct-coordinate averages, this reduces to
the usual conditional covariance formula for nuisance coordinates.

## Artifacts

- Harness: `notes/codex-refactor/run_round08_minos_covariance_starts.py`
- Results CSV: `notes/codex-refactor/round08_minos_covariance_starts_results.csv`
- Results JSONL: `notes/codex-refactor/round08_minos_covariance_starts_results.jsonl`
- Profile/candidate evaluations CSV: `notes/codex-refactor/round08_minos_covariance_starts_evaluations.csv`
- Profile/candidate evaluations JSONL: `notes/codex-refactor/round08_minos_covariance_starts_evaluations.jsonl`
- Failures CSV: `notes/codex-refactor/round08_minos_covariance_starts_failures.csv`
- Failures JSONL: `notes/codex-refactor/round08_minos_covariance_starts_failures.jsonl`
- Raw stdout: `notes/logs/round08_minos_covariance_starts_stdout.log`

The smoke stdout is also under `notes/logs/round08_minos_covariance_starts_smoke_stdout.log`.

## Commands

Common environment:

```bash
env PYTHONPATH=/root/pdgfits-private/src \
  PDGFITS_DATA_BACKEND=snapshot \
  PDGFITS_SNAPSHOT_DIR=/root/pdgfits-private/data/pdg-snapshot \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  /root/micromamba/micromamba run -r /root/micromamba-root -n pdg <command>
```

Syntax check, passed:

```bash
env PYTHONPATH=/root/pdgfits-private/src \
  PDGFITS_DATA_BACKEND=snapshot \
  PDGFITS_SNAPSHOT_DIR=/root/pdgfits-private/data/pdg-snapshot \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  /root/micromamba/micromamba run -r /root/micromamba-root -n pdg \
  python -m py_compile notes/codex-refactor/run_round08_minos_covariance_starts.py
```

B0S smoke run, passed:

```bash
prlimit --as=7800000000 --rss=7800000000 -- \
  env PYTHONPATH=/root/pdgfits-private/src \
  PDGFITS_DATA_BACKEND=snapshot \
  PDGFITS_SNAPSHOT_DIR=/root/pdgfits-private/data/pdg-snapshot \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  /root/micromamba/micromamba run -r /root/micromamba-root -n pdg \
  python notes/codex-refactor/run_round08_minos_covariance_starts.py \
    --case fit_b0s_s08637 \
    --results-csv /tmp/round08_smoke_results.csv \
    --results-jsonl /tmp/round08_smoke_results.jsonl \
    --evaluations-csv /tmp/round08_smoke_evaluations.csv \
    --evaluations-jsonl /tmp/round08_smoke_evaluations.jsonl \
    --failures-csv /tmp/round08_smoke_failures.csv \
    --failures-jsonl /tmp/round08_smoke_failures.jsonl \
    --stdout-log notes/logs/round08_minos_covariance_starts_smoke_stdout.log
```

Full requested run, passed with 16 result rows, 284 evaluation rows, and 0
failures:

```bash
prlimit --as=7800000000 --rss=7800000000 -- \
  env PYTHONPATH=/root/pdgfits-private/src \
  PDGFITS_DATA_BACKEND=snapshot \
  PDGFITS_SNAPSHOT_DIR=/root/pdgfits-private/data/pdg-snapshot \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  /root/micromamba/micromamba run -r /root/micromamba-root -n pdg \
  python notes/codex-refactor/run_round08_minos_covariance_starts.py \
    --results-csv notes/codex-refactor/round08_minos_covariance_starts_results.csv \
    --results-jsonl notes/codex-refactor/round08_minos_covariance_starts_results.jsonl \
    --evaluations-csv notes/codex-refactor/round08_minos_covariance_starts_evaluations.csv \
    --evaluations-jsonl notes/codex-refactor/round08_minos_covariance_starts_evaluations.jsonl \
    --failures-csv notes/codex-refactor/round08_minos_covariance_starts_failures.csv \
    --failures-jsonl notes/codex-refactor/round08_minos_covariance_starts_failures.jsonl \
    --stdout-log notes/logs/round08_minos_covariance_starts_stdout.log
```

Final syntax check, passed:

```bash
env PYTHONPATH=/root/pdgfits-private/src \
  PDGFITS_DATA_BACKEND=snapshot \
  PDGFITS_SNAPSHOT_DIR=/root/pdgfits-private/data/pdg-snapshot \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  /root/micromamba/micromamba run -r /root/micromamba-root -n pdg \
  python -m py_compile notes/codex-refactor/run_round08_minos_covariance_starts.py
```

## Policies Compared

- `covariance_only`: use only the covariance-predicted, exact-target-projected
  start with the existing SLSQP/trust-constr/KKT/descent-check stack.
- `current_plus_covariance`: evaluate the current continuation/MLE-projection
  starts plus the covariance-predicted start, then compare fixed-target chi2.

The production baseline is the unmodified current code path.

## Production Baseline

| Case | Side | Endpoint | Residual | Method | nfev | Root calls |
| --- | --- | ---: | ---: | --- | ---: | ---: |
| `fit_b0_s042b95` | lower | 0.275540404781 | 0.00116 | `trust-constr-exact-hess+KKT+descent-check` | 65 | 4 |
| `fit_b0_s042b95` | upper | 0.306070010091 | -0.00414 | `trust-constr-exact-hess+KKT` | 30 | 4 |
| `fit_b0s_s08637` | lower | 0.000213390571884 | -0.000857 | `SLSQP+descent-check` | 102 | 20 |
| `fit_b0s_s08637` | upper | 0.000238200533120 | 0.00115 | `SLSQP+descent-check` | 103 | 20 |
| `fit_eta_m026w` | lower | 30.0376184893 | 0.000708 | `SLSQP+descent-check` | 222 | 4 |
| `fit_eta_m026w` | upper | 30.9069444669 | -0.000935 | `SLSQP+descent-check` | 241 | 4 |
| `avg_m026r08` | lower | 0.139853603605 | 0.00172 | `avg-fixed` |  | 19 |
| `avg_m026r08` | upper | 0.179269747493 | 0.00129 | `avg-fixed` |  | 19 |

## Covariance-Start Results

| Case | Side | Policy | Endpoint | Delta vs prod | Residual | Method | nfev | Root calls | Cov - current chi2 | Chart warning |
| --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- |
| `fit_b0_s042b95` | lower | `covariance_only` | 0.275540404781 | 0 | 0.00116 | `SLSQP+descent-check` | 370 | 4 |  | True |
| `fit_b0_s042b95` | upper | `covariance_only` | 0.306070010091 | 0 | -0.00414 | `trust-constr-exact-hess+KKT` | 30 | 4 |  | True |
| `fit_b0_s042b95` | lower | `current_plus_covariance` | 0.275540404781 | 0 | 0.00116 | `trust-constr-exact-hess+KKT+descent-check` | 65 | 4 | 2.55e-06 | True |
| `fit_b0_s042b95` | upper | `current_plus_covariance` | 0.306070010091 | 0 | -0.00414 | `trust-constr-exact-hess+KKT` | 30 | 4 | -1.42e-14 | True |
| `fit_b0s_s08637` | lower | `covariance_only` | 0.000213390571884 | 0 | -0.000857 | `SLSQP+descent-check` | 100 | 20 |  | False |
| `fit_b0s_s08637` | upper | `covariance_only` | 0.000238200533120 | 0 | 0.00115 | `SLSQP+descent-check` | 101 | 20 |  | False |
| `fit_b0s_s08637` | lower | `current_plus_covariance` | 0.000213390571884 | 0 | -0.000857 | `SLSQP` | 100 | 20 | 1.71e-13 | False |
| `fit_b0s_s08637` | upper | `current_plus_covariance` | 0.000238200533120 | 0 | 0.00115 | `SLSQP+descent-check` | 101 | 20 | -1.05e-11 | False |
| `fit_eta_m026w` | lower | `covariance_only` | 30.0376184893 | 0 | 0.000707 | `SLSQP+descent-check` | 177 | 4 |  | False |
| `fit_eta_m026w` | upper | `covariance_only` | 30.9069444669 | 0 | -0.000935 | `SLSQP+descent-check` | 202 | 4 |  | False |
| `fit_eta_m026w` | lower | `current_plus_covariance` | 30.0376184893 | 0 | 0.000707 | `SLSQP+descent-check` | 177 | 4 | -9.32e-10 | False |
| `fit_eta_m026w` | upper | `current_plus_covariance` | 30.9069444669 | 0 | -0.000935 | `SLSQP+descent-check` | 241 | 4 | 8.00e-10 | False |
| `avg_m026r08` | lower | `covariance_only` | 0.139834515628 | -1.91e-05 | 0.00370 | `BFGS` | 12 | 20 |  |  |
| `avg_m026r08` | upper | `covariance_only` | 0.179289151844 | 1.94e-05 | 0.00322 | `BFGS` | 61 | 20 |  |  |
| `avg_m026r08` | lower | `current_plus_covariance` | 0.139834515628 | -1.91e-05 | 0.00370 | `Nelder-Mead` | 126 | 20 | 1.55e-15 |  |
| `avg_m026r08` | upper | `current_plus_covariance` | 0.179289151844 | 1.94e-05 | 0.00322 | `Nelder-Mead` | 117 | 20 | 0 |  |

Fit endpoints were unchanged to displayed precision. `M026R08` average endpoints
shifted by about `1.9e-05`, but both shifted endpoints still verify
`chi2_profile = chi2_min + 1` within the existing `5e-3` bisection tolerance.
This is a bracket/bisection tolerance artifact in the notes-only average
harness, not evidence of a lower fixed-target profile.

## Fixed-Target Candidate Comparison

For `current_plus_covariance`, every profile call records the best chi2 from
current starts and from the covariance start at the same fixed target value.

| Case | Profile comparisons | Min cov-current chi2 | Max cov-current chi2 | Count cov lower < -1e-8 | Count cov lower < -1e-6 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `fit_b0_s042b95` | 4 | -2.37e-09 | 2.55e-06 | 0 | 0 |
| `fit_b0s_s08637` | 20 | -2.16e-11 | 1.07e-11 | 0 | 0 |
| `fit_eta_m026w` | 4 | -1.75e-07 | 6.99e-08 | 1 | 0 |
| `avg_m026r08` | 20 | -2.00e-15 | 1.55e-15 | 0 | 0 |

The covariance start did not find a materially lower fixed-target chi2. The
largest covariance improvement was `1.75e-07` chi2 on `eta_c J/psi psi(2S)::M026W`,
well below the pre-existing `1e-4` descent-check scale and far below a
scientific concern threshold.

## Chart And Boundary Diagnostics

`B0::S042B95` is the clear negative case. The selected/profiled points had
minimum decay parameter about `3.39e-7` and maximum absolute fitted decay
coordinate about `93.89`, so the arctan BRU chart is deeply saturated. On the
lower endpoint, `covariance_only` preserved the endpoint but required `370`
endpoint-solve function evaluations versus `65` in production and changed the
certificate to `SLSQP+descent-check`.

`B0S-BR::S086.37` had no chart-saturation warning: minimum decay parameter was
about `1.82e-5`, maximum decay parameter about `0.00631`, and maximum absolute
fitted decay coordinate about `1.75`. Here covariance-only matched production
and shaved two endpoint `nfev` on each side, but did not find lower chi2.

`eta_c J/psi psi(2S)::M026W` also had no chart-saturation warning. It preserved
endpoints and reduced endpoint `nfev` on the covariance-only run (`222 -> 177`
lower, `241 -> 202` upper), but the fixed-target chi2 comparison showed only
sub-`1e-6` differences.

## Interpretation

This round did not find a real endpoint-correctness issue. Returned endpoints
continue to satisfy the profile residual invariant.

It also did not find a robust simplification. A universal covariance-only start
policy is rejected: it behaves well for B0S and eta, but it is misleading near
the saturated B0 BRU chart and can be much more expensive.

There is a small performance/certificate hint in well-conditioned cases:
`B0S-BR::S086.37` and `eta_c J/psi psi(2S)::M026W` preserve endpoints with
slightly lower endpoint `nfev`, and B0S lower avoids the descent-check suffix in
`current_plus_covariance`. This is not strong enough for production because the
same rule regresses `B0::S042B95` lower.

## Rejected Designs / Negative Results

- Rejected replacing continuation/MLE-projection starts with covariance-only
  starts. B0 lower is a concrete counterexample: same endpoint, worse solve
  behavior, severe chart saturation.
- Rejected always adding the covariance start in production. It rarely finds a
  lower fixed-target chi2 and would add extra solves unless the profile solver
  is reorganized around start ordering and early acceptance.
- Rejected treating the covariance predictor as a boundary-aware method. It is
  a local fitted-coordinate quadratic approximation and is not reliable near
  BR/BRU arctan saturation.
- Did not change bracketed endpoint bisection, asymmetric interpolation, or the
  solver stack. No lower fixed-target chi2 justified that.

## Recommendation

Do not ship covariance starts as a production behavior change yet.

The useful next direction is a gated diagnostic or stratified follow-up: use
HESSE/profile disagreement plus chart-saturation metrics to identify cases
where covariance starts are expected to be well-conditioned. If Anthony wants a
method-change spike rather than diagnostics, the next candidate should be a
chart-aware BRU experiment for saturated cases like `B0::S042B95`, not a
universal MINOS-style covariance start.
