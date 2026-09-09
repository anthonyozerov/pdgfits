# Fit-Side Profile Diagnostics

Date: 2026-06-24

Branch: `codex-cloud-asym-avg-local-experiments`

Environment prefix used for snapshot-backed commands:

```bash
env PYTHONPATH=/root/pdgfits-private/src \
  PDGFITS_DATA_BACKEND=snapshot \
  PDGFITS_SNAPSHOT_DIR=/root/pdgfits-private/data/pdg-snapshot \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  /root/micromamba/micromamba run -r /root/micromamba-root -n pdg
```

## Summary

The `Lam-b-0` failure was reproduced exactly. It is not a feasibility problem:
the failed constrained point has scaled equality violation around `1e-11`.
It is an optimizer-quality problem: the old SLSQP plus quasi-Newton
`trust-constr` path stopped at a feasible point with a large projected
gradient and much higher chi2 than a better constrained stationary point.

A candidate fix is applied in `src/pdgfits/asym_errors.py`:

- keep SLSQP as the fast first attempt;
- add exact objective and constraint Hessians to the `trust-constr` fallback;
- polish feasible candidates by solving the scalar-equality KKT equations;
- require feasibility plus projected-gradient/KKT stationarity for acceptance,
  instead of accepting `optimizer.success` alone;
- include projected-gradient norm in the failure message.

This fixes the requested `Lam-b-0` local validation and keeps `Upsilon(2S)`,
the asymmetric-error unit tests, the full non-DB test suite, and a small
average sanity check passing. Runtime is materially higher on hard
`Lam-b-0` targets, especially `S040.29`/`S040R29` and `S040.9`/`S040R9`.

I would call this a locally validated candidate fix, not a final PR-ready
claim for all PDG fits without a broader snapshot sweep.

## Reproduction

Command:

```bash
env PYTHONPATH=/root/pdgfits-private/src \
  PDGFITS_DATA_BACKEND=snapshot \
  PDGFITS_SNAPSHOT_DIR=/root/pdgfits-private/data/pdg-snapshot \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  /root/micromamba/micromamba run -r /root/micromamba-root -n pdg \
  python -m pdgfits.run_fits --fit_label 'Lam-b-0' --calc_asym_errors
```

Pre-fix result:

```text
RuntimeError: Profile minimization failed for S040.10=3.310979027615594e-06:
success=False, finite=True, constraint_violation=1.22e-17,
scaled_constraint_violation=1.06e-11,
message=The maximum number of function evaluations is exceeded.
```

The diagnostic harness in `notes/fit_profile_diagnostics.py` replayed the
current continuation order and isolated the failure to the first lower-side
bracket call:

```text
phase: lower_expand_0
target: S040.10
target_kind: parameter
chi2_min: 14.321408287472362
chi2_min + 1: 15.321408287472362
target_value: 5.620626900476748e-06
target_std: 1.154823936430577e-06
initial_lb / failing fixed value: 3.310979027615594e-06
initial_ub: 7.930274773337903e-06
```

Raw diagnostic output is stored in:

```text
notes/fit-side-profile-diagnostics.json
```

## Failing-Point Diagnostics

At the failing lower bracket `S040.10 = 3.310979027615594e-06`, the previous
current-path best candidate was feasible but not profiled:

| Attempt | chi2 | chi2 - (min+1) | scaled constraint | projected grad | status |
| --- | ---: | ---: | ---: | ---: | --- |
| projected continuation start | 96.054317 | 80.732909 | 2.46e-09 | 6.73e+05 | start only |
| SLSQP from continuation | 93.038511 | 77.717102 | 1.06e-11 | 5.49e+06 | rank-deficient |
| old trust-constr fallback | 29.669365 | 14.347957 | 1.06e-11 | 22.553386 | max eval |
| trust-constr BFGS, 5000 iter | 21.241907 | 5.920499 | 1.06e-11 | 57.047632 | max eval |
| exact-Hessian trust-constr | 20.232226 | 4.910818 | 1.06e-11 | 0.477168 | xtol |
| KKT root polish | 20.232033 | 4.910625 | 1.33e-10 | 2.57e-04 | converged |

Important observations:

- The old failure should not be accepted. Its feasibility is excellent, but its
  projected gradient is `22.55`, and a lower constrained stationary point exists.
- `optimizer.success=True` is not enough by itself. A start from the MLE with
  exact-Hessian `trust-constr` reported `xtol` success at `chi2=20.388637`, but
  had projected gradient `114.009`.
- The reduced nullspace Powell experiment reached essentially the same chi2 as
  the KKT root (`20.232033`) but did not certify stationarity
  (`projected_grad=14.63`), so derivative-free termination alone is not a good
  acceptance criterion.
- The useful combination was exact curvature to reach the correct basin, then
  KKT polishing to certify the local constrained optimum.

## Patch Rationale

The patch is intentionally conservative:

- SLSQP remains the first attempt for easy profile calls.
- If SLSQP is not certified, exact-Hessian `trust-constr` is run from the
  projected continuation start. This directly addresses the observed
  quasi-Newton fallback stall.
- A KKT root polish solves:

```text
grad chi2(x) + lambda * grad c_scaled(x) = 0
c_scaled(x) = 0
```

  The polished point is kept only if it is finite, feasible, and does not
  increase chi2 relative to the candidate being polished.
- Acceptance now requires scaled feasibility plus projected stationarity:

```text
scaled constraint violation <= 1e-7
projected_grad_norm <= max(1e-3, 1e-6 * objective_grad_norm)
```

This is stricter than accepting `success=False` because the constraint is small,
and also stricter than accepting `success=True` with poor KKT evidence.

## Validation

### Unit Tests

Command:

```bash
env PYTHONPATH=/root/pdgfits-private/src \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  /root/micromamba/micromamba run -r /root/micromamba-root -n pdg \
  python -m pytest tests/test_asym_errors.py -q
```

Result:

```text
3 passed in 4.77s
```

Full non-DB suite:

```bash
env PYTHONPATH=/root/pdgfits-private/src \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  /root/micromamba/micromamba run -r /root/micromamba-root -n pdg \
  python -m pytest tests/ -q
```

Result:

```text
159 passed, 2 skipped in 15.07s
```

### Upsilon(2S)

Command:

```bash
env PYTHONPATH=/root/pdgfits-private/src \
  PDGFITS_DATA_BACKEND=snapshot \
  PDGFITS_SNAPSHOT_DIR=/root/pdgfits-private/data/pdg-snapshot \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  /root/micromamba/micromamba run -r /root/micromamba-root -n pdg \
  python -m pdgfits.run_fits --fit_label 'Upsilon(2S)' --calc_asym_errors
```

Result: completed. Endpoint residuals:

```text
M052.4 residuals=(-0.00051,0.00055)
M052.6 residuals=(-0.0011,0.0018)
M052R22 residuals=(-0.0018,0.0023)
M052R4 residuals=(-0.00051,0.00055)
M052R6 residuals=(-0.0011,0.0018)
nuisance_M048.8 residuals=(-0.0035,0.0033)
```

### Lam-b-0

Command:

```bash
env PYTHONPATH=/root/pdgfits-private/src \
  PDGFITS_DATA_BACKEND=snapshot \
  PDGFITS_SNAPSHOT_DIR=/root/pdgfits-private/data/pdg-snapshot \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  /root/micromamba/micromamba run -r /root/micromamba-root -n pdg \
  python -m pdgfits.run_fits --fit_label 'Lam-b-0' --calc_asym_errors
```

Result: completed. Endpoint residuals:

```text
S040.10 residuals=(-0.00059,0.0033)
S040.11 residuals=(0.0017,-0.00069)
S040.15 residuals=(0.0021,-0.0041)
S040.29 residuals=(0.00057,-0.0049)
S040.4 residuals=(-0.0018,0.0035)
S040.9 residuals=(0.0031,-0.003)
S040R04 residuals=(5.5e-05,-0.00095)
S040R05 residuals=(-0.00059,0.00072)
S040R10 residuals=(-0.00059,0.0033)
S040R11 residuals=(0.0017,-0.00069)
S040R15 residuals=(0.0021,-0.0041)
S040R20 residuals=(0.00059,0.0013)
S040R23 residuals=(-0.0025,0.0026)
S040R29 residuals=(0.00057,-0.0049)
S040R4 residuals=(-0.0018,0.0035)
S040R9 residuals=(0.0031,-0.003)
nuisance_S042.10 residuals=(-0.00052,0.00056)
nuisance_S042.30 residuals=(-0.0028,0.0031)
nuisance_S051.2 residuals=(-0.00045,0.00047)
nuisance_S051.4 residuals=(-0.0024,-0.0026)
```

Max absolute printed endpoint residual: `0.0049`, inside the current
`binary_search_error` verification tolerance.

### Average Sanity Check

Command:

```bash
env PYTHONPATH=/root/pdgfits-private/src \
  PDGFITS_DATA_BACKEND=snapshot \
  PDGFITS_SNAPSHOT_DIR=/root/pdgfits-private/data/pdg-snapshot \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  /root/micromamba/micromamba run -r /root/micromamba-root -n pdg \
  python -m pdgfits.run_avgs --node M026R08 --output notes/avg_m026r08_fit_profile_sanity.csv
```

Result: completed; reported finite average-side asymmetric errors:

```text
0.019546087791061734 0.01987005609699094
```

## Recommendation

Keep the candidate fix, but treat it as needing a broader fit-side snapshot
sweep before claiming the whole asymmetric-error issue is closed. The local
evidence supports the approach: the exact failing profile point is now solved by
a lower-chi2 KKT-certified constrained optimum, and the full `Lam-b-0` run passes
endpoint verification. The remaining concern is performance and coverage across
larger or more pathological arbitrary target functions.
