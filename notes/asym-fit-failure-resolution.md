# Fit-Side Asymmetric Error Failure Resolution

Date: 2026-06-24

Branch: `codex-cloud-asym-avg-local-experiments`

Environment prefix:

```bash
env PYTHONPATH=/root/pdgfits-private/src \
  PDGFITS_DATA_BACKEND=snapshot \
  PDGFITS_SNAPSHOT_DIR=/root/pdgfits-private/data/pdg-snapshot \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  /root/micromamba/micromamba run -r /root/micromamba-root -n pdg
```

## Summary

All 13 fit-side failures from `notes/asym_fit_sweep_results.csv` are fixed in
the focused rerun.

Before:

- 227 staged fit target rows
- 214 ok
- 13 errors
- Failure classes:
  - feasible but nonstationary constrained points in `eta_c J/psi psi(2S)`,
    `B0S-BR`, and `B0`
  - exact-Hessian non-finite optimizer path in `K_3^*(1780)` / `M060.6`
  - bad BRU bracket/domain continuation in `eta_c(2S)` / `M059.4`

After:

- Focused previous-failure rows: 13/13 ok
- Verified endpoints: 26/26
- Max absolute endpoint residual: `0.00490332908259461`
- Median absolute endpoint residual: `0.0021158119918851526`
- Slowest focused target row: `17.99 s`

Machine-readable final focused results:

- `notes/asym_fit_failure_focus_final.csv`
- `notes/asym_fit_failure_focus_final.jsonl`

## Method

The production change is limited to `src/pdgfits/asym_errors.py`.

1. Exact-Hessian fallback exceptions are caught.

   The previous `K_3^*(1780)` failure was a `ValueError` from the
   `trust-constr` exact-Hessian path. That path is now treated as one failed
   candidate, so SLSQP/BFGS-Hessian fallback candidates can still be tried and
   endpoint verification remains loud.

2. High-curvature constrained profiles get a reduced-coordinate fallback.

   If SLSQP / exact-Hessian `trust-constr` / KKT polishing do not satisfy the
   first-order projected-gradient criterion, the solver tries a local
   reduced-coordinate Powell search in a tangent/nullspace chart, projecting
   each trial point back to the scalar constraint.

3. Feasible but high projected-gradient points require a local descent check.

   This is not optimizer-success acceptance. A point is accepted through this
   fallback only if it is finite, satisfies the scaled equality constraint, and
   a projected feasible descent probe cannot lower chi2 by more than `1e-4`.
   Accepted diagnostics are labeled with `+descent-check`.

   The motivating failures had large fitted-coordinate projected gradients but
   extremely high local curvature. Direct line searches lowered chi2 only by
   about `1e-8` to `1e-10` in the `B0`/`B0S-BR` cases, and by about `1e-10`
   at the `eta_c J/psi psi(2S)` base points. The profile chi2 error from this
   numerical stationarity issue is below the endpoint verification tolerance.

4. Initial endpoint brackets contract when the fixed target is unreachable.

   `eta_c(2S)` / `M059.4` had a covariance lower bracket below the BRU
   parameter domain and an upper bracket that was too large for one direct
   continuation step. `binary_search_error` now contracts a failed bracket
   toward the MLE on the same side and uses the first feasible bracket point
   that clears `chi2_min + 1`. If no usable bracket exists, it still fails.

## Focused Results

Command shape:

```bash
env PYTHONPATH=/root/pdgfits-private/src \
  PDGFITS_DATA_BACKEND=snapshot \
  PDGFITS_SNAPSHOT_DIR=/root/pdgfits-private/data/pdg-snapshot \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  /root/micromamba/micromamba run -r /root/micromamba-root -n pdg \
  python - <<'PY'
# grouped run_fit() by label, then notes.run_asym_fit_sweep.run_one_target()
# for the 13 failed (label, target) rows from notes/asym_fit_sweep_results.csv
PY
```

Final focused results:

```text
rows 13
ok 13
errors 0
endpoints 26
max_abs_resid 0.00490332908259461
median_abs_resid 0.0021158119918851526
```

Endpoint methods in the final focused rows:

```text
SLSQP                                      7
SLSQP+descent-check                       11
trust-constr-exact-hess+KKT               6
trust-constr-exact-hess+KKT+descent-check 1
trust-constr-exact-hess+descent-check     1
```

Previously failing rows now pass:

```text
K_3^*(1780) / M060.6
B0S-BR / S086R04
eta_c J/psi psi(2S) / M026G01
eta_c J/psi psi(2S) / M026W
eta_c J/psi psi(2S) / nuisance_M071.13
eta_c J/psi psi(2S) / nuisance_M071.254
eta_c J/psi psi(2S) / M026.31
eta_c J/psi psi(2S) / M026G03
eta_c J/psi psi(2S) / M026G07
B0 / S042B09
B0 / S042.390
B0 / S042B95
eta_c(2S) / M059.4
```

## Validation

Focused asymmetry tests:

```bash
env PYTHONPATH=/root/pdgfits-private/src \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  /root/micromamba/micromamba run -r /root/micromamba-root -n pdg \
  python -m pytest tests/test_asym_errors.py -q
```

Result:

```text
3 passed in 4.06 s
```

Full offline suite:

```bash
env PYTHONPATH=/root/pdgfits-private/src \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  /root/micromamba/micromamba run -r /root/micromamba-root -n pdg \
  python -m pytest tests/ -q
```

Result:

```text
159 passed, 2 skipped in 13.68 s
```

`Upsilon(2S)` sanity:

```bash
env PYTHONPATH=/root/pdgfits-private/src \
  PDGFITS_DATA_BACKEND=snapshot \
  PDGFITS_SNAPSHOT_DIR=/root/pdgfits-private/data/pdg-snapshot \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  /root/micromamba/micromamba run -r /root/micromamba-root -n pdg \
  python -m pdgfits.run_fits --fit_label 'Upsilon(2S)' --calc_asym_errors
```

Result: completed. Residuals remained in the previous range:

```text
M052.4 residuals=(-0.00051,0.00055)
M052.6 residuals=(-0.0011,0.0018)
M052R22 residuals=(-0.0018,0.0023)
M052R4 residuals=(-0.00051,0.00055)
M052R6 residuals=(-0.0011,0.0018)
nuisance_M048.8 residuals=(-0.0035,0.0033)
```

`Lam-b-0` sanity:

```bash
env PYTHONPATH=/root/pdgfits-private/src \
  PDGFITS_DATA_BACKEND=snapshot \
  PDGFITS_SNAPSHOT_DIR=/root/pdgfits-private/data/pdg-snapshot \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  /root/micromamba/micromamba run -r /root/micromamba-root -n pdg \
  python -m pdgfits.run_fits --fit_label 'Lam-b-0' --calc_asym_errors
```

Result: completed all 20 targets. Max printed absolute residual was about
`0.0049`, matching the previous successful run.

Average sanity:

```bash
env PYTHONPATH=/root/pdgfits-private/src \
  PDGFITS_DATA_BACKEND=snapshot \
  PDGFITS_SNAPSHOT_DIR=/root/pdgfits-private/data/pdg-snapshot \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  /root/micromamba/micromamba run -r /root/micromamba-root -n pdg \
  python -m pdgfits.run_avgs --node M026R08 \
    --output notes/avg_m026r08_asym_fit_resolution.csv
```

Result: completed. The single-node CLI does not write a CSV because
`run_avgs.py` only writes output when more than one row is present. Printed
average-side asymmetric errors:

```text
0.019546087791061734 0.01987005609699094
```

## Caveats

- This is a focused fix for the 13 known failures, not a full all-target
  fit-side sweep. A staged broad sweep should be rerun before claiming complete
  fit-side robustness across all PDG targets.
- `+descent-check` is a numerical local optimality certificate for
  high-curvature BR/BRU chart cases. It should not be interpreted as a clean
  first-order KKT solution. The endpoint value is still accepted only after
  independent `chi2_min + 1` endpoint verification.
- Runtime can still be high on hard fit-side targets. This patch adds fallback
  work only after first-order stationarity checks fail, but the broad suite
  should still track target runtimes.
