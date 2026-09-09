# Simplified Fit-Side Asymmetric Error Broad Sweep

Date: 2026-06-24

Branch: `codex-cloud-asym-avg-local-experiments`

Baseline commit noted before this run: `850b09a refactor: simplify profile solver fallbacks`.

## Summary

The simplified fit-side profile solver completed the same staged broad target
set as the previous fit-side broad sweep.

- Rows before resume: 205 CSV/JSONL rows, all `ok`.
- Rows after resume: 227 CSV rows and 227 JSONL rows, all `ok`.
- Duplicate target keys: 0.
- Target key set versus previous broad sweep: identical, 227 / 227 keys.
- Successful endpoints verified: 454, upper and lower for each successful row.
- Previous broad-sweep fit-side errors: 13 / 13 now fixed.
- Regressions among previous successful target rows: 0.

The scientific acceptance criterion was endpoint verification:
`profile_chi2(endpoint) - (chi2_min + 1)` after profiling fitted and nuisance
parameters. Optimizer success and feasibility alone were not treated as
sufficient.

## Artifacts

- Simplified sweep CSV: `notes/asym_fit_sweep_results_simplified.csv`
- Simplified sweep JSONL: `notes/asym_fit_sweep_results_simplified.jsonl`
- Simplified stdout/stderr log: `notes/logs/asym_fit_sweep_simplified_stdout.log`
- Previous broad-sweep report: `notes/asym-errors-broad-sweep.md`
- Sweep driver: `notes/run_asym_fit_sweep.py`

## Resource Mode

The sweep used the local snapshot backend and the `pdg` micromamba
environment. It ran sequentially through the existing sweep driver. No new
parallelism was added.

Runtime safeguards:

- `XLA_PYTHON_CLIENT_PREALLOCATE=false`
- Snapshot backend: `PDGFITS_DATA_BACKEND=snapshot`
- Snapshot directory: `/root/pdgfits-private/data/pdg-snapshot`
- Per-target timeout: 900 seconds
- Resume mode: default enabled, using the existing JSONL completed-key set

## Commands

The follow-up command was resumed against the existing simplified outputs:

```bash
env PYTHONPATH=/root/pdgfits-private/src \
  PDGFITS_DATA_BACKEND=snapshot \
  PDGFITS_SNAPSHOT_DIR=/root/pdgfits-private/data/pdg-snapshot \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  /root/micromamba/micromamba run -r /root/micromamba-root -n pdg \
  python notes/run_asym_fit_sweep.py \
    --stage representative \
    --fit-label 'chi_c012 psi(2S)' \
    --fit-label 'D0' \
    --fit-label 'Lambda_c' \
    --fit-label 'B0' \
    --fit-label 'eta_c J/psi psi(2S)' \
    --fit-label 'B0S-BR' \
    --fit-label 'B+J/psi' \
    --max-nodes 3 \
    --max-params 3 \
    --max-nuisance 3 \
    --jsonl notes/asym_fit_sweep_results_simplified.jsonl \
    --csv notes/asym_fit_sweep_results_simplified.csv \
    --stdout-log notes/logs/asym_fit_sweep_simplified_stdout.log \
    --target-timeout-sec 900
```

The resumed process reported:

```text
fit labels selected=7, completed target rows=205
```

That confirms the existing output files were used for resume/deduplication
instead of rerunning completed target rows unnecessarily.

## Coverage

| Quantity | Value |
| --- | ---: |
| Target rows | 227 |
| `ok` rows | 227 |
| Error rows | 0 |
| Endpoint checks | 454 |
| Fit labels represented | 81 |
| Duplicate `(stage, label, target)` keys | 0 |

Known and high-interest labels:

| Label | Rows | Status | Max target runtime s |
| --- | ---: | --- | ---: |
| `G(2000),G(1800)` | 3 | 3 ok | 0.987 |
| `Upsilon(2S)` | 6 | 6 ok | 2.483 |
| `Lam-b-0` | 20 | 20 ok | 87.575 |
| `chi_c012 psi(2S)` | 9 | 9 ok | 324.937 |
| `D0` | 7 | 7 ok | 27.032 |
| `Lambda_c` | 9 | 9 ok | 16.294 |
| `B0` | 9 | 9 ok | 9.763 |
| `eta_c J/psi psi(2S)` | 8 | 8 ok | 17.206 |
| `B0S-BR` | 9 | 9 ok | 15.708 |
| `B+J/psi` | 9 | 9 ok | 24.834 |
| `K_3^*(1780)` | 2 | 2 ok | 2.967 |
| `eta_c(2S)` | 2 | 2 ok | 4.166 |

## Endpoint Residuals

Residual definition:
`profile_chi2(endpoint) - (chi2_min + 1)`.

Combined absolute residuals across all 454 verified endpoints:

| Quantity | Value |
| --- | ---: |
| Count | 454 |
| Median | 0.000919 |
| Mean | 0.001384 |
| 90% | 0.003541 |
| 95% | 0.004249 |
| 99% | 0.004914 |
| 99.9% | 0.004974 |
| Max abs | 0.004978 |

Signed residuals:

| Side | Median | Mean | Min | Max |
| --- | ---: | ---: | ---: | ---: |
| Upper | -0.00000751 | -0.000149 | -0.004916 | 0.004978 |
| Lower | 0.0000101 | 0.000158 | -0.004969 | 0.004848 |

The largest residuals are at the binary-search tolerance boundary, not
scientific endpoint failures.

## Runtime

Target runtime distribution for the 227 successful rows:

| Quantity | Seconds |
| --- | ---: |
| Median | 1.875 |
| Mean | 8.148 |
| 90% | 12.672 |
| 95% | 25.465 |
| 99% | 133.291 |
| 99.9% | 293.123 |
| Max | 324.937 |

Slowest successful targets:

| Label | Target | Runtime s | n_params | n_nodes | n_nuisance |
| --- | --- | ---: | ---: | ---: | ---: |
| `chi_c012 psi(2S)` | `M056.6` | 324.937 | 52 | 118 | 3 |
| `chi_c012 psi(2S)` | `M055B1` | 184.166 | 52 | 118 | 3 |
| `chi_c012 psi(2S)` | `M056.11` | 149.354 | 52 | 118 | 3 |
| `Lam-b-0` | `S040R29` | 87.575 | 10 | 10 | 4 |
| `Lam-b-0` | `S040.29` | 86.142 | 10 | 10 | 4 |
| `chi_c012 psi(2S)` | `M055B11` | 73.022 | 52 | 118 | 3 |
| `chi_c012 psi(2S)` | `M055B10` | 47.745 | 52 | 118 | 3 |
| `Lam-b-0` | `S040.10` | 43.537 | 10 | 10 | 4 |
| `Lam-b-0` | `S040R10` | 41.743 | 10 | 10 | 4 |
| `D0` | `S032B03` | 27.032 | 36 | 71 | 1 |

The simplified solver remains numerically conservative but is faster on the
worst `chi_c012 psi(2S)` rows than the previous broad sweep, where two targets
were about 631 to 632 seconds.

## Method Usage

Accepted endpoint method distribution:

| Method | Endpoints |
| --- | ---: |
| `SLSQP` | 280 |
| `SLSQP+descent-check` | 110 |
| `trust-constr-exact-hess+KKT` | 36 |
| `projected-start` | 26 |
| `trust-constr-exact-hess+descent-check` | 1 |
| `trust-constr-exact-hess+KKT+descent-check` | 1 |

All profile-point method counts accumulated during successful rows:

| Method | Profile points |
| --- | ---: |
| `SLSQP` | 1164 |
| `SLSQP+descent-check` | 709 |
| `projected-start` | 564 |
| `trust-constr-exact-hess+KKT` | 280 |
| `projected-start+descent-check` | 158 |
| `trust-constr-exact-hess+KKT+descent-check` | 13 |
| `trust-constr-exact-hess+descent-check` | 6 |
| `trust-constr-exact-hess` | 2 |

KKT and descent-check usage:

- Rows with at least one KKT-polished profile point: 54 / 227.
- KKT-polished profile points: 293.
- Rows with at least one accepted descent-check endpoint: 73 / 227.
- Accepted endpoints using descent check: 112 / 454.
- Profile points using descent check: 886.

No final method string used the removed BFGS-Hessian `trust-constr` fallback
or the removed reduced/nullspace Powell fallback.

## Comparison To Previous Broad Sweep

Previous fit-side broad sweep from `notes/asym-errors-broad-sweep.md`:

- Target rows: 227.
- Successful rows: 214.
- Error rows: 13.
- Successful endpoints verified: 428.
- Max absolute residual among successful endpoints: 0.004978.

Simplified sweep:

- Target rows: 227.
- Successful rows: 227.
- Error rows: 0.
- Successful endpoints verified: 454.
- Max absolute residual among successful endpoints: 0.004978.

The target key set is identical between the old and simplified sweeps. All 13
previous errors are now `ok`:

| Previous failure | Old class | Simplified status |
| --- | --- | --- |
| `K_3^*(1780)` / `M060.6` | `ValueError` | ok |
| `B0S-BR` / `S086R04` | `RuntimeError` | ok |
| `eta_c J/psi psi(2S)` / `M026G01` | `RuntimeError` | ok |
| `eta_c J/psi psi(2S)` / `M026W` | `RuntimeError` | ok |
| `eta_c J/psi psi(2S)` / `nuisance_M071.13` | `RuntimeError` | ok |
| `eta_c J/psi psi(2S)` / `nuisance_M071.254` | `RuntimeError` | ok |
| `B0` / `S042B09` | `RuntimeError` | ok |
| `eta_c(2S)` / `M059.4` | `RuntimeError` | ok |
| `eta_c J/psi psi(2S)` / `M026.31` | `RuntimeError` | ok |
| `eta_c J/psi psi(2S)` / `M026G03` | `RuntimeError` | ok |
| `eta_c J/psi psi(2S)` / `M026G07` | `RuntimeError` | ok |
| `B0` / `S042.390` | `RuntimeError` | ok |
| `B0` / `S042B95` | `RuntimeError` | ok |

Regressions among the 214 previous successful rows: none.

## Failures And Caveats

There were no row-level errors in this simplified staged broad sweep.

Caveats:

- This is still a staged representative plus capped follow-up fit-side sweep,
  not an all-target sweep over every node and parameter in every fit.
- Runtime remains high for the largest `chi_c012 psi(2S)` targets and selected
  `Lam-b-0` targets.
- The descent check remains an important numerical acceptance certificate:
  112 accepted endpoints used it in this sweep. Removing it was already shown
  to fail known hard cases in the simplification ablation notes.
- The exact-Hessian `trust-constr` fallback remains necessary for some
  accepted endpoints, though no removed fallback was used.

## Recommendation

Keep the simplified fit-side solver from commit `850b09a`. The broad staged
snapshot evidence is stronger than the previous broad sweep: the same 227
targets all pass endpoint verification, all 13 previous failures are fixed,
and no previous successes regressed.

The next validation step, if more confidence is required, should be an
expanded all-target fit sweep for a bounded set of high-risk labels rather
than more changes to the solver.
