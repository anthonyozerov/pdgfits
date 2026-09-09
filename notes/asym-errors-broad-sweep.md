# Asymmetric Error Broad Sweep

This report records a local snapshot validation run on branch
`codex-cloud-asym-avg-local-experiments`.

The scientific acceptance criterion used here is endpoint verification:
for a reported 1-sigma endpoint, the profiled chi2 at the fixed target
must be `chi2_min + 1` after profiling other fitted and nuisance
parameters. Optimizer success and small feasibility violation alone were
not treated as sufficient.

## Environment

Common environment:

```bash
PYTHONPATH=/root/pdgfits-private/src
PDGFITS_DATA_BACKEND=snapshot
PDGFITS_SNAPSHOT_DIR=/root/pdgfits-private/data/pdg-snapshot
XLA_PYTHON_CLIENT_PREALLOCATE=false
/root/micromamba/micromamba run -r /root/micromamba-root -n pdg <command>
```

Snapshot discovery:

- Average input rows: 9116
- Unique average nodes: 2647
- Fit table rows: 98
- Fit algorithms: `BR=33`, `SPECIAL=20`, `IGNORE=16`, `MASS=13`, `BRU=13`,
  `BR (NO MATRIX)=1`, `BR PRINT=1`, `SPECIALT=1`
- Fit labels selected for sweep: 81 non-`IGNORE`, non-`tauhflav` labels

## Machine-Readable Outputs

- Average sweep JSONL: `notes/asym_avg_sweep_results.jsonl` (2647 rows)
- Average sweep CSV: `notes/asym_avg_sweep_results.csv` (2647 data rows)
- Average stdout/stderr log: `notes/logs/asym_avg_sweep_stdout.log`
- Fit sweep JSONL: `notes/asym_fit_sweep_results.jsonl` (227 rows)
- Fit sweep CSV: `notes/asym_fit_sweep_results.csv` (227 data rows)
- Fit stdout/stderr log: `notes/logs/asym_fit_sweep_stdout.log`
- Additional session log: `notes/logs/local-codex-broad-sweep-20260623-234150.log`
- Sweep drivers: `notes/run_asym_avg_sweep.py`,
  `notes/run_asym_fit_sweep.py`

The fit CSV contains multiline exception text for a few rows, so JSONL
line count is the cleaner row-count check.

## Commands Run

Full average sweep:

```bash
env PYTHONPATH=/root/pdgfits-private/src \
  PDGFITS_DATA_BACKEND=snapshot \
  PDGFITS_SNAPSHOT_DIR=/root/pdgfits-private/data/pdg-snapshot \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  /root/micromamba/micromamba run -r /root/micromamba-root -n pdg \
  python notes/run_asym_avg_sweep.py \
    --jsonl notes/asym_avg_sweep_results.jsonl \
    --csv notes/asym_avg_sweep_results.csv \
    --stdout-log notes/logs/asym_avg_sweep_stdout.log
```

Representative fit sweep:

```bash
env PYTHONPATH=/root/pdgfits-private/src \
  PDGFITS_DATA_BACKEND=snapshot \
  PDGFITS_SNAPSHOT_DIR=/root/pdgfits-private/data/pdg-snapshot \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  /root/micromamba/micromamba run -r /root/micromamba-root -n pdg \
  python notes/run_asym_fit_sweep.py \
    --stage representative \
    --jsonl notes/asym_fit_sweep_results.jsonl \
    --csv notes/asym_fit_sweep_results.csv \
    --stdout-log notes/logs/asym_fit_sweep_stdout.log \
    --target-timeout-sec 900
```

Capped follow-up on higher-dimensional and nuisance-heavy fits:

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
    --jsonl notes/asym_fit_sweep_results.jsonl \
    --csv notes/asym_fit_sweep_results.csv \
    --stdout-log notes/logs/asym_fit_sweep_stdout.log \
    --target-timeout-sec 900
```

Sanity commands:

```bash
env PYTHONPATH=/root/pdgfits-private/src \
  PDGFITS_DATA_BACKEND=snapshot \
  PDGFITS_SNAPSHOT_DIR=/root/pdgfits-private/data/pdg-snapshot \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  /root/micromamba/micromamba run -r /root/micromamba-root -n pdg \
  python -m pdgfits.run_avgs --node M026R08 \
    --output notes/avg_m026r08_broad_sweep_sanity.csv
```

```bash
env PYTHONPATH=/root/pdgfits-private/src \
  PDGFITS_DATA_BACKEND=snapshot \
  PDGFITS_SNAPSHOT_DIR=/root/pdgfits-private/data/pdg-snapshot \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  /root/micromamba/micromamba run -r /root/micromamba-root -n pdg \
  python -m pdgfits.run_fits --fit_label 'Upsilon(2S)' --calc_asym_errors
```

`Lam-b-0` was not rerun through the CLI after the sweep because the sweep
already covered all 20 selected `Lam-b-0` targets after the code changes,
with target runtimes up to 151.5 s.

Tests:

```bash
env PYTHONPATH=/root/pdgfits-private/src \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  /root/micromamba/micromamba run -r /root/micromamba-root -n pdg \
  python -m pytest tests/test_asym_errors.py -q
```

Result: `3 passed in 4.45s`.

```bash
env PYTHONPATH=/root/pdgfits-private/src \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  /root/micromamba/micromamba run -r /root/micromamba-root -n pdg \
  python -m pytest tests/ -q
```

Result: `159 passed, 2 skipped in 14.20s`.

## Step 1: Full Snapshot Average Sweep

Coverage:

- Nodes attempted: 2647
- Successful rows: 2647
- Error/skipped rows: 0
- Fresh independent endpoint checks: 5294 endpoints
- Fresh check statuses: 2647 upper `ok`, 2647 lower `ok`
- Nodes with `br_adjust`: 220
- Nodes with nuisance parameters: 229
- Maximum nuisance count observed: 4
- Nodes with `dep_meas` in this snapshot: 0
- Nodes with input correlation rows in this snapshot: 0

For averages, the fresh check fixes the primary coordinate directly and
profiles only nuisance coordinates. This is independent of the endpoint
search bookkeeping and uses direct chi2 evaluation for one-parameter
averages.

Fresh endpoint residual distribution, using
`profile_chi2(endpoint) - (chi2_min + 1)`:

| Quantity | Count | Median | Mean | 90% | 95% | 99% | 99.9% | Max abs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Combined abs residual | 5294 | 0.001984 | 0.002173 | 0.004272 | 0.004632 | 0.004924 | 0.004993 | 0.0049997 |

Signed residuals:

| Side | Median | Mean | Min | Max |
| --- | ---: | ---: | ---: | ---: |
| Upper | 0.0000269 | 0.0000113 | -0.004999 | 0.004987 |
| Lower | -0.0001056 | -0.0000741 | -0.005000 | 0.004983 |

Runtime distribution per node:

| Quantity | Median | Mean | 90% | 95% | 99% | 99.9% | Max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Runtime seconds | 0.360 | 0.881 | 1.906 | 2.383 | 5.229 | 10.222 | 11.204 |

Endpoint profile call distribution:

| Quantity | Median | Mean | 90% | 95% | 99% | 99.9% | Max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Profile calls per node | 26 | 26.61 | 31 | 32 | 36 | 48 | 52 |

Slowest average nodes:

| Node | Runtime s | n_params | n_nuisance | has_adjust |
| --- | ---: | ---: | ---: | --- |
| `S041B46` | 11.204 | 5 | 4 | True |
| `M070R84` | 10.999 | 5 | 4 | True |
| `S051R05` | 10.279 | 5 | 4 | True |
| `S042R2` | 10.191 | 4 | 3 | True |
| `S086R46` | 9.899 | 5 | 4 | True |

Largest absolute fresh residuals were at the binary-search tolerance
boundary, not failure outliers. Examples: `M002W` lower residual
`-0.004999656`, `M057B18` upper residual `-0.004998958`,
`S042CKS` upper residual `-0.004998656`.

Average conclusion: the current average-side fixed-coordinate nuisance
profiling looks robust on the full snapshot sweep. I found no average
endpoint failures, no skipped fresh checks, and no residuals beyond the
expected binary-search tolerance.

## Step 2: Fit-Side Sweep

The fit sweep is staged and resumable. It does not compute all targets for
all fits. The first stage selects a representative target set per fit:
regular parameters, nuisance parameters, and nodes. Known hard fits
`G(2000),G(1800)`, `Upsilon(2S)`, and `Lam-b-0` are expanded to all
targets by the script. A second capped pass expanded selected
high-dimensional or nuisance-heavy labels with up to three nodes, three
regular parameters, and three nuisance parameters.

Coverage:

- Selected fit labels: 81
- Target rows written: 227
- Successful target rows: 214
- Error target rows: 13
- Successful endpoints verified by endpoint residuals: 428

Successful endpoint residual distribution:

| Quantity | Count | Median | Mean | 90% | 95% | 99% | 99.9% | Max abs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Combined abs residual | 428 | 0.000895 | 0.001343 | 0.003487 | 0.004059 | 0.004915 | 0.004974 | 0.004978 |

Signed residuals:

| Side | Median | Mean | Min | Max |
| --- | ---: | ---: | ---: | ---: |
| Upper | -0.00000923 | -0.000215 | -0.004916 | 0.004978 |
| Lower | 0.00000685 | 0.000146 | -0.004969 | 0.004848 |

Runtime distribution for successful target rows:

| Quantity | Median | Mean | 90% | 95% | 99% | 99.9% | Max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Target runtime seconds | 2.056 | 20.402 | 46.561 | 111.056 | 203.132 | 632.082 | 632.346 |

Profile calls for successful target rows:

| Quantity | Median | Mean | 90% | 95% | 99% | Max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Profile calls | 7 | 12.49 | 22 | 23 | 23 | 23 |

KKT polish and fallback usage:

- Successful rows requiring at least one KKT-polished profile point: 113 / 214
- Total KKT-polished profile points in successful rows: 982
- Endpoint/profile methods among successful endpoints:
  `SLSQP=252`, `trust-constr-exact-hess+KKT=148`,
  `projected-start=26`, `projected-start+KKT=1`,
  `trust-constr-bfgs-hess+KKT=1`

Known/hard labels:

| Label | Rows | Status | Max target runtime s |
| --- | ---: | --- | ---: |
| `G(2000),G(1800)` | 3 | 3 ok | 0.961 |
| `Upsilon(2S)` | 6 | 6 ok | 2.672 |
| `Lam-b-0` | 20 | 20 ok | 151.495 |
| `chi_c012 psi(2S)` | 9 | 9 ok | 632.346 |
| `D0` | 7 | 7 ok | 106.719 |
| `Lambda_c` | 9 | 9 ok | 54.862 |
| `B0` | 9 | 6 ok, 3 error | 9.013 |
| `eta_c J/psi psi(2S)` | 8 | 1 ok, 7 error | 177.723 |
| `B0S-BR` | 9 | 8 ok, 1 error | 26.630 |
| `B+J/psi` | 9 | 9 ok | 136.073 |

Slowest successful target rows:

| Label | Target | Runtime s | n_params | n_nodes | n_nuisance |
| --- | --- | ---: | ---: | ---: | ---: |
| `chi_c012 psi(2S)` | `M056.11` | 632.346 | 52 | 118 | 3 |
| `chi_c012 psi(2S)` | `M056.6` | 631.107 | 52 | 118 | 3 |
| `chi_c012 psi(2S)` | `M055B1` | 206.928 | 52 | 118 | 3 |
| `eta_c J/psi psi(2S)` | `M026.45` | 177.723 | 21 | 50 | 2 |
| `Lam-b-0` | `S040.29` | 151.495 | 10 | 10 | 4 |

Largest residuals among successful fit endpoints were still within the
binary-search tolerance. Examples: `a_2(1320)` / `M012R1` upper residual
`0.004978`, `pi_2(1670)` / `M034.8` lower residual `-0.004969`,
`Lam-b-0` / `S040.29` lower residual `-0.004936`.

## Step 3: Outlier And Failure Analysis

The 13 fit-side target failures group into the following classes.

1. Feasible but nonstationary constrained profiles.

This is the dominant failure class. The constrained point can satisfy the
target equality to tiny scaled violation, and the optimizer may report
success, but the projected/KKT stationarity residual is too large. These
are scientifically unacceptable as asymmetric endpoints and are correctly
loud failures.

Affected rows:

- `B0S-BR` / `S086R04`: scaled constraint violation `8.04e-12`,
  projected gradient norm `88.7`, optimizer message `xtol`.
- `eta_c J/psi psi(2S)` / `M026G01`, `M026W`, `nuisance_M071.13`,
  `nuisance_M071.254`, `M026.31`, `M026G03`, `M026G07`: projected
  gradient norms about `1.56` to `2.13`.
- `B0` / `S042B09`, `S042.390`, `S042B95`: scaled constraint violations
  near `1e-14`, but projected gradient norms around `4.96e3` or `955`.

2. Non-finite optimizer/Hessian path.

- `K_3^*(1780)` / `M060.6`: failure was `ValueError: array must not
  contain infs or NaNs` inside the trust-constr exact-Hessian fallback
  during lower-side endpoint search. The fit covariance itself was finite.
  This looks like a profile optimizer/domain handling failure, possibly
  near a constrained map boundary.

3. Constraint infeasible or severe target violation.

- `eta_c(2S)` / `M059.4`: upper-side expansion failed with scaled
  constraint violation `80.3` and projected gradient norm `inf`. The
  optimizer could not satisfy the target constraint for the attempted
  point.

4. Runtime outliers.

Runtime is a separate operational concern. Several successful endpoints
require expensive fallback/KKT work: `chi_c012 psi(2S)` exceeded 10 minutes
per target for two targets, and multiple `Lam-b-0`, `B+J/psi`, `D0`, and
`Lambda_c` targets took tens to hundreds of seconds. This is why the fit
sweep remained staged and capped rather than all targets for all fits.

No successful fit endpoint had a high endpoint residual after endpoint
verification. The failures are mostly profile quality failures before a
scientifically valid endpoint can be accepted.

## Step 4: Design Recommendation

I made one small, principled code change: `ProfilePoint` now carries shared
profile diagnostics (`scaled_constraint_violation`, projected gradient
norm, objective gradient norm, method label, iteration/evaluation counts,
runtime), and `binary_search_error.last_diagnostics` records the actual
upper/lower endpoint coordinates. This supports common sweep/reporting
logic without changing the mathematical acceptance rule.

I did not attempt a broader algorithm rewrite in this pass. The evidence
does not justify declaring the fit-side machinery robust yet:

- Average-side validation is strong and should be preserved. Direct
  fixed-primary-coordinate nuisance profiling is simpler and appropriate
  for averages.
- Fit-side successful endpoints look good when the profiler accepts them,
  including the known hard `Upsilon(2S)` and `Lam-b-0` cases.
- Fit-side failure clusters show that exact-Hessian trust-constr plus KKT
  polishing is not sufficient as a universal method. Feasible but
  nonstationary constrained profiles remain the main hard problem.
- KKT polishing is common in successful hard fits, so it is not just an
  exceptional fallback. That argues for making the fit-side profile solver
  more principled, not merely increasing iteration limits.

Recommended next design step:

- Keep average-side reduced/fixed-coordinate nuisance profiling.
- Keep loud endpoint verification and projected/KKT stationarity checks.
- For fits, investigate a reduced-coordinate or chart-aware constrained
  profiler for branch-fraction/simplex maps, with better handling of
  map-domain boundaries and non-finite Hessians.
- Use the failing clusters above as regression cases before broadening the
  sweep. In particular: `eta_c J/psi psi(2S)`, `B0`, `B0S-BR`,
  `K_3^*(1780)`, and `eta_c(2S)`.

Bottom line: averages appear close to reliable on this snapshot; fits are
improved and diagnostic, but not yet broadly reliable.
