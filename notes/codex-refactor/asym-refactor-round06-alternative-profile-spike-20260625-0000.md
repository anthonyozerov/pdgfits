# Asym Refactor Round 06 Alternative Profile Spike

Date: 2026-06-25 00:00 PDT

Branch: `codex/asym-refactor-simplify-20260624-221852`

Production code changed: no. `src/pdgfits/build_chi2.py` was not edited.

## Scope

This was a notes-only alternative-method spike.  The goal was to compare the
current fitted-coordinate constrained profile solver against an independent
fixed-target solve in physical parameter space for hard BR/BRU/profile targets.

New notes-only harness:

- `notes/codex-refactor/run_round06_physical_profile_comparator.py`

Artifacts:

- `notes/codex-refactor/round06_physical_profile_summary.csv`
- `notes/codex-refactor/round06_physical_profile_summary.jsonl`
- `notes/codex-refactor/round06_physical_profile_attempts.csv`
- `notes/codex-refactor/round06_physical_profile_attempts.jsonl`
- `notes/codex-refactor/round06_physical_profile_b0_extra_summary.csv`
- `notes/codex-refactor/round06_physical_profile_b0_extra_summary.jsonl`
- `notes/codex-refactor/round06_physical_profile_b0_extra_attempts.csv`
- `notes/codex-refactor/round06_physical_profile_b0_extra_attempts.jsonl`

Raw stdout logs are under `notes/logs/` and should remain uncommitted.

## Comparator Formulation

For each selected current endpoint, the harness fixes the same physical target
value and minimizes the same chi2, but with optimization variables equal to the
physical parameters:

```text
minimize chi2(params_to_fitted_params(params))
subject to target_func(params) = endpoint
```

For decay/branching-fraction parameters, it applies explicit physical bounds:

```text
1e-12 <= BR/BRU parameter <= 1 - 1e-12
```

For single-particle BR/simplex fits the harness can add sum-to-one linear
constraints, but all targets in this run were `BRU`, so no simplex constraints
were active.

Starts were projected MLE and physical-parameter covariance perturbations.  The
main run used one covariance eigen-direction pair and two random covariance
starts, with both SLSQP and exact-Hessian trust-constr in the physical chart.
The B0 follow-up used three covariance eigen-direction pairs and eight random
starts with SLSQP only.

Acceptance for comparison was based on feasibility, not optimizer `success`:

- finite chi2;
- scaled fixed-target violation `<= 1e-7`;
- bound violation `<= 1e-10`;
- simplex violation `<= 1e-7` if applicable.

Predeclared interpretation threshold:

- `> 1e-3` lower chi2 than the current fixed-endpoint profile is concerning;
- `<= 1e-6` is numerical noise unless systematic;
- a higher comparator chi2 is not evidence against the current method, but it
  may show the comparator chart is too poorly conditioned to be a replacement.

## Commands

Common environment:

```bash
env PYTHONPATH=/root/pdgfits-private/src \
  PDGFITS_DATA_BACKEND=snapshot \
  PDGFITS_SNAPSHOT_DIR=/root/pdgfits-private/data/pdg-snapshot \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  /root/micromamba/micromamba run -r /root/micromamba-root -n pdg <command>
```

Initial compile:

```bash
python -m py_compile notes/codex-refactor/run_round06_physical_profile_comparator.py
```

Smoke run:

```bash
prlimit --as=7800000000 --rss=7800000000 -- \
  env PYTHONPATH=/root/pdgfits-private/src \
  PDGFITS_DATA_BACKEND=snapshot \
  PDGFITS_SNAPSHOT_DIR=/root/pdgfits-private/data/pdg-snapshot \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  /root/micromamba/micromamba run -r /root/micromamba-root -n pdg \
  python notes/codex-refactor/run_round06_physical_profile_comparator.py \
    --endpoint 'B0S-BR::S086.37::upper' \
    --summary-csv /tmp/round06_smoke_summary.csv \
    --summary-jsonl /tmp/round06_smoke_summary.jsonl \
    --attempts-csv /tmp/round06_smoke_attempts.csv \
    --attempts-jsonl /tmp/round06_smoke_attempts.jsonl \
    --stdout-log notes/logs/round06_physical_profile_smoke_stdout.log \
    --target-timeout-sec 180 \
    --cov-dirs 1 \
    --random-starts 1 \
    --solvers slsqp \
    --seed 20260625
```

Main hard-target run:

```bash
prlimit --as=7800000000 --rss=7800000000 -- \
  env PYTHONPATH=/root/pdgfits-private/src \
  PDGFITS_DATA_BACKEND=snapshot \
  PDGFITS_SNAPSHOT_DIR=/root/pdgfits-private/data/pdg-snapshot \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  /root/micromamba/micromamba run -r /root/micromamba-root -n pdg \
  python notes/codex-refactor/run_round06_physical_profile_comparator.py \
    --summary-csv notes/codex-refactor/round06_physical_profile_summary.csv \
    --summary-jsonl notes/codex-refactor/round06_physical_profile_summary.jsonl \
    --attempts-csv notes/codex-refactor/round06_physical_profile_attempts.csv \
    --attempts-jsonl notes/codex-refactor/round06_physical_profile_attempts.jsonl \
    --stdout-log notes/logs/round06_physical_profile_comparator_stdout.log \
    --target-timeout-sec 420 \
    --cov-dirs 1 \
    --random-starts 2 \
    --solvers slsqp,trust-constr \
    --seed 20260625
```

B0 extra-start follow-up:

```bash
prlimit --as=7800000000 --rss=7800000000 -- \
  env PYTHONPATH=/root/pdgfits-private/src \
  PDGFITS_DATA_BACKEND=snapshot \
  PDGFITS_SNAPSHOT_DIR=/root/pdgfits-private/data/pdg-snapshot \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  /root/micromamba/micromamba run -r /root/micromamba-root -n pdg \
  python notes/codex-refactor/run_round06_physical_profile_comparator.py \
    --endpoint 'B0::S042B95::lower' \
    --summary-csv notes/codex-refactor/round06_physical_profile_b0_extra_summary.csv \
    --summary-jsonl notes/codex-refactor/round06_physical_profile_b0_extra_summary.jsonl \
    --attempts-csv notes/codex-refactor/round06_physical_profile_b0_extra_attempts.csv \
    --attempts-jsonl notes/codex-refactor/round06_physical_profile_b0_extra_attempts.jsonl \
    --stdout-log notes/logs/round06_physical_profile_b0_extra_stdout.log \
    --target-timeout-sec 300 \
    --cov-dirs 3 \
    --random-starts 8 \
    --solvers slsqp \
    --seed 20260626
```

Final compile:

```bash
python -m py_compile notes/codex-refactor/run_round06_physical_profile_comparator.py
```

Result: completed successfully.

## Main Results

`improvement_vs_current = current_profile_chi2 - best_comparator_chi2`.
Positive values would mean the physical comparator found a lower fixed-target
chi2.

| Label | Target | Side | Current method | Current chi2 | Current residual | Best comparator chi2 | Comparator residual | Improvement | Feasible attempts | Best solver/start |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `B0` | `S042B95` | upper | `trust-constr-exact-hess+KKT` | 83.897856 | -0.004138 | 84.102657 | 0.200663 | -0.204800 | 2/10 | `slsqp` / `param_cov_eig1_minus` |
| `B0` | `S042B95` | lower | `trust-constr-exact-hess+KKT+descent-check` | 83.903150 | 0.001156 | 83.946276 | 0.044283 | -0.043127 | 4/10 | `slsqp` / `mle_projected` |
| `B0S-BR` | `S086.37` | upper | `SLSQP+descent-check` | 26.786638 | 0.001152 | 26.786638 | 0.001152 | 3.65e-12 | 10/10 | `trust-constr` / `mle_projected` |
| `B0S-BR` | `S086.37` | lower | `SLSQP+descent-check` | 26.784630 | -0.000857 | 26.784630 | -0.000857 | -2.35e-11 | 10/10 | `trust-constr` / `param_cov_eig1_plus` |
| `eta_c J/psi psi(2S)` | `M026W` | upper | `SLSQP+descent-check` | 186.509964 | -0.000935 | 186.510011 | -0.000888 | -4.73e-05 | 10/10 | `trust-constr` / `param_cov_eig1_plus` |
| `eta_c J/psi psi(2S)` | `M026W` | lower | `SLSQP+descent-check` | 186.511606 | 0.000707 | 186.511606 | 0.000707 | -6.34e-11 | 10/10 | `slsqp` / `random_mle_2` |

No endpoint had a positive chi2 improvement above `1e-3`.  The cleanest BRU
case was `B0S-BR::S086.37`: both sides matched the current fixed-target chi2 to
about `1e-11`.

The `eta_c J/psi psi(2S)::M026W` target is not itself a decay parameter, but it
does sit inside a BRU fit with 18 bounded decay parameters.  The physical
comparator matched the lower side to numerical precision and the upper side to
`4.7e-5` chi2, below the concern threshold.

## B0 Chart Diagnostics

`B0::S042B95` was the hardest target for the direct physical chart.  It is a
node with equation type `/` involving `S042.181` and `S042.390`, in a BRU fit
with 22 bounded decay parameters.  The smallest decay parameters at the fit are
around `3.4e-7`, which maps to very large arctan inverse coordinates.  The best
main-run B0 attempts had:

| Side | Best comparator chi2 | Improvement | Eq projected grad norm | Min decay param | Max decay param | Max abs fitted decay coord |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| upper | 84.102657 | -0.204800 | 6.70e5 | 3.39e-7 | 0.0510 | 93.82 |
| lower | 83.946276 | -0.043127 | 2.49e5 | 3.39e-7 | 0.0510 | 93.89 |

The best strictly feasible physical-chart upper attempt was much higher than
production, but the lowest near-feasible upper attempt was:

| Start | Solver | Success | Feasible | Chi2 | Improvement | Scaled target violation |
| --- | --- | --- | --- | ---: | ---: | ---: |
| `mle_projected` | `slsqp` | false | false | 83.898632 | -0.000776 | 6.50e-7 |

That is still above the current profile chi2, and the target violation is only
slightly above the strict `1e-7` comparator feasibility cutoff.

The extra-start B0 lower follow-up improved the best physical-chart value but
did not approach or beat the production profile:

| Target | Side | Attempts | Best comparator chi2 | Comparator residual | Improvement |
| --- | --- | ---: | ---: | ---: | ---: |
| `B0::S042B95` | lower | 15 | 83.930677 | 0.028683 | -0.027527 |

Interpretation: direct physical-parameter SLSQP/trust-constr is not a credible
replacement chart for this B0 ratio endpoint.  It is useful as a stress test,
and it found no lower constrained chi2, but its large tangent residuals and
chart saturation show that it is poorly conditioned here.

## Boundary And Chart Activity

No selected target had active decay bounds under the `1e-8` activity check, and
no simplex constraints were present because all three fits were `BRU`.

Chart saturation was target-dependent:

| Label | Target | Decay params | Simplex groups | Best max abs fitted decay coord | Comment |
| --- | --- | ---: | ---: | ---: | --- |
| `B0` | `S042B95` | 22 | 0 | ~93.9 | Severe arctan inverse saturation from tiny BRU parameters; direct physical chart poorly conditioned. |
| `B0S-BR` | `S086.37` | 7 | 0 | ~1.75 | Well conditioned enough for physical trust-constr to match production. |
| `eta_c J/psi psi(2S)` | `M026W` | 18 | 0 | ~0.20 | Well conditioned decay map, but target is a non-decay parameter. |

## Rejected Ideas

- I did not copy the production fitted-space endpoint solver just to recover the
  production endpoint coordinates as physical starts.  That would be useful for
  polishing diagnostics, but it would make this round less independent.
- I did not add a logit or reduced-coordinate replacement chart.  B0 suggests
  such a chart could be worth a later spike, but this round already shows that
  naive direct physical parameters are not a drop-in replacement.
- I did not change endpoint bisection, profile tolerances, or asymmetric
  interpolation.  No lower fixed-target chi2 was found.
- I did not broaden to a sweep over all descent-check/KKT rows.  The requested
  hard targets were covered first, and the B0 conditioning issue is specific
  enough that a broader run with this same physical chart would mostly measure
  comparator failures.

## Bottom Line

The physical-space comparator found no hidden lower constrained-profile chi2 on
the requested hard targets.  `B0S-BR::S086.37` is strong negative evidence: an
independent bounded physical trust-constr solve matches the current
fixed-target profile to numerical precision.  `eta_c J/psi psi(2S)::M026W` also
does not show a material discrepancy.

`B0::S042B95` is not evidence of a production failure.  The comparator fails on
conditioning before it challenges the current solution: large arctan inverse
coordinates, high physical-chart projected gradients, and worse feasible chi2
values.  Recommendation: keep the current method.  If Anthony wants a
principled replacement spike, the next candidate should be a deliberately
scaled bounded/reduced BRU chart, not naive direct physical-parameter SLSQP.
