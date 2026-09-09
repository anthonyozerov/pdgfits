# Asym Refactor Round 03 Supervised

Date: 2026-06-24 23:07 PDT

Branch: `codex/asym-refactor-simplify-20260624-221852`

## Scope

This round was a focused robustness/evidence pass for fit-side constrained
profile endpoints accepted through `+descent-check`, the least theorem-like
acceptance path in the current asymmetric-error solver.

I did not change production code. I added a notes-only diagnostic harness:

- `notes/codex-refactor/run_asym_descent_multistart.py`

Diagnostic outputs:

- `notes/codex-refactor/round03_descent_multistart_summary.csv`
- `notes/codex-refactor/round03_descent_multistart_summary.jsonl`
- `notes/codex-refactor/round03_descent_multistart_starts.jsonl`

The bulky stdout log is under `notes/logs/` and should remain uncommitted.

## Hard Endpoints Selected

The source table was `notes/asym_fit_sweep_results_simplified.csv`. I selected
fixed endpoint coordinates, not whole target endpoint searches:

| Label | Target | Side | Why selected |
| --- | --- | --- | --- |
| `Lam-b-0` | `S040.29` | lower | Slow hard Lam-b endpoint; accepted by `SLSQP+descent-check`; residual sits near the bisection tolerance boundary. The duplicate node `S040R29` was not repeated. |
| `B0` | `S042B95` | lower | Accepted by `trust-constr-exact-hess+KKT+descent-check`; large projected gradient; known previous failure label. |
| `B0S-BR` | `S086.37` | upper | Hard BR label; accepted by plain `SLSQP+descent-check` with sizable projected gradient. |
| `eta_c J/psi psi(2S)` | `M026W` | lower | Known no-descent ablation failure target; accepted by `SLSQP+descent-check`. |
| `eta_c J/psi psi(2S)` | `nuisance_M071.254` | lower | Accepted by `trust-constr-exact-hess+descent-check`; previous failure target; high objective-gradient scale. |

## Diagnostic Design

For each fixed endpoint, the harness rebuilt the same constrained profile
problem:

```text
target_func(fitted_params_to_params(fp)) = endpoint
```

It then ran the current SLSQP -> exact-Hessian `trust-constr` -> KKT-polish ->
stationarity/descent acceptance logic from explicit starts.

Starts per endpoint: 11 total.

- 1 MLE-projected start;
- plus/minus top 2 fitted-space covariance eigen-directions, scale `0.5`;
- 3 deterministic random covariance perturbations around the MLE, scale `0.5`;
- 3 deterministic random covariance perturbations around the first fixed-endpoint solution, scale `0.25`.

Seed: `20260624`.

Predeclared material-improvement threshold: `1e-3` in fixed-endpoint chi2.
This is below the endpoint bisection residual tolerance `5e-3` but well above
roundoff-level differences seen in repeat solves.

## Commands

Common environment:

```bash
env PYTHONPATH=/root/pdgfits-private/src \
  PDGFITS_DATA_BACKEND=snapshot \
  PDGFITS_SNAPSHOT_DIR=/root/pdgfits-private/data/pdg-snapshot \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  /root/micromamba/micromamba run -r /root/micromamba-root -n pdg <command>
```

Smoke check:

```bash
python notes/codex-refactor/run_asym_descent_multistart.py \
  --summary-csv /tmp/round03_smoke_summary.csv \
  --summary-jsonl /tmp/round03_smoke_summary.jsonl \
  --start-jsonl /tmp/round03_smoke_starts.jsonl \
  --stdout-log notes/logs/round03_descent_multistart_smoke_stdout.log \
  --endpoint 'B0::S042B95::lower' \
  --cov-dirs 1 \
  --random-starts 1 \
  --endpoint-random-starts 1 \
  --target-timeout-sec 120
```

Main diagnostic run:

```bash
prlimit --as=7800000000 --rss=7800000000 -- \
  env PYTHONPATH=/root/pdgfits-private/src \
  PDGFITS_DATA_BACKEND=snapshot \
  PDGFITS_SNAPSHOT_DIR=/root/pdgfits-private/data/pdg-snapshot \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  /root/micromamba/micromamba run -r /root/micromamba-root -n pdg \
  python notes/codex-refactor/run_asym_descent_multistart.py \
    --endpoint 'Lam-b-0::S040.29::lower' \
    --endpoint 'B0::S042B95::lower' \
    --endpoint 'B0S-BR::S086.37::upper' \
    --endpoint 'eta_c J/psi psi(2S)::M026W::lower' \
    --endpoint 'eta_c J/psi psi(2S)::nuisance_M071.254::lower' \
    --target-timeout-sec 300 \
    --cov-dirs 2 \
    --random-starts 3 \
    --endpoint-random-starts 3 \
    --seed 20260624
```

Validation:

```bash
python -m py_compile notes/codex-refactor/run_asym_descent_multistart.py
```

Result: completed in the `pdg` environment.

I also checked the summary artifact programmatically: 5 rows, all `ok`, all
11 starts accepted per endpoint, no material improvements, and all improvements
below `1e-6`.

## Results

Residual means `profile_chi2(endpoint) - (chi2_min + 1)`. Improvement means
`accepted_profile_chi2 - best_multistart_chi2`; positive values would indicate
that an alternative start found a lower fixed-endpoint profile chi2.

| Label | Target | Side | Accepted method | Endpoint residual | Accepted profile chi2 | Best multistart chi2 | Improvement | Starts ok | Runtime s | Status |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `B0` | `S042B95` | lower | `trust-constr-exact-hess+KKT+descent-check` | 0.00115614 | 83.903149735565 | 83.903149732963 | 2.60e-09 | 11/11 | 5.91 | ok |
| `B0S-BR` | `S086.37` | upper | `SLSQP+descent-check` | 0.00115176 | 26.786638436322 | 26.786638436319 | 2.50e-12 | 11/11 | 4.77 | ok |
| `Lam-b-0` | `S040.29` | lower | `SLSQP+descent-check` | -0.00493586 | 15.316472423170 | 15.316472423162 | 7.85e-12 | 11/11 | 10.15 | ok |
| `eta_c J/psi psi(2S)` | `M026W` | lower | `SLSQP+descent-check` | 0.000706727 | 186.511605656217 | 186.511605656072 | 1.45e-10 | 11/11 | 3.86 | ok |
| `eta_c J/psi psi(2S)` | `nuisance_M071.254` | lower | `trust-constr-exact-hess+descent-check` | 0.00188467 | 186.512783595976 | 186.512783571062 | 2.49e-08 | 11/11 | 6.88 | ok |

Maximum observed improvement was `2.49e-08`, from
`eta_c J/psi psi(2S) / nuisance_M071.254 / lower`. This is far below the
`1e-3` diagnostic threshold and below endpoint-relevant scale.

All 55 fixed-target solves were accepted by the same local criteria used by the
current solver. Some alternative starts returned higher local constrained
points, but none returned a materially lower one.

## Interpretation

This focused diagnostic supports keeping the current default descent-check path.
For these hard endpoints, the concern was that `+descent-check` might be hiding
a local non-minimum. The multistart evidence did not find such a failure:
independent projected starts converged back to the accepted fixed-endpoint
profile chi2 to numerical precision.

This does not turn the descent check into a theorem. It strengthens the
empirical story for the specific hard endpoints most likely to expose a local
basin problem.

## Rejected Fixes Or Ideas

- I did not add solver machinery. There was no fixed-endpoint improvement that
would justify another fallback path.
- I did not loosen endpoint residual or profile acceptance tolerances.
- I did not change `build_chi2.py`, asymmetric interpolation, or the endpoint
root search.
- I did not run a massive sweep. A focused hard-endpoint multistart check was
the higher-value scientific test for this round.
- I did not commit `notes/logs/`.

## Remaining Risks

- This is a five-endpoint diagnostic, not a global proof over all fit targets.
- The harness mirrors production acceptance logic in notes-only code; it is an
independent-start test, not an independent mathematical formulation.
- Random starts are local covariance-scale perturbations. They are good for
nearby basin checks, but they do not exhaust disconnected feasible components.
- The endpoint residual tolerance remains the existing `5e-3`; this round did
not test stricter endpoint bisection tolerances.
- Boundary/Wilks-regularity concerns for BR/BRU targets remain statistical
modeling risks even when the numerical profile solve is stable.

## Bottom Line

No concrete profile-likelihood failure was found. The evidence from this round
supports leaving the production fit-side descent-check path unchanged and using
optional multistart diagnostics as evidence tooling for future hard endpoints.
