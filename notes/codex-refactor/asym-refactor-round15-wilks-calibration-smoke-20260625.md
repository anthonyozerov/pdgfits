# Asym Refactor Round 15 Wilks-Calibration Smoke Test

Date: 2026-06-25

Production code changed: no. `src/pdgfits/build_chi2.py` was not edited.

## Question

Round 13 found no numerical endpoint failure, but flagged Wilks/profile-likelihood
regularity risks in boundary/tiny-parameter and asymmetric/profile-shape strata.
This round asked whether those labels produce visible calibration symptoms in a
tiny, explicitly exploratory toy setting.

This is not a coverage study. It is a hard-capped smoke test.

## Design

Cases:

| Case | Stratum | Reason |
| --- | --- | --- |
| `B0::S042B95` | boundary/tiny-parameter | round-13 high-risk BRU boundary/chart-saturation case |
| `B0S-BR::S086.37` | asymmetric/profile-shape | cheaper round-13 medium-high nonlinear/asymmetric case |
| `G(2000),G(1800)::K002M` | clean control | locally quadratic interior direct-coordinate control |

Toy generation used the fitted snapshot model mean `mu_adjust(param_values)`.
Noise was local Gaussian with each reported asymmetric error symmetrized at zero
residual, `2 sigma_minus sigma_plus / (sigma_minus + sigma_plus)`, and the case
correlation matrix eigenvalue-clipped for sampling if needed. This is a
pragmatic smoke DGP, not the full asymmetric interpolation as a generative
likelihood.

For each toy:

- refit the toy chi2 using `chi2_open(..., y_toy)`;
- profile the original fitted target value under the toy refit;
- record `q = chi2_profile(truth_target) - chi2_min_toy`;
- compare `q <= 1` to the chi-square-1 one-sigma heuristic;
- for the first 2 toys per case, compute full lower/upper profile endpoints and
  record endpoint residual/certification diagnostics.

## Artifacts

- Harness: `notes/codex-refactor/run_round15_wilks_calibration_smoke.py`
- Toy CSV/JSONL: `notes/codex-refactor/round15_wilks_smoke_toys.csv`, `notes/codex-refactor/round15_wilks_smoke_toys.jsonl`
- Endpoint CSV/JSONL: `notes/codex-refactor/round15_wilks_smoke_endpoints.csv`, `notes/codex-refactor/round15_wilks_smoke_endpoints.jsonl`
- Failure CSV/JSONL: `notes/codex-refactor/round15_wilks_smoke_failures.csv`, `notes/codex-refactor/round15_wilks_smoke_failures.jsonl`
- Summary JSON: `notes/codex-refactor/round15_wilks_smoke_summary.json`

Exact row counts:

| Artifact | Rows |
| --- | ---: |
| Toy rows | 24 |
| Endpoint rows | 12 |
| Failure rows | 0 |

Runtime was `87.8s` for 8 toys per case and endpoint computation on the first
2 toys per case.

## Commands

Syntax check, passed:

```bash
env PYTHONPATH=/root/pdgfits-private/src \
  PDGFITS_DATA_BACKEND=snapshot \
  PDGFITS_SNAPSHOT_DIR=/root/pdgfits-private/data/pdg-snapshot \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  /root/micromamba/micromamba run -r /root/micromamba-root -n pdg \
  python -m py_compile notes/codex-refactor/run_round15_wilks_calibration_smoke.py
```

Bounded smoke run, passed:

```bash
prlimit --as=7800000000 --rss=7800000000 -- \
  env PYTHONPATH=/root/pdgfits-private/src \
  PDGFITS_DATA_BACKEND=snapshot \
  PDGFITS_SNAPSHOT_DIR=/root/pdgfits-private/data/pdg-snapshot \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  /root/micromamba/micromamba run -r /root/micromamba-root -n pdg \
  python notes/codex-refactor/run_round15_wilks_calibration_smoke.py \
    --toys-per-case 8 \
    --endpoint-toys-per-case 2 \
    --max-runtime-sec 600 \
    --overwrite
```

## Results

Expected chi-square-1 reference values:

- `P(chi2_1 <= 1) = 0.6827`;
- median `chi2_1 = 0.4549`.

| Case | Stratum | Toys | `q <= 1` | Mean q | Median q | Max q | Endpoint toys containing truth | Boundary/chart signal |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `B0::S042B95` | boundary/tiny-parameter | 8 | 6/8 | 0.459 | 0.129 | 1.675 | 1/2 | 8/8 chart-saturated; min BRU coord `2.69e-7`, max fitted coord `118` |
| `B0S-BR::S086.37` | asymmetric/profile-shape | 8 | 5/8 | 1.364 | 0.734 | 3.989 | 2/2 | no chart saturation; min BRU coord `1.62e-5` |
| `G(2000),G(1800)::K002M` | clean control | 8 | 6/8 | 0.837 | 0.439 | 2.548 | 1/2 | no boundary metric |

Per-case q values:

| Case | q values |
| --- | --- |
| `B0::S042B95` | `1.140, 0.139, 0.057, 0.100, 0.120, 1.675, 0.437, 0.007` |
| `B0S-BR::S086.37` | `0.649, 0.676, 3.989, 0.792, 1.014, 0.022, 3.753, 0.019` |
| `G(2000),G(1800)::K002M` | `0.786, 2.548, 0.378, 0.500, 0.000, 2.090, 0.195, 0.197` |

Endpoint residuals over the 12 computed endpoints were all within the existing
`5e-3` tolerance. Max absolute endpoint residual was `0.004777`.

Endpoint profile methods:

| Method | Endpoint rows |
| --- | ---: |
| `SLSQP` | 4 |
| `SLSQP+descent-check` | 3 |
| `trust-constr-exact-hess+KKT` | 5 |

There were no profile consistency failures (`q < -1e-4`: `0/24`). One B0 toy
used the scipy fallback after Minuit did not report success
(`fit_b0_s042b95`, toy 6, seed `2026062521`); the row still had a certified
profile point, `q=0.437`, scaled constraint violation `2.9e-14`, and no
consistency failure.

## Answers

1. Did the smoke test find any numerical endpoint-correctness failure?

No. There were `0` failure rows, `0/24` profile consistency failures, and all
12 computed endpoints verified `chi2_profile(endpoint) = chi2_min + 1` within
the existing tolerance. The largest absolute endpoint residual was `0.004777`.

2. Did the Wilks-risk strata show visible calibration symptoms?

Partly, but only at smoke-test strength.

`B0::S042B95` visibly reproduced the boundary/tiny-parameter condition: all
8 toy MLEs stayed chart-saturated, with tiny BRU coordinates around `1e-7`.
Its q distribution did not look worse than the clean control in 8 toys
(`q<=1` was `6/8`, same as control), so this round shows boundary activation,
not a clear calibration deviation.

`B0S-BR::S086.37` had the clearest visible q-tail signal: `q<=1` was `5/8`,
mean q was `1.364`, median q was `0.734`, and two toys had q near `4`. The
clean control was closer to the chi-square-1 median (`0.439` vs expected
`0.455`) and had `6/8` below 1. With only 8 toys, this is a symptom worth
studying, not a coverage estimate.

3. Are results strong enough to justify a production code change?

No. The endpoints remain numerically certified, no lower-profile or endpoint
failure was found, and the toy count is intentionally tiny. There is no
production solver, map, or `build_chi2.py` change justified by this round.

4. Concrete next step?

If Anthony wants calibration evidence, the next step is a serious notes-only
calibration study, not a targeted production fix: more toys per stratum,
include `eta_c J/psi psi(2S)::M026G01`, add at least one more clean control, and
compare multiple DGPs including the local Gaussian used here and an explicitly
asymmetric/split-normal toy. If the immediate goal is solver-method work, stop
and synthesize rounds 7-15 instead; the numerical-method evidence has plateaued
without finding an endpoint-correctness failure.

5. Which ideas were rejected/limited, quantitatively?

- Full coverage claims: rejected. This is only `8` toys per case.
- Endpoints for every toy: limited to `2` toys per case because endpoint toys
  cost about `8s` for B0S and `12-16s` for B0 in probes, while q-only toys were
  much cheaper.
- Treating local Gaussian toys as the real asymmetric likelihood DGP: rejected.
  The smoke DGP uses symmetrized errors and clipped correlations only for a
  cheap empirical check.
- Production changes from q-tail symptoms: rejected. `0` numerical endpoint
  failures and max endpoint residual `0.004777` do not support changing the
  production method.

## Bottom Line

Round 15 found no numerical endpoint-correctness failure. It did find a small
visible calibration-smoke signal in the asymmetric/profile-shape B0S stratum
and persistent boundary activation in B0. This supports a later serious
calibration study if desired; it does not support another optimizer fallback or
production code change.
