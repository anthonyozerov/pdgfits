# Asym Refactor Round 11 BR/BRU Chart Conditioning

Date: 2026-06-25

Production code changed: no. `src/pdgfits/build_chi2.py` was not edited.

## Question

Can an alternative bounded or physical chart for saturated BR/BRU profile solves
produce the same verified endpoints with cleaner stationarity/certificates or
lower cost, without changing the scientific chi2 model?

This round tested chart changes only inside a notes-only fixed-target profile
harness. The chi2 and target functions remained the production ones. Alternative
chart coordinates were mapped to physical parameters and then back through the
existing `params_to_fitted_params` composition before evaluating chi2.

Charts tested:

- `logit`: decay parameters use `p = sigmoid(z)`.
- `log_positive`: decay parameters use `p = exp(z)`, with `z` bounded to keep
  `p` in `(1e-12, 1 - 1e-12)`.
- `direct_physical`: decay parameters are optimized directly in bounded
  physical coordinates.

Thresholds:

- Endpoint verification tolerance: `abs(profile_chi2 - (chi2_min + 1)) <= 5e-3`.
- Numerical tie: lower fixed-target chi2 improvement `<= 1e-5`.
- Material lower constrained-profile improvement: `> 1e-4`.

## Artifacts

- Harness: `notes/codex-refactor/run_round11_brbru_chart_conditioning.py`
- Endpoint results CSV/JSONL: `notes/codex-refactor/round11_brbru_chart_conditioning_results.csv`, `notes/codex-refactor/round11_brbru_chart_conditioning_results.jsonl`
- Fixed production-endpoint checks CSV/JSONL: `notes/codex-refactor/round11_brbru_chart_conditioning_fixed_checks.csv`, `notes/codex-refactor/round11_brbru_chart_conditioning_fixed_checks.jsonl`
- Profile/candidate evaluations CSV/JSONL: `notes/codex-refactor/round11_brbru_chart_conditioning_evaluations.csv`, `notes/codex-refactor/round11_brbru_chart_conditioning_evaluations.jsonl`
- Failures CSV/JSONL: `notes/codex-refactor/round11_brbru_chart_conditioning_failures.csv`, `notes/codex-refactor/round11_brbru_chart_conditioning_failures.jsonl`
- Raw stdout: `notes/logs/round11_brbru_chart_conditioning_stdout.log`

Exact row counts:

| Table | Rows |
| --- | ---: |
| Endpoint results | 12 |
| Fixed production-endpoint checks | 12 |
| Profile/candidate evaluations | 246 |
| Failures | 0 |

The failures files are empty because no harness or chart root failed.

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
  python -m py_compile notes/codex-refactor/run_round11_brbru_chart_conditioning.py
```

Smoke run, passed with 2 endpoint rows, 2 fixed checks, 65 evaluation rows, and
0 failures:

```bash
prlimit --as=7800000000 --rss=7800000000 -- \
  env PYTHONPATH=/root/pdgfits-private/src \
  PDGFITS_DATA_BACKEND=snapshot \
  PDGFITS_SNAPSHOT_DIR=/root/pdgfits-private/data/pdg-snapshot \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  /root/micromamba/micromamba run -r /root/micromamba-root -n pdg \
  python notes/codex-refactor/run_round11_brbru_chart_conditioning.py \
    --case fit_b0s_s08637 \
    --chart logit \
    --results-csv /tmp/round11_smoke_results.csv \
    --results-jsonl /tmp/round11_smoke_results.jsonl \
    --evaluations-csv /tmp/round11_smoke_evaluations.csv \
    --evaluations-jsonl /tmp/round11_smoke_evaluations.jsonl \
    --fixed-checks-csv /tmp/round11_smoke_fixed.csv \
    --fixed-checks-jsonl /tmp/round11_smoke_fixed.jsonl \
    --failures-csv /tmp/round11_smoke_failures.csv \
    --failures-jsonl /tmp/round11_smoke_failures.jsonl \
    --stdout-log notes/logs/round11_brbru_chart_conditioning_smoke_stdout.log
```

Full run, passed:

```bash
prlimit --as=7800000000 --rss=7800000000 -- \
  env PYTHONPATH=/root/pdgfits-private/src \
  PDGFITS_DATA_BACKEND=snapshot \
  PDGFITS_SNAPSHOT_DIR=/root/pdgfits-private/data/pdg-snapshot \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  /root/micromamba/micromamba run -r /root/micromamba-root -n pdg \
  python notes/codex-refactor/run_round11_brbru_chart_conditioning.py \
    --results-csv notes/codex-refactor/round11_brbru_chart_conditioning_results.csv \
    --results-jsonl notes/codex-refactor/round11_brbru_chart_conditioning_results.jsonl \
    --evaluations-csv notes/codex-refactor/round11_brbru_chart_conditioning_evaluations.csv \
    --evaluations-jsonl notes/codex-refactor/round11_brbru_chart_conditioning_evaluations.jsonl \
    --fixed-checks-csv notes/codex-refactor/round11_brbru_chart_conditioning_fixed_checks.csv \
    --fixed-checks-jsonl notes/codex-refactor/round11_brbru_chart_conditioning_fixed_checks.jsonl \
    --failures-csv notes/codex-refactor/round11_brbru_chart_conditioning_failures.csv \
    --failures-jsonl notes/codex-refactor/round11_brbru_chart_conditioning_failures.jsonl \
    --stdout-log notes/logs/round11_brbru_chart_conditioning_stdout.log
```

## Endpoint Results

All tested charts returned the same endpoint values as production to displayed
precision. All endpoint residuals stayed within the `5e-3` verification
tolerance.

| Case | Chart | Side | Endpoint delta | Prod residual | Chart residual | Prod method/nfev | Chart method/nfev | Min BRU | Max chart coord | Max original fitted coord |
| --- | --- | --- | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: |
| `B0::S042B95` | `logit` | lower | 0 | 0.001156 | 0.001156 | `trust-constr-exact-hess+KKT+descent-check` / 65 | `SLSQP+descent-check` / 160 | 3.39e-7 | 14.90 | 93.89 |
| `B0::S042B95` | `logit` | upper | 0 | -0.004138 | -0.004138 | `trust-constr-exact-hess+KKT` / 30 | `SLSQP` / 117 | 3.39e-7 | 14.90 | 93.89 |
| `B0::S042B95` | `log_positive` | lower | 0 | 0.001156 | 0.001156 | `trust-constr-exact-hess+KKT+descent-check` / 65 | `SLSQP+descent-check` / 182 | 3.39e-7 | 14.90 | 93.89 |
| `B0::S042B95` | `log_positive` | upper | 0 | -0.004138 | -0.004138 | `trust-constr-exact-hess+KKT` / 30 | `SLSQP` / 113 | 3.39e-7 | 14.90 | 93.89 |
| `B0::S042B95` | `direct_physical` | lower | 0 | 0.001156 | 0.001204 | `trust-constr-exact-hess+KKT+descent-check` / 65 | `projected-start+KKT+descent-check` / 67 | 3.39e-7 | 0.0508 | 93.89 |
| `B0::S042B95` | `direct_physical` | upper | 0 | -0.004138 | -0.004138 | `trust-constr-exact-hess+KKT` / 30 | `projected-start+KKT+descent-check` / 63 | 3.39e-7 | 0.0507 | 93.89 |
| `B0S-BR::S086.37` | `logit` | lower | 0 | -0.000857 | -0.000857 | `SLSQP+descent-check` / 102 | `SLSQP+descent-check` / 86 | 1.82e-5 | 10.92 | 1.75 |
| `B0S-BR::S086.37` | `logit` | upper | 0 | 0.001152 | 0.001152 | `SLSQP+descent-check` / 103 | `SLSQP+descent-check` / 86 | 1.82e-5 | 10.92 | 1.75 |
| `B0S-BR::S086.37` | `log_positive` | lower | 0 | -0.000857 | -0.000857 | `SLSQP+descent-check` / 102 | `SLSQP+descent-check` / 82 | 1.82e-5 | 10.92 | 1.75 |
| `B0S-BR::S086.37` | `log_positive` | upper | 0 | 0.001152 | 0.001152 | `SLSQP+descent-check` / 103 | `SLSQP+descent-check` / 81 | 1.82e-5 | 10.92 | 1.75 |
| `B0S-BR::S086.37` | `direct_physical` | lower | 0 | -0.000857 | -0.000857 | `SLSQP+descent-check` / 102 | `projected-start+KKT` / 23 | 1.82e-5 | 0.00595 | 1.75 |
| `B0S-BR::S086.37` | `direct_physical` | upper | 0 | 0.001152 | 0.001152 | `SLSQP+descent-check` / 103 | `trust-constr-exact-hess+KKT` / 19 | 1.82e-5 | 0.00631 | 1.75 |

The selected endpoint `nfev` numbers alone make direct physical coordinates look
attractive for `B0S-BR::S086.37`, but wall-clock/root cost does not: the direct
physical B0S root took `158.3s` versus `6.1s` for `log_positive` and `6.4s` for
`logit`. Direct physical B0 took `48.6s` versus about `2.4-2.5s` for the log
charts.

## Fixed Production-Endpoint Checks

The fixed checks reran each alternative chart at the production endpoint value,
which is the direct test for hidden lower constrained profiles at the same
reported endpoint.

| Case | Chart | Max lower chi2 improvement | Worst chart chi2 change | Material lower profiles |
| --- | --- | ---: | ---: | ---: |
| `B0::S042B95` | `logit` | 4.95e-9 | 4.95e-9 | 0 |
| `B0::S042B95` | `log_positive` | 4.91e-9 | -2.05e-12 | 0 |
| `B0::S042B95` | `direct_physical` | -4.83e-5 | -6.35e-4 | 0 |
| `B0S-BR::S086.37` | `logit` | 1.32e-12 | -2.19e-12 | 0 |
| `B0S-BR::S086.37` | `log_positive` | 3.13e-13 | 8.88e-14 | 0 |
| `B0S-BR::S086.37` | `direct_physical` | 1.32e-12 | 3.73e-13 | 0 |

Across all 12 fixed checks, the largest lower chi2 improvement was only
`4.95e-9`, far below the `1e-5` numerical-tie threshold and the `1e-4` material
threshold. No chart found an endpoint-correctness issue.

Direct physical coordinates were sometimes worse at the same fixed value. The
clearest example is `B0::S042B95` upper, where the direct physical fixed check
returned a profile chi2 `6.35e-4` higher than the production profile at the same
endpoint and had a very large chart-coordinate projected gradient. That is not a
scientific failure, but it is evidence against using direct physical coordinates
as a saturated-case replacement chart.

## Stationarity And Cost

For saturated `B0::S042B95`, log charts reduce the reported chart-coordinate
projected-gradient scale, but they do not produce a robust certificate or cost
win:

- Lower side production projected gradient in the original fitted chart:
  `4189`; logit/log-positive chart projected gradients: `1.40` and `1.45`.
- Both lower log-chart endpoints still require `+descent-check`.
- Selected endpoint `nfev` worsens from `65` to `160`/`182` on the lower side
  and from `30` to `117`/`113` on the upper side.
- The tiny BRU parameter remains physically tiny, about `3.39e-7`; the original
  arctan fitted coordinate is still about `93.9` when the physical point is
  mapped back into production coordinates.

For `B0S-BR::S086.37`, direct physical coordinates give cleaner selected
endpoint certificates (`KKT` rather than `descent-check`) and lower selected
endpoint `nfev` (`102/103` to `23/19`). This is a real local hint that a physical
chart can be nicer when the BRU parameters are not saturated. It is not a
production-ready replacement because it does not generalize to `B0::S042B95`,
and its root runtime is much worse in this harness.

## Answers

1. Did an alternative chart find a real endpoint-correctness issue or materially lower constrained profile?

No. All chart endpoints verified within the existing `5e-3` tolerance, all
endpoint values matched production to displayed precision, and the largest lower
fixed-target chi2 improvement was `4.95e-9`. There were `0 / 12` material lower
constrained profiles.

2. Did any chart produce the same endpoint with cleaner stationarity/certificates or meaningfully lower cost?

Partially, but not robustly. `direct_physical` gave cleaner KKT-style selected
certificates for the less-saturated `B0S-BR::S086.37` endpoints. The saturated
`B0::S042B95` lower side still needed descent-check under every tested chart,
and log/log-positive charts made selected endpoint `nfev` worse. Direct physical
coordinates were slow and less reliable on B0. There is no cross-case
simplification.

3. Which chart idea is rejected and why, quantitatively?

Rejected as production replacements:

- `logit` / `log_positive` for saturated BRU replacement: same endpoints and no
  lower chi2, but B0 selected endpoint `nfev` worsened from `65/30` to
  `160/117` and `182/113`; B0 lower still needed descent-check.
- `direct_physical` as a general BRU replacement: same endpoints, but B0 direct
  root took `48.6s`, B0S direct root took `158.3s`, and the B0 upper fixed check
  was worse than production by `6.35e-4` chi2 at the same endpoint.

4. Is any production change justified now?

No. The current arctan/BRU map is ugly in saturated cases, but the endpoints are
stable and verified. The alternatives did not find a scientific correctness
issue or a robust certificate/cost improvement. Keep production unchanged.

5. What should the next supervised round do if the overnight loop continues?

Do not pursue a blanket BR/BRU chart replacement. If another method-change spike
is needed, the only chart-related direction with a remaining signal is a
reduced-coordinate physical-profile experiment for well-conditioned BRU cases
like `B0S-BR::S086.37`, explicitly treating it as non-general unless it also
handles saturated `B0::S042B95`. Otherwise, move away from numerical chart
experiments and synthesize the negative evidence or examine statistical boundary
regularity/Wilks risk for tiny BRU parameters.

## Bottom Line

Round 11 rejects chart replacement quantitatively. Alternative charts reproduce
the current endpoints and do not expose a hidden lower profile. The current
arctan chart remains numerically unattractive for `B0::S042B95`, but the evidence
continues to support "bad conditioning with stable verified endpoints," not a
production method change.
