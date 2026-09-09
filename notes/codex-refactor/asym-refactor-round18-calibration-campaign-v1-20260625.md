# Asym Refactor Round 18 Calibration Campaign v1

Date: 2026-06-25

Production code changed: no. `src/pdgfits/build_chi2.py` was not edited.

## Question

Round 17 made the scientifically honest next step explicit: stop adding
optimizer fallbacks and run a larger, DGP-labeled calibration campaign for the
five core risk/control cases. The primary statistic is

```text
q = chi2_profile(truth_target) - chi2_min_toy
```

Numerical endpoint correctness is separate: endpoint roots are valid only when
the profiled endpoint chi2 verifies `chi2_min + 1` within the endpoint residual
tolerance.

## Design

Cases:

| Case | Stratum |
| --- | --- |
| `B0::S042B95` | boundary/tiny-parameter |
| `B0S-BR::S086.37` | asymmetric/profile-shape |
| `eta_c J/psi psi(2S)::M026G01` | asymmetric/profile-shape |
| `G(2000),G(1800)::K002M` | clean control |
| `G(2000),G(1800)::K003M` | clean control |

DGPs:

- `local_gaussian_sym`;
- `split_normal_pdg_resid`.

Run size:

- `500` toys per case/DGP;
- `5` cases x `2` DGPs x `500` toys = `5000` toy rows;
- endpoint roots on toy indices `0`, `100`, `200`, `300`, `400` for each
  case/DGP = `50` endpoint toy rows = `100` endpoint-side rows;
- base seed `2026062518`;
- endpoint residual tolerance `5e-3`;
- runtime `8885.49s` by harness metadata, `2:28:07` wall clock by
  `/usr/bin/time -v`;
- max RSS `1,312,712 KB`, well below the 8 GB cap.

Reference chi-square-1 thresholds:

| Quantity | Expected |
| --- | ---: |
| `P(q <= median)` at `0.454936423119572` | `0.500` |
| `P(q <= 1)` | `0.682689` |
| `P(q > q90)` at `2.705543454095404` | `0.100` |
| `P(q > q95)` at `3.841458820694124` | `0.050` |

Intervals below are Wilson 95% binomial intervals.

## Artifacts

| Artifact | Rows |
| --- | ---: |
| `notes/codex-refactor/round18_calibration_campaign_v1_toys.csv` | 5000 |
| `notes/codex-refactor/round18_calibration_campaign_v1_toys.jsonl` | 5000 |
| `notes/codex-refactor/round18_calibration_campaign_v1_endpoints.csv` | 100 |
| `notes/codex-refactor/round18_calibration_campaign_v1_endpoints.jsonl` | 100 |
| `notes/codex-refactor/round18_calibration_campaign_v1_failures.csv` | 0 |
| `notes/codex-refactor/round18_calibration_campaign_v1_failures.jsonl` | 0 |
| `notes/codex-refactor/round18_calibration_campaign_v1_summary.json` | 1 summary object |

Raw stdout/stderr and `/usr/bin/time -v` output are under
`notes/logs/codex-round18-calibration-campaign-v1-run-20260625.log` and are
not part of the committed artifact set.

## Commands

Syntax check:

```bash
env PYTHONPATH=/root/pdgfits-private/src \
  PDGFITS_DATA_BACKEND=snapshot \
  PDGFITS_SNAPSHOT_DIR=/root/pdgfits-private/data/pdg-snapshot \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  /root/micromamba/micromamba run -r /root/micromamba-root -n pdg \
  python -m py_compile notes/codex-refactor/run_round16_calibration_pilot.py
```

Campaign:

```bash
/usr/bin/time -v prlimit --as=7800000000 --rss=7800000000 -- \
  env PYTHONPATH=/root/pdgfits-private/src \
  PDGFITS_DATA_BACKEND=snapshot \
  PDGFITS_SNAPSHOT_DIR=/root/pdgfits-private/data/pdg-snapshot \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  /root/micromamba/micromamba run -r /root/micromamba-root -n pdg \
  python notes/codex-refactor/run_round16_calibration_pilot.py \
    --toys-per-case-dgp 500 \
    --endpoint-toys-per-case-dgp 0 \
    --endpoint-stride 100 \
    --endpoint-stride-max-toys 5 \
    --base-seed 2026062518 \
    --max-runtime-sec 12000 \
    --overwrite \
    --toy-csv notes/codex-refactor/round18_calibration_campaign_v1_toys.csv \
    --toy-jsonl notes/codex-refactor/round18_calibration_campaign_v1_toys.jsonl \
    --endpoint-csv notes/codex-refactor/round18_calibration_campaign_v1_endpoints.csv \
    --endpoint-jsonl notes/codex-refactor/round18_calibration_campaign_v1_endpoints.jsonl \
    --failure-csv notes/codex-refactor/round18_calibration_campaign_v1_failures.csv \
    --failure-jsonl notes/codex-refactor/round18_calibration_campaign_v1_failures.jsonl \
    --summary-json notes/codex-refactor/round18_calibration_campaign_v1_summary.json \
  > notes/logs/codex-round18-calibration-campaign-v1-run-20260625.log 2>&1
```

A focused deterministic rerun of the one negative-q row
(`B0::S042B95`, split-normal toy `84`, seed `2026162602`) reproduced the same
`q=-0.0010571834498449562`.

## Numerical Endpoint Diagnostics

Endpoint-root correctness held on the predeclared subset:

- harness failure rows: `0`;
- endpoint-side rows: `100`;
- endpoint toy rows: `50`;
- endpoint profile `success=False`: `0`;
- endpoint residuals outside `5e-3`: `0`;
- max absolute endpoint residual: `0.0049933978126581735`.

Endpoint profile methods:

| Method | Endpoint-side rows |
| --- | ---: |
| `SLSQP` | 55 |
| `SLSQP+descent-check` | 26 |
| `trust-constr-exact-hess+KKT` | 18 |
| `projected-start+descent-check` | 1 |

Largest endpoint residual:

| DGP | Case | Toy | Side | Residual | Method |
| --- | --- | ---: | --- | ---: | --- |
| split normal | `B0S-BR::S086.37` | 200 | lower | `-0.0049933978126581735` | `SLSQP+descent-check` |

One q-profile consistency failure did appear, but it was not an endpoint-root
failure and was not in the endpoint subset:

| DGP | Case | Toy | Seed | q | Profile method | Refit |
| --- | --- | ---: | ---: | ---: | --- | --- |
| split normal | `B0::S042B95` | 84 | 2026162602 | `-0.0010571834498449562` | `SLSQP` | Minuit valid/accurate |

Details for that row:

- profiled chi2 at truth target: `76.24028994923016`;
- toy refit chi2 minimum: `76.24134713268`;
- scaled constraint violation: `7.27e-15`;
- projected gradient norm: `0.002125`;
- Minuit EDM: `1.25e-08`;
- chart saturated at the toy MLE: min decay parameter `3.48e-7`,
  max absolute fitted decay coordinate `91.51`.

Interpretation: this is a refit/profile consistency warning in one
chart-saturated B0 toy, not evidence that any reported endpoint root crossed at
the wrong chi2 level. It should be kept visible if these toys are used for a
publication-grade calibration study.

## Q Results By Case

| DGP | Case | `q <= med` | `q <= 1` | `q > q90` | `q > q95` | Median q | Max q | Chart sat. |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| local Gaussian | `B0::S042B95` | 257/500 (0.514, [0.470,0.558]) | 344/500 (0.688, [0.646,0.727]) | 50/500 (0.100, [0.077,0.129]) | 30/500 (0.060, [0.042,0.084]) | 0.428 | 13.722 | 500/500 |
| local Gaussian | `B0S-BR::S086.37` | 221/500 (0.442, [0.399,0.486]) | 322/500 (0.644, [0.601,0.685]) | 55/500 (0.110, [0.085,0.140]) | 19/500 (0.038, [0.024,0.059]) | 0.583 | 8.787 | 0/500 |
| local Gaussian | `eta_c...::M026G01` | 239/500 (0.478, [0.435,0.522]) | 327/500 (0.654, [0.611,0.694]) | 54/500 (0.108, [0.084,0.138]) | 25/500 (0.050, [0.034,0.073]) | 0.488 | 7.496 | 0/500 |
| local Gaussian | `G...::K002M` | 275/500 (0.550, [0.506,0.593]) | 354/500 (0.708, [0.667,0.746]) | 39/500 (0.078, [0.058,0.105]) | 15/500 (0.030, [0.018,0.049]) | 0.359 | 10.447 | 0/500 |
| local Gaussian | `G...::K003M` | 233/500 (0.466, [0.423,0.510]) | 340/500 (0.680, [0.638,0.719]) | 49/500 (0.098, [0.075,0.127]) | 28/500 (0.056, [0.039,0.080]) | 0.512 | 15.854 | 0/500 |
| split normal | `B0::S042B95` | 265/500 (0.530, [0.486,0.573]) | 354/500 (0.708, [0.667,0.746]) | 47/500 (0.094, [0.071,0.123]) | 21/500 (0.042, [0.028,0.063]) | 0.405 | 15.358 | 500/500 |
| split normal | `B0S-BR::S086.37` | 256/500 (0.512, [0.468,0.556]) | 350/500 (0.700, [0.658,0.739]) | 33/500 (0.066, [0.047,0.091]) | 14/500 (0.028, [0.017,0.046]) | 0.417 | 7.412 | 0/500 |
| split normal | `eta_c...::M026G01` | 229/500 (0.458, [0.415,0.502]) | 332/500 (0.664, [0.621,0.704]) | 44/500 (0.088, [0.066,0.116]) | 21/500 (0.042, [0.028,0.063]) | 0.513 | 11.659 | 0/500 |
| split normal | `G...::K002M` | 234/500 (0.468, [0.425,0.512]) | 339/500 (0.678, [0.636,0.717]) | 48/500 (0.096, [0.073,0.125]) | 28/500 (0.056, [0.039,0.080]) | 0.538 | 11.803 | 0/500 |
| split normal | `G...::K003M` | 261/500 (0.522, [0.478,0.565]) | 355/500 (0.710, [0.669,0.748]) | 44/500 (0.088, [0.066,0.116]) | 26/500 (0.052, [0.036,0.075]) | 0.405 | 11.636 | 0/500 |

Aggregated by stratum:

| DGP | Stratum | `q <= med` | `q <= 1` | `q > q90` | `q > q95` | Median q | Max q |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| local Gaussian | asymmetric/profile-shape | 460/1000 (0.460, [0.429,0.491]) | 649/1000 (0.649, [0.619,0.678]) | 109/1000 (0.109, [0.091,0.130]) | 44/1000 (0.044, [0.033,0.059]) | 0.535 | 8.787 |
| local Gaussian | boundary/tiny-parameter | 257/500 (0.514, [0.470,0.558]) | 344/500 (0.688, [0.646,0.727]) | 50/500 (0.100, [0.077,0.129]) | 30/500 (0.060, [0.042,0.084]) | 0.428 | 13.722 |
| local Gaussian | clean controls | 508/1000 (0.508, [0.477,0.539]) | 694/1000 (0.694, [0.665,0.722]) | 88/1000 (0.088, [0.072,0.107]) | 43/1000 (0.043, [0.032,0.057]) | 0.443 | 15.854 |
| split normal | asymmetric/profile-shape | 485/1000 (0.485, [0.454,0.516]) | 682/1000 (0.682, [0.652,0.710]) | 77/1000 (0.077, [0.062,0.095]) | 35/1000 (0.035, [0.025,0.048]) | 0.485 | 11.659 |
| split normal | boundary/tiny-parameter | 265/500 (0.530, [0.486,0.573]) | 354/500 (0.708, [0.667,0.746]) | 47/500 (0.094, [0.071,0.123]) | 21/500 (0.042, [0.028,0.063]) | 0.405 | 15.358 |
| split normal | clean controls | 495/1000 (0.495, [0.464,0.526]) | 694/1000 (0.694, [0.665,0.722]) | 92/1000 (0.092, [0.076,0.112]) | 54/1000 (0.054, [0.042,0.070]) | 0.472 | 11.803 |

## Required Answers

1. Did any numerical endpoint-correctness failure appear?

No endpoint-root correctness failure appeared. Endpoint-side rows were
`100/100` profile-successful, `0/100` exceeded the `5e-3` residual tolerance,
and the maximum endpoint residual was `0.0049933978126581735`.

There was one q-profile consistency failure outside the endpoint subset:
`B0::S042B95`, split-normal toy `84`, seed `2026162602`, with
`q=-0.0010571834498449562`. It was reproduced deterministically. This is a
numerical warning for toy refit/profile consistency in a chart-saturated B0
toy, but it is not an endpoint root at the wrong profile level.

2. Does the `B0S-BR::S086.37` q-tail/median-shift survive larger toy counts and
DGP choice?

The high-q tail does not survive as a robust DGP-invariant signal. Under
local-Gaussian toys, `q > q90` is `55/500` (0.110, Wilson
[0.085,0.140]) and `q > q95` is `19/500` (0.038, [0.024,0.059]), so the tail
is near or below chi-square-1 expectation. The local-Gaussian center/median
shift is more persistent: `q <= median` is `221/500` (0.442,
[0.399,0.486]) and `q <= 1` is `322/500` (0.644, [0.601,0.685]).

Under split-normal toys, the B0S shift mostly disappears or reverses:
`q <= median` is `256/500` (0.512, [0.468,0.556]),
`q <= 1` is `350/500` (0.700, [0.658,0.739]), and tail counts are below
nominal (`33/500` above q90, `14/500` above q95). The round-17 conclusion of
DGP sensitivity is strengthened.

3. Does eta show stable asymmetric/profile-shape miscalibration, or only
volatility/outliers?

Eta shows mild profile-shape volatility and outliers, not a stable
miscalibration case. Local-Gaussian eta has q90/q95 counts essentially nominal:
`54/500` above q90 and `25/500` above q95. Split-normal eta is also near or
below nominal in tails: `44/500` above q90 and `21/500` above q95. The median
side is mildly high-q in both DGPs (`q <= median` `239/500` and `229/500`), but
the Wilson intervals touch or nearly touch 0.5. This is a risk flag, not a
production-change signal.

4. Does persistent B0 boundary/chart activation translate into stable
q-calibration symptoms?

No. B0 chart saturation is perfectly persistent (`1000/1000` toys across both
DGPs), but q calibration is close to chi-square-1 expectations. Local-Gaussian
B0 is `50/500` above q90 and `30/500` above q95; split-normal B0 is `47/500`
above q90 and `21/500` above q95. The boundary/chart issue remains a Wilks
regularity warning and now also owns the single profile-consistency failure,
but it does not translate into a stable q-tail or median-shift symptom.

5. Are clean controls within pilot/campaign binomial noise?

Yes, in aggregate. Local-Gaussian clean controls have `88/1000` above q90 and
`43/1000` above q95, with Wilson intervals containing the chi-square
expectations. Split-normal clean controls have `92/1000` above q90 and
`54/1000` above q95, also compatible. Individual clean controls still produce
large maxima (`15.854`, `11.803`, `11.636`), so isolated high q values are not
by themselves evidence of case-specific miscalibration.

6. Are any production changes justified?

No production solver, chart, or asymmetric-chi2 change is justified by this
round. Endpoint residual verification held on the endpoint subset, and the
statistical calibration symptoms are DGP-sensitive or within campaign-scale
binomial uncertainty. The one B0 split-normal profile-consistency failure is
worth follow-up, but it does not justify changing production endpoint logic in
this round.

7. What is the statistically honest next step?

For a near-term methods decision, stop and synthesize: rounds 7-18 have found
strong negative evidence against an endpoint-correctness or solver-change
story.

If calibration evidence itself remains important, the next step should not be a
new optimizer fallback. It should be one of:

- audit the single B0 split-normal profile-consistency failure with a targeted
  refit/globality diagnostic;
- add a recentered split-normal sensitivity DGP, since the current split-normal
  DGP is mode-centered and can shift marginal means;
- continue only the scientifically live strata to `1000-2000` toys per
  case/DGP, especially B0S and eta, with the clean controls retained for
  comparison;
- add an average/control stratum if Anthony wants to connect this fit-side
  calibration evidence to the broader PDG workload.

## Bottom Line

Round 18 completed the preferred 500-toy campaign v1. No endpoint-root
correctness failure appeared. One chart-saturated B0 split-normal toy produced
a reproducible q-profile consistency failure of about `-0.00106`, which should
be documented but does not implicate endpoint-root residual verification.

Scientifically, the earlier B0S high-tail concern mostly collapses into a
local-Gaussian median/CDF shift that is not reproduced by the split-normal DGP.
Eta remains profile-shape volatile but not clearly miscalibrated. B0 remains
boundary-saturated in every toy, but its q distribution is close to nominal
apart from the one consistency warning. Clean controls are compatible with
campaign-scale binomial noise. Production changes are not justified.
