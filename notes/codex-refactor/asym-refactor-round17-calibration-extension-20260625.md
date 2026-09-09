# Asym Refactor Round 17 Calibration Extension

Date: 2026-06-25

Production code changed: no. `src/pdgfits/build_chi2.py` was not edited.

## Question

Round 16 found no numerical endpoint failure, but left a small, DGP-sensitive
calibration signal for `B0S-BR::S086.37` under local-Gaussian toys. This round
asked whether that signal survives a slightly more honest pilot that includes
the missing round-13 asymmetric/profile-shape case
`eta_c J/psi psi(2S)::M026G01`.

This is still a pilot, not a coverage study.

## Design

The round-16 harness was extended rather than replaced:

- added `fit_eta_m026g01` for `eta_c J/psi psi(2S)::M026G01`;
- added endpoint subset selection by stride;
- added q-threshold binomial summaries with standard errors and Wilson 95%
  intervals to the summary JSON.

Cases:

| Case | Stratum |
| --- | --- |
| `B0::S042B95` | boundary/tiny-parameter |
| `B0S-BR::S086.37` | asymmetric/profile-shape |
| `eta_c J/psi psi(2S)::M026G01` | asymmetric/profile-shape |
| `G(2000),G(1800)::K002M` | clean control |
| `G(2000),G(1800)::K003M` | clean control |

DGPs:

- `local_gaussian_sym`: fitted model mean plus correlated Gaussian noise using
  zero-residual symmetrized reported errors.
- `split_normal_pdg_resid`: Gaussian-copula two-piece-normal toy using the
  production residual sign convention.

Run size:

- `5` cases x `2` DGPs x `64` toys = `640` toy rows;
- endpoints on toy indices `0`, `16`, `32`, and `48` for each case/DGP =
  `40` endpoint roots = `80` endpoint-side rows;
- base seed `2026062517`;
- runtime `1292.39s` with one worker and 8 GB `prlimit`.

Reference `chi2_1` thresholds:

| Quantity | Expected |
| --- | ---: |
| `P(q <= median)` | `0.500` |
| `P(q <= 1)` | `0.6827` |
| `P(q > q90)` | `0.100` |
| `P(q > q95)` | `0.050` |

## Artifacts

| Artifact | Rows |
| --- | ---: |
| `notes/codex-refactor/round17_calibration_extension_toys.csv` | 640 |
| `notes/codex-refactor/round17_calibration_extension_toys.jsonl` | 640 |
| `notes/codex-refactor/round17_calibration_extension_endpoints.csv` | 80 |
| `notes/codex-refactor/round17_calibration_extension_endpoints.jsonl` | 80 |
| `notes/codex-refactor/round17_calibration_extension_failures.csv` | 0 |
| `notes/codex-refactor/round17_calibration_extension_failures.jsonl` | 0 |
| `notes/codex-refactor/round17_calibration_extension_summary.json` | 1 summary object |

The only raw dry-run outputs are under `notes/logs/` and are intentionally not
part of the durable artifact set.

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

Full run:

```bash
prlimit --as=7800000000 --rss=7800000000 -- \
  env PYTHONPATH=/root/pdgfits-private/src \
  PDGFITS_DATA_BACKEND=snapshot \
  PDGFITS_SNAPSHOT_DIR=/root/pdgfits-private/data/pdg-snapshot \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  /root/micromamba/micromamba run -r /root/micromamba-root -n pdg \
  python notes/codex-refactor/run_round16_calibration_pilot.py \
    --toys-per-case-dgp 64 \
    --endpoint-toys-per-case-dgp 0 \
    --endpoint-stride 16 \
    --endpoint-stride-max-toys 4 \
    --base-seed 2026062517 \
    --max-runtime-sec 3600 \
    --overwrite \
    --toy-csv notes/codex-refactor/round17_calibration_extension_toys.csv \
    --toy-jsonl notes/codex-refactor/round17_calibration_extension_toys.jsonl \
    --endpoint-csv notes/codex-refactor/round17_calibration_extension_endpoints.csv \
    --endpoint-jsonl notes/codex-refactor/round17_calibration_extension_endpoints.jsonl \
    --failure-csv notes/codex-refactor/round17_calibration_extension_failures.csv \
    --failure-jsonl notes/codex-refactor/round17_calibration_extension_failures.jsonl \
    --summary-json notes/codex-refactor/round17_calibration_extension_summary.json
```

## Numerical Endpoint Diagnostics

No numerical endpoint-correctness failure appeared.

Counts:

- toy rows: `640`;
- endpoint-side rows: `80`;
- failure rows: `0`;
- profile consistency failures (`q < -1e-4`): `0`;
- endpoint residuals outside the `5e-3` tolerance: `0`;
- maximum absolute endpoint residual: `0.004916835028730304`.

Endpoint profile methods:

| Method | Endpoint-side rows |
| --- | ---: |
| `SLSQP` | 50 |
| `SLSQP+descent-check` | 21 |
| `trust-constr-exact-hess+KKT` | 9 |

The largest residual was `0.004916835` on
`B0::S042B95`, local-Gaussian toy 32 upper, using
`trust-constr-exact-hess+KKT`. It is inside the current tolerance.

There was one raw negative q row:
`eta_c...::M026G01`, split-normal toy 54, `q=-2.303e-05`. It is above the
predeclared `-1e-4` profile-consistency failure cutoff and had scaled
constraint violation `7.19e-13`, so it is treated as numerical tolerance noise,
not a failure.

One B0 local-Gaussian toy used the scipy fallback after Minuit did not validate
(`toy 4`, q `0.4373`); it still produced a normal profile row and no failure.

## Q Threshold Results

Entries are `count/64 (fraction +- binomial SE)`.

| DGP | Case | `q <= med` | `q <= 1` | `q > q90` | `q > q95` | Median q | Max q | Chart sat. |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| local Gaussian | `B0::S042B95` | `41/64 (0.641 +- 0.060)` | `46/64 (0.719 +- 0.056)` | `6/64 (0.094 +- 0.036)` | `5/64 (0.078 +- 0.034)` | `0.305` | `13.722` | `64/64` |
| local Gaussian | `B0S-BR::S086.37` | `26/64 (0.406 +- 0.061)` | `43/64 (0.672 +- 0.059)` | `7/64 (0.109 +- 0.039)` | `1/64 (0.016 +- 0.016)` | `0.645` | `6.666` | `0/64` |
| local Gaussian | `eta_c...::M026G01` | `30/64 (0.469 +- 0.062)` | `41/64 (0.641 +- 0.060)` | `8/64 (0.125 +- 0.041)` | `4/64 (0.062 +- 0.030)` | `0.553` | `7.496` | `0/64` |
| local Gaussian | `G...::K002M` | `32/64 (0.500 +- 0.062)` | `41/64 (0.641 +- 0.060)` | `8/64 (0.125 +- 0.041)` | `3/64 (0.047 +- 0.026)` | `0.437` | `4.894` | `0/64` |
| local Gaussian | `G...::K003M` | `27/64 (0.422 +- 0.062)` | `41/64 (0.641 +- 0.060)` | `4/64 (0.062 +- 0.030)` | `4/64 (0.062 +- 0.030)` | `0.626` | `6.329` | `0/64` |
| split normal | `B0::S042B95` | `34/64 (0.531 +- 0.062)` | `48/64 (0.750 +- 0.054)` | `4/64 (0.062 +- 0.030)` | `0/64 (0.000 +- 0.000)` | `0.403` | `3.709` | `64/64` |
| split normal | `B0S-BR::S086.37` | `34/64 (0.531 +- 0.062)` | `47/64 (0.734 +- 0.055)` | `1/64 (0.016 +- 0.016)` | `1/64 (0.016 +- 0.016)` | `0.399` | `4.285` | `0/64` |
| split normal | `eta_c...::M026G01` | `26/64 (0.406 +- 0.061)` | `40/64 (0.625 +- 0.061)` | `6/64 (0.094 +- 0.036)` | `3/64 (0.047 +- 0.026)` | `0.706` | `11.659` | `0/64` |
| split normal | `G...::K002M` | `30/64 (0.469 +- 0.062)` | `46/64 (0.719 +- 0.056)` | `8/64 (0.125 +- 0.041)` | `6/64 (0.094 +- 0.036)` | `0.485` | `9.853` | `0/64` |
| split normal | `G...::K003M` | `38/64 (0.594 +- 0.061)` | `49/64 (0.766 +- 0.053)` | `2/64 (0.031 +- 0.022)` | `2/64 (0.031 +- 0.022)` | `0.231` | `6.013` | `0/64` |

Aggregated by stratum:

| DGP | Stratum | `q <= 1` | `q > q90` | `q > q95` | Median q | Max q |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| local Gaussian | asymmetric/profile-shape | `84/128 (0.656 +- 0.042)` | `15/128 (0.117 +- 0.028)` | `5/128 (0.039 +- 0.017)` | `0.601` | `7.496` |
| local Gaussian | boundary/tiny-parameter | `46/64 (0.719 +- 0.056)` | `6/64 (0.094 +- 0.036)` | `5/64 (0.078 +- 0.034)` | `0.305` | `13.722` |
| local Gaussian | clean controls | `82/128 (0.641 +- 0.042)` | `12/128 (0.094 +- 0.026)` | `7/128 (0.055 +- 0.020)` | `0.527` | `6.329` |
| split normal | asymmetric/profile-shape | `87/128 (0.680 +- 0.041)` | `7/128 (0.055 +- 0.020)` | `4/128 (0.031 +- 0.015)` | `0.518` | `11.659` |
| split normal | boundary/tiny-parameter | `48/64 (0.750 +- 0.054)` | `4/64 (0.062 +- 0.030)` | `0/64 (0.000 +- 0.000)` | `0.403` | `3.709` |
| split normal | clean controls | `95/128 (0.742 +- 0.039)` | `10/128 (0.078 +- 0.024)` | `8/128 (0.062 +- 0.021)` | `0.335` | `9.853` |

Wilson intervals for every threshold are recorded in
`round17_calibration_extension_summary.json`.

## Interpretation

### Endpoint correctness

The numerical invariant held on the endpoint subset. There were no failure rows,
no endpoint residuals outside tolerance, no failed endpoint profile statuses,
and no q consistency failures. The results do not justify a solver, parameter
map, or asymmetric-chi2 production change.

### `B0S-BR::S086.37`

The round-15/16 B0S high-tail signal does not survive as a robust high-q tail.

Under local-Gaussian toys, `q > q90` was `7/64`, which is close to the
chi-square expectation of `6.4/64`, and `q > q95` was only `1/64`, below the
expected `3.2/64`. There is still a mild center-shift symptom:
`q <= median` was `26/64`, lower than the expected `32/64`, and the median q
was `0.645`. With binomial SE `0.061`, this is suggestive but not strong.

Under split-normal toys, the tail mostly disappeared: `q > q90` was `1/64` and
`q <= 1` was `47/64`. This confirms the round-16 conclusion that the B0S signal
is DGP-sensitive and that the earlier `3/16` local-Gaussian tail was mostly
pilot volatility, not a stable calibration failure.

### `eta_c J/psi psi(2S)::M026G01`

Adding eta did not reveal a numerical endpoint problem. It did show profile
volatility, but not a clean calibration failure.

Under local-Gaussian toys, eta had `8/64` above q90 and `4/64` above q95,
basically near the chi-square expectations. Under split-normal toys, eta had
`6/64` above q90 and `3/64` above q95, again near expectation. The median q was
high in both DGPs (`0.553`, `0.706`), and there were a few large outliers,
including max `11.659` under split-normal toys.

So eta is qualitatively profile-shape/asymmetry-sensitive, but it is not the
same pattern as B0S. B0S's high-q tail weakens under split-normal toys; eta's
tail counts stay near nominal under both DGPs while the median remains somewhat
high. With 64 toys, this is a useful risk flag, not evidence for a production
method change.

### `B0::S042B95`

B0 remains the clearest boundary/chart-activation case:

| DGP | Chart-saturated toys | Min BR/BRU coordinate | Max fitted coordinate |
| --- | ---: | ---: | ---: |
| local Gaussian | `64/64` | `2.45e-7` | `130.0` |
| split normal | `64/64` | `2.94e-7` | `108.3` |

That activation still does not translate into a stable q-calibration symptom.
The local-Gaussian run had some large outliers (`q_max=13.722`, `5/64` above
q95), but split-normal had `0/64` above q95 and `q_max=3.709`. The durable
finding is boundary/chart activation, not a reproducible q-calibration failure.

### Clean controls

The clean controls behave plausibly in aggregate, but they are noisy at
`64` toys per case. Local-Gaussian controls combined had `12/128` above q90 and
`7/128` above q95, close to expectation. Split-normal controls combined had
`10/128` above q90 and `8/128` above q95, also broadly plausible. Individual
control cases still produced large maxima (`9.853` for `K002M` under
split-normal), which is a useful reminder not to overinterpret single-case
pilot tails.

## Required Answers

1. Did any numerical endpoint-correctness failure appear?

No. There were `0` failure rows, `0` endpoint residuals outside tolerance,
`0` endpoint profile failures, and `0` profile consistency failures. The max
absolute endpoint residual was `0.004916835`, inside the current `5e-3`
tolerance.

2. Does the `B0S-BR::S086.37` q-tail survive more toys and DGP choice?

No, not as a robust tail. The local-Gaussian q90 count became `7/64`, close to
nominal, and the split-normal q90 count was only `1/64`. The earlier signal was
mostly pilot noise plus DGP sensitivity. A mild median/CDF shift remains worth
remembering, but it is not a production-change signal.

3. Does adding `eta_c...::M026G01` show the same asymmetric/profile-shape q
behavior as B0S?

Not exactly. Eta has profile-shape volatility and some large q outliers, but
its q90/q95 tail counts are near chi-square expectations under both DGPs. It
does not replicate the B0S DGP-sensitive high-tail pattern.

4. Does `B0::S042B95` boundary activation translate into q-calibration symptoms?

Mostly no. Boundary/chart activation is persistent (`128/128` toys), but q
symptoms are DGP-sensitive: local-Gaussian has several high outliers, while
split-normal has no q95 exceedance. This remains a Wilks regularity warning,
not an endpoint or calibration failure.

5. Are clean controls behaving as expected within pilot-scale binomial noise?

Yes, in aggregate. They are not perfectly quiet, and individual controls have
tail outliers, but combined q90/q95 counts are close enough to chi-square
expectation for a `128`-toy pilot. This reinforces the need for larger samples
before making coverage claims.

6. Are production changes justified?

No. The numerical endpoint invariant held, and the calibration symptoms are
pilot-scale and DGP-sensitive. There is no justified solver, chart, or
`build_chi2.py` change.

7. What is the statistically honest next step?

Stop numerical-method grinding and synthesize the evidence if Anthony needs a
near-term decision. If calibration evidence itself is important, run a larger
DGP-labeled calibration campaign, not another solver round. A reasonable next
campaign is at least `500` toys per case/DGP for the five core cases, with the
same predeclared q thresholds, endpoint checks on a fixed subset, and explicit
reporting of DGP sensitivity. Publication-grade tail claims likely need
`1000-2000` toys per case/DGP.

## Bottom Line

Round 17 found no endpoint-correctness failure. The B0S tail that motivated the
extension mostly collapses when expanded to 64 toys and compared across DGPs.
The added eta case is profile-shape volatile but not a clean miscalibration
case. B0 remains boundary-saturated in every toy, but its q symptoms are not
stable across DGPs. Production changes are not justified; the honest next step
is synthesis or a larger calibration campaign.
