# Asym Refactor Round 16 Calibration Study Design

Date: 2026-06-25

Production code changed: no. `src/pdgfits/build_chi2.py` was not edited.

## Question

Round 15 found no numerical endpoint-correctness failure, but it did show a
small calibration-smoke signal in `B0S-BR::S086.37`: `q <= 1` was `5/8`, with
two q values near 4. The boundary/tiny-parameter case `B0::S042B95` stayed
chart-saturated but did not look worse than a clean control in only 8 toys.

This round asked whether that is enough evidence to justify a serious later
calibration study, and what the design should be.

## Round 15 Inspection

Round-15 artifacts were checked directly:

| Artifact | Lines | Data rows |
| --- | ---: | ---: |
| `round15_wilks_smoke_toys.csv` | 25 | 24 |
| `round15_wilks_smoke_endpoints.csv` | 13 | 12 |
| `round15_wilks_smoke_failures.csv` | 1 | 0 |
| `round15_wilks_smoke_toys.jsonl` | 24 | 24 |
| `round15_wilks_smoke_endpoints.jsonl` | 12 | 12 |
| `round15_wilks_smoke_failures.jsonl` | 0 | 0 |

The report and summary JSON agree: 3 cases x 8 toys = 24 toy rows, endpoints
for the first 2 toys per case = 12 endpoint rows, and 0 failures.

Important limitations in the round-15 harness:

- only one DGP was used: local Gaussian toys with errors symmetrized at zero
  residual;
- the DGP was a pragmatic sampling model, not the full asymmetric chi2
  interpreted as a normalized generative likelihood;
- only 8 toys per case were run;
- endpoint computation was intentionally sparse and diagnostic;
- boundary metrics were recorded at the toy MLE, not at every profiled endpoint
  point.

## Round 16 Pilot Design

I added a notes-only harness:

- `notes/codex-refactor/run_round16_calibration_pilot.py`

It reuses the round-15 refit/profile workflow but adds `dgp` as an explicit
factor and writes resumable artifacts.

Cases:

| Case | Stratum | Reason |
| --- | --- | --- |
| `B0::S042B95` | boundary/tiny-parameter | round-13 high-risk BRU boundary/chart-saturation case |
| `B0S-BR::S086.37` | asymmetric/profile-shape | round-15 q-tail smoke signal and round-13 medium-high profile-shape risk |
| `G(2000),G(1800)::K002M` | clean control | locally quadratic interior direct-coordinate control |
| `G(2000),G(1800)::K003M` | clean control | second locally quadratic interior coordinate in the same clean fit |

DGPs:

| DGP | Definition | Intended use |
| --- | --- | --- |
| `local_gaussian_sym` | Round-15 DGP: fitted model mean plus correlated Gaussian noise using `2 sigma_minus sigma_plus / (sigma_minus + sigma_plus)` and eigenvalue-clipped covariance. | Continuity with round 15 and local quadratic reference. |
| `split_normal_pdg_resid` | Gaussian-copula two-piece-normal toy. Positive production residual `y - mu` uses `error_n`; negative residual uses `error_p`, matching the sign convention of the PDG chi2 residual. | Asymmetric-error-inspired sensitivity check. |

The split-normal DGP is deliberately described as "asymmetric-error-inspired",
not exact. It is mode-centered at the fitted model mean; asymmetric scales imply
nonzero marginal means. It is not a normalized likelihood derived from the
production asymmetric chi2.

## Artifacts

| Artifact | Rows |
| --- | ---: |
| `notes/codex-refactor/round16_calibration_pilot_toys.csv` | 128 |
| `notes/codex-refactor/round16_calibration_pilot_toys.jsonl` | 128 |
| `notes/codex-refactor/round16_calibration_pilot_endpoints.csv` | 16 |
| `notes/codex-refactor/round16_calibration_pilot_endpoints.jsonl` | 16 |
| `notes/codex-refactor/round16_calibration_pilot_failures.csv` | 0 |
| `notes/codex-refactor/round16_calibration_pilot_failures.jsonl` | 0 |
| `notes/codex-refactor/round16_calibration_pilot_summary.json` | 1 summary object |

Pilot size: 4 cases x 2 DGPs x 16 toys = 128 toy rows. Endpoint roots were
computed for toy 0 in each case/DGP, giving 8 roots = 16 endpoint-side rows.

Runtime: `227.39s` for the full pilot with one worker and an 8 GB `prlimit`
cap. Full run cap was `900s`.

Seeds: base seed `2026062516`, with deterministic offsets
`base + 1_000_000 * case_index + 100_000 * dgp_index + toy_index`.

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

Throwaway one-toy dry run under `notes/logs/`:

```bash
prlimit --as=7800000000 --rss=7800000000 -- \
  env PYTHONPATH=/root/pdgfits-private/src \
  PDGFITS_DATA_BACKEND=snapshot \
  PDGFITS_SNAPSHOT_DIR=/root/pdgfits-private/data/pdg-snapshot \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  /root/micromamba/micromamba run -r /root/micromamba-root -n pdg \
  python notes/codex-refactor/run_round16_calibration_pilot.py \
    --case fit_g2000_k002m \
    --dgp split_normal_pdg_resid \
    --toys-per-case-dgp 1 \
    --endpoint-toys-per-case-dgp 0 \
    --max-runtime-sec 120 \
    --overwrite \
    --toy-csv notes/logs/round16_dryrun_toys.csv \
    --toy-jsonl notes/logs/round16_dryrun_toys.jsonl \
    --endpoint-csv notes/logs/round16_dryrun_endpoints.csv \
    --endpoint-jsonl notes/logs/round16_dryrun_endpoints.jsonl \
    --failure-csv notes/logs/round16_dryrun_failures.csv \
    --failure-jsonl notes/logs/round16_dryrun_failures.jsonl \
    --summary-json notes/logs/round16_dryrun_summary.json
```

Full pilot:

```bash
prlimit --as=7800000000 --rss=7800000000 -- \
  env PYTHONPATH=/root/pdgfits-private/src \
  PDGFITS_DATA_BACKEND=snapshot \
  PDGFITS_SNAPSHOT_DIR=/root/pdgfits-private/data/pdg-snapshot \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  /root/micromamba/micromamba run -r /root/micromamba-root -n pdg \
  python notes/codex-refactor/run_round16_calibration_pilot.py \
    --toys-per-case-dgp 16 \
    --endpoint-toys-per-case-dgp 1 \
    --max-runtime-sec 900 \
    --overwrite
```

## Numerical Endpoint Correctness

No numerical endpoint-correctness failure appeared.

Counts:

- toy rows: `128`;
- endpoint rows: `16`;
- failure rows: `0`;
- profile consistency failures (`q < -1e-4`): `0`;
- maximum absolute endpoint residual: `0.00475926360994805`, within the
  current `5e-3` endpoint residual tolerance.

Endpoint profile methods over the 16 endpoint rows:

| Method | Rows |
| --- | ---: |
| `SLSQP` | 8 |
| `SLSQP+descent-check` | 5 |
| `trust-constr-exact-hess+KKT` | 3 |

## Q Distribution Pilot Results

Reference values for `chi2_1`:

- `P(q <= 1) = 0.682689`;
- median `q = 0.454936`;
- 90th percentile `q = 2.705543`;
- 95th percentile `q = 3.841459`.

Per case/DGP:

| DGP | Case | Toys | `q <= 1` | `q > q90` | `q > q95` | Mean q | Median q | Max q |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| local Gaussian | `B0::S042B95` | 16 | 13/16 | 1/16 | 1/16 | 0.647 | 0.146 | 4.296 |
| local Gaussian | `B0S-BR::S086.37` | 16 | 11/16 | 3/16 | 0/16 | 0.994 | 0.626 | 3.770 |
| local Gaussian | `G...::K002M` | 16 | 10/16 | 0/16 | 0/16 | 0.980 | 0.944 | 2.594 |
| local Gaussian | `G...::K003M` | 16 | 10/16 | 1/16 | 0/16 | 0.966 | 0.668 | 3.388 |
| split normal | `B0::S042B95` | 16 | 11/16 | 1/16 | 0/16 | 0.872 | 0.729 | 2.937 |
| split normal | `B0S-BR::S086.37` | 16 | 11/16 | 0/16 | 0/16 | 0.792 | 0.635 | 2.500 |
| split normal | `G...::K002M` | 16 | 14/16 | 1/16 | 0/16 | 0.504 | 0.267 | 2.742 |
| split normal | `G...::K003M` | 16 | 11/16 | 1/16 | 0/16 | 0.850 | 0.456 | 3.431 |

By stratum:

| DGP | Stratum | Toys | `q <= 1` | `q > q90` | `q > q95` | Mean q | Median q | Max q |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| local Gaussian | asymmetric/profile-shape | 16 | 11/16 | 3/16 | 0/16 | 0.994 | 0.626 | 3.770 |
| local Gaussian | boundary/tiny-parameter | 16 | 13/16 | 1/16 | 1/16 | 0.647 | 0.146 | 4.296 |
| local Gaussian | clean controls | 32 | 20/32 | 1/32 | 0/32 | 0.973 | 0.911 | 3.388 |
| split normal | asymmetric/profile-shape | 16 | 11/16 | 0/16 | 0/16 | 0.792 | 0.635 | 2.500 |
| split normal | boundary/tiny-parameter | 16 | 11/16 | 1/16 | 0/16 | 0.872 | 0.729 | 2.937 |
| split normal | clean controls | 32 | 25/32 | 2/32 | 0/32 | 0.677 | 0.303 | 3.431 |

## Interpretation

### Endpoint correctness

The pilot found no endpoint-correctness failure. All endpoint residuals stayed
inside the existing tolerance, and there were no failed rows or negative-q
profile consistency failures.

### B0S tail stability

The `B0S-BR::S086.37` tail is real enough to justify studying, but not stable
enough to treat as an established calibration failure.

Evidence:

- round 15 local-Gaussian smoke: `q <= 1` was `5/8`, with two q values near 4;
- round 16 local-Gaussian pilot: `q <= 1` was `11/16`, with `3/16` above the
  chi-square-1 90th percentile and max `3.770`;
- round 16 split-normal pilot: `q <= 1` was also `11/16`, but `0/16` were above
  the chi-square-1 90th percentile and max was only `2.500`.

So the asymmetric/profile-shape case still shows a local-Gaussian upper-tail
signal, but that signal weakened relative to round 15 and did not survive the
split-normal DGP. With 16 toys per DGP, this is design evidence, not calibration
evidence.

### B0 boundary activation

`B0::S042B95` reproduced boundary/chart activation under both DGPs:

| DGP | Chart-saturated toys | Minimum BRU coordinate range | Max fitted-coordinate range |
| --- | ---: | ---: | ---: |
| local Gaussian | 16/16 | `2.69e-7` to `4.29e-7` | `74.3` to `118.4` |
| split normal | 16/16 | `2.94e-7` to `3.75e-7` | `84.8` to `108.2` |

That boundary activation did not translate into a clear q-calibration symptom.
Local Gaussian had `q <= 1` in `13/16`, with one q value above the 95th
percentile. Split-normal had `q <= 1` in `11/16`, no q above the 95th
percentile, and a max q of `2.937`. The boundary flags are persistent, but the q
distribution is not visibly worse than the controls at this pilot scale.

### Clean controls

The two clean controls are useful because they show how noisy 16 toys still are.
For example, under local Gaussian the combined controls had `20/32` q values
below 1 and median q `0.911`, while under split-normal they had `25/32` below 1
and median q `0.303`. Neither should be over-interpreted as true calibration or
miscalibration; they are a warning that pilot-scale q summaries are volatile.

## Rejected Or Limited Ideas

- Production code changes: rejected. There is no numerical endpoint failure and
  no lower-profile evidence.
- Editing `build_chi2.py`: rejected by instruction and by evidence.
- Treating this as a coverage study: rejected. `16` toys per case/DGP is only a
  pilot.
- Exact asymmetric-chi2 generative model: rejected for this round. The
  production chi2 is a clipped residual-dependent sigma objective, not a
  normalized off-the-shelf likelihood; pretending otherwise would be less honest
  than using explicit sensitivity DGPs.
- Endpoints for every toy: limited by cost. The primary calibration statistic is
  q at the truth-target profile; endpoint roots were computed on a deterministic
  subset to check residual/certification behavior.
- Adding `eta_c J/psi psi(2S)::M026G01` to this bounded pilot: deferred. It
  belongs in the serious study because round 13 flagged it, but the round-16 cap
  prioritized DGP comparison and two clean controls first.

## Recommended Serious Calibration Study

If Anthony wants publication-grade calibration evidence, run a notes-only,
resumable, stratified parametric toy study. Do not mix this with optimizer
wrapper cleanup.

Minimum scientifically honest design:

1. Cases:
   - boundary/tiny-parameter: `B0::S042B95`;
   - asymmetric/profile-shape: `B0S-BR::S086.37`;
   - asymmetric/profile-shape second case: `eta_c J/psi psi(2S)::M026G01`;
   - clean controls: `G(2000),G(1800)::K002M` and `G(2000),G(1800)::K003M`;
   - optional average/control stratum: one average with nuisance adjustments,
     such as `M026R08`, to distinguish fit-specific behavior from average-side
     behavior.
2. DGPs:
   - local Gaussian with symmetrized errors at zero residual, matching round 15;
   - Gaussian-copula split-normal using the production residual sign convention;
   - optional sensitivity variant: recentered split-normal, explicitly labeled
     as a mean-centered sensitivity check rather than a mode-centered one.
3. Primary statistic:
   - for each toy, refit and compute `q = chi2_profile(truth_target) -
     chi2_min_toy`;
   - compare the empirical CDF to `chi2_1` at fixed thresholds
     `0.454936`, `1`, `2.705543`, and `3.841459`;
   - report QQ/CDF plots and binomial intervals for threshold counts.
4. Numerical diagnostics:
   - record refit status, profile status, profile method, q consistency
     failures, profile true-point stationarity/constraint metrics;
   - compute endpoints on a predeclared subset, e.g. every 10th toy or the first
     50 per case/DGP, and always record endpoint residuals;
   - record boundary/chart activation at toy MLEs, and if feasible extend the
     profile-point diagnostics to expose boundary metrics at truth-profile and
     endpoint-profile points.
5. Toy counts:
   - pilot extension: 64 toys per case/DGP, including `M026G01`, to estimate
     cost and failure rate;
   - serious study: at least 500 toys per case/DGP for threshold estimates;
   - publication-grade tails: 1000-2000 toys per case/DGP if 90th/95th
     percentile behavior is central.
6. Runtime/cost estimate:
   - this pilot ran 128 q rows plus 16 endpoint rows in `227s`;
   - observed q-only costs were roughly `3-4s` per B0 toy, `1.3-1.5s` per B0S
     toy, and `0.3-0.4s` per G-control toy;
   - for the four pilot cases, 500 toys per case/DGP should be on the order of
     2-3 single-worker hours plus endpoint-subset overhead;
   - adding `M026G01`, optional averages, and 10% endpoint roots likely moves
     this to a multi-hour or overnight run;
   - shard by case/DGP with resumable CSV/JSONL outputs, using one worker by
     default and at most two workers on the Hermes VPS.

Predeclared analysis rules should include:

- no production-code change unless numerical endpoint residual failures,
  profile consistency failures, or reproducible lower-profile solutions appear;
- calibration claims require DGP-labeled results and uncertainty intervals;
- DGP sensitivity is a scientific result, not a nuisance to hide;
- boundary activation without q symptoms should be reported as a Wilks
  regularity warning, not as an endpoint failure.

## Required Answers

1. Did any numerical endpoint-correctness failure appear?

No. There were `0` failures, `0` profile consistency failures, and all 16
endpoint rows verified within the current `5e-3` residual tolerance.

2. Did any q distribution visibly deviate from chi-square-1 beyond smoke-test
noise?

Not convincingly. The `B0S-BR::S086.37` local-Gaussian tail remains visible
(`3/16` above the chi-square-1 90th percentile), but clean controls were also
noisy at 16 toys and the split-normal DGP did not reproduce that tail. This is
evidence for a serious study design, not evidence of established miscalibration.

3. Is the apparent `B0S-BR::S086.37` tail stable under more toys or DGP choice?

Only partly. It persists as a mild local-Gaussian tail but weakens relative to
round 15 and disappears under the split-normal DGP at this pilot size.

4. Does `B0::S042B95` boundary activation translate into q calibration symptoms,
or just boundary flags?

At this pilot scale, just boundary flags. `B0::S042B95` was chart-saturated in
`32/32` toys across both DGPs, but its q distribution did not look worse than
the clean controls in a stable way.

5. Are production changes justified?

No. The correct action is notes-only calibration/design work. There is no
evidence for another optimizer fallback, chart replacement, or asymmetric-chi2
formula change.

6. What should Anthony run next for publication-grade calibration evidence?

Run the serious calibration study above: at least the five core cases
(`B0::S042B95`, `B0S-BR::S086.37`, `eta_c...::M026G01`, `G...::K002M`,
`G...::K003M`) under at least the two DGPs used here, with 500+ toys per
case/DGP, predeclared q-threshold summaries, endpoint residual checks on a
subset, and boundary metrics. Treat DGP sensitivity as part of the result.

## Bottom Line

Round 16 produced a notes-only diagnostic/design and a bounded quantitative
pilot. It found no concrete numerical endpoint failure and no production method
change. It did produce enough evidence to justify a serious later calibration
study if Anthony wants calibration claims: the B0S tail is DGP-sensitive, B0
boundary activation is persistent but not clearly miscalibrated, and pilot-scale
controls are noisy enough that publication-grade evidence needs a predeclared,
larger, DGP-labeled study.
