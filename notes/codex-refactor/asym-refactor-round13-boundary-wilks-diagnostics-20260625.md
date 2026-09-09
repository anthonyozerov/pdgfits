# Asym Refactor Round 13 Boundary/Wilks Diagnostics

Date: 2026-06-25

Production code changed: no. `src/pdgfits/build_chi2.py` was not edited.

## Question

Among endpoints that are numerically stable and pass
`profile_chi2(endpoint) = chi2_min + 1`, which ones are statistically delicate
for a Wilks/profile-likelihood one-sigma interpretation?

This round deliberately did not search for another optimizer fallback. It
reused round 9-12 artifacts and added a cheap parsed-input asymmetry scan. The
diagnostic separates:

- endpoint numerical correctness and cross-checks;
- HESSE/parabolic approximation quality;
- physical/chart boundary proximity and tiny BR/BRU coordinates;
- lower/upper profile asymmetry and sampled profile nonquadraticity;
- active or near-active one-sided boundary behavior;
- cases where asymmetric reported-error interpolation/profile shape appears to
  dominate over optimizer behavior.

## Artifacts

- Harness: `notes/codex-refactor/run_round13_boundary_wilks_diagnostics.py`
- Results CSV: `notes/codex-refactor/round13_boundary_wilks_diagnostics_results.csv`
- Results JSONL: `notes/codex-refactor/round13_boundary_wilks_diagnostics_results.jsonl`
- Failures CSV: `notes/codex-refactor/round13_boundary_wilks_diagnostics_failures.csv`
- Failures JSONL: `notes/codex-refactor/round13_boundary_wilks_diagnostics_failures.jsonl`
- Summary JSON: `notes/codex-refactor/round13_boundary_wilks_diagnostics_summary.json`
- Raw stdout: `notes/logs/round13_boundary_wilks_diagnostics_stdout.log`

Exact row counts:

| Table | Rows |
| --- | ---: |
| Result endpoints | 20 |
| Failures | 0 |

Risk counts over the 20 endpoints:

| Wilks/statistical regularity risk | Rows |
| --- | ---: |
| high | 2 |
| medium-high | 2 |
| medium | 4 |
| low-medium | 1 |
| low | 11 |

Dominant-risk counts:

| Dominant driver | Rows |
| --- | ---: |
| locally-quadratic-interior | 11 |
| asymmetric-chi2/profile-shape | 4 |
| boundary/tiny-parameter | 2 |
| tiny-parameter-boundary | 2 |
| endpoint-tolerance-not-regularity | 1 |

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
  python -m py_compile notes/codex-refactor/run_round13_boundary_wilks_diagnostics.py
```

Full run, passed:

```bash
prlimit --as=7800000000 --rss=7800000000 -- \
  env PYTHONPATH=/root/pdgfits-private/src \
  PDGFITS_DATA_BACKEND=snapshot \
  PDGFITS_SNAPSHOT_DIR=/root/pdgfits-private/data/pdg-snapshot \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  /root/micromamba/micromamba run -r /root/micromamba-root -n pdg \
  python notes/codex-refactor/run_round13_boundary_wilks_diagnostics.py \
    --results-csv notes/codex-refactor/round13_boundary_wilks_diagnostics_results.csv \
    --results-jsonl notes/codex-refactor/round13_boundary_wilks_diagnostics_results.jsonl \
    --failures-csv notes/codex-refactor/round13_boundary_wilks_diagnostics_failures.csv \
    --failures-jsonl notes/codex-refactor/round13_boundary_wilks_diagnostics_failures.jsonl \
    --summary-json notes/codex-refactor/round13_boundary_wilks_diagnostics_summary.json \
    --stdout-log notes/logs/round13_boundary_wilks_diagnostics_stdout.log
```

Runtime was about `19s`. The input scan reran existing fit/average setup for the 10
round-9 cases to summarize parsed reported-error asymmetry; it did not change
the chi2 model or run a new endpoint method.

## Main Risk Table

| Rank | Case | Side | Risk | Driver | Endpoint residual | HESSE rel diff | HESSE endpoint residual | Curve RMS | Min BR/BRU coord | Max fitted coord | Target/error to zero | Input max asym |
| ---: | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `B0S-BR::S086.37` | lower | medium-high | asymmetric-chi2/profile-shape | -8.57e-4 | 1.99e-2 | 4.01e-2 | 2.69e-3 | 1.82e-5 | 1.75 | 18.5 | 1.10 |
| 2 | `B0S-BR::S086.37` | upper | medium-high | asymmetric-chi2/profile-shape | 1.15e-3 | 1.92e-2 | -3.62e-2 | 3.64e-3 | 1.82e-5 | 1.75 | 18.5 | 1.10 |
| 3 | `B0::S042B95` | lower | high | boundary/tiny-parameter | 1.16e-3 | 9.09e-16 | 1.16e-3 | 1.16e-3 | 3.39e-7 | 93.9 | 19.1 | 0.607 |
| 4 | `eta_c...::M026G01` | upper | medium | asymmetric-chi2/profile-shape | 4.76e-3 | 2.29e-2 | -4.01e-2 | 4.27e-3 | 1.64e-4 | 0.194 | 14.0 | 0.824 |
| 5 | `B0::S042B95` | upper | high | boundary/tiny-parameter | -4.14e-3 | 9.09e-16 | -4.14e-3 | 4.14e-3 | 3.39e-7 | 93.9 | 19.1 | 0.607 |
| 6 | `eta_c...::M026G01` | lower | medium | asymmetric-chi2/profile-shape | -2.18e-3 | 1.19e-2 | 2.19e-2 | 2.98e-3 | 1.61e-4 | 0.198 | 14.0 | 0.824 |
| 7 | `B0S-BR::S086R04` | lower | medium | tiny-parameter-boundary | 8.81e-5 | 3.92e-3 | 7.99e-3 | 7.74e-4 | 1.82e-5 | 1.75 | 29.3 | 1.10 |
| 8 | `B0S-BR::S086R04` | upper | medium | tiny-parameter-boundary | -5.25e-5 | 3.89e-3 | -7.78e-3 | 1.49e-3 | 1.82e-5 | 1.75 | 29.3 | 1.10 |
| 9 | `avg_m002w` | lower | low-medium | endpoint-tolerance-not-regularity | -5.00e-3 | 6.38e-3 | 7.69e-3 | 5.62e-3 | n/a | n/a | 10.8 | 0 |

The remaining 11 endpoints were classified low risk. The four
`G(2000),G(1800)::K002M/K003M` control endpoints are the clean controls:
endpoint residuals around `1e-10`, HESSE/profile relative disagreement below
`5e-15`, profile asymmetry ratio `1.0`, and no boundary metric.

## Interpretation

### 1. Numerical endpoint correctness

No numerical endpoint-correctness issue was found.

All 20 endpoints remained within the existing `5e-3` endpoint residual
tolerance. Round-10 globality checks found no material lower constrained
profiles: largest improvement over the previous selected fixed-target profile
was `1.26e-6`, below the `1e-4` material threshold. Round-11 chart checks for
`B0::S042B95` and `B0S-BR::S086.37` found no lower chart profile; largest
improvement was `4.95e-9`. Round-12 VM verification, where available, rechecked
VM-polished endpoints with production profile solves.

This round is therefore a statistical regularity triage, not a solver failure.

### 2. Boundary and tiny-parameter risk

`B0::S042B95` lower/upper are the clearest Wilks-regularity risks. The reported
node itself is not close to zero in one-sigma units (`target/error ~= 19.1`),
but the profiled BR/BRU coordinates include a physical parameter near
`3.39e-7`, and the production fitted coordinate is saturated at about `93.9`.
That is exactly the regime where regular interior asymptotics and fitted-chart
covariance can be misleading.

Important nuance: the profile is numerically stable here. HESSE/profile
endpoint disagreement is essentially zero, chart alternatives reproduced the
same endpoint, and broad starts returned to the same fixed-target profile. The
risk is not "bad endpoint"; it is "Wilks chi-square-1 may be a delicate
statistical approximation because the profiled solution lives near a physical
boundary."

`B0S-BR::S086R04` lower/upper are lower-grade boundary risks. Their minimum
BR/BRU coordinate is about `1.82e-5`, not saturated like B0, and profiles are
locally close to quadratic. They are flagged medium because tiny branching
coordinates are present, not because the endpoint solver struggled.

### 3. HESSE/profile nonlinearity and asymmetric interpolation

`B0S-BR::S086.37` lower/upper are the main medium-high nonlinearity cases. They
are not chart-saturated (`max fitted coord ~= 1.75`), but HESSE endpoints miss
the profiled level by about `0.036-0.040` chi2, HESSE/profile relative endpoint
disagreement is about `1.9-2.0%`, and the parsed input scan found strongly
asymmetric reported errors in the fit (`max fractional error asymmetry ~= 1.10`).
Round-11 chart alternatives and round-12 VM checks verified the same endpoint.
This points away from optimizer behavior and toward the model/profile shape:
nonlinear target/mapping plus the asymmetric-error chi2 interpolation.

`eta_c...::M026G01` lower/upper show a similar but smaller pattern. The upper
side has HESSE/profile relative disagreement `2.29%`, HESSE endpoint residual
`-0.0401`, endpoint residual near the current tolerance edge (`0.00476`), and
target-row reported-error asymmetry up to `0.367`. This is also a profile-shape
risk rather than an endpoint-correctness failure.

### 4. Active constraints and one-sided target behavior

No tested reported target is itself close enough to zero to look like a
classical one-sided boundary interval. The smallest positive
`target_value / min(side_error)` ratios in this set are:

- `avg_m026r08`: `8.16`;
- `avg_m002w`: `10.8`;
- `eta_c...::M026G01`: `14.0`;
- `B0S-BR::S086.37`: `18.5`;
- `B0::S042B95`: `19.1`.

So the one-sided-boundary concern is not that a reported endpoint is about to
cross zero. It is that some profiled nuisance/BR/BRU coordinates are tiny or
chart-saturated, which can still violate the regular interior conditions behind
Wilks' theorem.

### 5. Endpoint tolerance edge

`avg_m002w` lower is a numerical-tolerance edge, not a Wilks boundary case. The
production residual is `-0.004999656`, right at the current bisection tolerance.
Round 12's VM polish moved the endpoint by only `5.36e-5` and verified the level
to `5.39e-11`. The average has symmetric input errors in the parsed scan and no
boundary metrics. This justifies treating it as low-medium operational risk:
tighter endpoint polishing may be useful, but it does not indicate fragile
Wilks regularity.

## Required Answers

1. Did this find a numerical endpoint-correctness issue?

No. Result rows: `20`; failures: `0`; numerical endpoint issue rows: `0`.
Every reported endpoint remained certified within existing residual tolerance,
and no round-10/11 cross-check found a material lower constrained profile.

2. Which endpoints are statistically/regularity-risky despite numerical stability?

Highest risk:

- `B0::S042B95` lower/upper: high, boundary/tiny-parameter dominated.
- `B0S-BR::S086.37` lower/upper: medium-high, asymmetric/profile-shape dominated
  with tiny BR/BRU coordinates.
- `eta_c...::M026G01` lower/upper: medium, asymmetric/profile-shape dominated.
- `B0S-BR::S086R04` lower/upper: medium, tiny-parameter boundary risk but
  closer to local quadratic behavior.

`avg_m002w` lower is low-medium, but because of endpoint residual tolerance, not
because of boundary/Wilks regularity. `eta_c...::M026W` and
`eta_c...::nuisance_M071.254` retain numerical certificate tags, but this round
classifies them as low statistical regularity risk because their profiles are
locally quadratic/interior at this diagnostic resolution.

3. Are risks dominated by boundary/tiny-parameter behavior, HESSE/profile nonlinearity, asymmetric chi2 interpolation, or something else?

They split by stratum:

- Boundary/tiny-parameter dominated: `B0::S042B95`.
- Asymmetric chi2/profile-shape dominated: `B0S-BR::S086.37` and
  `eta_c...::M026G01`.
- Tiny-parameter but locally near-quadratic: `B0S-BR::S086R04`.
- Endpoint tolerance, not regularity: `avg_m002w` lower.
- Locally quadratic/interior: `G(2000),G(1800)::K002M/K003M` controls and most
  direct eta/nuisance endpoints.

4. Does this justify a production code change now?

No. There is no endpoint-correctness failure, no lower constrained profile, and
no evidence that changing the production solver or parameter map would improve
scientific correctness. The useful output is a risk table and a clearer
distinction between numerical certification and Wilks/statistical regularity.

5. What exact next round, if any, is highest-value before the 09:00 PT stop?

Stop changing production mechanics. The highest-value next bounded round would
be a tiny, explicitly exploratory Wilks-calibration smoke test on two strata:

- boundary/tiny-parameter: `B0::S042B95`;
- asymmetric/profile-shape: `B0S-BR::S086.37` or `eta_c...::M026G01`.

Cap it hard, e.g. `20-50` parametric toys per stratum under the current fitted
chi2 model, one worker, and report only the empirical distribution of
`2 Delta log L`/profile coverage signals and boundary activations. This would
not prove coverage, but it would test whether the risk table corresponds to a
visible Wilks-calibration symptom. If that is too much for the remaining window,
the next best action is to stop and synthesize rounds 7-13 for Anthony.

## Bottom Line

Round 13 found no numerical endpoint failure. It did identify a small set of
statistically delicate endpoints: B0 is boundary/tiny-parameter dominated, while
B0S `S086.37` and eta `M026G01` are dominated by nonquadratic/asymmetric profile
shape under the existing asymmetric-error chi2 model. These are reasons to
qualify Wilks one-sigma interpretation, not reasons to change production code.
