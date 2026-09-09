# Asym Refactor Round 10 MnCross Status + Stratified Globality

Date: 2026-06-25

Production code changed: no. `src/pdgfits/build_chi2.py` was not edited.

## Question

Round 09 found structured risk signals but no endpoint-correctness failure:
descent-check/high-gradient endpoints, chart saturation, call-heavy endpoints,
HESSE/profile nonlinear cases, residual-tolerance-edge endpoints, and three
sub-`1e-5` lower fixed-target chi2 candidates. This round asked whether those
signals escalate under a targeted, broader fixed-target search.

Thresholds used here:

- Endpoint verification tolerance: `abs(profile_chi2 - (chi2_min + 1)) <= 5e-3`.
- Numerical tie threshold: lower fixed-target improvement `<= 1e-5`.
- Material lower constrained profile threshold: improvement `> 1e-4`, matching
  the existing descent-check function tolerance scale.
- Endpoint concern threshold: a production endpoint fixed-target rerun both
  falls outside the `5e-3` residual tolerance and has a meaningful lower chi2.

## Artifacts

- Harness: `notes/codex-refactor/run_round10_mncross_stratified_globality.py`
- Structured status CSV: `notes/codex-refactor/round10_mncross_status_summary.csv`
- Structured status JSONL: `notes/codex-refactor/round10_mncross_status_summary.jsonl`
- Stratified result CSV: `notes/codex-refactor/round10_stratified_globality_results.csv`
- Stratified result JSONL: `notes/codex-refactor/round10_stratified_globality_results.jsonl`
- Candidate-start CSV: `notes/codex-refactor/round10_stratified_globality_candidates.csv`
- Candidate-start JSONL: `notes/codex-refactor/round10_stratified_globality_candidates.jsonl`
- Failures CSV/JSONL: `notes/codex-refactor/round10_stratified_globality_failures.csv`, `notes/codex-refactor/round10_stratified_globality_failures.jsonl`
- Raw stdout: `notes/logs/round10_mncross_stratified_globality_stdout.log`

Exact row counts:

| Table | Rows |
| --- | ---: |
| Status summary | 20 |
| Stratified fixed-target checks | 11 |
| Candidate starts | 181 |
| Harness failures | 0 |

The failures files are empty because no case-level harness failure occurred.

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
python -m py_compile notes/codex-refactor/run_round10_mncross_stratified_globality.py
```

Smoke average check, passed:

```bash
python notes/codex-refactor/run_round10_mncross_stratified_globality.py \
  --case avg_m002w \
  --status-csv /tmp/round10_status_smoke.csv \
  --status-jsonl /tmp/round10_status_smoke.jsonl \
  --results-csv /tmp/round10_results_smoke.csv \
  --results-jsonl /tmp/round10_results_smoke.jsonl \
  --candidates-csv /tmp/round10_candidates_smoke.csv \
  --candidates-jsonl /tmp/round10_candidates_smoke.jsonl \
  --failures-csv /tmp/round10_failures_smoke.csv \
  --failures-jsonl /tmp/round10_failures_smoke.jsonl \
  --stdout-log notes/logs/round10_mncross_stratified_globality_smoke_stdout.log
```

Full run, passed:

```bash
prlimit --as=7800000000 --rss=7800000000 -- \
  env PYTHONPATH=/root/pdgfits-private/src \
  PDGFITS_DATA_BACKEND=snapshot \
  PDGFITS_SNAPSHOT_DIR=/root/pdgfits-private/data/pdg-snapshot \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  /root/micromamba/micromamba run -r /root/micromamba-root -n pdg \
  python notes/codex-refactor/run_round10_mncross_stratified_globality.py \
    --status-csv notes/codex-refactor/round10_mncross_status_summary.csv \
    --status-jsonl notes/codex-refactor/round10_mncross_status_summary.jsonl \
    --results-csv notes/codex-refactor/round10_stratified_globality_results.csv \
    --results-jsonl notes/codex-refactor/round10_stratified_globality_results.jsonl \
    --candidates-csv notes/codex-refactor/round10_stratified_globality_candidates.csv \
    --candidates-jsonl notes/codex-refactor/round10_stratified_globality_candidates.jsonl \
    --failures-csv notes/codex-refactor/round10_stratified_globality_failures.csv \
    --failures-jsonl notes/codex-refactor/round10_stratified_globality_failures.jsonl \
    --stdout-log notes/logs/round10_mncross_stratified_globality_stdout.log
```

## MnCross-Style Status Summary

The status table classifies each Round 09 endpoint into explicit crossing/status
categories: valid residual, lower fixed-target chi2, chart/boundary, call-heavy,
invalid profile solve, residual tolerance edge, descent-check, high projected
gradient, HESSE/profile nonlinear, covariance-start suspect, and
globality-suspect.

Counts over the 20 Round 09 endpoints:

| Status | Count |
| --- | ---: |
| Valid residual | 20 |
| Endpoint-correctness failure | 0 |
| Lower fixed-target chi2 | 3 |
| Material lower fixed-target chi2 `>1e-4` | 0 |
| Parameter/chart boundary | 2 |
| Call-heavy/call-limit-like | 12 |
| Invalid profile solve | 0 |
| Residual-tolerance-edge | 2 |
| Descent-check | 8 |
| High projected/KKT gradient | 8 |
| HESSE-profile nonlinear | 3 |
| Covariance-start suspect | 11 |
| Disconnected/globality-suspect | 7 |

Top status rows:

| Risk | Case | Side | Main statuses | Prod residual | HESSE residual | Cov-current chi2 |
| ---: | --- | --- | --- | ---: | ---: | ---: |
| 135 | `fit_eta_nuisance_m071254` | lower | lower-chi2, call-heavy, descent/high-gradient | 0.00189 | -0.00592 | -1.26e-6 |
| 135 | `fit_eta_nuisance_m071254` | upper | lower-chi2, call-heavy, descent/high-gradient | -0.000600 | 0.00723 | -1.46e-6 |
| 125 | `fit_eta_m026w` | lower | lower-chi2, descent/high-gradient | 0.000708 | 0.000707 | -2.01e-6 |
| 55 | `fit_b0_s042b95` | lower | chart boundary, descent/high-gradient | 0.00116 | 0.00116 | 2.54e-6 |
| 47 | `fit_eta_m026g01` | upper | residual edge, HESSE nonlinear, descent/high-gradient | 0.00476 | -0.0401 | -2.85e-11 |
| 42 | `fit_b0_s042b95` | upper | chart boundary, residual near tolerance | -0.00414 | -0.00414 | -4.26e-14 |
| 35 | `fit_b0s_s08637` | lower | call-heavy, HESSE nonlinear, descent/high-gradient | -0.000857 | 0.0401 | -2.91e-11 |
| 35 | `fit_b0s_s08637` | upper | call-heavy, HESSE nonlinear, descent/high-gradient | 0.00115 | -0.0362 | -1.05e-11 |
| 30 | `avg_m002w` | lower | call-heavy, residual tolerance edge | -0.00500 | 0.00769 | n/a |
| 25 | `fit_eta_m026w` | upper | descent/high-gradient | -0.000935 | -0.000935 | -1.82e-7 |

## Stratified Fixed-Target Search

Target set:

- All Round 09 endpoints tagged descent-check plus high projected gradient.
- The three Round 09 sub-`1e-5` lower fixed-target chi2 candidates:
  `eta_c...::M026W` lower production endpoint,
  `eta_c...::nuisance_M071.254` lower production endpoint, and
  `eta_c...::nuisance_M071.254` upper parabolic endpoint.
- Chart-saturated `B0::S042B95` lower/upper.
- Mild nonlinear `B0S-BR::S086.37` lower/upper and `eta_c...::M026G01` upper.
- Residual-edge average `M002W` lower.

Fit-side starts were deliberately broader than production: MLE/current projected,
covariance eigenvector perturbations, MINOS-style covariance-predicted starts,
random covariance perturbations, endpoint perturbations after the first accepted
solve, and BR/BRU chart-boundary-biased starts. Average `M002W` is a
one-parameter direct profile, so the fixed target is a direct chi2 evaluation.

| Case | Side | Fixed value | Strata | Best residual | Improve vs previous | Improve vs current | Best start | OK/starts | Max chart coord |
| --- | --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: |
| `fit_eta_nuisance_m071254` | lower | production | descent/high-gradient, sub-`1e-5` | 0.00188 | 1.26e-6 | 1.26e-6 | `random_mle_3` | 18/18 | 0.194 |
| `fit_eta_nuisance_m071254` | upper | production | descent/high-gradient | -0.000600 | -4.58e-10 | -2.52e-10 | `random_endpoint_1` | 18/18 | 0.194 |
| `fit_eta_m026w` | lower | production | descent/high-gradient, sub-`1e-5` | 0.000707 | 1.21e-6 | 1.10e-9 | `all_decay_coords_expand` | 18/18 | 0.191 |
| `fit_b0_s042b95` | lower | production | descent/high-gradient, chart-saturated | 0.00116 | -4.47e-9 | 8.27e-11 | `random_endpoint_4` | 18/18 | 93.9 |
| `fit_eta_m026g01` | upper | production | descent/high-gradient, nonlinear | 0.00476 | 2.85e-11 | 2.85e-11 | `covariance_predicted_raw` | 17/18 | 0.189 |
| `fit_b0s_s08637` | lower | production | descent/high-gradient, nonlinear | -0.000857 | 3.77e-13 | 7.11e-15 | `cov_eig1_plus` | 18/18 | 1.75 |
| `fit_b0s_s08637` | upper | production | descent/high-gradient, nonlinear | 0.00115 | 3.40e-11 | 4.39e-11 | `random_mle_1` | 18/18 | 1.75 |
| `fit_eta_m026w` | upper | production | descent/high-gradient | -0.000935 | 6.54e-11 | 6.54e-11 | `all_decay_coords_expand` | 18/18 | 0.196 |
| `fit_eta_nuisance_m071254` | upper | parabolic | sub-`1e-5` parabolic | 0.00723 | 9.64e-9 | 1.47e-6 | `random_endpoint_2` | 18/18 | 0.194 |
| `fit_b0_s042b95` | upper | production | chart-saturated | -0.00414 | -2.84e-14 | 1.42e-14 | `covariance_predicted_raw` | 18/18 | 93.9 |
| `avg_m002w` | lower | production | residual edge | -0.00500 | 0 | 0 | `direct_average_coordinate` | 1/1 | n/a |

Candidate-level extrema:

- Largest improvement over the previous best selected profile chi2:
  `1.2582032695718226e-6` on `eta_c...::nuisance_M071.254` lower production
  endpoint.
- Largest improvement over the current-start baseline:
  `1.473681436436891e-6` on `eta_c...::nuisance_M071.254` upper parabolic
  endpoint.
- Candidate improvements greater than `1e-5`: `0 / 181`.
- Candidate improvements greater than `1e-4`: `0 / 181`.
- Production fixed-target endpoint residual failures: `0 / 10` production
  endpoints checked.

The only fixed-target residual above `5e-3` is the deliberately tested
`nuisance_M071.254` upper parabolic/HESSE diagnostic value (`0.00723`). That is
not a reported endpoint; it reinforces the Round 09 conclusion that HESSE
targets are diagnostics only and cannot replace profile endpoints.

One aggressive exploratory candidate failed locally:
`eta_c...::M026G01` upper from `boundary_min_decay_more_saturated` landed at
chi2 `6745`, with scaled constraint violation `2.66e-11`, and was not selected.
The case-level check still had 17 accepted starts and a stable best residual.

## Answers

1. Did any risk tag escalate into a real endpoint-correctness failure or a lower constrained profile big enough to matter?

No. All 20 Round 09 endpoints had valid production residuals. In the Round 10
fixed-target reruns, all 10 production endpoints remained within the `5e-3`
endpoint tolerance. The largest lower constrained-profile improvement over the
previous selected best was `1.26e-6`, and no candidate exceeded `1e-5`, let
alone the `1e-4` material threshold.

2. Did the structured status summary reveal a robust simplification or production change?

No. It produced useful triage labels, but not a method change. The statuses are
heterogeneous: chart saturation dominates `B0::S042B95`, HESSE/profile
nonlinearity dominates `B0S-BR::S086.37` and `M026G01`, and the eta direct
coordinate/nuisance cases show only tiny fixed-target ties. There is no single
solver simplification or start policy that is supported across these strata.

3. Are the sub-`1e-5` lower fixed-target candidates from Round 9 numerical ties, or do they reproduce/escalate under broader starts?

They reproduce only as numerical ties and do not escalate:

- `eta_c...::M026W` lower production endpoint: best improvement `1.21e-6`;
  residual remains `0.000707`.
- `eta_c...::nuisance_M071.254` lower production endpoint: best improvement
  `1.26e-6`; residual remains `0.00188`.
- `eta_c...::nuisance_M071.254` upper parabolic endpoint: best improvement
  over current baseline `1.47e-6`; this is a parabolic diagnostic target, not a
  reported endpoint, and the residual remains `0.00723`.

4. Do chart-saturated BR/BRU endpoints show disconnected feasible-component/globality problems?

No evidence of that in this targeted search. `B0::S042B95` remains severely
chart-saturated (`max_abs_decay_fitted_coord ~= 93.9`, min decay parameter
`~3.39e-7`), but 18/18 broader starts on each side returned to the same
production fixed-target profile within numerical noise. The lower side best was
actually `4.47e-9` higher than the previous selected chi2, and the upper side
matched to `~1e-14`. This looks like bad conditioning with stable endpoints,
not disconnected feasible components.

5. What exact idea is rejected, and why?

Rejected: escalating MnCross-style risk tags directly into a production solver
change, such as adding universal covariance starts, trusting HESSE/parabolic
endpoints, or adding broader random/chart-boundary multistarts to production.

Reason: the broad starts found no material lower constrained profile. The risk
tags are diagnostic triage, not evidence that the current endpoint machinery is
wrong. HESSE/parabolic fixed values can miss the profile target by `0.036` to
`0.040` chi2 on nonlinear cases and `0.00723` in the eta nuisance parabolic
check, so they remain unsuitable as reported endpoints. Chart-boundary starts
are useful for stress testing but not as a production default.

6. What should the next round do if supervision continues?

Do not add more optimizer fallbacks without a failure case. The next useful
round should choose one of two focused directions:

- If the goal remains numerical correctness, test whether a tighter endpoint
  residual target, for example `1e-3`, is affordable and stable on residual-edge
  cases (`M002W` lower and `M026G01` upper) and then on a small stratified set.
- If the goal is a method-change spike, run a chart-aware BR/BRU conditioning
  experiment only for saturated cases like `B0::S042B95`, requiring identical
  endpoint chi2 with cleaner stationarity/certificates before considering any
  production parameter-map change.

## Bottom Line

Round 10 did not find a hidden lower constrained profile or endpoint residual
failure. The structured statuses are useful for reporting and triage, but they
support keeping the current bracketed profile endpoint method unchanged for now.
