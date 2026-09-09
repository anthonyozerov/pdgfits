# Asym Refactor Round 09 HESSE/Profile Diagnostics

Date: 2026-06-25

Production code changed: no. `src/pdgfits/build_chi2.py` was not edited.

## Question

Can cheap local diagnostics stratify pdgfits asymmetric-error endpoints into
benign, nonlinear, or numerically risky cases that deserve expensive
profile-curve/globality checks?

Diagnostics recorded here:

- HESSE/parabolic one-sigma target prediction from the local fitted-coordinate
  covariance.
- Explicit profile evaluation at the parabolic endpoint.
- Current production endpoint residual/method/call diagnostics.
- MINOS-style covariance-start candidate comparison from the notes-only round 8
  profile wrapper.
- BR/BRU chart metrics: minimum physical decay parameter and maximum absolute
  fitted decay coordinate.
- A structured `MnCross`-style status tag summary.

HESSE endpoints are diagnostics only. They are not acceptable reported errors
unless profile verification supports them.

## Artifacts

- Harness: `notes/codex-refactor/run_round09_hesse_profile_diagnostics.py`
- Results CSV: `notes/codex-refactor/round09_hesse_profile_diagnostics_results.csv`
- Results JSONL: `notes/codex-refactor/round09_hesse_profile_diagnostics_results.jsonl`
- Evaluation/candidate CSV: `notes/codex-refactor/round09_hesse_profile_diagnostics_evaluations.csv`
- Evaluation/candidate JSONL: `notes/codex-refactor/round09_hesse_profile_diagnostics_evaluations.jsonl`
- Failures CSV: `notes/codex-refactor/round09_hesse_profile_diagnostics_failures.csv`
- Failures JSONL: `notes/codex-refactor/round09_hesse_profile_diagnostics_failures.jsonl`
- Summary JSON: `notes/codex-refactor/round09_hesse_profile_diagnostics_summary.json`
- Raw stdout: `notes/logs/round09_hesse_profile_diagnostics_stdout.log`
- Smoke stdout: `notes/logs/round09_hesse_profile_diagnostics_smoke_stdout.log`

Exact row counts:

| Table | Rows |
| --- | ---: |
| Results | 20 |
| Evaluation/candidate rows | 527 |
| Failures | 0 |

The 20 result rows are 10 stratified cases times two endpoint sides.

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
python -m py_compile notes/codex-refactor/run_round09_hesse_profile_diagnostics.py
```

Smoke run, passed with 2 result rows and 21 evaluation rows:

```bash
prlimit --as=7800000000 --rss=7800000000 -- \
  env PYTHONPATH=/root/pdgfits-private/src \
  PDGFITS_DATA_BACKEND=snapshot \
  PDGFITS_SNAPSHOT_DIR=/root/pdgfits-private/data/pdg-snapshot \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  /root/micromamba/micromamba run -r /root/micromamba-root -n pdg \
  python notes/codex-refactor/run_round09_hesse_profile_diagnostics.py \
    --case fit_g2000_k002m \
    --stdout-log notes/logs/round09_hesse_profile_diagnostics_smoke_stdout.log
```

Full run, passed with 20 result rows, 527 evaluation rows, and 0 failures:

```bash
prlimit --as=7800000000 --rss=7800000000 -- \
  env PYTHONPATH=/root/pdgfits-private/src \
  PDGFITS_DATA_BACKEND=snapshot \
  PDGFITS_SNAPSHOT_DIR=/root/pdgfits-private/data/pdg-snapshot \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  /root/micromamba/micromamba run -r /root/micromamba-root -n pdg \
  python notes/codex-refactor/run_round09_hesse_profile_diagnostics.py \
    --results-csv notes/codex-refactor/round09_hesse_profile_diagnostics_results.csv \
    --results-jsonl notes/codex-refactor/round09_hesse_profile_diagnostics_results.jsonl \
    --evaluations-csv notes/codex-refactor/round09_hesse_profile_diagnostics_evaluations.csv \
    --evaluations-jsonl notes/codex-refactor/round09_hesse_profile_diagnostics_evaluations.jsonl \
    --failures-csv notes/codex-refactor/round09_hesse_profile_diagnostics_failures.csv \
    --failures-jsonl notes/codex-refactor/round09_hesse_profile_diagnostics_failures.jsonl \
    --summary-json notes/codex-refactor/round09_hesse_profile_diagnostics_summary.json \
    --stdout-log notes/logs/round09_hesse_profile_diagnostics_stdout.log
```

## Case Set

Required direct-coordinate cases:

- `M002W` average.
- `M026R08` average.
- `G(2000),G(1800)::K002M`.
- `G(2000),G(1800)::K003M`.
- `eta_c J/psi psi(2S)::M026W`.

Required arbitrary/mapped cases:

- `B0::S042B95`.
- `B0S-BR::S086.37`.

Extra stratified candidates from existing hard-case artifacts:

- `B0S-BR::S086R04`, call-heavy/KKT-heavy but well-conditioned.
- `eta_c J/psi psi(2S)::M026G01`, tiny BRU node with high endpoint residual.
- `eta_c J/psi psi(2S)::nuisance_M071.254`, high-gradient/descent-check nuisance endpoint.

## Ranked Risk Table

| Rank | Case | Side | Risk | Main status tags | Prod residual | HESSE rel diff | HESSE endpoint residual | Min decay | Max fitted decay coord | Best cov-current chi2 |
| ---: | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `fit_eta_nuisance_m071254` | lower | 135 | `new-lower-chi2`, `call-heavy`, `descent-check`, `high-projected-gradient` | 0.00189 | 0.00389 | -0.00592 | 1.64e-4 | 0.194 | -1.26e-6 |
| 2 | `fit_eta_nuisance_m071254` | upper | 135 | `new-lower-chi2`, `call-heavy`, `descent-check`, `high-projected-gradient` | -0.000600 | 0.00392 | 0.00723 | 1.64e-4 | 0.194 | -1.46e-6 |
| 3 | `fit_eta_m026w` | lower | 125 | `new-lower-chi2`, `descent-check`, `high-projected-gradient` | 0.000708 | 1.28e-16 | 0.000707 | 1.64e-4 | 0.194 | -2.01e-6 |
| 4 | `fit_b0_s042b95` | lower | 55 | `boundary/chart-saturation`, `descent-check`, `high-projected-gradient` | 0.00116 | 9.09e-16 | 0.00116 | 3.39e-7 | 93.9 | 2.54e-6 |
| 5 | `fit_eta_m026g01` | upper | 47 | `descent-check`, `high-projected-gradient` | 0.00476 | 0.0229 | -0.0401 | 1.64e-4 | 0.194 | -2.85e-11 |
| 6 | `fit_b0_s042b95` | upper | 42 | `boundary/chart-saturation` | -0.00414 | 9.09e-16 | -0.00414 | 3.39e-7 | 93.9 | -4.26e-14 |
| 7 | `fit_b0s_s08637` | lower | 35 | `call-heavy`, `descent-check`, `high-projected-gradient` | -0.000857 | 0.0199 | 0.0401 | 1.82e-5 | 1.75 | -2.91e-11 |
| 8 | `fit_b0s_s08637` | upper | 35 | `call-heavy`, `descent-check`, `high-projected-gradient` | 0.00115 | 0.0192 | -0.0362 | 1.82e-5 | 1.75 | -1.05e-11 |
| 9 | `avg_m002w` | lower | 30 | `call-heavy` | -0.00500 | 0.00638 | 0.00769 | n/a | n/a | n/a |
| 10 | `fit_eta_m026w` | upper | 25 | `descent-check`, `high-projected-gradient` | -0.000935 | 1.28e-16 | -0.000935 | 1.64e-4 | 0.194 | -1.82e-7 |

All 20 production endpoints still satisfy the profile residual invariant within
the current endpoint tolerance. The largest absolute production residual was
`0.0049997`, on `avg_m002w` lower, exactly at the existing bisection tolerance
edge.

## Lower Fixed-Target Chi2 Flags

The harness intentionally flagged covariance/current candidate differences below
`-1e-6`. It found three such rows:

| Case | Side | Fixed target | Best lower chi2 delta | Context |
| --- | --- | ---: | ---: | --- |
| `eta_c J/psi psi(2S)::M026W` | lower | production endpoint | -2.01e-6 | covariance start vs current start at the same fixed target |
| `eta_c J/psi psi(2S)::nuisance_M071.254` | lower | production endpoint | -1.26e-6 | covariance start vs current start at the same fixed target |
| `eta_c J/psi psi(2S)::nuisance_M071.254` | upper | parabolic endpoint | -1.46e-6 | covariance start vs current start at the same fixed target |

These are lower fixed-target chi2 candidates, but they are not endpoint
correctness failures:

- Endpoint values did not move in the diagnostic root rerun.
- The largest delta is `2.01e-6`, far below the existing `1e-4`
  descent-check function tolerance.
- The affected endpoints already carry descent-check/high-gradient status, so
  this is a useful risk/status signal, not evidence that the reported endpoint
  misses `chi2_min + 1`.

Answer to the required acceptance question: this round did **not** find a real
endpoint-correctness issue. It did find sub-`1e-5` lower fixed-target chi2
candidates in notes-only comparisons, and those should be carried forward as
risk markers.

## HESSE/Profile Findings

HESSE/parabolic disagreement stratified cases usefully, but not by itself.

- Nearly quadratic direct-coordinate fits are benign:
  `G(2000),G(1800)::K002M/K003M` had relative HESSE/profile disagreement below
  `5e-15` and parabolic endpoint residuals below `2e-10`.
- Well-conditioned but mildly nonlinear cases are visible:
  `B0S-BR::S086.37` had about `1.9%` HESSE/profile error disagreement and the
  parabolic endpoint missed the profiled target by `0.036` to `0.040` chi2.
- The tiny eta node `M026G01` had the largest relative HESSE/profile
  disagreement, `2.29%`, and a parabolic endpoint residual of `-0.0401`.
- Direct `eta_c...::M026W` had essentially zero HESSE/profile endpoint
  disagreement, which explains why covariance starts were locally reasonable
  there, but the endpoint still has high projected-gradient/descent-check status.
- `B0::S042B95` also had essentially zero HESSE/profile endpoint disagreement,
  but the chart metrics dominate: minimum decay parameter `3.39e-7` and maximum
  fitted decay coordinate `93.9`.

Rejected quantitatively: using HESSE/parabolic endpoints as a replacement for
profile endpoints. The B0S and eta node cases miss the profile target by up to
about `0.04` chi2 at the parabolic endpoint, while production endpoints verify
against `chi2_min + 1`.

## Round 8 Behavior Explained

The round 8 covariance-start behavior is mostly explained by chart and local
conditioning diagnostics:

- `B0::S042B95` is the clear chart-saturation negative case. HESSE/profile
  target displacement looks locally symmetric, but the fitted chart is saturated
  (`max_abs_decay_fitted_coord ~= 93.9`, `min_decay_param ~= 3.39e-7`). This
  explains why covariance-only starts can preserve the endpoint but give worse
  solve behavior and less trustworthy certificates.
- `B0S-BR::S086.37` has no severe chart saturation (`min_decay_param ~=
  1.82e-5`, `max_abs_decay_fitted_coord ~= 1.75`) but does show mild
  HESSE/profile nonlinearity (`~2%`, parabolic residual `~0.04`) and remains
  call-heavy/descent-check. This supports "mixed; verify profile" rather than
  "blind covariance start."
- `eta_c...::M026W` is well behaved in local HESSE target space and not chart
  saturated. The tiny lower fixed-target chi2 candidate and round 8 nfev hint
  are consistent with covariance starts being locally useful there, but the
  effect is far too small to justify a production change.

## Production Recommendation

No production method change is justified.

Reasons:

- No endpoint-correctness failure was found.
- The only lower fixed-target chi2 candidates are at the `1e-6` to `2e-6`
  level, below the solver's existing descent-check tolerance and with unchanged
  endpoints.
- HESSE/parabolic endpoints fail as a general substitute on nonlinear cases.
- Universal covariance starts remain rejected: chart-saturated `B0::S042B95`
  is still the counterexample, while well-conditioned wins are small and
  certificate-level only.

The practical outcome is diagnostic: risk tags are useful and should become
more systematic before any method change.

## Recommended Next Round

Do a structured `MnCross` status summary and targeted stratified failure/globality
search.

Concrete target set:

- All descent-check/high-projected-gradient endpoints.
- The three sub-`1e-5` lower fixed-target chi2 candidates above.
- Chart-saturated BR/BRU endpoints like `B0::S042B95`.
- Mild HESSE/profile nonlinear cases like `B0S-BR::S086.37` and
  `eta_c...::M026G01`.

Goal: determine whether these risk tags ever escalate from numerical-tie/status
signals into a real lower constrained profile or endpoint residual failure.
