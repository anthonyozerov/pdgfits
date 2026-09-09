# Asym Refactor Round 01

Date: 2026-06-24 22:28 PDT

Branch: `codex/asym-refactor-simplify-20260624-221852`

Code commit: `8fa8c56 perf: streamline asymmetry profiling`

## Scope

Round 1 was intentionally conservative: read the scientific/method notes, inspect
the inherited dirty worktree, decide whether the carried JAX/asymmetric-error
efficiency edits should survive, and make at most a low-risk improvement after
validation.

Required context read:

- `SCIENTIFIC_CONTEXT.md`
- `AGENTS.md`
- `src/pdgfits/CLAUDE.md`
- `notes/asymmetric-error-algorithm-evidence.md`
- `notes/asym-profile-simplification.md`
- `notes/asym-fit-simplified-broad-sweep.md`
- `notes/endpoint-search/asym-endpoint-search-exploration.md`
- `notes/asym-jax-efficiency-20260624-190946.md`
- `notes/asym-efficiency-round2-20260624-194504.md`

## Worktree Triage

Initial state had modified `src/pdgfits/asym_errors.py` and
`src/pdgfits/avg.py`, plus untracked efficiency reports/CSV/JSONL artifacts and
untracked `notes/logs/`.

Decision:

- Keep and commit the carried source changes after validation. They preserve the
  profile-likelihood endpoint invariant and have focused timing evidence.
- Commit the small markdown/CSV/JSONL efficiency artifacts because they are the
  evidence for the carried changes.
- Do not commit `notes/logs/`; it includes historical bulky logs and current
  generated logs. The useful results are summarized in committed markdown/CSV.
- Do not revert the carried changes. They are not speculative cleanup; they are
  focused performance changes with unchanged residual behavior on the tested
  hard cases.

## Algorithm Map

Current fit-side asymmetric errors remain profile-likelihood endpoints:

1. For each scalar target, use a constrained profile solver in fitted-parameter
   space.
2. Project starts to the fixed-target constraint.
3. Try SLSQP first, then exact-Hessian `trust-constr` when needed.
4. KKT-polish exact-Hessian candidates.
5. Accept only finite, scaled-feasible points with projected/KKT stationarity or
   the existing no-meaningful-feasible-descent certificate.
6. Use bracketed bisection for endpoints and verify
   `profile_chi2(endpoint) = chi2_min + 1` within tolerance.

Average-side profiling remains the direct fixed-primary-coordinate nuisance
profile. That is still the simpler mathematical problem for averages and avoids
generic nonlinear equality constraints.

No change was made to `build_chi2.py`, the asymmetric-error interpolation
formula, endpoint tolerance, bracketed bisection logic, or acceptance criteria.

## Changes Kept Or Added

From the carried work:

- `avg.py`: average nuisance-profile BFGS now uses a JIT-compiled
  `jax.value_and_grad(chi2)` callable with `jac=True`, avoiding separate value
  and gradient evaluations.
- `avg.py`: Nelder-Mead is still available but skipped when BFGS succeeds or
  when the existing nuisance-gradient certificate already accepts the profile
  point.
- `asym_errors.py`: `calc_asym_errors()` builds the objective gradient and
  Hessian JIT wrappers once per fit and shares them across target profiles.
- `asym_errors.py`: candidate acceptance checks cache their boolean result on
  the immutable profile candidate object to avoid repeated projected-gradient or
  descent-check work.
- `asym_errors.py`: `calc_asym_errors()` no longer solves a constrained profile
  at the MLE target value. The unconstrained optimum already satisfies the MLE
  target; the code now checks target consistency and chi2 at the fitted optimum.

Round 1 added one tiny refinement:

- Move the `chi2(fitted_values)` MLE check outside the target loop in
  `calc_asym_errors()`. This preserves the same check and avoids recomputing the
  identical base chi2 once per target.

## Evidence From Carried Reports

`notes/asym-jax-efficiency-20260624-190946.md`:

- Four average-node probe: `4/4` ok, total average runtime `18.526 s` to
  `17.686 s`, max abs fresh residual unchanged at `0.004999656`.
- `Upsilon(2S)` six-target derivative-reuse probe: asym runtime `8.836 s` to
  `6.881 s`, endpoint function evals unchanged at `45`, max abs residual
  unchanged at `0.003450197`.
- Rejected `build_chi2(..., use_jit=False)` for averages: `18.283 s` to
  `142.495 s` on the same four average nodes.

`notes/asym-efficiency-round2-20260624-194504.md`:

- Four average-node BFGS-gated nuisance-profile probe: `4/4` ok, total average
  runtime `17.743 s` to `12.468 s`, max abs fresh residual unchanged at
  `0.004999656`.
- Twelve-node focused average validation: `12/12` ok, max abs fresh residual
  `0.004999656`.
- `Upsilon(2S)` direct-MLE base-check probe: asym runtime `6.926 s` to
  `6.647 s`, max abs residual unchanged at `0.003450197`.
- Focused fit validation after candidate-result cache: `7/7` ok, max abs
  residual `0.003450197`.

## Round 1 Validation

All commands used the snapshot backend:

```bash
env PYTHONPATH=/root/pdgfits-private/src \
  PDGFITS_DATA_BACKEND=snapshot \
  PDGFITS_SNAPSHOT_DIR=/root/pdgfits-private/data/pdg-snapshot \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  /root/micromamba/micromamba run -r /root/micromamba-root -n pdg <command>
```

Targeted unit tests:

```bash
python -m pytest tests/test_asym_errors.py tests/test_build_chi2.py tests/test_build_funcs.py -q
```

Result: `21 passed, 1 skipped in 7.19s`.

Full non-DB suite:

```bash
python -m pytest tests/ -q
```

Result: `160 passed, 2 skipped in 13.48s`.

Focused fit/average validation harness:

```bash
python notes/endpoint-search/run_asym_endpoint_search_benchmark.py \
  --variant codex-round01 \
  --fit-csv notes/codex-refactor/round01_fit_validation.csv \
  --fit-jsonl notes/codex-refactor/round01_fit_validation.jsonl \
  --avg-csv notes/codex-refactor/round01_avg_validation.csv \
  --avg-jsonl notes/codex-refactor/round01_avg_validation.jsonl \
  --stdout-log notes/logs/codex_refactor_round01_validation_stdout.log \
  --fit-target 'Upsilon(2S)::M052.4' \
  --fit-target 'Upsilon(2S)::M052.6' \
  --fit-target 'Upsilon(2S)::M052R22' \
  --fit-target 'Upsilon(2S)::M052R4' \
  --fit-target 'Upsilon(2S)::M052R6' \
  --fit-target 'Upsilon(2S)::nuisance_M048.8' \
  --fit-target 'B0::S042B09' \
  --avg-node M002W --avg-node S042CKS --avg-node M026R08 --avg-node S086R46 \
  --target-timeout-sec 300
```

Fit results:

- `7/7` rows ok, `0` errors.
- Max abs endpoint residual: `0.003450196954705831`.
- Runtime sum over target rows: `16.549223863024963 s`.
- Total profile calls: `70`.

Average results:

- `4/4` rows ok, `0` errors.
- Max abs fresh endpoint residual: `0.004999656283968035`.
- Runtime sum: `12.630297037016135 s`.
- Total profile calls: `88`.

Representative public fit CLI:

```bash
python -m pdgfits.run_fits --fit_label 'Upsilon(2S)' --calc_asym_errors
```

Result: completed successfully. Six printed targets were verified; max printed
absolute residual was about `0.0035`.

Representative public average CLI:

```bash
python -m pdgfits.run_avgs --node M026R08 --output notes/codex-refactor/round01_m026r08_run_avgs.csv
```

Result: completed successfully. Printed final unscaled asymmetric errors
`0.019546087791061734` and `0.01987005609699094`. Existing CLI behavior writes
`--output` only when more than one result row exists, so this single-node run did
not create the requested CSV; I did not patch that unrelated behavior in this
round.

## Rejected Ideas

- Rejected changing `build_chi2` or asymmetric interpolation. No blocking
  scientific issue was found.
- Rejected changing endpoint root search. Prior endpoint-search evidence already
  found the safe win: final bisection-value reuse. A Brent/secant style change
  would add another tolerance and needs a dedicated hard-case benchmark.
- Rejected average central-value BFGS-first optimization, per the carried
  round-2 report: it caused small but real central value/chi2 drift and was not
  consistently faster.
- Rejected direct-coordinate special casing for fit profiles in this round. The
  generic constrained fit profiler has stronger hard-case evidence, and the safe
  direct-coordinate simplification only clearly applies to averages.

## Remaining Risks

- This round did not rerun the staged broad fit sweep. The source changes were
  validated on focused hard/representative targets plus tests.
- The average-side BFGS gating is supported on focused nuisance-heavy cases, not
  a fresh full average snapshot sweep.
- The direct MLE base check in `calc_asym_errors()` is mathematically
  straightforward, but it removes an independent constrained solve at the MLE
  target. Endpoint residual verification remains the real endpoint check.
- The fit-side constrained profile solver still provides local stationarity or
  descent evidence, not a proof of global constrained optimality.
- The least theorem-like acceptance route remains `+descent-check`; optional
  multistart diagnostics for those endpoints remain a high-value future task.

## Next High-Value Work

The next round should not add solver machinery by default. Better options:

- Run a focused multistart diagnostic on endpoints accepted via
  `+descent-check`, especially hard `Lam-b-0`, `B0`, and
  `eta_c J/psi psi(2S)` targets.
- If average-side performance remains a priority, rerun the full average sweep
  after the BFGS-gated nuisance-profile change and compare residuals and
  central values against the previous full sweep.
- Consider making `run_avgs --node ... --output ...` write a single-row CSV, but
  treat that as CLI hygiene rather than asymmetric-error methodology.
