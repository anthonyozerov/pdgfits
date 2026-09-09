# Asymmetric Profile Simplification

Date: 2026-06-24

Branch: `codex-cloud-asym-avg-local-experiments`

Baseline context: the previous committed/pushed baseline was
`a809476 fix: resolve focused fit-side asymmetry failures`.

## Summary

The fit-side profile solver was simplified. The focused hard set passes without
the BFGS-Hessian `trust-constr` fallback and without the reduced/nullspace
Powell fallback, so both paths were removed from `src/pdgfits/asym_errors.py`.

The remaining fit-side method is:

1. Project the continuation and best-fit starts onto the scalar target
   constraint.
2. Try SLSQP first.
3. If SLSQP is not accepted, try exact-Hessian `trust-constr`.
4. KKT-polish exact-Hessian candidates.
5. Accept only finite, scaled-feasible profiles with either projected/KKT
   stationarity or a no-meaningful-feasible-descent certificate.
6. Endpoint-search brackets may contract toward the MLE when an outward
   bracket target is unreachable.
7. Every returned endpoint is still verified against `chi2_min + 1`.

Averages were left on the simpler direct fixed-primary-coordinate nuisance
profile. That path was already full-sweep validated and remains conceptually
cleaner for single-node averages.

## Code Changes

- Removed fit-side BFGS-Hessian `trust-constr` fallback.
- Removed fit-side reduced/nullspace Powell fallback.
- Added `ProfileSolverOptions` so validation harnesses can ablate SLSQP,
  exact-Hessian fallback, KKT polish, descent check, and bracket contraction
  without source edits.
- Extended `notes/run_asym_fit_sweep.py::run_one_target()` to accept those
  solver options.
- Added focused harness: `notes/run_asym_profile_ablation.py`.

Machine-readable outputs from this pass:

- `notes/asym_profile_ablation_fit_minimal_final.csv`
- `notes/asym_profile_ablation_fit_minimal_final.jsonl`
- `notes/asym_profile_ablation_avg.csv`
- `notes/asym_profile_ablation_avg.jsonl`
- `notes/asym_profile_ablation_fit_restrictive.csv`
- `notes/asym_profile_ablation_fit_nodescent_fast.csv`
- `notes/asym_profile_ablation_fit_no_contract_eta2s.csv`
- `notes/asym_profile_ablation_fit_exact_kkt_b0s.csv`

## Focused Hard-Case Set

Fit focused set, final run:

- Previous 13 focused failures from `notes/asym_fit_failure_focus_final.csv`.
- All `Upsilon(2S)` targets from the existing sweep.
- Hard `Lam-b-0` subset: `S040.10`, `S040.29`, `S040.9`,
  `S040R29`, `S040R9`, `nuisance_S042.30`.
- `eta_c J/psi psi(2S) / M026.45` as a slow previous success.

Average focused set:

- `M026R08`.
- Slowest and nuisance-heavy rows from `notes/asym_avg_sweep_results.csv`.
- Largest residual-boundary examples from `notes/asym_avg_sweep_results.csv`.

## Ablation Results

| Variant | Paths enabled | Scope run | Status | Max abs endpoint residual | Slowest target | Failure reason |
| --- | --- | ---: | --- | ---: | ---: | --- |
| `minimal` | SLSQP, exact-Hessian `trust-constr`, KKT, descent check, bracket contraction | 26 fit targets | 26 ok | 0.0049359 | 87.90 s | None |
| average direct profile | fixed primary coordinate, BFGS/Nelder-Mead nuisance profile | 12 averages | 12 ok | 0.0049997 | 12.28 s | None |
| `slsqp_kkt` | SLSQP + KKT only | first 2 targeted rows | 2 errors | n/a | n/a | `B0` feasible/KKT-polished points still had projected gradient about `4.96e3` |
| `no_descent` | SLSQP, exact-Hessian `trust-constr`, KKT, no descent certificate | 2 targeted rows | 2 errors | n/a | n/a | `eta_c J/psi psi(2S)` feasible optimizer-success points had projected gradient about `1.89` |
| `minimal_no_contract` | minimal solver, no bracket contraction | 1 targeted row | 1 error | n/a | n/a | `eta_c(2S) / M059.4` upper bracket reached scaled constraint violation `80.3` |
| `exact_kkt` | exact-Hessian `trust-constr` + KKT only | 1 targeted row | 1 ok | 0.0000881 | 20.46 s | Passed `B0S-BR / S086R04`, but was not practical as sole first-line method in broader targeted attempt |

Endpoint methods in the final 26-target minimal fit run:

| Method | Endpoint count |
| --- | ---: |
| `SLSQP+descent-check` | 25 |
| `SLSQP` | 17 |
| `trust-constr-exact-hess+KKT` | 8 |
| `trust-constr-exact-hess+KKT+descent-check` | 1 |
| `trust-constr-exact-hess+descent-check` | 1 |

No final accepted endpoint used BFGS-Hessian `trust-constr` or reduced Powell.

## Commands

Common environment:

```bash
env PYTHONPATH=/root/pdgfits-private/src \
  PDGFITS_DATA_BACKEND=snapshot \
  PDGFITS_SNAPSHOT_DIR=/root/pdgfits-private/data/pdg-snapshot \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  /root/micromamba/micromamba run -r /root/micromamba-root -n pdg <command>
```

Focused final fit harness:

```bash
python notes/run_asym_profile_ablation.py \
  --skip-avgs \
  --variant minimal \
  --fit-csv notes/asym_profile_ablation_fit_minimal_final.csv \
  --fit-jsonl notes/asym_profile_ablation_fit_minimal_final.jsonl \
  --stdout-log notes/logs/asym_profile_ablation_minimal_final_stdout.log \
  --target-timeout-sec 240
```

Focused average harness:

```bash
python notes/run_asym_profile_ablation.py \
  --skip-fits \
  --avg-csv notes/asym_profile_ablation_avg.csv \
  --avg-jsonl notes/asym_profile_ablation_avg.jsonl \
  --stdout-log notes/logs/asym_profile_ablation_avg_stdout.log
```

Selected negative ablations:

```bash
python notes/run_asym_profile_ablation.py \
  --skip-avgs \
  --variant slsqp_kkt \
  --fit-target 'B0::S042.390' \
  --fit-target 'B0::S042B09' \
  --fit-csv notes/asym_profile_ablation_fit_restrictive.csv \
  --fit-jsonl notes/asym_profile_ablation_fit_restrictive.jsonl \
  --stdout-log notes/logs/asym_profile_ablation_restrictive_stdout.log \
  --target-timeout-sec 120
```

```bash
python notes/run_asym_profile_ablation.py \
  --skip-avgs \
  --variant no_descent \
  --fit-target 'eta_c J/psi psi(2S)::M026W' \
  --fit-target 'eta_c J/psi psi(2S)::nuisance_M071.254' \
  --fit-csv notes/asym_profile_ablation_fit_nodescent_fast.csv \
  --fit-jsonl notes/asym_profile_ablation_fit_nodescent_fast.jsonl \
  --stdout-log notes/logs/asym_profile_ablation_nodescent_fast_stdout.log \
  --target-timeout-sec 120
```

```bash
python notes/run_asym_profile_ablation.py \
  --skip-avgs \
  --variant minimal_no_contract \
  --fit-target 'eta_c(2S)::M059.4' \
  --fit-csv notes/asym_profile_ablation_fit_no_contract_eta2s.csv \
  --fit-jsonl notes/asym_profile_ablation_fit_no_contract_eta2s.jsonl \
  --stdout-log notes/logs/asym_profile_ablation_no_contract_eta2s_stdout.log \
  --target-timeout-sec 120
```

```bash
python notes/run_asym_profile_ablation.py \
  --skip-avgs \
  --variant exact_kkt \
  --fit-target 'B0S-BR::S086R04' \
  --fit-csv notes/asym_profile_ablation_fit_exact_kkt_b0s.csv \
  --fit-jsonl notes/asym_profile_ablation_fit_exact_kkt_b0s.jsonl \
  --stdout-log notes/logs/asym_profile_ablation_exact_kkt_b0s_stdout.log \
  --target-timeout-sec 120
```

Validation:

```bash
python -m pytest tests/test_asym_errors.py -q
python -m pytest tests/ -q
python -m pdgfits.run_avgs --node M026R08 --output notes/avg_m026r08_profile_simplification.csv
python -m pdgfits.run_fits --fit_label 'Upsilon(2S)' --calc_asym_errors
python -m pdgfits.run_fits --fit_label 'Lam-b-0' --calc_asym_errors
```

## Validation Results

- `tests/test_asym_errors.py`: `3 passed in 4.03s`.
- `tests/`: `159 passed, 2 skipped in 13.90s`.
- `M026R08`: completed; printed scaled asymmetric errors
  `0.019546087791061734 0.01987005609699094`.
- `Upsilon(2S)`: completed; max printed residual about `0.0035`.
- `Lam-b-0`: completed all 20 reported targets; max printed residual about
  `0.0049`.

## Caveats

- This is focused hard-case validation, not a full all-target fit sweep.
- Runtime is still high for `Lam-b-0 / S040.29` and `S040R29`
  (about 88 seconds each in the focused final run).
- The descent certificate is still a numerical local certificate, not a clean
  analytic KKT solution. It is retained because removing it immediately fails
  known `eta_c J/psi psi(2S)` hard targets.
- Exact-Hessian `trust-constr` is retained as fallback because SLSQP+KKT fails
  known `B0` hard targets. It is not used as the sole first-line method because
  it can be materially slower.
- Bracket contraction is retained because `eta_c(2S) / M059.4` fails without it.

## Next Recommended Sweep

Rerun a staged broad fit-side sweep next, using the simplified solver:

```bash
python notes/run_asym_fit_sweep.py \
  --stage representative \
  --jsonl notes/asym_fit_sweep_results_simplified.jsonl \
  --csv notes/asym_fit_sweep_results_simplified.csv \
  --stdout-log notes/logs/asym_fit_sweep_simplified_stdout.log \
  --target-timeout-sec 900
```

Do not rerun the full average sweep unless average-side code changes; the
current average path was already full-sweep validated and the focused average
set still passes.
