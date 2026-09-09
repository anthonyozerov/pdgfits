# Asym Refactor Round 05 ProfileRoot Prototype

Date: 2026-06-24 23:35 PDT

Branch: `codex/asym-refactor-simplify-20260624-221852`

## Summary

This round made a small production refactor around the existing asymmetric-error
machinery. The scientific model, fit-side constrained solver, average-side
fixed-coordinate profile, endpoint bisection, bracket contraction, and endpoint
residual tolerance are unchanged. `build_chi2.py` was not edited.

The new interface layer makes profile evaluation and endpoint root search
explicit:

- `ProfilePoint`: existing per-profile-evaluation certificate record.
- `ProfileProblem`: protocol for `evaluate(value) -> ProfilePoint`.
- `CallableProfileProblem`: adapter for existing profile closures.
- `ProfileEndpoint`: one verified endpoint plus the endpoint `ProfilePoint`.
- `ProfileRoot`: both sides of the bracketed profile root search plus counters.
- `find_profile_root(...)`: explicit root-search entry point.
- `binary_search_error(...)`: compatibility wrapper returning `(error_p,
  error_n)` and filling `last_diagnostics` / `last_root`.

## Files Changed

- `src/pdgfits/asym_errors.py`
  - Added `ProfileProblem`, `CallableProfileProblem`, `ProfileEndpoint`,
    `ProfileRoot`, and `find_profile_root`.
  - Moved the existing bisection implementation behind `find_profile_root`.
  - Kept `binary_search_error` as a compatibility wrapper with the same return
    values and legacy diagnostics keys.
  - Added nested endpoint point diagnostics:
    `upper_profile_point` and `lower_profile_point`.
  - Wrapped `calc_asym_errors` target profiles in `CallableProfileProblem`.
- `src/pdgfits/avg.py`
  - Set `profile_chi2.last_point` for average profile evaluations.
  - Wrapped the direct average profile closure in `CallableProfileProblem`.
- `tests/test_asym_errors.py`
  - Added tests for `find_profile_root` and endpoint point diagnostics.
- `src/pdgfits/CLAUDE.md`
  - Documented the new profile/root interface.
- Validation artifacts:
  - `notes/codex-refactor/round05_profileroot_fit_hard.csv`
  - `notes/codex-refactor/round05_profileroot_fit_hard.jsonl`
  - `notes/codex-refactor/round05_profileroot_avg_m026r08.csv`
  - `notes/codex-refactor/round05_profileroot_avg_m026r08.jsonl`
  - this report

Raw stdout logs were written under `notes/logs/` and should remain uncommitted.

## Commands Run

Common environment:

```bash
env PYTHONPATH=/root/pdgfits-private/src \
  PDGFITS_DATA_BACKEND=snapshot \
  PDGFITS_SNAPSHOT_DIR=/root/pdgfits-private/data/pdg-snapshot \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  /root/micromamba/micromamba run -r /root/micromamba-root -n pdg <command>
```

Focused tests:

```bash
python -m pytest tests/test_asym_errors.py -q
```

Final result after the last code edit: `5 passed in 4.20s`.

Full tests:

```bash
python -m pytest tests/ -q
```

Final result after the last code edit: `161 passed, 2 skipped in 13.84s`.

Focused fit hard set:

```bash
prlimit --as=7800000000 --rss=7800000000 -- \
  env PYTHONPATH=/root/pdgfits-private/src \
  PDGFITS_DATA_BACKEND=snapshot \
  PDGFITS_SNAPSHOT_DIR=/root/pdgfits-private/data/pdg-snapshot \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  /root/micromamba/micromamba run -r /root/micromamba-root -n pdg \
  python notes/run_asym_profile_ablation.py \
    --skip-avgs \
    --variant minimal \
    --fit-target 'B0::S042B95' \
    --fit-target 'B0S-BR::S086.37' \
    --fit-target 'eta_c J/psi psi(2S)::M026W' \
    --fit-csv notes/codex-refactor/round05_profileroot_fit_hard.csv \
    --fit-jsonl notes/codex-refactor/round05_profileroot_fit_hard.jsonl \
    --stdout-log notes/logs/round05_profileroot_fit_hard_stdout.log \
    --target-timeout-sec 240
```

Focused average validation:

```bash
prlimit --as=7800000000 --rss=7800000000 -- \
  env PYTHONPATH=/root/pdgfits-private/src \
  PDGFITS_DATA_BACKEND=snapshot \
  PDGFITS_SNAPSHOT_DIR=/root/pdgfits-private/data/pdg-snapshot \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  /root/micromamba/micromamba run -r /root/micromamba-root -n pdg \
  python notes/endpoint-search/run_asym_endpoint_search_benchmark.py \
    --variant profileroot-round05 \
    --skip-fits \
    --avg-node M026R08 \
    --avg-csv notes/codex-refactor/round05_profileroot_avg_m026r08.csv \
    --avg-jsonl notes/codex-refactor/round05_profileroot_avg_m026r08.jsonl \
    --stdout-log notes/logs/round05_profileroot_avg_m026r08_stdout.log
```

Public `calc_asym_errors` call-site check:

```bash
prlimit --as=7800000000 --rss=7800000000 -- \
  env PYTHONPATH=/root/pdgfits-private/src \
  PDGFITS_DATA_BACKEND=snapshot \
  PDGFITS_SNAPSHOT_DIR=/root/pdgfits-private/data/pdg-snapshot \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  /root/micromamba/micromamba run -r /root/micromamba-root -n pdg \
  python - <<'PY'
import jax
jax.config.update('jax_enable_x64', True)
from pdgfits.fit import run_fit
from pdgfits.asym_errors import calc_asym_errors
fit = run_fit('B0', verbose=False)
res = calc_asym_errors(fit, targets=['S042B95'])
row = res['S042B95']
print('S042B95', row['error_p'], row['error_n'], row['upper_residual'], row['lower_residual'], row['function_evals'])
print('upper_method', row['upper_profile_point']['method'])
print('lower_method', row['lower_profile_point']['method'])
PY
```

Result:

```text
S042B95 0.015264802654805842 0.015264802654805842 -0.0041375141125143955 0.0011561377639424109 4
upper_method trust-constr-exact-hess+KKT
lower_method trust-constr-exact-hess+KKT+descent-check
```

Comparison scripts:

```bash
python - <<'PY'
import pandas as pd
new = pd.read_csv('notes/codex-refactor/round05_profileroot_fit_hard.csv')
old = pd.read_csv('notes/endpoint-search/asym_endpoint_search_fit_cache_reuse.csv')
cols = ['error_p','error_n','upper_endpoint','lower_endpoint','upper_residual','lower_residual','upper_chi2','lower_chi2','profile_calls']
merged = new.merge(old[['label','target'] + cols], on=['label','target'], suffixes=('_new','_old'))
for _, row in merged.iterrows():
    diffs = {col: float(row[f'{col}_new']) - float(row[f'{col}_old']) for col in cols}
    print(f"{row['label']}::{row['target']} " + ' '.join(f"{k}_diff={v:.3g}" for k, v in diffs.items()))
PY
```

```bash
python - <<'PY'
import pandas as pd
avg_new = pd.read_csv('notes/codex-refactor/round05_profileroot_avg_m026r08.csv')
avg_old = pd.read_csv('notes/codex-refactor/round02_avg_direct_mle_check.csv')
row_new = avg_new.iloc[0]
row_old = avg_old[avg_old['node'] == 'M026R08'].iloc[0]
for col in ['error_p','error_n','upper_endpoint','lower_endpoint','upper_residual','lower_residual','fresh_upper_residual','fresh_lower_residual','endpoint_function_evals','profile_calls']:
    print(col, float(row_new[col]) - float(row_old[col]))
PY
```

## Quantitative Results

Fit hard set, compared with
`notes/endpoint-search/asym_endpoint_search_fit_cache_reuse.csv`:

| Target | Status | Endpoint/value diffs | Residual diffs | Profile call diff |
| --- | --- | ---: | ---: | ---: |
| `B0::S042B95` | ok | all `0` | upper `0`, lower `0` | `0` |
| `B0S-BR::S086.37` | ok | all `0` | upper `0`, lower `0` | `0` |
| `eta_c J/psi psi(2S)::M026W` | ok | all `0` | upper `0`, lower `0` | `0` |

Fit hard-set endpoint residuals:

| Target | Upper residual | Lower residual | Upper method | Lower method |
| --- | ---: | ---: | --- | --- |
| `B0::S042B95` | `-0.0041375141125143955` | `0.0011561377639424109` | `trust-constr-exact-hess+KKT` | `trust-constr-exact-hess+KKT+descent-check` |
| `B0S-BR::S086.37` | `0.001151760087164888` | `-0.0008567282674860621` | `SLSQP+descent-check` | `SLSQP+descent-check` |
| `eta_c J/psi psi(2S)::M026W` | `-0.0009350525845945867` | `0.0007079364693538537` | `SLSQP+descent-check` | `SLSQP+descent-check` |

Average validation for `M026R08`, compared with
`notes/codex-refactor/round02_avg_direct_mle_check.csv`:

| Quantity | Round 05 | Prior | Diff |
| --- | ---: | ---: | ---: |
| `error_p` | `0.01987005609699094` | `0.01987005609699094` | `0` |
| `error_n` | `0.019546087791061734` | `0.019546087791061734` | `0` |
| `upper_endpoint` | `0.17926974749256475` | `0.17926974749256475` | `0` |
| `lower_endpoint` | `0.13985360360451207` | `0.13985360360451207` | `0` |
| `upper_residual` | `0.0012911859348374577` | `0.0012911859348374577` | `0` |
| `lower_residual` | `0.0017222474233671292` | `0.0017222474233671292` | `0` |
| `fresh_upper_residual` | `0.0012911859348374577` | `0.0012911859348374577` | `0` |
| `fresh_lower_residual` | `0.0017222474233666851` | `0.0017222474233666851` | `0` |
| `endpoint_function_evals` | `19` | `19` | `0` |
| `profile_calls` | `19` | `19` | `0` |

## Behavior Preserved

- Endpoint invariant preserved:
  `profile_chi2(endpoint) = chi2_min + 1` within the existing bisection
  tolerance.
- Fit-side profile solver preserved:
  SLSQP, exact-Hessian `trust-constr`, KKT polish, descent check, and bracket
  contraction are unchanged.
- Average-side profile solver preserved:
  direct fixed-primary-coordinate nuisance profiling remains separate from the
  generic equality-constrained fit profiler.
- Endpoint search preserved:
  bracketed bisection remains the default and only root algorithm.
- Public compatibility preserved:
  existing callers of `binary_search_error(...)` still receive `(error_p,
  error_n)` and the old top-level diagnostics keys.

## Rejected Designs

- Did not force averages through `build_constrained_profile_chi2`. The direct
  fixed-coordinate average profile is simpler and separately validated.
- Did not add solver fallbacks. The round-03 and round-04 evidence did not show
  a concrete failure requiring more machinery.
- Did not replace bisection with secant, Newton, or Brent. Root-finder changes
  would not address profile-minimization quality and need a separate hard-case
  benchmark.
- Did not split fit and average implementations into new full production
  classes. The closure adapter gives the endpoint root search explicit
  `ProfilePoint` objects with much lower migration risk.
- Did not edit `build_chi2.py` or the asymmetric interpolation formula.

## Remaining Risks

- This is an interface prototype, not a global-optimality proof. Fit-side
  constrained profile certificates are still local KKT/projected-gradient or
  local no-meaningful-descent certificates.
- `CallableProfileProblem` still wraps closures. A later round can turn the fit
  and average profilers into first-class problem classes once there is a clear
  diagnostic or migration benefit.
- The new nested `upper_profile_point` / `lower_profile_point` diagnostics are
  serializable with the repository's existing `clean(...)` helpers, but any
  external strict-schema consumer of `last_diagnostics` should ignore unknown
  keys or adopt the new fields deliberately.
- Validation here is focused, not a broad all-target sweep. It covered the
  requested hard fit targets, one average target, unit tests, full tests, and a
  public `calc_asym_errors` hard-target call.

## Recommended Next Round

Use the new `ProfileRoot` object in diagnostic harnesses before changing more
production behavior. A good next step is a notes-only diagnostic that writes
endpoint `ProfilePoint` fields directly from `binary_search_error.last_root`
instead of re-matching endpoint coordinates against `profile.diagnostics`.
That would reduce fragile post-hoc matching and make future solver comparisons
cleaner without changing the numerical algorithm.
