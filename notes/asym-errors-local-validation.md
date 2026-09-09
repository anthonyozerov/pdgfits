# Local asymmetric-error validation

Date: 2026-06-23

Branch: `codex-cloud-asym-avg-local-experiments`

Applied cloud task under review: `task_e_6a3b26f71bf08333a99f0705d1fb1be7`

Environment:

- Python: `/root/micromamba/micromamba run -r /root/micromamba-root -n pdg python -V` -> `Python 3.11.15`
- Package install: `pip show pdgfits` in the `pdg` env reports editable location `/root/pdgfits-private`
- Snapshot backend: `data/pdg-snapshot`

## Commands and Results

Focused tests:

```bash
env PYTHONPATH=/root/pdgfits-private/src \
  PDGFITS_DATA_BACKEND=snapshot \
  PDGFITS_SNAPSHOT_DIR=/root/pdgfits-private/data/pdg-snapshot \
  /root/micromamba/micromamba run -r /root/micromamba-root -n pdg \
  python -m pytest tests/test_asym_errors.py -q
```

Result after local patches:

```text
3 passed in 4.25s
```

Broader offline tests:

```bash
env PYTHONPATH=/root/pdgfits-private/src \
  PDGFITS_DATA_BACKEND=snapshot \
  PDGFITS_SNAPSHOT_DIR=/root/pdgfits-private/data/pdg-snapshot \
  /root/micromamba/micromamba run -r /root/micromamba-root -n pdg \
  python -m pytest tests/ -q
```

Final result:

```text
159 passed, 2 skipped in 13.51s
```

Snapshot average scan:

```bash
env PYTHONPATH=/root/pdgfits-private/src \
  PDGFITS_DATA_BACKEND=snapshot \
  PDGFITS_SNAPSHOT_DIR=/root/pdgfits-private/data/pdg-snapshot \
  /root/micromamba/micromamba run -r /root/micromamba-root -n pdg \
  python - <<'PY'
# Scanned avg_queries() nodes, ran preprocess() for each node, and counted
# nuisance/adjust/dependent-measurement cases.
PY
```

Result:

```text
nodes 2647
preprocess_errors 0
nuisance nodes count 229
adjust nodes count 220
dep nodes count 9
correlated nodes count 0
```

Average endpoint validation command shape:

```bash
env PYTHONPATH=/root/pdgfits-private/src \
  PDGFITS_DATA_BACKEND=snapshot \
  PDGFITS_SNAPSHOT_DIR=/root/pdgfits-private/data/pdg-snapshot \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  /root/micromamba/micromamba run -r /root/micromamba-root -n pdg \
  python - <<'PY'
# Mixed deterministic sample:
# - all avg-node-ex.yaml "interesting" nodes present in the snapshot
# - top 25 nuisance averages by nuisance count/raw measurement count
# - all dependent-measurement averages in the snapshot sample cap
# - 20 smallest-scale averages
# - 12 largest-scale averages
# - 25 deterministic random remaining averages, seed 20260623
#
# For each run_avg() result, checked both returned endpoints by independently
# profiling chi2 at value +/- returned error. Wrote:
# notes/asym_avg_validation_final.csv
PY
```

Final average endpoint validation:

```text
sample_nodes 105
ok_averages 105
skipped 0
errors 0
nuisance_ok 52
dep_ok 9
post_corr_ok 16
endpoints_checked 210
median abs endpoint residual 0.00207857862999945
max abs endpoint residual 0.0049547342096245
```

Output CSV:

```text
notes/asym_avg_validation_final.csv
sha256 2111e2753f514a94fec3076e17cc14d35ef6fdc28ba931747805482f630c4caa
```

Before local patches, the same mixed average sample failed 22/105 averages. Most failures were tiny-scale averages where the constrained profiler accepted or searched at an absolute scale around `1e-9` for quantities with natural errors around `1e-15` to `1e-21`. After only dimensionless constraint scaling, failures dropped to 4/105, but fresh endpoint rechecks still showed path-dependent nuisance profiles. The final average-side direct nuisance profiler eliminated those failures in this sample.

Fit-side smoke checks:

```bash
env PYTHONPATH=/root/pdgfits-private/src \
  PDGFITS_DATA_BACKEND=snapshot \
  PDGFITS_SNAPSHOT_DIR=/root/pdgfits-private/data/pdg-snapshot \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  /root/micromamba/micromamba run -r /root/micromamba-root -n pdg \
  python -m pdgfits.run_fits --fit_label 'Upsilon(2S)' --calc_asym_errors
```

Result: completed. Reported endpoint residuals:

```text
M052.4 residuals=(-0.00051,0.00055)
M052.6 residuals=(-0.0011,0.0018)
M052R22 residuals=(-0.0018,0.0023)
M052R4 residuals=(-0.00051,0.00055)
M052R6 residuals=(-0.0011,0.0018)
nuisance_M048.8 residuals=(-0.0035,0.0033)
```

Nuisance-heavy fit-side blocker:

```bash
env PYTHONPATH=/root/pdgfits-private/src \
  PDGFITS_DATA_BACKEND=snapshot \
  PDGFITS_SNAPSHOT_DIR=/root/pdgfits-private/data/pdg-snapshot \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  /root/micromamba/micromamba run -r /root/micromamba-root -n pdg \
  python -m pdgfits.run_fits --fit_label 'Lam-b-0' --calc_asym_errors
```

Result: failed in fit-side constrained profiling. A retry after raising the constrained-profile iteration budget still failed:

```text
RuntimeError: Profile minimization failed for S040.10=3.310979027615594e-06:
success=False, finite=True, constraint_violation=1.22e-17,
scaled_constraint_violation=1.06e-11,
message=The maximum number of function evaluations is exceeded.
```

I did not weaken this into "success" because the stricter projected-gradient check did not certify a constrained optimum.

## Local Code Changes

- Fixed `query.py` f-string syntax that Python 3.11 rejects by precomputing quoted SQL lists. This was necessary for imports/tests in the supplied `pdg` environment.
- Added endpoint residual diagnostics and a tiny-scale regression test in `tests/test_asym_errors.py`.
- Changed fit-side constrained profile optimization to:
  - scale equality constraints dimensionlessly by the target's natural scale;
  - try continuation and projected best-fit starts;
  - use SLSQP with trust-constr fallback;
  - reject non-success statuses unless constraint feasibility and a projected-gradient/KKT check pass;
  - verify binary-search endpoints against `chi2_min + 1`.
- Changed average-side profiling to a direct fixed-primary-coordinate nuisance minimization instead of SLSQP equality constraints. This is mathematically equivalent for averages because the target is a direct primary parameter, and it removed the observed path-dependence for nuisance/tiny-scale averages.

## Assessment

The applied cloud diff was not scientifically robust as received. The main average-side issue was that returned endpoints could be finite without being verified against the true profiled `chi2_min + 1` crossing under fresh profiling. Tiny-scale averages also broke because constraint tolerances were effectively absolute rather than natural-scale relative.

The local average-side patch is much stronger: the final mixed snapshot sample checked 210 endpoints across ordinary, nuisance, dependent-measurement, tiny-scale, large-scale, and random averages, with zero failures and max absolute endpoint residual below `0.005`.

This is still not PR-ready for the whole issue. Fit-side arbitrary target profiling remains a blocker: `Upsilon(2S)` passes, but `Lam-b-0` still fails under a stricter optimizer certification. I would treat the current state as a candidate average-side fix plus a useful fit-side diagnostic hardening, not as a complete closure of asymmetric errors in fits and averages.
