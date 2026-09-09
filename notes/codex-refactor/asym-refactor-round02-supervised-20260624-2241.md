# Asym Refactor Round 02 Supervised

Date: 2026-06-24 22:41 PDT

Branch: `codex/asym-refactor-simplify-20260624-221852`

Code/evidence commit: `b546b20 perf: avoid redundant avg profile solve`

## Scope

This round reconciled the killed deterministic round 2 only. I did not start
new solver diagnostics, multistart work, or fit-profile changes.

Required context read:

- `SCIENTIFIC_CONTEXT.md`
- `src/pdgfits/CLAUDE.md`
- round-1 report `notes/codex-refactor/asym-refactor-round01-20260624-2228.md`
- killed round-2 logs under `notes/logs/codex-asym-refactor-20260624-221852-round02.log`
- round-2 CSV/JSONL average artifacts under `notes/codex-refactor/`

## Killed-Round State

The handoff was accurate. Initial status had:

- one partial source change in `src/pdgfits/avg.py`;
- four untracked round-2 average benchmark artifacts:
  `round02_avg_baseline.{csv,jsonl}` and
  `round02_avg_direct_mle_check.{csv,jsonl}`;
- accidental tracked deletions under `src/pdgfits/__pycache__/`;
- untracked `notes/logs/`.

The tracked pycache deletions were restored with `git restore -- src/pdgfits/__pycache__`.
No `notes/logs/` files were committed.

The killed round had also produced complete validation before it was stopped:
the baseline average harness, patched average harness, targeted tests, full
non-DB tests, and one public `run_avgs` smoke run all completed. What was
missing was the supervised decision, report, and commit.

## Decision

I kept the average direct-MLE check.

The source change replaces:

```python
base_profile_chi2 = profile_chi2(val)
```

with a direct evaluation of the fitted optimum:

```python
base_chi2 = float(chi2_val(jnp.array(param_values, dtype=jnp.float64)))
```

For averages, the profiled scalar is the primary parameter coordinate itself.
At the MLE target value `val`, the full fitted parameter vector is feasible for
the fixed-primary profile, so the profiled chi2 minimum should equal the
unconstrained MLE chi2. The direct check preserves the base-MLE consistency
check without spending a central profile solve.

This does weaken one internal smoke test: `profile_chi2(val)` used to exercise
the average profiling closure at the easiest possible target before endpoint
search. I judged that acceptable because the scientific invariant for
asymmetric errors is endpoint verification, not a central profile diagnostic.
The benchmark re-evaluates fresh profiled chi2 at both endpoints and verifies
the residuals against `chi2_min + 1`. Those endpoint checks are where a broken
profile solve would affect reported errors.

I added a short comment at the direct check so the intent is explicit.

## Validation Evidence

All project commands used the snapshot backend:

```bash
env PYTHONPATH=/root/pdgfits-private/src \
  PDGFITS_DATA_BACKEND=snapshot \
  PDGFITS_SNAPSHOT_DIR=/root/pdgfits-private/data/pdg-snapshot \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  /root/micromamba/micromamba run -r /root/micromamba-root -n pdg <command>
```

Baseline average hard-case harness from the killed round:

```bash
python notes/endpoint-search/run_asym_endpoint_search_benchmark.py \
  --variant codex-round02-avg-baseline \
  --skip-fits \
  --avg-csv notes/codex-refactor/round02_avg_baseline.csv \
  --avg-jsonl notes/codex-refactor/round02_avg_baseline.jsonl \
  --stdout-log notes/logs/codex_refactor_round02_avg_baseline_stdout.log \
  --avg-node M002W --avg-node M026R08 --avg-node M056R50 \
  --avg-node M057B18 --avg-node M070R84 --avg-node S032B94 \
  --avg-node S041B46 --avg-node S042CKS --avg-node S042R2 \
  --avg-node S051R05 --avg-node S051R06 --avg-node S086R46 \
  --target-timeout-sec 300
```

Result: `12/12` average rows ok, `0` errors, total runtime
`33.30470114995842 s`, total profile calls `298`, max absolute fresh endpoint
residual `0.004999656283968035`.

Patched average hard-case harness from the killed round:

```bash
python notes/endpoint-search/run_asym_endpoint_search_benchmark.py \
  --variant codex-round02-avg-direct-mle-check \
  --skip-fits \
  --avg-csv notes/codex-refactor/round02_avg_direct_mle_check.csv \
  --avg-jsonl notes/codex-refactor/round02_avg_direct_mle_check.jsonl \
  --stdout-log notes/logs/codex_refactor_round02_avg_direct_mle_check_stdout.log \
  --avg-node M002W --avg-node M026R08 --avg-node M056R50 \
  --avg-node M057B18 --avg-node M070R84 --avg-node S032B94 \
  --avg-node S041B46 --avg-node S042CKS --avg-node S042R2 \
  --avg-node S051R05 --avg-node S051R06 --avg-node S086R46 \
  --target-timeout-sec 300
```

Result: `12/12` average rows ok, `0` errors, total runtime
`33.84293695598899 s`, total profile calls `286`, max absolute fresh endpoint
residual `0.004999656283968035`.

CSV reconciliation in this supervised pass showed:

- same 12 nodes in both files;
- no differences in status, measurement counts, parameter counts, central
  values, `chi2_min`, errors, endpoints, endpoint residuals, fresh endpoint
  chi2 values, or endpoint-search counters;
- exactly one fewer profile call and one fewer profile success per node;
- no profile failures in either run.

Targeted unit tests from the killed round:

```bash
python -m pytest tests/test_asym_errors.py tests/test_build_chi2.py tests/test_build_funcs.py -q
```

Result: `21 passed, 1 skipped in 7.02s`.

Full non-DB suite rerun in this supervised pass:

```bash
python -m pytest tests/ -q
```

Result: `160 passed, 2 skipped in 13.68s`.

Public average CLI smoke run from the killed round:

```bash
python -m pdgfits.run_avgs --node M026R08 \
  --output notes/codex-refactor/round02_m026r08_run_avgs.csv
```

Result: completed successfully and printed the same unscaled `M026R08`
asymmetric errors as round 1:

- negative: `0.019546087791061734`
- positive: `0.01987005609699094`

As in round 1, the existing single-node CLI behavior did not leave the requested
CSV output; I did not patch that unrelated behavior.

## Worth-It Assessment

This is worth keeping, but narrowly.

The defensible win is deterministic call reduction and simpler semantics:
average profiling no longer performs a central profile solve that cannot change
the endpoint result. The effect is exact in the saved diagnostics:
`298 -> 286` profile calls over 12 hard average nodes.

This should not be sold as a wall-time improvement. The patched 12-node run was
slightly slower overall (`33.84 s` versus `33.30 s`), which is consistent with
noise dominating a one-call change. The reason to keep it is that it removes a
redundant calculation while preserving the verified endpoint outputs, not that
it demonstrates a speedup in this small benchmark.

## Remaining Risks

- The removed `profile_chi2(val)` call no longer smoke-tests the average
  profiler at the central MLE target. Endpoint fresh-residual verification
  mitigates the scientifically relevant risk, but the diagnostic surface is
  slightly smaller.
- This round reused the completed killed-round 12-node benchmark rather than
  rerunning it. I did rerun the full non-DB test suite after reconciliation.
- No full average snapshot sweep was run in this supervised round.
