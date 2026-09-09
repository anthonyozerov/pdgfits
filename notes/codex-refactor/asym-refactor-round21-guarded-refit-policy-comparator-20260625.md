# Round 21 guarded B0/BR-BRU refit policy comparator

Date: 2026-06-25

Branch: `codex/asym-refactor-simplify-20260624-221852`

Scope: notes-only offline comparator. No production code was changed, and `src/pdgfits/build_chi2.py` was not touched.

## Question

Evaluate whether a minimal guarded unconstrained-refit policy for saturated BR/BRU/B0-like charts can catch the round-19/20 lower local minima without perturbing stable actual snapshot fits or clean controls.

The production-relevant invariant remains unchanged: asymmetric-error endpoints must verify the profiled chi2 endpoint equation. This round only audits the *unconstrained MLE/globality* refit before profile use.

## Artifacts

Harness:

- `notes/codex-refactor/run_round21_guarded_refit_policy_comparator.py`

Main outputs:

- `notes/codex-refactor/round21_guarded_refit_policy_comparator_selection.csv`
- `notes/codex-refactor/round21_guarded_refit_policy_comparator_rows.csv`
- `notes/codex-refactor/round21_guarded_refit_policy_comparator_attempts.csv`
- `notes/codex-refactor/round21_guarded_refit_policy_comparator_policy_rows.csv`
- `notes/codex-refactor/round21_guarded_refit_policy_comparator_summary.json`
- `notes/codex-refactor/round21_guarded_refit_policy_comparator_actual_snapshot.json`
- `notes/codex-refactor/round21_guarded_refit_policy_comparator_actual_snapshot_policy_rows.csv`

JSONL mirrors were also written for the CSV artifacts. `notes/logs/codex-round21-guarded-refit-policy-comparator-20260625.log` was kept out of the commit.

Exact full command:

```bash
/usr/bin/time -v prlimit --as=7800000000 --rss=7800000000 -- \
  env PYTHONPATH=/root/pdgfits-private/src:/root/pdgfits-private/notes/codex-refactor \
  PDGFITS_DATA_BACKEND=snapshot \
  PDGFITS_SNAPSHOT_DIR=/root/pdgfits-private/data/pdg-snapshot \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  /root/micromamba/micromamba run -r /root/micromamba-root -n pdg \
  python notes/codex-refactor/run_round21_guarded_refit_policy_comparator.py \
    --overwrite \
    --output-prefix notes/codex-refactor/round21_guarded_refit_policy_comparator \
  > notes/logs/codex-round21-guarded-refit-policy-comparator-20260625.log 2>&1
```

Resource summary:

- Selected rows: 1104
- Completed rows: 1104
- Failure rows: 0
- Attempt rows: 10312
- Policy rows: 8832
- Actual snapshot policy rows: 8
- Wall time: 2:17:05
- Max RSS: 2.2 GB

## Selection

The harness reused the round-20 refit machinery but disabled fixed-truth/profile starts, because production cannot use toy truth or profile-root information.

Selected rows:

- All 1000 `fit_b0_s042b95` round-18 calibration-campaign-v1 toys: 500 local Gaussian and 500 split-normal.
- 80 representative B0S/eta rows: smallest-q and q-near-1 profile-shape rows for both DGPs.
- 24 clean direct-coordinate controls from `fit_g2000_k002m` and `fit_g2000_k003m`.
- Actual snapshot `B0::S042B95` sanity check with observed data.

## Compared policies

The policy rows compare:

- `baseline_reported`: round-18 reported/reference value.
- `returned_mle_restart`: returned-MLE restart with Minuit/scipy.
- `loose_saturation_floor_only`: deterministic decay-floor/tiny-parameter starts if loose saturation fires.
- `loose_saturation_returned_plus_floor`: returned-MLE restart plus deterministic floor/tiny starts.
- `extreme_saturation_returned_plus_floor`: narrower production-eligible saturation guard plus the same starts.
- `cheap_hint_1e_5_returned_then_floor`: run floors only if returned-MLE restart improves by `>1e-5`.
- `cheap_hint_1e_4_returned_then_floor`: run floors only if returned-MLE restart improves by `>1e-4`.
- `oracle_small_q_returned_plus_floor`: not production eligible; included only to test the value of small-q/profile-symptom narrowing.

Deterministic starts:

- Decay floors: `1e-6`, `3e-6`, `1e-5`, `1e-4`.
- Tiny-decay multiplicative perturbations: `0.5`, `2`, `10`.

Candidate production-eligible saturation triggers:

- Loose: `min_decay_param_mle <= 1e-6` and `max_abs_decay_fitted_coord_mle >= 50`.
- Extreme: `min_decay_param_mle <= 5e-7` and `max_abs_decay_fitted_coord_mle >= 80`.

Both use only post-fit diagnostics available without toy truth.

## Main B0 results

Combined across all 1000 B0 campaign rows:

| Policy | Triggered rows | Extra attempts/row | >1e-4 | >1e-3 | >1e-2 | Max lowering |
|---|---:|---:|---:|---:|---:|---:|
| returned-MLE only | 1000 | 2.000 | 4 | 4 | 3 | 0.0191831006 |
| loose floor only | 1000 | 7.000 | 22 | 20 | 12 | 0.0347463256 |
| loose returned+floor | 1000 | 9.000 | 22 | 20 | 12 | 0.0347463256 |
| extreme returned+floor | 965 | 8.755 | 22 | 20 | 12 | 0.0347463256 |
| cheap hint `>1e-5` | 19 | 2.133 | 4 | 4 | 3 | 0.0191831006 |
| cheap hint `>1e-4` | 4 | 2.028 | 4 | 4 | 3 | 0.0191831006 |
| oracle small-q | 107 | 2.749 | 6 | 6 | 4 | 0.0191831006 |

By DGP for the full loose returned+floor policy:

| DGP | Rows | Triggered | >1e-4 | >1e-3 | >1e-2 | Max lowering |
|---|---:|---:|---:|---:|---:|---:|
| local Gaussian | 500 | 500 | 14 | 13 | 7 | 0.0347463256 |
| split-normal | 500 | 500 | 8 | 7 | 5 | 0.0302582128 |

The narrower `extreme_saturation_returned_plus_floor` trigger caught every B0 material row while skipping 35 of 1000 B0 rows. All 22 B0 rows with lowering `>1e-4` had both loose and extreme saturation diagnostics before looking at toy truth or q.

Best starts among those 22 B0 material rows:

- `clip_decay_floor_3e-06::minuit`: 9
- `clip_decay_floor_1e-05::minuit`: 5
- `returned_mle_restart::scipy`: 2
- `tiny_decay_factor_0.5::minuit`: 2
- `tiny_decay_factor_10::minuit`: 2
- `clip_decay_floor_0.0001::minuit`: 1
- `clip_decay_floor_1e-06::minuit`: 1

This is direct negative evidence against a returned-MLE-only policy. It also rejects cheap returned-MLE hint triggers: the floor/tiny starts found most material rows that returned-MLE restarts did not indicate.

It also rejects small-q/profile-symptom narrowing as the main trigger. Only 3 of the 22 B0 material rows had original `q < 0.02`, and material misses extended up to `q = 2.2576551068`. The oracle small-q policy still caught only 6 B0 rows `>1e-4`.

## Actual snapshot and controls

Actual snapshot `B0::S042B95`:

- `min_decay_param_mle = 3.3901979930e-07`
- `max_abs_decay_fitted_coord_mle = 93.8912378663`
- Loose saturation trigger: true
- Extreme saturation trigger: true
- Best lowering under returned+floor policies: `7.5076087569e-09`

So the actual snapshot fit remains stable even when the guarded policy fires.

Clean controls:

| Case | DGP | Rows | Triggered | >1e-4 | >1e-3 | >1e-2 | Max lowering |
|---|---|---:|---:|---:|---:|---:|---:|
| `fit_g2000_k002m` | local Gaussian | 6 | 0 | 0 | 0 | 0 | 0 |
| `fit_g2000_k002m` | split-normal | 6 | 0 | 0 | 0 | 0 | 2.43e-10 |
| `fit_g2000_k003m` | local Gaussian | 6 | 0 | 0 | 0 | 0 | 0 |
| `fit_g2000_k003m` | split-normal | 6 | 0 | 0 | 0 | 0 | 1.42e-08 |

Clean controls remain stable. No clean row crossed `1e-4`, `1e-3`, or `1e-2`.

Representative non-B0 profile-shape rows did not satisfy the saturation trigger. Returned-MLE-only comparator attempts found three borderline `>1e-4` lowerings, all below `1e-3`:

- B0S local: 2 of 20, max `1.2765e-04`
- eta split-normal: 1 of 20, max `1.1538e-04`

Those are not evidence for the saturated BR/BRU floor policy; the production guard considered here would not run floor starts for them.

## Calibration impact

Correcting the B0 refits does not change the round-18 calibration conclusions.

For the full loose returned+floor policy:

- Local Gaussian B0:
  - `q <= chi2_1 median`: 257 -> 256, delta `-1`
  - `q <= 1`: unchanged at 344
  - `q > chi2_1 90%`: unchanged at 50
  - `q > chi2_1 95%`: unchanged at 30
- Split-normal B0:
  - All threshold counts unchanged.

The round-19 negative-q split-normal row is fixed: toy 84 moves from `q = -0.0010571834` to `q = 0.0111855899` under the full policy, with lowering `0.0122427733`.

## Interpretation

This round found a real issue in unconstrained refit/globality stability, not in endpoint profiling. It is broader than the one round-19 row:

- `22/1000` B0 calibration toys lower by `>1e-4`.
- `20/1000` lower by `>1e-3`.
- `12/1000` lower by `>1e-2`.
- All material B0 rows are in the extreme saturated chart-risk regime.

This is not a robust simplification or performance win. It is a guarded robustness fix candidate with quantitative evidence.

The evidence is also bounded:

- The actual snapshot `B0::S042B95` fit remains stable.
- Clean direct-coordinate controls remain stable.
- Representative B0S/eta rows do not fire the saturation guard.
- Calibration threshold-count conclusions do not materially change.

## Recommendation

A production patch is justified, but should be a focused next-round patch rather than part of this comparator round.

Recommended guard for `fit.run_fit()`:

1. After the normal unconstrained fit, compute fitted decay/BR/BRU chart diagnostics.
2. If `min_decay_param_mle <= 5e-7` and `max_abs_decay_fitted_coord_mle >= 80`, run a deterministic guarded refit set:
   - returned-MLE restart with existing Minuit/scipy refinement;
   - decay floor starts at `1e-6`, `3e-6`, `1e-5`, `1e-4`;
   - tiny-decay multiplicative starts `0.5`, `2`, `10` for parameters already at or below the tiny-decay regime.
3. Accept a finite lower result only past an explicit numerical-noise guard, and emit diagnostics recording trigger, attempted starts, best start, optimizer, and chi2 lowering.
4. Do not use toy truth, profile q, endpoint status, or profile-root information in the trigger.

Rows caught/missed by this trigger in the offline comparator:

- Caught all 22 B0 rows with lowering `>1e-4`.
- Missed no B0 rows with lowering `>1e-4`, `>1e-3`, or `>1e-2`.
- Fired on the actual snapshot `B0::S042B95`, but found only `7.5e-09` lowering.
- Fired on 965 of 1000 B0 campaign rows, so the runtime cost is real for this saturated chart class.
- Did not fire on clean controls or representative B0S/eta rows.

Recommended next round: implement the guarded `fit.run_fit()` patch with concise diagnostics and focused regression evidence. Do not change endpoint/profile-root mechanics and do not edit `build_chi2.py`.
