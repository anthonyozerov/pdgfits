# Asym Refactor Round 20 B0 Refit Stability Audit

Date: 2026-06-25

Production code changed: no. `src/pdgfits/build_chi2.py` was not edited.

## Question

Round 19 showed that the negative-q row from round 18 was not an endpoint or
fixed-target profile failure. The fixed-truth profile was stable, but the
unconstrained toy refit had missed a lower point by `0.012242773842658039`
despite Minuit reporting valid/accurate.

This round asks whether that valid-Minuit unconstrained-refit miss is isolated
or recurrent enough to justify a minimal guarded production change.

## Design

This was a refit/globality audit only. It did not run endpoint searches.

Selection from `round18_calibration_campaign_v1_toys.csv`:

| Stratum | Rows |
| --- | ---: |
| `B0::S042B95` local-Gaussian, all `q < 0.02` plus saturation/q-near-1 reps | 61 |
| `B0::S042B95` split-normal, all `q < 0.02`, negative-q neighbors, plus saturation/q-near-1 reps | 74 |
| B0S/eta small-q comparison rows | 16 |
| clean-control small-q and q-near-1 rows | 24 |
| Total | 175 |

For each row the harness regenerated the same toy and reran:

- production/reference Minuit start;
- Minuit restart and existing scipy refinement from the returned MLE;
- B0-only deterministic decay floor/tiny-parameter perturbation starts;
- fixed-truth profile solution starts for B0 rows with `q < 0.001`.

The fixed-truth profile starts are diagnostic only; an ordinary production
unconstrained fit would not already have that profile point available.

## Artifacts

| Artifact | Rows / Notes |
| --- | ---: |
| `notes/codex-refactor/run_round20_b0_refit_stability_audit.py` | notes-only harness |
| `notes/codex-refactor/round20_b0_refit_stability_audit_selection.csv` / `.jsonl` | 175 |
| `notes/codex-refactor/round20_b0_refit_stability_audit_rows.csv` / `.jsonl` | 175 |
| `notes/codex-refactor/round20_b0_refit_stability_audit_attempts.csv` / `.jsonl` | 1665 |
| `notes/codex-refactor/round20_b0_refit_stability_audit_failures.csv` / `.jsonl` | 0 failures |
| `notes/codex-refactor/round20_b0_refit_stability_audit_summary.json` | summary |
| `notes/codex-refactor/round20_b0_refit_stability_audit_actual_fit_check.json` | actual B0 snapshot sanity check |

Runtime was `19:27.54` wall clock with max RSS `1,547,840 KB`.

## Commands

Syntax check:

```bash
env PYTHONPATH=/root/pdgfits-private/src:/root/pdgfits-private/notes/codex-refactor \
  PDGFITS_DATA_BACKEND=snapshot \
  PDGFITS_SNAPSHOT_DIR=/root/pdgfits-private/data/pdg-snapshot \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  /root/micromamba/micromamba run -r /root/micromamba-root -n pdg \
  python -m py_compile notes/codex-refactor/run_round20_b0_refit_stability_audit.py
```

Audit:

```bash
/usr/bin/time -v prlimit --as=7800000000 --rss=7800000000 -- \
  env PYTHONPATH=/root/pdgfits-private/src:/root/pdgfits-private/notes/codex-refactor \
  PDGFITS_DATA_BACKEND=snapshot \
  PDGFITS_SNAPSHOT_DIR=/root/pdgfits-private/data/pdg-snapshot \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  /root/micromamba/micromamba run -r /root/micromamba-root -n pdg \
  python notes/codex-refactor/run_round20_b0_refit_stability_audit.py \
    --overwrite \
    --output-prefix notes/codex-refactor/round20_b0_refit_stability_audit \
  > notes/logs/codex-round20-b0-refit-stability-audit-20260625.log 2>&1
```

## Results

Overall restart/refinement lowerings:

| Group | Rows | `>1e-4` | `>1e-3` | `>1e-2` | Max improvement |
| --- | ---: | ---: | ---: | ---: | ---: |
| All selected rows | 175 | 5 | 3 | 2 | `0.01734243433776328` |
| B0 selected rows | 135 | 3 | 3 | 2 | `0.01734243433776328` |
| Non-B0 comparison rows | 40 | 2 | 0 | 0 | `0.00012765345078591395` |
| Clean controls | 24 | 0 | 0 | 0 | `2.43e-10` on `K002M`, `1.42e-08` on `K003M` |

Material rows:

| Case / DGP / Toy | Original q | Improvement | q after best refit | Best start / optimizer | Min decay param | Max abs fitted decay coord |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| B0 split toy 44 | `2.198357965710329e-05` | `0.01734243433776328` | `0.017364417917420383` | fixed-truth profile / scipy | `3.8992211678611967e-07` | `81.63422192043552` |
| B0 split toy 84 | `-0.0010571834498449562` | `0.012242773842658039` | `0.011185590392813083` | fixed-truth profile / scipy | `3.478448910221638e-07` | `91.50914512767227` |
| B0 local toy 51 | `0.0007890492927913328` | `0.006347549703917821` | `0.007136598996709154` | decay floor `1e-05` / Minuit | `3.036830298505598e-07` | `104.8164878602176` |
| B0S local toy 84 | `-6.844407167960753e-05` | `0.00012765345078591395` | `5.9209379106306415e-05` | returned MLE / scipy | `1.7316424676820323e-05` | `1.8381963471687002` |
| eta split toy 240 | `-9.713845398096055e-05` | `0.00011538325490789703` | `1.8244800926936477e-05` | returned MLE / scipy | `0.00015322677160116763` | `0.2077377590553663` |

All five rows started from Minuit fits that were reported valid/accurate. The
two non-B0 rows are just above `1e-4` and below `1e-3`; the only `>1e-3` and
`>1e-2` lowerings are B0 rows with severe BR/BRU chart saturation.

The original round-19 row is reproduced exactly in this broader audit. Its
profile-consistency failure disappears after the better refit:

```text
toy 84 q before = -0.0010571834498449562
toy 84 q after  =  0.011185590392813083
```

The actual snapshot `B0::S042B95` fit was also checked with the same returned
MLE and B0 floor restarts. It is stable: best lowering was only
`7.507608756895934e-09`, with min decay parameter `3.390197993036817e-07` and
max fitted decay coordinate `93.89123786630213`.

## Interpretation

The round-19 miss is not isolated. There are two additional material B0 toy
baseline misses in the selected audit set, including one in the local-Gaussian
DGP and one larger split-normal row. Among B0 rows with `q < 0.02`, the count
is `3/107` above `1e-4`, `3/107` above `1e-3`, and `2/107` above `1e-2`.
Among B0 rows with `q < 0.001`, the count is `3/30` at `>1e-3`, so the
recurrence is concentrated in the very-small-q tail.

This is not purely a q-tail artifact across all cases: B0S and eta small-q
rows have two tiny `>1e-4` lowerings but no `>1e-3`, and clean controls have
none. It is also not simply "largest fitted coordinate fails": representative
B0 rows with `q >= 0.02`, including highly saturated rows, had no `>1e-4`
lowering.

The practical mechanism matters. A simple returned-MLE restart or scipy
refinement would catch the two non-B0 borderline rows, but it would not catch
the two largest B0 misses. Those required either a fixed-truth profile start
or deterministic B0 decay-floor starts. The fixed-truth profile start is not
available in normal production refits, so a production guard that actually
catches the B0 failures would need to include chart-specific B0/BR/BRU floor
perturbations, not only an MLE restart.

## Calibration Impact

Correcting the audited baseline misses does not change the round-18 scientific
calibration conclusions. For the affected round-18 case/DGP summaries, the
counts at the chi-square-1 median, `q <= 1`, q90, and q95 thresholds are
unchanged.

Examples:

| Case / DGP | q min before | q min after audited corrections | Threshold-count changes |
| --- | ---: | ---: | ---: |
| B0 split | `-0.0010571834498449562` | `3.2011643511964394e-05` | 0 |
| B0 local | `1.2748646938121055e-05` | unchanged | 0 |
| B0S local | `-6.844407167960753e-05` | `1.067953490441198e-05` | 0 |
| eta split | `-9.713845398096055e-05` | `-3.95718879673268e-05` | 0 |

The correction removes the single `q < -1e-4` B0 profile-consistency failure,
but all corrected values remain in the very-low-q region. The round-18
conclusion remains: B0 is chart/boundary-saturated but its aggregate q
calibration is close to nominal at campaign scale.

## Production Implication

A production change is not justified in this round.

Reason: recurrence is real, but the simple candidate policy from round 19
(`returned MLE restart and/or existing scipy refinement`) is insufficient for
the largest B0 misses. The policy that would catch them is more chart-specific:
for saturated BR/BRU fits, try deterministic decay-floor/tiny-parameter starts,
run Minuit and possibly scipy from the best finite candidate, and accept a
lower finite chi2 only above a predeclared threshold while emitting diagnostics.

That policy is plausible, but it should be evaluated as its own before/after
production patch because:

- the evidence here is from synthetic toys, not a failure of the actual B0
  snapshot fit;
- clean controls are stable;
- non-B0 lowerings are below `1e-3`;
- the actual B0 snapshot fit is stable under the same floor-start sanity check;
- adding BR/BRU floor starts to `fit.run_fit()` is more than wrapper cleanup and
  needs focused tests/harness evidence.

## Required Answers

1. Was the round-19 unconstrained refit/globality miss isolated or recurrent?

Recurrent, but sparse and concentrated in saturated B0 small-q toys. The audit
found 3 B0 rows above `1e-3`, including the round-19 row.

2. Did any restart/refinement find lower chi2 by `>1e-4`, `>1e-3`, or `>1e-2`?

Yes. Overall: `5/175` above `1e-4`, `3/175` above `1e-3`, and `2/175` above
`1e-2`. B0 only: `3/135`, `3/135`, and `2/135`. Clean controls: `0/24` at all
three thresholds.

3. Are failures specific to B0 chart saturation, DGP, q-tail selection, or also
seen in controls?

The material `>1e-3` failures are specific to B0 chart-saturated small-q rows.
They occur in both local-Gaussian and split-normal DGPs. Non-B0 comparison rows
show only borderline `>1e-4` lowerings, and clean controls show none.

4. Does correcting the refit baseline change the scientific calibration
conclusions from round 18?

No. The q threshold counts used in round 18 are unchanged. The correction fixes
the one `q < -1e-4` consistency failure and moves a few very-small-q values
slightly upward, but it does not alter the campaign-level calibration picture.

5. Is a production change justified now?

No. A production policy is now worth designing, but not shipping from this
round's evidence. The smallest plausible policy would be a saturated-BR/BRU
guard that tries returned-MLE/scipy plus deterministic decay-floor starts and
accepts only finite lower chi2 improvements above a threshold, with diagnostics.
It should be tested in a dedicated patch round.

6. Recommended next round?

Run a focused guarded-refit policy comparator before editing production:

- apply the candidate guard to all B0 round-18 rows, not just the enriched
  selection;
- include the actual snapshot fit and several non-B0/clean controls;
- compare old vs guarded `chi2_min`, target MLEs, q statistics, and runtime;
- only then consider a minimal `fit.run_fit()` patch with tests.

## Bottom Line

The baseline/globality issue is real enough to track: valid/accurate Minuit can
miss lower unconstrained B0 toy minima by `0.006` to `0.017` chi2 in severely
saturated BR/BRU charts. It is not an endpoint-root or fixed-profile failure,
and it does not change round-18 calibration conclusions. Because the actual B0
snapshot fit is stable and the largest toy misses need chart-specific floor
starts, this round stays notes-only and rejects an immediate production edit.
