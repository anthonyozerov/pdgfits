# Asym Refactor Round 19 B0 Consistency Audit

Date: 2026-06-25

Production code changed: no. `src/pdgfits/build_chi2.py` was not edited.

## Question

Round 18 found one reproducible negative profile statistic outside the endpoint
subset:

```text
B0::S042B95, split_normal_pdg_resid, toy 84, seed 2026162602
q = chi2_profile(truth target) - chi2_min_toy = -0.0010571834498449562
```

A fixed-target profile should not beat a true unconstrained global minimum. This
round audited whether that row was a profile-solver/accounting problem, an
unconstrained-refit/globality problem, a chart-tolerance symptom, or a toy-DGP
artifact.

## Commands

Syntax check:

```bash
env PYTHONPATH=/root/pdgfits-private/src \
  PDGFITS_DATA_BACKEND=snapshot \
  PDGFITS_SNAPSHOT_DIR=/root/pdgfits-private/data/pdg-snapshot \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  /root/micromamba/micromamba run -r /root/micromamba-root -n pdg \
  python -m py_compile notes/codex-refactor/run_round19_b0_consistency_audit.py
```

Audit run:

```bash
/usr/bin/time -v prlimit --as=7800000000 --rss=7800000000 -- \
  env PYTHONPATH=/root/pdgfits-private/src \
  PDGFITS_DATA_BACKEND=snapshot \
  PDGFITS_SNAPSHOT_DIR=/root/pdgfits-private/data/pdg-snapshot \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  /root/micromamba/micromamba run -r /root/micromamba-root -n pdg \
  python notes/codex-refactor/run_round19_b0_consistency_audit.py \
    --toy-indices 83,84,85 \
    --audit-toy 84 \
    --run-endpoint-check \
    --output-prefix notes/codex-refactor/round19_b0_consistency_audit \
  > notes/logs/codex-round19-b0-consistency-audit-20260625.log 2>&1
```

Runtime was `1:28.10` wall clock with max RSS `779,036 KB`.

## Artifacts

| Artifact | Rows |
| --- | ---: |
| `notes/codex-refactor/run_round19_b0_consistency_audit.py` | notes-only harness |
| `notes/codex-refactor/round19_b0_consistency_audit_reproduction.csv` / `.jsonl` | 3 |
| `notes/codex-refactor/round19_b0_consistency_audit_unconstrained.csv` / `.jsonl` | 23 |
| `notes/codex-refactor/round19_b0_consistency_audit_profiles.csv` / `.jsonl` | 13 |
| `notes/codex-refactor/round19_b0_consistency_audit_endpoints.csv` / `.jsonl` | 2 |
| `notes/codex-refactor/round19_b0_consistency_audit_summary.json` | 1 object |

The raw command log is under `notes/logs/` and should remain uncommitted.

## Reproduction

The deterministic toy 84 row reproduced the round-18 numbers:

| Quantity | Value |
| --- | ---: |
| truth target | `0.29080520743599103` |
| toy target MLE, reported refit | `0.28920216871277615` |
| reported toy `chi2_min` | `76.24134713268` |
| production fixed-truth profile chi2 | `76.24028994923016` |
| reported q | `-0.0010571834498449562` |
| scaled constraint violation | `7.27e-15` |
| projected gradient norm | `0.002125214416619866` |
| min decay parameter at reported MLE | `3.478448910221638e-7` |
| max abs decay fitted coordinate at reported MLE | `91.50914512767227` |

The audit profile implementation, which records coordinates, matched the
production fixed-truth profile to `2.13e-11` chi2:

```text
production profile chi2 = 76.24028994923016
best audit profile chi2 = 76.24028994920884
```

This rules out a material profile-accounting mismatch for the fixed-truth
profile value.

## Unconstrained Refit Check

The negative q is explained by the reported toy refit missing a lower
unconstrained point. Several starts find the same lower chi2:

| Start | Optimizer | chi2 | Improvement vs reported | Status / certificate |
| --- | --- | ---: | ---: | --- |
| production reference fit start | Minuit | `76.24134713268` | `0` | valid/accurate, EDM `1.25e-08` |
| reported refit MLE | Minuit restart | `76.22910442984346` | `0.01224270283654505` | valid/accurate, EDM `7.24e-08` |
| fixed-truth profile solution | Minuit | `76.2291043588962` | `0.01224277378381089` | valid/accurate, EDM `5.25e-12` |
| fixed-truth profile solution | scipy | `76.22910435883735` | `0.012242773842658039` | success, grad norm `4.14e-05` |
| tiny/floor decay starts | Minuit | `76.22910436` to `76.22910443` | about `0.0122427` | recurrent |

Using the best lower refit, the fixed-truth statistic becomes positive:

```text
q_corrected = 76.24028994923016 - 76.22910435883735
            = 0.011185590392813083
```

So the q<0 row is an unconstrained refit/globality failure, not evidence that
the fixed-truth constrained profile is below the true minimum.

## Fixed-Truth Profile Check

The fixed-truth profile solve itself was stable under the targeted profile
multistart.

| Start | Method | chi2 | Improvement vs production profile | Projected grad | Scaled violation |
| --- | --- | ---: | ---: | ---: | ---: |
| reported refit MLE | SLSQP | `76.24028994923016` | `0` | `0.002125` | `7.27e-15` |
| production reference start | SLSQP | `76.24028994921045` | `1.97e-11` | `0.033532` | `4.00e-14` |
| `clip_decay_floor_1e-05` | trust-constr+KKT | `76.24028994920884` | `2.13e-11` | `2.10e-06` | `6.55e-14` |
| profile solution restart | SLSQP | `76.24028994923025` | `-8.5e-14` | `0.001957` | `2.55e-14` |

No lower fixed-truth profile was found. The cleaner KKT-polished row only
improves the fixed profile by numerical noise, while reducing the stationarity
diagnostic.

## Endpoint Check

Toy 84 was not in the round-18 endpoint subset, but this round computed its
toy endpoints. With the reported local refit baseline, endpoint residuals pass
the existing `5e-3` residual tolerance:

| Side | Endpoint | Profile chi2 | Residual vs reported `chi2_min+1` | Residual vs corrected `chi2_min+1` |
| --- | ---: | ---: | ---: | ---: |
| lower | `0.2739522730918128` | `77.24170474082855` | `0.0003576081485476834` | `0.012600381991205722` |
| upper | `0.30449678543526726` | `77.24574663267562` | `0.004399499995614065` | `0.016642273838272104` |

Interpretation: no endpoint-root search failure was found relative to the
baseline chi2 it was given. But if this toy's endpoints were reported, the
wrong local refit baseline would make them high relative to the corrected lower
minimum. That is still a refit/globality failure, not a bisection/profile-root
failure.

## Nearby Toys

The tiny comparison set suggests this row is isolated in the immediate
neighborhood:

| Toy | Reported q | Best refit improvement | q after refit check |
| ---: | ---: | ---: | ---: |
| 83 | `0.600057347147299` | `3.45e-08` | `0.6000573816194361` |
| 84 | `-0.0010571834498449562` | `0.012242773842658039` | `0.011185590392813083` |
| 85 | `0.6194723886662956` | `7.83e-09` | `0.6194723964947997` |

The existing round-18 CSV also has only one `q < -1e-4` B0 split-normal row
among 500 toys, and it is this toy 84 row.

## Mechanism

This is not a toy/DGP accounting artifact. The same toy observations and same
production chi2 were used for the reported refit, the fixed-truth profile, and
the restart checks. The split-normal DGP only generated the toy data.

The mechanism is:

1. The first Minuit refit from the production reference start returns
   valid/accurate at `chi2=76.24134713268`.
2. The fixed-truth profile solve, using the same chi2, finds a feasible point at
   `chi2=76.24028994923016`.
3. Restarting the unconstrained refit from the reported MLE or the profile point
   finds `chi2=76.22910435883735`.
4. Therefore the fixed-truth profile is above the better unconstrained minimum,
   and the negative q disappears.

The chart saturation is probably the enabling numerical condition: the B0 toy
MLE has tiny decay parameters and fitted decay coordinates around magnitude
`91.5`. However, this is more than a harmless endpoint tolerance artifact: the
best refit improvement is `0.01224` chi2, and endpoint residuals relative to the
corrected baseline would shift by `0.0126` to `0.0166`.

## Answers

1. Endpoint-root correctness failure: no root-solver failure was found. Toy-84
   endpoints pass residual verification against the reported local `chi2_min`;
   they fail relative to the corrected lower refit baseline because the baseline
   was wrong.
2. Unconstrained refit/globality failure: yes. A restart/refinement finds a
   lower unconstrained chi2 by `0.012242773842658039`.
3. Profile-solver/certificate issue: no material lower fixed-truth profile was
   found; the audit profile matched production within `2.13e-11`.
4. DGP/accounting artifact: no evidence. The issue is in the refit on this toy,
   not in the split-normal toy bookkeeping.
5. Tolerance-scale blip: not just tolerance. The original negative q is small,
   but the refit miss is `0.01224` chi2 and survives several starts.

## Production Implication

No production asymmetric-error/profile-root method change is justified by this
round. The hard invariant remains endpoint residual verification, and the
fixed-target profile machinery behaved consistently here.

This does justify a focused next-round audit of unconstrained fit/refit
stability in chart-saturated B0-like cases. A minimal candidate fix, if the
failure recurs, would be a guarded unconstrained-refit restart/refinement check:
after a valid Minuit fit in saturated BR/BRU charts, restart from the returned
point and/or run the existing scipy refinement, then accept a lower finite chi2
only above a predeclared improvement threshold. That should be tested broadly
before touching production `fit.run_fit()`.

## Recommended Next Round

Run a small notes-first refit-stability audit, not another endpoint/profile
fallback experiment:

- scan the B0 split-normal campaign rows with small q or high chart saturation;
- rerun only unconstrained refit restarts/refinements, not endpoints;
- quantify recurrence of `valid Minuit but restart lowers chi2 > 1e-4`;
- if recurrent on real snapshot fits or endpoint-producing toys, propose the
  smallest guarded refit restart change with before/after evidence.
