# Asym Refactor Round 12 Robust VM Endpoint-Equation Spike

Date: 2026-06-25

Production code changed: no. `src/pdgfits/build_chi2.py` was not edited.

## Question

Can a Venzon-Moolgavkar-style endpoint-equation formulation, with robust
trust-region / damping safeguards, match current bracketed profile endpoints on
smooth low-dimensional cases with fewer profile solves or cleaner certificates?
Or should it be rejected for pdgfits because it is brittle, too problem-specific,
or weaker than verified bisection plus profiling?

The notes-only harness solves the fitted-coordinate KKT/level system

```text
grad chi2(x) + lambda * grad target(x) = 0
chi2(x) = chi2_min + 1
```

where `target(x)` is left free and becomes the endpoint.  This is the direct VM
endpoint-equation analogue of a fixed-target profile optimum.  It is not used as
a reported endpoint unless a fresh production profile solve at the VM endpoint
verifies `profile_chi2 = chi2_min + 1`.

## Artifacts

- Harness: `notes/codex-refactor/run_round12_robust_vm_endpoint_spike.py`
- Result CSV/JSONL: `notes/codex-refactor/round12_robust_vm_endpoint_results.csv`, `notes/codex-refactor/round12_robust_vm_endpoint_results.jsonl`
- Iteration CSV/JSONL: `notes/codex-refactor/round12_robust_vm_endpoint_iterations.csv`, `notes/codex-refactor/round12_robust_vm_endpoint_iterations.jsonl`
- Failure CSV/JSONL: `notes/codex-refactor/round12_robust_vm_endpoint_failures.csv`, `notes/codex-refactor/round12_robust_vm_endpoint_failures.jsonl`

Exact row counts:

| Table | Rows |
| --- | ---: |
| Endpoint solver results | 40 |
| VM iteration trace rows | 92 |
| Harness failures | 0 |

The 40 result rows are 5 cases x 2 sides x 2 starts x 2 solvers.

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
env PYTHONPATH=/root/pdgfits-private/src \
  PDGFITS_DATA_BACKEND=snapshot \
  PDGFITS_SNAPSHOT_DIR=/root/pdgfits-private/data/pdg-snapshot \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  /root/micromamba/micromamba run -r /root/micromamba-root -n pdg \
  python -m py_compile notes/codex-refactor/run_round12_robust_vm_endpoint_spike.py
```

Smoke run, passed with 8 result rows, 8 iteration rows, and 0 failures:

```bash
prlimit --as=7800000000 --rss=7800000000 -- \
  env PYTHONPATH=/root/pdgfits-private/src \
  PDGFITS_DATA_BACKEND=snapshot \
  PDGFITS_SNAPSHOT_DIR=/root/pdgfits-private/data/pdg-snapshot \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  /root/micromamba/micromamba run -r /root/micromamba-root -n pdg \
  python notes/codex-refactor/run_round12_robust_vm_endpoint_spike.py \
    --case fit_g2000_k002m \
    --results-csv /tmp/round12_smoke_results.csv \
    --results-jsonl /tmp/round12_smoke_results.jsonl \
    --iterations-csv /tmp/round12_smoke_iterations.csv \
    --iterations-jsonl /tmp/round12_smoke_iterations.jsonl \
    --failures-csv /tmp/round12_smoke_failures.csv \
    --failures-jsonl /tmp/round12_smoke_failures.jsonl
```

Full run, passed with the required cases plus bounded optional cases:

```bash
prlimit --as=7800000000 --rss=7800000000 -- \
  env PYTHONPATH=/root/pdgfits-private/src \
  PDGFITS_DATA_BACKEND=snapshot \
  PDGFITS_SNAPSHOT_DIR=/root/pdgfits-private/data/pdg-snapshot \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  /root/micromamba/micromamba run -r /root/micromamba-root -n pdg \
  python notes/codex-refactor/run_round12_robust_vm_endpoint_spike.py \
    --include-optional \
    --keep-going \
    --results-csv notes/codex-refactor/round12_robust_vm_endpoint_results.csv \
    --results-jsonl notes/codex-refactor/round12_robust_vm_endpoint_results.jsonl \
    --iterations-csv notes/codex-refactor/round12_robust_vm_endpoint_iterations.csv \
    --iterations-jsonl notes/codex-refactor/round12_robust_vm_endpoint_iterations.jsonl \
    --failures-csv notes/codex-refactor/round12_robust_vm_endpoint_failures.csv \
    --failures-jsonl notes/codex-refactor/round12_robust_vm_endpoint_failures.jsonl \
    --stdout-log notes/logs/round12_robust_vm_endpoint_full_stdout.log
```

## Case Set

Required cases:

- `G(2000),G(1800)::K002M`
- `G(2000),G(1800)::K003M`
- `avg_m002w`

Bounded optional cases included:

- `avg_m026r08`
- `B0S-BR::S086.37`

Each side was solved from two covariance starts:

- `hesse_one_sigma`: pure local parabolic one-sigma displacement from the MLE.
- `production_displacement_cov`: covariance-predicted point using the production
  endpoint scalar displacement.  This is a convergence-back check, not a
  standalone endpoint method.

Each start was run with raw `plain_newton` and a damped/trust-region
`robust_trust` variant.

## Main Results

All rows converged:

| Solver | Successes / rows |
| --- | ---: |
| `plain_newton` | 20 / 20 |
| `robust_trust` | 20 / 20 |

Robust safeguards did not save any solve:

| Event | Count over robust rows |
| --- | ---: |
| Raw Newton failure rescued by robust VM | 0 |
| Damping events | 0 |
| Trust-region clips | 0 |
| Rejected steps | 0 |
| Singular linear steps | 0 |

This matters: the successful rows show the endpoint equations are benign on
these cases, but they do not provide evidence that Fischer-Lewis-style
safeguards solve a pdgfits hard case.

### Robust VM From HESSE Starts

| Case | Side | VM iterations | Endpoint delta vs production | VM level residual | Fresh profile residual | Production root evals |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `fit_g2000_k002m` | lower | 0 | 0 | 3.58e-11 | 3.58e-11 | 4 |
| `fit_g2000_k002m` | upper | 0 | 0 | 3.63e-11 | 3.63e-11 | 4 |
| `fit_g2000_k003m` | lower | 0 | 0 | 1.90e-10 | 1.90e-10 | 4 |
| `fit_g2000_k003m` | upper | 0 | 0 | 1.90e-10 | 1.90e-10 | 4 |
| `avg_m002w` | lower | 2 | -5.36e-5 | 5.39e-11 | 5.39e-11 | 24 |
| `avg_m002w` | upper | 2 | 3.42e-6 | 5.56e-11 | 5.56e-11 | 24 |
| `avg_m026r08` | lower | 2 | 1.66e-5 | 6.25e-8 | 6.25e-8 | 19 |
| `avg_m026r08` | upper | 2 | -1.30e-5 | 3.11e-8 | 3.11e-8 | 19 |
| `fit_b0s_s08637` | lower | 2 | -5.13e-9 | 7.18e-7 | 7.18e-7 | 20 |
| `fit_b0s_s08637` | upper | 3 | -7.40e-9 | 3.15e-12 | 1.43e-11 | 20 |

The fresh production profile verification at each robust VM endpoint agreed
with the VM chi2 to at worst `1.45e-11` chi2, and the worst fresh profile
residual was `7.18e-7`, well inside the existing `5e-3` endpoint tolerance and
also inside the tighter `1e-3` diagnostic target.

## Interpretation

The two `G(2000),G(1800)` direct-coordinate fits are exactly the case where VM
should look good.  The HESSE/parabolic point already satisfies the endpoint
equations to numerical precision, so VM needs zero nonlinear iterations and
matches production exactly.  This reproduces the round 9 finding that these
profiles are almost perfectly quadratic.

For `avg_m002w`, VM from the HESSE start moved the lower endpoint by
`-5.36e-5` and the upper endpoint by `3.42e-6` relative to production.  This is
not an endpoint-correctness failure.  The production lower residual was
`-0.004999656`, i.e. right at the accepted bisection tolerance edge, while the VM
endpoint and a fresh direct profile check hit the level equation to `~5e-11`.
VM found a tighter root inside the current tolerance, not a lower constrained
profile at the same fixed target.

`avg_m026r08` showed the same pattern at smaller scale: endpoint deltas of
`1.66e-5` and `-1.30e-5`, fresh profile residuals below `7e-8`, and production
residuals of about `0.0017` and `0.0013`.

The optional nonlinear mapped BRU parameter `B0S-BR::S086.37` also converged
from the HESSE start in 2-3 VM iterations and matched production to about
`7.4e-9` in endpoint value.  This is useful evidence that the equation form can
work on at least one well-conditioned mapped target, but it is not enough to
generalize to saturated BR/BRU cases such as `B0::S042B95`.

## Answers

1. Did VM/robust-VM find a real endpoint-correctness issue or materially lower constrained profile?

No.  Every robust VM endpoint that converged was rechecked with the production
profile solver.  The largest fresh profile residual was `7.18e-7`, and the
fresh profile chi2 matched the VM chi2 within `1.45e-11`.  Differences against
production endpoints are explained by production's accepted bisection residuals,
especially `avg_m002w` lower at `-0.004999656`.

2. Did it match current endpoints with fewer profile solves/calls or cleaner certificates?

Partially, but not in a production-justifying way.  On the smooth direct fits,
VM needed zero nonlinear iterations from HESSE starts and no outer profile
solves, versus 4 production profile evaluations.  On averages and B0S it needed
2-3 VM iterations, again with no outer profile bisection.  However, the VM
certificate alone is only a local KKT/level certificate.  To be scientifically
safe in pdgfits, the harness still used production profile verification.  Once
that safety rail is retained, VM is a possible diagnostic or polish, not a
cleaner replacement.

3. Which parts are promising to borrow, and which parts are rejected quantitatively?

Promising, narrowly:

- VM endpoint equations are a clean diagnostic for smooth direct-coordinate
  cases.
- VM can polish residual-edge bisection endpoints to a tighter level residual
  without changing the underlying profile optimum.

Rejected for production replacement:

- Robust VM as a demonstrated robustness improvement.  There were zero damping,
  trust-region, rejected-step, singular-step, or raw-Newton-rescue events.
- Unbracketed endpoint equations as a replacement for bracketed bisection plus
  profile verification.  Root side selection depends on the start, and KKT
  stationarity does not by itself prove that the constrained profile is the
  relevant minimum.
- Generalizing from the well-conditioned optional `B0S-BR::S086.37` row to
  saturated BR/BRU or arbitrary scalar targets.

4. Is any production change justified now?

No.  VM did not expose an endpoint-correctness failure, did not find a material
lower constrained profile, and did not demonstrate a safeguard advantage.  The
current bracketed profile endpoint method should remain unchanged.

5. What should the next supervised round do if the overnight loop continues?

Do not productionize VM.  If another numerical round is useful, the only
reasonable VM-related follow-up is a notes-only residual-polish diagnostic on
residual-edge endpoints to ask whether a tighter `1e-3` endpoint tolerance is
cheap enough.  Otherwise, move away from solver mechanics: synthesize the
negative numerical evidence from rounds 7-12 or investigate statistical
regularity/Wilks risk near tiny BR/BRU boundaries.

## Bottom Line

VM endpoint equations are mathematically attractive on smooth direct-coordinate
cases, and they reproduce current endpoints after profile verification.  This
round did not find a correctness issue or a robust-method win.  For pdgfits,
unbracketed VM is weaker than the current verified bisection-plus-profile method
unless it is treated only as a diagnostic or optional endpoint polish.
