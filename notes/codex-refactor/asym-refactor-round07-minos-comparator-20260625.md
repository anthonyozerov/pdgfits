# Asym Refactor Round 07 MINOS Comparator

Date: 2026-06-25

Production code changed: no. `src/pdgfits/build_chi2.py` was not edited.

## Question

Anthony asked whether pdgfits should borrow from Minuit/MINOS. This round ran a
notes-only `iminuit` comparator where MINOS is scientifically fair: the profiled
quantity is an actual optimizer coordinate, not an arbitrary scalar node
function.

`iminuit` was available in the `pdg` environment:

```text
iminuit_available=true
iminuit_version=2.32.0
```

## Artifacts

- Harness: `notes/codex-refactor/run_round07_minos_comparator.py`
- Results: `notes/codex-refactor/round07_minos_comparator_results.csv`
- Results JSONL: `notes/codex-refactor/round07_minos_comparator_results.jsonl`
- Rejections: `notes/codex-refactor/round07_minos_comparator_rejections.csv`
- Rejections JSONL: `notes/codex-refactor/round07_minos_comparator_rejections.jsonl`
- Raw stdout: `notes/logs/round07_minos_comparator_stdout.log`

## Commands

Common environment:

```bash
env PYTHONPATH=/root/pdgfits-private/src \
  PDGFITS_DATA_BACKEND=snapshot \
  PDGFITS_SNAPSHOT_DIR=/root/pdgfits-private/data/pdg-snapshot \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  /root/micromamba/micromamba run -r /root/micromamba-root -n pdg <command>
```

Availability check, passed:

```bash
python - <<'PY'
try:
    import iminuit
    print('iminuit_available=true')
    print('iminuit_version=' + iminuit.__version__)
except Exception as exc:
    print('iminuit_available=false')
    print(type(exc).__name__ + ': ' + str(exc))
PY
```

Syntax checks, passed:

```bash
python -m py_compile notes/codex-refactor/run_round07_minos_comparator.py
```

Smoke comparator, passed:

```bash
prlimit --as=7800000000 --rss=7800000000 -- \
  env PYTHONPATH=/root/pdgfits-private/src \
  PDGFITS_DATA_BACKEND=snapshot \
  PDGFITS_SNAPSHOT_DIR=/root/pdgfits-private/data/pdg-snapshot \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  /root/micromamba/micromamba run -r /root/micromamba-root -n pdg \
  python notes/codex-refactor/run_round07_minos_comparator.py \
    --case avg_m002w \
    --results-csv /tmp/round07_smoke_results.csv \
    --results-jsonl /tmp/round07_smoke_results.jsonl \
    --failures-csv /tmp/round07_smoke_failures.csv \
    --failures-jsonl /tmp/round07_smoke_failures.jsonl \
    --rejections-csv /tmp/round07_smoke_rejections.csv \
    --rejections-jsonl /tmp/round07_smoke_rejections.jsonl \
    --stdout-log notes/logs/round07_minos_smoke_stdout.log
```

Full comparator, passed:

```bash
prlimit --as=7800000000 --rss=7800000000 -- \
  env PYTHONPATH=/root/pdgfits-private/src \
  PDGFITS_DATA_BACKEND=snapshot \
  PDGFITS_SNAPSHOT_DIR=/root/pdgfits-private/data/pdg-snapshot \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  /root/micromamba/micromamba run -r /root/micromamba-root -n pdg \
  python notes/codex-refactor/run_round07_minos_comparator.py \
    --results-csv notes/codex-refactor/round07_minos_comparator_results.csv \
    --results-jsonl notes/codex-refactor/round07_minos_comparator_results.jsonl \
    --failures-csv notes/codex-refactor/round07_minos_comparator_failures.csv \
    --failures-jsonl notes/codex-refactor/round07_minos_comparator_failures.jsonl \
    --rejections-csv notes/codex-refactor/round07_minos_comparator_rejections.csv \
    --rejections-jsonl notes/codex-refactor/round07_minos_comparator_rejections.jsonl \
    --stdout-log notes/logs/round07_minos_comparator_stdout.log
```

## Case Selection

Accepted fair cases:

| Case | Why fair |
| --- | --- |
| `M002W` average | One-parameter average; the profiled value is the only coordinate. |
| `M026R08` average | Primary average coordinate with two nuisance adjustments; MINOS fixes the same primary coordinate. |
| `G(2000),G(1800)::K002M` | Two-parameter `MASS` fit; target is a fitted coordinate. |
| `G(2000),G(1800)::K003M` | Same `MASS` fit; second fitted coordinate. |
| `eta_c J/psi psi(2S)::M026W` | Hard round-05/06 target, but `M026W` is a non-decay fit parameter, so MINOS profiles the same coordinate honestly. |

Rejected cases/ideas:

| Case | Reason |
| --- | --- |
| `B0::S042B95` | `S042B95` is a relationship/node ratio, not a Minuit coordinate. MINOS on any one underlying decay parameter would answer a different question. |
| `B0S-BR::S086.37` in default chart | `S086.37` is a physical BRU parameter, but production profiles it through the sigmoid fitted chart. MINOS on the default fitted coordinate gives an internal chart-coordinate interval; MINOS on bounded physical coordinates changes the profile formulation. Worth a later caveated test, not a clean oracle here. |
| Unbracketed MINOS replacement | Rejected. The current invariant is endpoint verification against `chi2_min + 1`; MINOS status is useful evidence but not a substitute for residual checks. |

## Endpoint Results

Residuals below are fresh fixed-coordinate Minuit profile residuals relative to
the pdgfits target `chi2_min + 1`, except `pdg_resid_repo`, which is the
repository endpoint-search residual.

| Case | Side | pdg endpoint | MINOS endpoint | Delta | pdg resid repo | fresh MINOS resid | pdg err | HESSE err | MINOS err |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `avg_m002w` | lower | 0.208769531 | 0.208716113 | -5.34e-05 | -4.9997e-03 | -1.48e-05 | 0.0212695 | 0.0214052 | 0.0214052 |
| `avg_m002w` | upper | 0.251523437 | 0.251526702 | 3.26e-06 | -3.1976e-04 | -1.48e-05 | 0.0214844 | 0.0214052 | 0.0214052 |
| `avg_m026r08` | lower | 0.139853604 | 0.139870337 | 1.67e-05 | 1.7222e-03 | -1.17e-05 | 0.0195461 | 0.0196892 | 0.0195068 |
| `avg_m026r08` | upper | 0.179269747 | 0.179256673 | -1.31e-05 | 1.2912e-03 | -9.98e-06 | 0.0198701 | 0.0196892 | 0.0198795 |
| `fit_g2000_k002m` | lower | 1787.491371561 | 1787.491371561 | 1.33e-10 | 3.58e-11 | -3.46e-14 | 7.39652 | 7.39652 | 7.39652 |
| `fit_g2000_k002m` | upper | 1802.284413193 | 1802.284413193 | -1.34e-10 | 3.63e-11 | -2.35e-14 | 7.39652 | 7.39652 | 7.39652 |
| `fit_g2000_k003m` | lower | 1992.211129053 | 1992.211129053 | 7.51e-10 | 1.90e-10 | -9.33e-15 | 7.92340 | 7.92340 | 7.92340 |
| `fit_g2000_k003m` | upper | 2008.057929243 | 2008.057929243 | -7.51e-10 | 1.90e-10 | 1.29e-14 | 7.92340 | 7.92340 | 7.92340 |
| `fit_eta_m026w` | lower | 30.037618489 | 30.037778478 | 1.60e-04 | 7.08e-04 | 2.03e-05 | 0.434663 | 0.434635 | 0.434542 |
| `fit_eta_m026w` | upper | 30.906944467 | 30.907142581 | 1.98e-04 | -9.35e-04 | 8.67e-05 | 0.434663 | 0.434635 | 0.434822 |

All MINOS sides were valid. No fresh fixed-coordinate profile found a lower
endpoint crossing inconsistent with the current pdgfits profile endpoint. The
largest endpoint shift was `1.98e-04` on the hard `eta_c J/psi psi(2S)::M026W`
upper endpoint, below the current bisection residual scale.

## HESSE vs MINOS

The parabolic/HESSE error is identical to MINOS in the two-parameter `MASS` fit,
as expected for a nearly quadratic direct-coordinate profile. It differs in the
cases with asymmetric/nuisance/nonlinear structure:

- `M026R08`: HESSE differs from MINOS by about `1.8e-04` to `1.9e-04`, roughly
  1 percent of the error.
- `eta_c J/psi psi(2S)::M026W`: HESSE differs from MINOS by about `9.2e-05` to
  `1.9e-04`, less than 0.05 percent of the error but with visible side
  asymmetry.

This supports using HESSE/parabolic errors as a nonlinearity diagnostic and
start predictor, not as the reported asymmetric error when profile endpoints
are required.

## Interpretation

Methodological pieces worth borrowing from MINOS:

- Covariance/error-matrix predicted conditional-profile starts. The clean
  direct-coordinate cases show MINOS lands very close to the verified endpoint
  with fewer root-search artifacts than bisection tolerance alone.
- Structured crossing status. MINOS explicitly reports valid side, new minimum,
  parameter limit, and call-limit style failures; pdgfits should keep and extend
  first-class endpoint status rather than relying on optimizer success alone.
- HESSE-vs-MINOS comparison. The difference is an inexpensive local
  nonlinearity/asymmetry diagnostic.

Parts incompatible with pdgfits as a general replacement:

- MINOS profiles one coordinate. Many pdgfits targets are arbitrary scalar node
  functions of physical parameters, such as ratios or relationship-derived
  observables.
- MINOS coordinate intervals depend on the parameter chart. For BR/BRU physical
  parameters, the production sigmoid/softmax maps and physical boundary
  constraints are not interchangeable with Minuit's external box-limit
  transform.
- MINOS is still local. It does not replace fixed-target chi2 residual
  verification or the need to certify the profiled minimum.

## Bottom Line

Use `iminuit`/MINOS as a regression oracle for direct-coordinate average and fit
targets, including hard direct coordinates such as `M026W`. Do not use it as a
general oracle for arbitrary scalar or mapped physical targets.

The most plausible pdgfits borrowing is MINOS-style covariance-start prediction
plus richer endpoint status fields. Keep bracketed endpoint verification as the
scientific invariant.
