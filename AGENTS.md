# AGENTS.md

This file provides guidance to agents working in this repository. Keep it current when CLIs, validation workflows, or numerical methods change. `SCIENTIFIC_CONTEXT.md` explains the scientific purpose and should be read before changing fit/average methodology.

## Overview

`pdgfits` reproduces PDG least-squares fits and weighted averages from either the internal PDG PostgreSQL database or an offline snapshot backend. It builds chi-squared objectives with JAX/scipy/iminuit, supports nonlinear PDG relationship equations, nuisance parameters, BR/BRU parameter maps, and profile-likelihood asymmetric errors.

Online DB mode requires an SSH tunnel to `127.0.0.1:5433` plus `.env` values `DB_NAME`, `DB_USER`, `DB_PW`. For agent work, prefer the snapshot backend unless online DB access is explicitly needed.

## Environment

Use the existing micromamba env when available:

```bash
env PYTHONPATH=/root/pdgfits-private/src \
  PDGFITS_DATA_BACKEND=snapshot \
  PDGFITS_SNAPSHOT_DIR=/root/pdgfits-private/data/pdg-snapshot \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  /root/micromamba/micromamba run -r /root/micromamba-root -n pdg <command>
```

For long Codex/benchmark runs on the Hermes VPS, cap resources conservatively, e.g. `prlimit --as=7800000000 --rss=7800000000`, and avoid high parallelism; this host also runs Hermes/Mattermost.

## Commands

```bash
# Run a single fit
python -m pdgfits.run_fits --fit_label '<LABEL>'
python -m pdgfits.run_fits --fit_label '<LABEL>' --calc_asym_errors

# Run weighted averages
python -m pdgfits.run_avgs
python -m pdgfits.run_avgs --node <NODE>

# Capture/replay offline query snapshot
python -m pdgfits.snapshot capture --out data/pdg-snapshot
PDGFITS_DATA_BACKEND=snapshot PDGFITS_SNAPSHOT_DIR=data/pdg-snapshot python -m pdgfits.run_fits --fit_label '<LABEL>'

# Tests
python -m pytest tests/ -q
python -m pytest tests/ -m db --db -q   # include DB regression tests
```

## Current Asymmetric-Error Method

Asymmetric errors are profile-likelihood/Wilks endpoints: for each side, find a target value where the profiled chi-squared equals `chi2_min + 1` after profiling over all other fitted/nuisance parameters. Feasibility and optimizer success alone are not sufficient; endpoints must be verified by the profiled chi2 residual.

Fit-side profile solver, as of the simplified broad sweep:

1. project continuation/MLE starts onto the scalar target constraint;
2. try SLSQP;
3. fallback to exact-Hessian `trust-constr`;
4. KKT-polish exact-Hessian candidates;
5. accept only finite, scaled-feasible profile points with either projected/KKT stationarity or a no-meaningful-feasible-descent certificate;
6. allow endpoint bracket contraction when outward BR/BRU targets are unreachable;
7. always verify the endpoint residual against `chi2_min + 1`.

Removed fallbacks that were not needed on hard-case/broad staged validation: BFGS-Hessian `trust-constr` and reduced/nullspace Powell.

Average-side profiling remains the simpler direct fixed-primary-coordinate nuisance profile. It was full-snapshot validated separately.

Endpoint search remains bracketed bisection with endpoint verification. It now avoids the duplicate final profile solve by reusing the final bisection evaluation and records per-search counters/cache diagnostics. Do not replace this with unbracketed secant/Newton logic unless a hard-case benchmark proves it is both simpler and no less robust.

## Validation Artifacts

Important notes/results live under `notes/`:

- `notes/asym-profile-simplification.md` — ablation showing the minimal focused-passing fit solver.
- `notes/asym-fit-simplified-broad-sweep.md` — staged broad fit sweep: 227/227 rows ok, 454 endpoints, previous 13 failures fixed with no regressions.
- `notes/endpoint-search/asym-endpoint-search-exploration.md` — endpoint-search benchmark and small final-evaluation reuse optimization.
- `notes/run_asym_fit_sweep.py` — broad/staged fit sweep harness.
- `notes/run_asym_profile_ablation.py` — focused hard-case ablation harness.
- `notes/endpoint-search/run_asym_endpoint_search_benchmark.py` — focused endpoint-search benchmark harness.

Keep bulky logs under `notes/logs/` and do not commit giant logs unless the user explicitly asks.

## Architecture

### Fits

Pipeline: `query` → `preprocess` → `build_funcs` + `corr_mat` → `build_chi2` → `param_maps` → minimization, orchestrated by `fit.run_fit()`. CLI entry point: `pdgfits.run_fits`.

### Averages

Single-node weighted averages: `query.avg_queries()` → `avg.run_avg()` → standard preprocessing / model / chi2 pipeline with vectorized `mu`. CLI entry point: `pdgfits.run_avgs`.

Detailed module notes are in `src/pdgfits/CLAUDE.md`.