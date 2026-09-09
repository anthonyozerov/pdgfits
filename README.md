# PDG fits and averages

Python/JAX implementations of joint fits and weighted averages for Particle Data Group data. Both use the same measurement parsing, relationship functions, asymmetric-error objective and nuisance inputs. These are proposed methods; they do not exactly reproduce every current PDG prescription.

## Run locally

Install the package with `python -m pip install -e '.[test,plots]'`. Select the offline snapshot explicitly for reproducible work:

```bash
export PDGFITS_DATA_BACKEND=snapshot
export PDGFITS_SNAPSHOT_DIR=/absolute/path/to/pdg-snapshot
python -m pdgfits.run_fits --fit_label 'B0S-BR'
python -m pdgfits.run_fits --fit_label 'Upsilon(2S)' --calc_asym_errors
python -m pdgfits.run_avgs --node M026R08 --output results/M026R08.csv
python -m pytest
```

The installed commands are `pdgfits`, `pdgfits-averages` and `pdgfits-snapshot`. Install the `plots` extra for contours and the curated YAML node selection. A single average writes a one-row CSV. Its `n_meas` includes auxiliary constraint measurements, `n_primary_meas` counts primary-node observations, and `ndof = n_meas - n_params` is the nominal model count, not a proof of a chi-squared sampling law.

The default backend is the internal PostgreSQL database. It requires the PDG tunnel and local `.env` credentials. An authorized online session can capture an offline snapshot with `python -m pdgfits.snapshot capture --out data/pdg-snapshot`. Snapshots and captured regression fixtures are local data, excluded by `.gitignore`.

## Code map

- `query.py`, `snapshot.py`: database access and offline replay.
- `parser.py`, `preprocess.py`: measurement units, PDG relationships, dependent measurements and auxiliary inputs.
- `build_funcs.py`, `corr_mat.py`, `build_chi2.py`: predictions, measurement correlations and the common objective.
- `param_maps.py`: bounded branching fractions, sum-to-one coordinates and numerical scales.
- `fit.py`, `avg.py`: prepare and minimize a joint fit or an average; return plain result dictionaries.
- `profiles.py`: minimize while holding an arbitrary target, or a direct average coordinate, fixed.
- `asym_errors.py`: bracket and verify profile endpoints. `find_profile_root` returns the result directly; `binary_search_error` remains a compatibility wrapper for older experiments.
- `diagnostics.py`, `plotting.py`: comparisons, sensitivities and optional plots.

Reported profile endpoints satisfy `Q_profile(target) - Q_min = 1`, within the requested objective tolerance. `Q` uses PDG-style interpolation of reported asymmetric errors and an inverse correlation matrix. Treating it as a likelihood ratio with nominal coverage requires a justified statistical model; successful optimization alone does not establish that interpretation.

The profile solver retains projected starts, SLSQP, exact-Hessian fallback, KKT polishing, feasibility/stationarity checks and a small feasible-descent check. These safeguards were needed on the recorded difficult cases. The experimental `birge.block_birge` diagnostic is available explicitly but no longer runs or prints rescaled errors automatically for every average.

## Validation and provenance

`python -m pytest` runs the offline tests. `python -m pytest --db` opts into the existing database regression tests; use it only with the intended backend and data access.

`tools/check_snapshot.py OUTPUT.jsonl` captures all 81 snapshot fits, difficult profile targets, four SciPy cases, every nuisance-bearing average and a spread of direct averages. Run it in separate processes with `PYTHONPATH` set to the old and new package sources for a numerical refactor comparison. `tools/compare_snapshot.py BEFORE.jsonl AFTER.jsonl` checks their agreement.

Retained June experiments and raw results are under `notes/` and `outputs/`. They describe their original code revisions and sampling assumptions. The independent remote history through `3691d0e` was imported as squash commit `bf3d97c`; the untouched nested source remains ignored. The subsequent simplification and its validation are documented in `notes/simplification-20260908.md`.

## Experimental node scales

For a **linear, unconstrained Gaussian model with fixed symmetric errors**,
`birge.fit_linear_scales(y, X, V, nodes)` estimates a separate inflation factor
for each node using restricted maximum likelihood. It refits the mean as the
scales change and preserves the supplied correlations through `D_s V D_s`.
A single ordinary average recovers the clipped Birge ratio.

This is an opt-in reference calculation. It does not modify `run_fit` or
`run_avg`, implement the legacy weak-input rule, or establish coverage for the
asymmetric objective. Returned covariance is conditional on the estimated scales.
See [the node-scale note](notes/node-scales.md) for the exact assumptions and
remaining work. The companion `birge.linear_birge` returns residual diagnostics
and their variance-mixing matrix for independent nodes or correlated blocks.

`pdg_scaling.pdg_average` and `pdg_scaling.pdg_linear_fit` provide comparison
baselines for the PDG prescriptions. The first includes the existing asymmetric
average iteration and scale-only exclusion. The second covers linear symmetric
fits, separate pull scales, one exclusion/refit pass, original-center reporting,
and the existing nonsingular correlation-block adjustment. Use `exclude_weak=False`
to compare no exclusions. These are explicit comparator APIs, not automatic changes
to the main fitting pipeline. See [the method note](notes/node-scales.md).
