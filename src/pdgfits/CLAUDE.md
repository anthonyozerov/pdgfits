# AGENTS.md

> **Note to agents:** Keep this file updated whenever modules, APIs, or key design details change.

## Purpose

This subdirectory implements a general least-squares fitter for PDG particle physics data. It queries the internal PDG PostgreSQL database, constructs a chi-squared objective from measurements and their inter-relationships, and minimizes using `iminuit` or `scipy.optimize.minimize` + JAX autodiff.

## Database Connection

`query.py` connects via `psycopg2` to a local SSH tunnel (`127.0.0.1:5433`) into the PDG server. Credentials come from a `.env` file with `DB_NAME`, `DB_USER`, `DB_PW`. The tunnel must be running before querying.

For offline work, `query.py` also supports a simple snapshot backend. Capture
one while online:

```bash
python -m pdgfits.snapshot capture --out data/pdg-snapshot
```

Then replay it by setting `PDGFITS_DATA_BACKEND=snapshot` and
`PDGFITS_SNAPSHOT_DIR=data/pdg-snapshot`. The snapshot stores the public query
outputs (`all_fits`, `fit_queries`, `avg_queries`) plus the extra
`pdg_most_precise_value` and `nuisance_corr` lookups that preprocessing and
diagnostics need. It intentionally does not recreate a local SQL database.

## Data Model

The PDG database tables used:

- **`fit_control1`** — maps a `label` to a fit `algorithm`, `measurement_type`, `data_count`, and `chi_square` (PDG's reported chi2)
- **`fit_control2`** — lists the nodes (PDG IDs) and their equation types included in a fit
- **`relationship`** / **`relationship_equation`** — defines how nodes depend on parameters via linear combinations, with equation types (`+`, `G+`, `R+`, `lifetime`, `/`, `P`, `G*`, `P/`, `SR`, `SQ`, etc.)
- **`measurement`** / **`measurement_adjust`** — raw measurement strings and branching-ratio-adjusted versions; filtered to `place='U'`, `publication_status IS NULL`, `fit_flag IS NULL`, matching `measurement_type`, and `u.summary_year IS NULL`
- **`correlation`** — off-diagonal correlation coefficients between measurements
- **`fit_seed`** — initial parameter values for the optimizer
- **`units`** / **`pdgid`** — unit text and data type (e.g. `'T'` for lifetime) per node
- **`tree`** — data types for nodes (e.g. `'G'` for width, `'E'` for partial width)
- **`reference`** — source year for measurements
- **`ignore_minus`** — flags nodes where the absolute value of measurements should be taken
- **`fit_correlation_matrix`** — correlations between nuisance parameters (used when coefficient parameters are outside the fit)
- **`result_summary`** — PDG summary values for parameters; queried by `pdg_value` / `pdg_most_precise_value`

## Fitting Pipeline

1. **`query.fit_queries(fit_label)`** — queries all tables and returns `(algorithm, measurement_type, fit_df, rel_df, meas_df, corr_df, fit_seed_df, tree_df)`; `query.all_fits()` returns the full `fit_control1` table; `query.avg_queries()` returns `(avg_df, corr_df_dict)` for all average nodes (no relationship or seed tables)
2. **`preprocess.preprocess(fit_df, rel_df, meas_df, corr_df, fit_seed_df, tree_df, algorithm, measurement_type)`** — a thin orchestrator that applies a sequence of named, individually-testable helpers (each takes and returns the dataframes it touches; no shared state). PDG-specific node lists are module constants (`UNIVERSALITY_TRIGGER_NODES`, `UNIVERSALITY_FAKE_NODES`, `EQ_TYPE_FORCE_PLUS`, `PARTIAL_WIDTH_EQ_TYPES`). The steps, in order:
   - `_equate_universality_pairs` — e-mu universality (`measurement_type='UNIV'`) and omega-/antiomega+, Xi-/antiXi+ mass/lifetime equality: removes fake measurements and directly equates parameters
   - `_extract_dep_meas` — `dep_meas` measurements: offsets stored in `dep_meas_data`, actual value extracted
   - `_parse_measurements` — parses measurement strings (via `unadjust_measurement`); `ignore_minus` takes absolute value of measurements for flagged nodes; `br_adjust` measurements un-adjust the scaling, storing adjustment factors in `adjust_data`
   - `_apply_unit_scaling` — applies unit scaling (keV→MeV etc.)
   - `_force_plus_eq_types` — overrides equation type to `+` for the `EQ_TYPE_FORCE_PLUS` nodes
   - `_add_lifetime_relations` — lifetime nodes (`data_type='T'`): inserts `lifetime` equation type (`lifetime=1/width`) and derives width seed
   - `_add_partial_width_relations` — partial-width–to–total-width summation rows (`G+`, `G*`, `R+` nodes)
   - `_build_parameter_keys` — builds `parameter_key`/`coeff_parameter_key` columns and the `parameters`/`nodes` lists
   - `_add_nuisance_parameters` — nodes referenced as coefficient, br_adjust, or dep_meas targets but not in fit get fake measurements from `pdg_most_precise_value`, plus correlations from `fit_correlation_matrix`. Iterates `sorted(set(...))` so nuisance rows are appended in a **deterministic** order (a bare `set` previously made the output row order non-reproducible across runs; harmless to results but not reproducible)
   - `_add_clump2_correlations` — `systematic_error_clump2`: infers correlations among triplets of related measurements (COMER2 handling; uses `organize_triplet_clump`)
   - `_add_clump_correlations` — `systematic_error_clump`: adds pairwise correlations from shared systematics within a node
   - Returns `(fit_df, rel_df, meas_df, corr_df, fit_seed_df, dep_meas_data, adjust_data)`. Row order within the returned frames is **not** a contract (downstream is built consistently from `meas_df`); `tests/test_preprocess.py` compares outputs order-insensitively.
3. **`build_funcs`** — builds JAX functions mapping parameter arrays → predicted measurand values:
   - `get_node_funcs` / `get_parameter_funcs` → per-node/parameter JAX callables
   - `get_meas_funcs` / `get_mu` → combined `mu(params)` returning all predicted values
   - `get_mu_vectorized(parameters, nodes, meas_df, fit_df, rel_df, jit=True)` → faster alternative to `get_mu` that builds a single matrix multiply for all linear-simple `'+'` nodes and falls back to `func_factory` only for non-linear nodes; used by `avg.py`
   - `_build_node_func(node, parameters, eq_type_map, rel_df, jit=True)` → internal helper that builds a `func_factory` callable for a single node; used as the non-linear fallback inside `get_mu_vectorized`
   - `get_translate_dep(dep_meas_data, ...)` → returns `translate_dep(params)` array adding dependent-measurement offsets to residuals
   - `get_adjust(adjust_data, ...)` → returns `adjust(params)` array of per-measurement scale factors (for branching-ratio-adjusted measurements divided/multiplied by another node)
4. **`corr_mat.get_corr_mat`** — builds full correlation matrix as a JAX array (identity + off-diagonal from `corr_df`); inverse computed via `jnp.linalg.pinv` in the caller
5. **`build_chi2.build_chi2(y, mu, error_n, error_p, corr_mat_inv, fitted_params_to_params, translate_dep=None, adjust=None, use_jit=True)`** — constructs the chi-squared function with asymmetric errors (effective error interpolated between `error_n` and `error_p` based on residual sign, then clipped to `[error_min, error_max]`); returns `(chi2, chi2_grad, chi2_val, chi2_open)` where `chi2_open(fitted_params, y_arg)` takes measurements as an explicit argument (used for sensitivity analysis). Pass `use_jit=False` to skip `@jax.jit` on `chi2` and `chi2_open` (useful when running ~1000 short averages to avoid XLA compilation overhead).
6. **`build_chi2.build_chi2_c(...)`** — variant that also fits per-node error scale factors `c`; adds a `log(det) + 2*sum(log(error))` likelihood term; experimental
7. **`BR`/`BRU` algorithm handling** — param-mapping approaches in `param_maps.py`:
   - `build_param_map(parameters, fit_seed_df)` — **deprecated**; excludes the largest-seed decay per particle and reconstructs it as `1 - sum(others)`
   - `build_param_map_softmax(parameters)` — softmax reparametrization for single-particle `BR` fits (sum-to-1 enforced); `fixed_idx` returned for fixing one redundant parameter in Hesse
   - `build_param_map_sigmoid(parameters)` — arctan reparametrization for `BRU` algorithm or multi-particle `BR` fits; maps each decay parameter independently to `(0,1)` via `0.5 + atan(x)/π` with no sum-to-1 constraint; inverse is `tan(π*(x-0.5))`; no `fixed_idx` needed
   - `get_decay_info(parameters)` — returns `(decay_particles, decay_param_idxs)` for all decay particles; used in the constrained fit path
8. **Minimization** — two backends selectable via `optimizer` argument to `run_fit` (default `'minuit'`), and two fit spaces selectable via `fit_space` (default `'unconstrained'`):
   - `optimizer='minuit'`: `iminuit` with JAX gradient and Hesse; `errordef=1`; `strategy=0`; alternates `migrad()`/`simplex()` four times before final `hesse()`; covariance from Minuit's Hesse
   - `optimizer='scipy'`: `scipy.optimize.minimize`; unconstrained fits use `trust-ncg` with JAX Hessian-vector products, then a BFGS polish + trust-region retry if the first trust solve stalls, and finally a Nelder-Mead escape only if no successful finite result has been found. Constrained fits use a best-of constrained driver from the PDG-derived fit seed, trying `SLSQP` and `trust-constr`. Covariance is computed as `2 * pinv(H)` where H is the Hessian evaluated at the optimum via `jax.hessian(chi2)`.
   - `fit_space='unconstrained'`: decay params reparametrized via sigmoid (BRU/multi-particle BR) or softmax (single-particle BR); optimizer works in unconstrained space
   - `fit_space='constrained'`: decay params optimized directly in `[0,1]`; for minuit (sigmoid case only — softmax raises an error) sets `m.limits[i] = (0, 1)`; for scipy uses `Bounds` + `SLSQP`, with an additional `LinearConstraint` enforcing sum-to-1 in the softmax case
9. **Validation** — `run_fits.py` compares the fitted chi2 against `fit_control1.chi_square` (the PDG-reported chi2 stored in the DB). `fit_query.pdg_most_precise_value(node)` fetches the most precise PDG summary value for a node from `result_summary`, used for nuisance parameter initialization and optional parameter comparison.

## Scripts and Modules

- **`param_maps.py`** — all parameter-mapping logic for BR/BRU fits: `build_param_map` (deprecated), `build_param_map_sigmoid`, `build_param_map_softmax`, `get_decay_info`, `build_soft_pos_constraint`
- **`fit.py`** — `run_fit(label, verbose=True, optimizer='minuit', fit_space='unconstrained')` runs a single fit end-to-end and returns a plain dict with:
  - `label`, `algorithm`, `parameters`, `nodes`
  - `param_values`, `fitted_values`, `chi2_min`
  - `chi2`, `chi2_open`, `chi2_grad`, `fitted_params_to_params`, `params_to_fitted_params`
  - `node_funcs`, `parameter_funcs`, `mu`
  - `meas_df`, `rel_df`, `fit_df`
  - `covariance` (from Minuit Hesse, or `2 * pinv(jax.hessian(chi2))` for scipy)
  - `fit_valid`, `hesse_accurate` — `m.valid` and `m.accurate` from iminuit after Hesse; both `None` for scipy
  - Returns `None` if the fit is skipped (unsupported equation types)
  - **Seed-unit fixes**: `fit_seed` stores branching-fraction seeds inconsistently (some as proportions in (0,1), some as percent). Before building `param_init`, decay seeds are normalized per particle: `is_br` (single-particle, sum-to-1 / softmax) always divides by the group sum (matches the PDG FORTRAN `SUM_TO_1` branch, `sbrfit.f:697-716`); `is_bru` (multi-particle BR / BRU, sigmoid) divides any group containing a seed > 1 by 100 (a true proportion is never > 1). Divide-by-100 — not divide-by-sum — is used for `is_bru` because those branching fractions are not constrained to sum to 1. Directly measured non-decay parameters are also corrected when the seed and parsed measurements differ by an obvious power of ten, and lifetime-derived width seeds are corrected from parsed lifetime measurements when the raw lifetime seed was in display units. See `notes/fit-seed-units.md` for the branching-fraction justification.
- **`asym_errors.py`** — profile-likelihood asymmetric errors:
  - `ProfilePoint`, `ProfileProblem`, `ProfileEndpoint`, and `ProfileRoot` make the profile/root interface explicit. Profile implementations return certified `ProfilePoint` values; `find_profile_root(...)` owns bracket contraction/expansion, bisection, endpoint residual verification, counters, and endpoint `ProfilePoint` diagnostics. `binary_search_error(profile_chi2, val, chi2_min, lb, ub, ...)` is the compatibility wrapper returning `(error_p, error_n)` and filling `binary_search_error.last_diagnostics` / `last_root`.
  - `build_constrained_profile_chi2(...)` — builds a checked constrained-profile callable for fixed target values. Fit-side profile solves project starts onto the scalar constraint, try SLSQP, fallback to exact-Hessian `trust-constr`, KKT-polish exact-Hessian candidates, and accept only finite/scaled-feasible points with projected/KKT stationarity or a no-meaningful-feasible-descent certificate.
  - `calc_asym_errors(fit, targets=None)` — computes asymmetric errors for selected node/parameter targets; `targets=None` defaults to all nodes and fitted parameters. Endpoint correctness means profiled chi2 at the returned endpoint equals `chi2_min + 1`, not merely optimizer success.
- **`diagnostics.py`**:
  - `get_pdg_chi2(label)` — returns the PDG-reported chi2 for a label from `fit_control1`
  - `compare_to_pdg(fit)` — prints per-parameter comparison of fitted values against PDG summary values, and evaluates chi2 at the PDG point
  - `meas_diagnostics(fit)` — prints per-node measurement chi2 diagnostics (our fit vs. PDG values)
  - `meas_sensitivity(fit)` — for each measurement computes `d(mu_j)/d(y_j) * sigma_meas_j / sigma_fitted_j` via implicit differentiation of the optimality condition; also returns the full `(n_params, n_meas)` derivative matrix `d_params_dy`
- **`run_fits.py`** — CLI entry point; iterates over fits in `fit_control1` (skipping `IGNORE`) and calls `run_fit` + optional diagnostics. CLI args:
  - `--start_from LABEL` — resume from a given label
  - `--fit_type TYPE` — filter by algorithm (e.g. `BR`)
  - `--fit_label LABEL` — run a single fit
  - `--measurement_type TYPE` — filter by measurement type
  - `--compare_to_pdg` — print per-parameter comparison against PDG values
  - `--calc_asym_errors` — compute asymmetric errors (slow)
  - `--meas_diagnostics` — print per-node measurement chi2 diagnostics
  - `--meas_sensitivity` — print per-measurement sensitivity analysis
  - `--optimizer {minuit,scipy}` — select minimizer (default: `minuit`)
  - `--fit_space {unconstrained,constrained}` — select fit space (default: `unconstrained`)
- **`snapshot.py`** — captures a dependency-free offline snapshot of the
  DataFrame query layer. `python -m pdgfits.snapshot capture --out PATH` writes
  `all_fits.pkl`, one `fits/*.pkl` per captured label, optional `avg_queries.pkl`,
  and lookup caches for `pdg_most_precise_value` / `nuisance_corr`.
- **`fit_test.py`** — original monolithic script, kept for reference
- **`avg.py`** — `run_avg(node, meas_df_node, corr_df_node, skip_avg=False, contours=False, contours_dir=None)` runs a single weighted average. Constructs minimal synthetic `fit_df`/`rel_df`/`fit_seed_df` (equation type `+`, coefficient 1) and feeds them through the standard `preprocess` → `build_funcs` → `build_chi2` pipeline with `use_jit=True`. Uses `get_mu_vectorized` for the forward model. Minimizes with scipy Nelder-Mead; asymmetric errors via `binary_search_error` with profile chi2 (re-minimizing over nuisance parameters at each fixed primary-node value). When `contours=True`, also runs Minuit + `draw_mnmatrix()` and optionally saves the figure to `contours_dir`. Returns a dict `(node, parameters, nodes, param_values, chi2_min, n_meas, error_n, error_p, meas_df, rel_df, fit_df)`, or `None` if fewer than 2 measurements remain after preprocessing.
- **`run_avgs.py`** — CLI entry point for batch weighted averages. Calls `avg_queries()`, iterates over all nodes or a selected `--node`, calls `run_avg` per node, saves CSV output, and prints chi2/ndof diagnostics. CLI args include `--node NODE`, `--start_from NODE`, `--output PATH`, `--test-parse`, and `--interesting`.
- **`avg-node-ex.yaml`** — YAML config listing `interesting` nodes used by the average CLI's `--interesting` mode.

## Key Design Details

- **`func_factory.py`**: Supports equation types `+`, `G+`, `R+`, `lifetime`, `/`, `P`, `G*`, `P/`, `SR`, `SQ`. `func_factory(equation_type, coefficients, coeff_params, jit=True)` returns a function `f(params)`; with `jit=True` (default) it is `@jax.jit`-compiled, with `jit=False` it runs eagerly. `coeff_params` indexes into `[1, p0, p1, ...]`: index 0 means multiply by 1 (no coefficient parameter), otherwise multiply by the referenced parameter. At factory time, if all `coeff_params` entries are zero (the common case), a simpler `t(params, i) = jnp.dot(params, coefficients[i])` is used; otherwise a `jnp.where`-based gather selects the appropriate parameter multiplier. `G+`, `R+`, and `P` are identical (`t(0) * t(1)`) and handled by the same branch. `get_node_funcs` accepts `jit=True` and passes it through to `func_factory`.
- **`parser.py`**: `parse_measurement` parses PDG measurement strings like `1.234+0.05-0.03E-3` into `(value, pos_error, neg_error, last_err)`, combining multiple stat/syst errors in quadrature. Strings containing `@` are handled by stripping everything after the `@`. Also parses the structured measurement-annotation strings used in the DB (comma-separated records of space-separated tokens; each parser documents its format in a comment): `get_dep_meas_data`, `get_adjust_data`, and `br_adjust_node` (the last two share the `_node_key` helper, which dots a `'<par_code> <param>'` field into a `par_code.param` key). These are covered by `tests/test_parser.py`.
- **`parameter_key`**: Parameters are identified as `par_code.parameter` if `par_code` is non-null, else just `parameter`. Nuisance parameters from `coeff_parameter_key` or br_adjust targets are prefixed with `nuisance_`.
- **`corr_mat_inv`**: The inverse of the correlation matrix is computed once via `jnp.linalg.pinv` and passed into `build_chi2`; the covariance inverse is then `corr_mat_inv / outer(error, error)` (computed inside the chi2 function per call).
- JAX x64 mode must be enabled at startup: `jax.config.update("jax_enable_x64", True)`.
