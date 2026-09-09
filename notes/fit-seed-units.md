# Fit seed handling and units

Notes on how initial parameter values ("fit seeds") flow through the fitter, the
units problems they carry, and the fix applied in `fit.py` (June 2026).

## Where seeds come from

Seeds live in the `fit_seed` DB table (`par_code, parameter, seed`), queried by
`query.fit_queries` into `fit_seed_df`. They are starting values for the optimizer
only — they do not affect the final result if the fit converges, but a bad seed can
break the parameter reparametrization (see below) or send a hard fit to a bad basin.

## Seed transforms in the pipeline

1. **Lifetime → width** (`preprocess._add_lifetime_relations`): for `data_type='T'`
   nodes the fit works in width, so the width seed is derived as `1/lifetime_seed`.
   This mirrors the PDG FORTRAN (`sbrfit.f:253-254`, `ZSEED = 1/ZSEED`).
2. **Nuisance parameters** (`preprocess._add_nuisance_parameters`): seeds for nodes
   pulled in as nuisance params come from `pdg_most_precise_value`, not `fit_seed`.
3. **Decay (branching-fraction) units normalization** (`fit.py`, in `run_fit`): the
   subject of these notes.

## The units problem

The fit operates in standardized units. For *measurements*, `preprocess._apply_unit_scaling`
converts to those units (e.g. keV/eV → MeV via `parser.get_scale`). **Seeds are not
unit-scaled** — the code (like the PDG FORTRAN) trusts `fit_seed` to already be in
working units. That trust mostly holds, except for **branching fractions**, which are
stored inconsistently: some as proportions in (0,1), some as **percent** (summing to
~100, individual values up to ~98).

A percentage seed is fatal for us specifically: the `param_maps` reparametrizations
(`build_param_map_sigmoid` / `build_param_map_softmax`) require decay seeds in (0,1).
The sigmoid inverse is `tan(π·(val−0.5))`, which produces inf/NaN for `val > 1`,
tripping the `assert not isnan/isinf` in `run_fit`. (The PDG FORTRAN tolerates raw
percent seeds because its fit is a direct Gauss-Newton on a near-linear model, robust
to the starting point — it never maps into (0,1). We are not.)

### Where percent seeds actually occur (scan of `fit_seed`, June 2026)

Sigmoid path (BRU), all clearly percent (`seed_max` ≫ 1):

| label    | par_code | n_params | seed_sum | seed_max |
|----------|----------|----------|----------|----------|
| D+-      | S031     | 18       | 20.8     | 5.0      |
| D0       | S032     | 35       | 69.2     | 14.0     |
| Lambda_c | S033     | 26       | 45.0     | 7.0      |
| D_s      | S034     | 12       | 18.3     | 6.0      |

Softmax path (single-particle BR): ~14 groups stored as percent, summing to ~100
(eta'(958), phi(1020), omega(782), pi0, Lambda, K_L, ...).

Note the BRU groups do **not** sum to ~100: only a subset of each particle's decays
are fit parameters, so the proportions legitimately sum to < 1. This is why the old
"divide by the sum" hack was wrong for them (it forced sum-to-1, inflating each seed).

## The fix (`fit.py`, decay-seed block in `run_fit`)

Replaced the old `if decay_seed_sum >= 1: divide by sum` hack with a per-particle split
keyed on the already-computed `is_br` / `is_bru` flags:

- **`is_br`** (single-particle, sum-to-1 / softmax): **always** divide the group by its
  sum. Enforces the sum-to-1 starting point and absorbs percent entries as a side
  effect. Matches the PDG FORTRAN `SUM_TO_1` branch (`sbrfit.f:697-716`), which
  normalizes unconditionally rather than only when the sum ≥ 1.
- **`is_bru`** (multi-particle BR / BRU, sigmoid): branching fractions are NOT
  constrained to sum to 1, so divide-by-sum would distort them. Instead detect percent
  by `group.max() > 1` (a true proportion is never > 1) and divide that group by 100.

After either branch, seeds are clipped to `[1e-6, 1-1e-6]`. Proportion groups (max ≤ 1)
in the sigmoid path are left untouched apart from clipping.

These flags also cover the `fit_space='constrained'` sub-cases, where seeds likewise
need to be valid proportions.

## Verification

All affected fits run valid with chi2 matching PDG's stored value:

- Sigmoid / percent→proportion: D0 (chi2 150), D+- (65), Lambda_c (58.8), D_s (12.1)
- Softmax / normalize-by-sum: eta'(958) (69.6 vs PDG 69.5)

A throwaway scan script for re-detecting percent seeds was used during investigation
(grouped `fit_seed` by `par_code`, flagged `seed > 1`); not committed.

## PDG FORTRAN reference (`src/pdg_fortran_fit_code/`)

- `sbrfit.f:683` — `PP(IPAR) = ZSEED(Iseed)`: seeds used raw, no unit scaling.
- `sbrfit.f:697-716` — `SUM_TO_1` normalization (divide by sum), single-particle BR only.
- `sbrfit.f:328-349` — `SUM_TO_1` is True only for single-particle non-BRU `BR`.
- `sdofit.f:642-648` — `ZUNITS`/`DADJUST` computed per seed but used **only for display**
  (`sprtfit.f:166`, `ZFITPAR/PARUNITS`), confirming seeds aren't unit-converted for the fit.
- `sdofit.f:1084-1088` — measurements *are* scaled by `DADJUST` to working units.

## June 23 follow-up: direct and lifetime seed scales

The two remaining SciPy failures after the first conditioning pass,
`K_L eta+-,00 phase` and `K_L eta+-,00 ph CPT`, exposed the same general issue
outside branching fractions:

- `S013D` was seeded as `0.53`, while its parsed direct measurements are
  around `5.3e9` because the measurement strings carry `E10`.
- `S012W` was derived as `1 / 0.89 ~= 1.12` from the raw lifetime seed, while
  parsed `S012T` lifetime measurements are around `8.96e-11`, so the width seed
  should be around `1.12e10`.

`fit.py` now corrects only obvious power-of-ten mismatches before optimization:

- Directly measured non-decay parameters keep their seed mantissa but are
  multiplied by the nearest decimal scale when the seed and direct parsed
  measurements differ by at least `1e4`.
- Lifetime-derived width seeds are reset to the inverse parsed lifetime scale
  when the derived width seed differs by at least `1e4`.
- Large corrected non-decay coordinates are linearly scaled in fitted space,
  using the same `build_param_map_scaled()` machinery as the earlier
  lifetime-width preconditioner.

This preserves the chi2 objective and Minuit minima, but gives SciPy a
physically scaled starting point for future fits with scientific-notation
measurement strings.
