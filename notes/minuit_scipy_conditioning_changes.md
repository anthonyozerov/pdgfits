# Minuit/SciPy conditioning changes

**Date:** 2026-06-23
**Status:** Implemented in production code.
**Code touched:** `src/pdgfits/param_maps.py`, `src/pdgfits/fit.py`,
`tests/test_param_maps.py`.
**Related notes:** `notes/minuit_vs_scipy_comparison.md`,
`notes/minuit_vs_scipy_followup.md`, `notes/fit-seed-units.md`.

This note documents the optimizer-conditioning changes made after the
Minuit-vs-SciPy debugging work. It is intentionally detailed because several
plausible fixes were close to working, and future work may want to revisit
variants rather than rediscover the same failure modes.

## Executive summary

The original issue was that SciPy frequently failed to reach the same minimum
as Minuit, especially for branching-ratio fits. The follow-up debugging showed
that many failures were not caused by a wrong chi2 model or by Minuit finding a
different physical optimum. Instead, they were caused by poorly conditioned
optimizer coordinates:

- Tiny BRU branching-fraction seeds were mapped by the arctan inverse into huge
  fitted-space coordinates, often O(1e4).
- Lifetime-derived width parameters such as `S013W` and `S010W` lived on raw
  scales around O(1e7-1e8), while their lifetime contribution depends on
  `1 / width`, making the objective nearly flat in the raw coordinate.
- The SciPy path used a single `Newton-CG` solve, while the Minuit path already
  had repeated `migrad/simplex` alternation.

The implemented solution keeps one shared fitted-coordinate system and one
shared initial point for both Minuit and SciPy:

1. BRU/sigmoid decay coordinates keep the arctan map, but the fitted coordinate
   is scaled by `SIGMOID_ATAN_SCALE = 1e-4`.
2. Lifetime-derived width coordinates get a linear scale composed into
   `params_to_fitted_params` and `fitted_params_to_params`.
3. SciPy's unconstrained path uses `trust-ncg` with JAX Hessian-vector products,
   plus a small Nelder-Mead escape if the trust-region solve stalls.

The final code deliberately does **not** use best-of-two raw/clipped starts.
That option can be useful diagnostically, but Anthony explicitly preferred the
cleaner goal: make one setup well-conditioned enough that both Minuit and SciPy
work with it.

## Baseline behavior before these changes

The baseline logs were:

```text
outputs/minuit_2026-06-22.out
outputs/scipy_2026-06-22.out
```

Comparison with `outputs/compare.py`:

```text
minuit fits=81  scipy fits=81  union=81

chi2 agree (rel<= 1e-3): 58/81
chi2 differ:             23
scipy success:False:     19
minuit invalid:           0
```

Representative old SciPy failures:

```text
B0S-BR                 Minuit 25.8    SciPy 7.79e4
Lambda_c               Minuit 58.8    SciPy 6.12e3
D_s                    Minuit 12.1    SciPy 3.34e3
D+-                    Minuit 65.0    SciPy 568
K_L                    Minuit 37.4    SciPy 248
K+-                    Minuit 53.5    SciPy 56.6
chi_c012 psi(2S)       Minuit 394     SciPy 4.49e3
K_L eta+-,00 phase     Minuit 16.5    SciPy 1.14e27
K_L eta+-,00 ph CPT    Minuit 20.0    SciPy 1.84e30
```

There were also "silent" SciPy failures where `success=True` but the chi2 was
far from Minuit/PDG. That is why status alone is not a sufficient correctness
check in this codebase.

## Seed handling still matters

This implementation keeps the decay-seed unit fix from `notes/fit-seed-units.md`:

- Single-particle BR / softmax (`is_br`): normalize decay seeds by their group
  sum.
- Multi-particle BRU / sigmoid (`is_bru`): do **not** normalize by sum; if a
  group contains a seed greater than 1, treat it as percent and divide by 100.
- After the appropriate unit handling, clip decay seeds to `[1e-6, 1 - 1e-6]`
  so the inverse map is finite.

The key point is that the new conditioning changes are intended to improve the
optimizer coordinate system, not to silently change the physical parameter-space
starting values.

## BRU arctan scaling

### Original map

Before this change, the BRU unconstrained map was effectively:

```text
p = 0.5 + atan(x) / pi
x = tan(pi * (p - 0.5))
```

This map enforces `p in (0, 1)`, but tiny branching fractions invert to very
large fitted coordinates:

```text
p = 1e-5  ->  x ~= -1 / (pi * 1e-5) ~= -3.2e4
```

That is a poor optimizer coordinate. The derivative of the map is very small
in this saturated tail, so gradient/Hessian methods can see a direction that is
both important in parameter space and nearly dead in fitted space.

### Implemented map

The implemented map is:

```text
p = 0.5 + atan(x / s) / pi
x = s * tan(pi * (p - 0.5))
```

with:

```text
s = SIGMOID_ATAN_SCALE = 1e-4
```

This preserves the arctan family and the exact same parameter-space seed. It
only rescales fitted space. For `p = 1e-5`, the fitted coordinate becomes
roughly:

```text
x ~= -3.2
```

instead of `-3.2e4`.

This is guarded by:

```text
tests/test_param_maps.py::test_sigmoid_tiny_seed_stays_well_scaled
```

### Why this scale?

`1e-4` was chosen empirically from targeted checks. Scales around `1e-5`,
`3e-5`, and `1e-4` all fixed the representative failures we checked:

- Minuit `chi_c012 psi(2S)` stayed at the old good minimum, chi2 about `393.9`,
  with valid Hesse.
- SciPy `B0S-BR` moved from about `7.79e4` to about `25.8`.
- SciPy `Lambda_c` moved from about `6.12e3` to about `58.8`.
- SciPy `D_s` moved from about `3.34e3` to about `12.1`.

`1e-4` is less aggressive than `1e-5` while still shrinking the old saturated
coordinates by four orders of magnitude. This is not a mathematical constant;
it is a conservative production default based on the current fit suite.

Future work could tune this scale more systematically, possibly per fit or per
decay block. If doing that, check both SciPy and Minuit, and include
`chi_c012 psi(2S)` in the test set because it was sensitive to some alternative
maps.

## Logit experiments

We also tried replacing the BRU arctan map with a scaled logit:

```text
p = sigmoid(x / scale)
x = scale * logit(p)
```

This is a reasonable idea and should not be considered definitively ruled out.
It has attractive tail behavior: for `p = 1e-5`, unscaled `logit(p)` is only
about `-11.5`, much better than the original arctan inverse.

What happened in the checks we ran:

- Small-BRU SciPy failures such as `B0S-BR`, `Lambda_c`, and `D_s` were fixed
  for several logit scales.
- The large multi-particle fit `chi_c012 psi(2S)` was sensitive under Minuit.
  For scales `0.25`, `0.5`, `1.0`, and `2.0`, Minuit landed around chi2
  `1300` rather than the old `393.9` basin. Some larger scales got closer;
  scale `4.0` landed near `397.6` but still had invalid Minuit status/Hesse in
  the targeted check. Larger scales started to degrade `B0S-BR` under SciPy.

This does **not** prove logit cannot work. It only says that the simple scaled
logit variants we checked did not give as clean a one-setup result as scaled
arctan. A logit map with a different scale, better softmax gauge handling, a
different Minuit strategy, or a more careful treatment of multi-particle BRU
blocks might still be viable.

The reason scaled arctan was selected is pragmatic: it retained the arctan
geometry that already worked for Minuit while fixing the fitted-coordinate
scale that hurt SciPy.

## Seed clipping / best-of-starts experiments

The root-cause note suggested clipping fitted-space decay seeds, e.g.
`x0 = clip(x0, -10, 10)`. This did help several SciPy failures, and a
best-of-raw-and-clipped-starts driver can be a useful diagnostic.

However, hard clipping in fitted space changes the implied parameter-space
start. For the arctan map, `x = -10` corresponds to a branching fraction around
3 percent. That is much larger than physically ordinary seeds like `1e-5`.

When this was applied directly to both optimizers, several Minuit fits were
pushed into worse basins. A best-of-two-starts implementation recovered those,
but it no longer satisfied the cleaner design goal of giving both optimizers
the same well-conditioned setup.

This history is worth remembering:

- Clipping can rescue SciPy.
- It is not obviously wrong as a restart strategy.
- It is less clean than changing the coordinate map so the original
  parameter-space seed remains intact.

For production, the scaled arctan map was preferred.

## Lifetime-derived width scaling

The follow-up debugging found that `K_L` and `K+-` were not primarily BRU
sigmoid failures. They were softmax BR fits where the chi2 gap was mostly
carried by lifetime-derived width parameters:

```text
K+-: S010W around 8.1e7
K_L: S013W around 1.9e7
```

Lifetime nodes are represented as:

```text
lifetime = 1 / width
```

In raw width coordinates:

```text
d(1 / W) / dW = -1 / W^2
```

So a physically meaningful move in the lifetime prediction can correspond to a
large raw width move, while gradients in the raw coordinate look tiny. This
particularly confused SciPy's old Newton-CG path.

The implemented fix is to compose a linear preconditioner into the fitted
parameter maps:

```text
optimizer_coordinate = base_fitted_width / scale
base_fitted_width = optimizer_coordinate * scale
scale = max(abs(width_seed), 1.0)
```

This is implemented by `build_param_map_scaled(...)` in `param_maps.py`, and
the fit code detects width parameters by looking for preprocessed fit rows with
`type == 'lifetime'`.

Important properties:

- It does not change the scientific parameter value.
- It does not change the chi2 model.
- It changes derivatives with respect to optimizer coordinates, which is the
  desired preconditioning effect.
- It applies to both Minuit and SciPy through the shared parameter maps.

This fixed the representative width-scaling failures:

```text
K_L:  SciPy old 247.85 -> new 37.40
K+-:  SciPy old 56.58  -> new 53.47
```

## SciPy optimizer driver

The old unconstrained SciPy path was:

```text
scipy.optimize.minimize(..., method="Newton-CG", jac=True)
```

The new unconstrained SciPy path is:

```text
trust-ncg with exact JAX Hessian-vector products
```

The Hessian-vector product is computed with:

```python
jax.jvp(jax.grad(chi2), (x,), (p,))[1]
```

We first tried explicit full Hessians inside the trust-region iteration. That
was too expensive for the full suite, especially larger fits. Hessian-vector
products preserved the relevant second-order information while keeping runtime
reasonable. The full Hessian is still computed after convergence for the SciPy
covariance estimate, matching the old covariance approach:

```text
covariance = 2 * pinv(hessian)
```

The driver keeps the best finite result encountered from the single shared
start. If `trust-ncg` stalls, it runs a bounded Nelder-Mead escape and then
reruns `trust-ncg` from the escape point.

This is not meant to be a statement that this is the globally best SciPy
optimizer. It was the smallest robust change that addressed the observed
failure mode without changing the statistical objective.

## Result summary after implementation

Fresh full-suite logs:

```text
outputs/minuit_2026-06-23_after.out
outputs/scipy_2026-06-23_after.out
```

Final Minuit-vs-SciPy comparison:

```text
minuit fits=81  scipy fits=81  union=81

chi2 agree (rel<= 1e-3): 78/81
chi2 differ:              3
scipy success:False:      7
minuit invalid:           0
```

Old Minuit vs new Minuit:

```text
chi2 agree (rel<= 1e-3): 81/81
chi2 differ:              0
```

This is important: the chosen conditioning preserved the previous Minuit
solutions across the full suite while substantially improving SciPy.

Old SciPy vs new SciPy still differs in many places, but mostly because new
SciPy now reaches the Minuit/PDG basin:

```text
B0S-BR                 old 7.79e4  -> new 25.8
D_s                    old 3.34e3  -> new 12.1
B+J/psi                old 1.03e4  -> new 73.7
Lambda_c               old 6.12e3  -> new 58.8
B0                     old 4.26e3  -> new 82.9
Lam-b-0                old 632     -> new 14.3
eta                    old 558     -> new 46.4
Sigma+ decay param.    old 59.6    -> new 5.17
chi_c012 psi(2S)       old 4.49e3  -> new 393.9
D+-                    old 568     -> new 65.0
K_L                    old 248     -> new 37.4
eta'(958)              old 418     -> new 69.6
omega(782)             old 191     -> new 48.9
D0                     old 360     -> new 150
psi(3770)              old 46.8    -> new 20.1
rho_3(1690)            old 28.3    -> new 14.7
eta_c J/psi psi(2S)    old 302     -> new 186
K+-                    old 56.6    -> new 53.5
Upsilon(2S)            old 12.1    -> new 11.8
```

## Remaining discrepancies

The remaining Minuit/SciPy chi2 differences after the full rerun are:

```text
D0-topological          Minuit 1.43e-07   SciPy 1.24e-18
K_L eta+-,00 phase      Minuit 16.5       SciPy 2.72e+05
K_L eta+-,00 ph CPT     Minuit 20         SciPy 1.08e+10
```

`D0-topological` is effectively zero in both backends and should probably not
be treated as a substantive disagreement.

The two `K_L eta+-,00` phase fits remain genuine SciPy failures in this run.
They improved dramatically relative to the old `1e27` / `1e30` blowups, but
they are still far from Minuit. The earlier root-cause note suggested that
these may involve model blowup or a difficult lifetime/phase region. They
should be investigated separately.

Several SciPy fits still report `success=False` despite matching Minuit's chi2
exactly. In the final run, examples include:

```text
Upsilon(2S)
Omega mean lives
Lam-b-0
B0S-BR
eta_c J/psi psi(2S)
```

These are warnings about SciPy's local stopping criteria, not necessarily wrong
minima. For this project, chi2 agreement with Minuit/PDG and parameter-level
diagnostics are more informative than `result.success` alone.

## Targeted checks used while tuning

The most useful representative labels were:

```text
BRU / tiny-seed failures:
  B0S-BR
  Lambda_c
  D_s
  D+-

Large multi-particle BR/BRU sensitivity:
  chi_c012 psi(2S)

Softmax + lifetime-derived width scaling:
  K_L
  K+-

Softmax regressions to guard against:
  B0
  B+J/psi
```

When changing maps or optimizer settings, test both Minuit and SciPy on these
before launching the full suite. Some changes that looked good for SciPy alone
hurt Minuit, and some changes that fixed small BRU fits hurt the large
multi-particle fit.

## Verification commands used

All commands were run in the `pdg` conda environment. Full fit commands require
the PDG DB tunnel.

```bash
conda run -n pdg pytest tests/test_param_maps.py
conda run -n pdg python -m compileall src/pdgfits/fit.py src/pdgfits/param_maps.py tests/test_param_maps.py

conda run --no-capture-output -n pdg python -u -m pdgfits.run_fits --optimizer minuit > outputs/minuit_2026-06-23_after.out 2>&1
conda run --no-capture-output -n pdg python -u -m pdgfits.run_fits --optimizer scipy > outputs/scipy_2026-06-23_after.out 2>&1

conda run -n pdg python outputs/compare.py outputs/minuit_2026-06-23_after.out outputs/scipy_2026-06-23_after.out
conda run -n pdg python outputs/compare.py outputs/minuit_2026-06-22.out outputs/minuit_2026-06-23_after.out
conda run -n pdg python outputs/compare.py outputs/scipy_2026-06-22.out outputs/scipy_2026-06-23_after.out
```

One broader non-DB pytest subset still had unrelated failures in
`tests/test_build_chi2.py` because those tests call old `translate_dep` /
`adjust` keyword arguments that `build_chi2()` no longer accepts. That was not
part of this optimizer-conditioning change.

## Cautions for future changes

- Do not judge a candidate map only on SciPy. Check Minuit too.
- Do not judge a candidate map only on small BRU fits. Include
  `chi_c012 psi(2S)`.
- Avoid changing parameter-space seeds unless that is explicitly intended.
  Fitted-space clipping can change the implied physical starting fractions.
- Width scaling belongs in the parameter maps, not as a SciPy-only wrapper,
  because Minuit can benefit and because the returned fitted values/covariances
  should be in one consistent coordinate system.
- The scale constants are empirical engineering choices, not physics constants.
  If the DB changes or the fit suite expands, re-run the comparison.

## Follow-up: constrained SciPy direct-seed path

The original constrained SciPy path (`--optimizer scipy --fit_space constrained`)
used one bare `SLSQP` call from the PDG-derived constrained seed. A snapshot-backed
check on 2026-06-23 showed it was not reliable:

```text
minuit fits=81  scipy fits=81  union=81
chi2 agree (rel<= 1e-03): 50/81
chi2 differ:                31
scipy success:False:        15
```

The failures were concentrated in decay fits (`BR`/`BRU`). Several were silent
bad minima where SLSQP reported success.

A first experiment used the unconstrained sigmoid/softmax solution as a warm start
for the constrained solve. That showed the constrained objective could hold the
correct basin, but it was not a satisfying production path because it was no longer
really testing the constrained optimizer from the same fit seed.

The production constrained SciPy driver now starts from the same conditioned
PDG-derived constrained seed and tries both:

```text
SLSQP
trust-constr with JAX Hessian-vector products
```

It keeps the best finite result, preferring successful status when chi2 values are
effectively tied. This is enough to recover the constrained suite without using an
unconstrained warm start:

```text
PYTHONPATH=src PDGFITS_DATA_BACKEND=snapshot PDGFITS_SNAPSHOT_DIR=data/pdg-snapshot \
  conda run --no-capture-output -n pdg python -u -m pdgfits.run_fits \
  --optimizer scipy --fit_space constrained \
  > outputs/scipy_constrained_direct_2026-06-23.out 2>&1

PYTHONPATH=src conda run -n pdg python outputs/compare.py \
  outputs/minuit_2026-06-23_seedscale.out \
  outputs/scipy_constrained_direct_2026-06-23.out
```

Result:

```text
minuit fits=81  scipy fits=81  union=81
chi2 agree (rel<= 1e-03): 80/81
chi2 differ:                 1
scipy success:False:         0
minuit invalid:              0
```

The lone chi2 flag was again `D0-topological`, where both chi2 values are
effectively zero (`1.43e-07` vs `6.48e-11`).

## Follow-up: fixing the two K_L eta phase fits

The two remaining real SciPy failures from the first conditioning pass were:

```text
K_L eta+-,00 phase      Minuit 16.5   SciPy 2.72e+05
K_L eta+-,00 ph CPT     Minuit 20.0   SciPy 1.08e+10
```

The root cause was another seed-unit mismatch, not a new chi2-model problem.
Both fits use direct `S013D` measurements stored with `E10` in the measurement
strings, so parsed measurements are around `5.3e9` while the raw seed is
`0.53`. They also include the lifetime node `S012T`; preprocessing replaces it
with the width parameter `S012W = 1 / S012T`, but the raw lifetime seed
`0.89` produced a width seed near `1.12` instead of the parsed-measurement scale
near `1.12e10`.

The production fix is intentionally narrow:

1. After the existing branching-fraction seed normalization, `fit.py` calls
   `_condition_seed_units(...)`.
2. For directly measured non-decay parameters, it applies only obvious
   power-of-ten corrections: if the parsed direct measurement scale and seed
   differ by at least `1e4`, the seed keeps its mantissa and gets the nearest
   decimal scale.
3. For lifetime-derived width parameters, it uses the parsed lifetime
   measurement scale and seeds the width at `1 / median(abs(lifetime))` when
   the derived seed differs by at least `1e4`.
4. After unit conditioning, `_large_coordinate_scale_info(...)` linearly scales
   large non-decay optimizer coordinates through `build_param_map_scaled(...)`.
   This preserves the parameter-space chi2 and only improves optimizer
   conditioning.
5. The SciPy driver now tries a BFGS polish plus trust-region retry before the
   existing Nelder-Mead escape, and best-result bookkeeping prefers a successful
   result when chi2 values are effectively tied.

The simplified implementation deliberately avoids label-specific logic and
does not clip fitted coordinates. It corrects seeds into the same units as the
parsed measurements, then uses one generic linear preconditioner for large
non-decay coordinates.

Targeted regression labels used after this change:

```text
B0S-BR
Lambda_c
D_s
D+-
chi_c012 psi(2S)
K_L
K+-
B0
B+J/psi
K_L eta+-,00 phase
K_L eta+-,00 ph CPT
```

All targeted Minuit/SciPy chi2 values agreed within the existing `1e-3`
relative threshold. A full snapshot-backed rerun with the simplified code gave:

```text
minuit fits=81  scipy fits=81  union=81
chi2 agree (rel<= 1e-03): 80/81
chi2 differ:                 1
scipy success:False:         1
minuit invalid:              0
```

The lone chi2 flag was `D0-topological`, where both values are effectively zero
(`1.43e-07` vs `1.24e-18`). The lone SciPy status warning was
`eta_c J/psi psi(2S)`, with chi2 agreeing with Minuit. The two K_L eta phase
fits now converge under SciPy with successful status and chi2 matching Minuit.
