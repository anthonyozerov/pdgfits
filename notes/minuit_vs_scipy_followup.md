# minuit vs scipy — root-cause follow-up

**Date:** 2026-06-22
**Author:** Claude (Opus 4.8)
**Status:** Root cause identified and demonstrated experimentally. No production code changed.
Builds on `minuit_vs_scipy_comparison.md` (the first-pass observational report).
**Experiment scripts:** were run from a scratchpad (not committed); the four experiments
below are simple to reconstruct from this description. All used the `pdg` conda env with the
SSH tunnel up, calling only `run_fit()` (never touching the DB/tunnel directly).

## TL;DR

The 23 scipy disagreements are **not** a chi2-model bug and **not** a wrong-minimum problem.
They are a **starting-point conditioning problem created by the unbounded sigmoid/softmax
reparametrization** (`param_maps.py`). Tiny branching-fraction seeds get mapped into the
saturated tail of the reparametrization, where gradients vanish and the Hessian is indefinite,
and a single Newton-CG pass is trapped *at the seed*. Minuit survives only because its
simplex restarts crawl out of that region.

This answers the framing question ("more robust model or more robust optimizer?"): the
**optimum is already correct and well-conditioned** — the fix belongs at the **seed /
reparametrization**, with a trust-region + restart driver as the optimizer-side complement.

## Experiment A — does scipy hold minuit's minimum?

For each failing label, evaluate the (jitted) chi2 at minuit's converged point and restart
scipy *from* that point.

**Result (every label tested):** scipy started at minuit's solution stays put.
`Newton-CG` and `L-BFGS-B` both hold it with `|move| ~ 1e-6…1e-10` and chi2 unchanged.

| label | minuit chi2 | scipy-from-seed | scipy-from-minuit (Newton-CG) |
|---|---|---|---|
| B0S-BR | 25.79 | 77856 | **25.79** (`|move|`=1e-10) |
| Lambda_c | 58.76 | 6123 | **58.76** |
| omega(782) | 48.90 | 190.6 | **48.90** |
| K_L eta+-,00 phase | 16.46 | 1.14e27 | **16.46** |
| Upsilon(2S) | 11.78 | 12.10 | **11.78** |

⇒ The chi2 model and its minima are fine. The failure is entirely about *reaching* the basin
from the seed, not the basin itself. (This also rules out the original report's hypotheses 3–4
as *optimum* problems — the "Hessian not positive definite" warnings are about the seed region,
not the solution.)

## Experiment B — the seed itself is catastrophic, and Newton-CG can't move from it

Captured the actual seed `x0` (fitted space) by spying on the `scipy_minimize` call in
`fit.py`, then measured chi2 there and replayed several strategies.

- **chi2 at the seed** is enormous: B0S-BR 77857, Lambda_c 6123, omega 6666,
  K_L eta phase **1.1e27** (vs minima ~26, ~59, ~49, ~16).
- **Plain Newton-CG barely moves**: B0S-BR 77857→77856, Lambda_c 6123→6123. The CG inner
  solve on an indefinite/ill-conditioned Hessian yields a ≈0 step, so it reports "success"
  sitting essentially at the start.
- No single off-the-shelf method fixed all cases from the raw seed.

## Experiment C — why the seed is catastrophic (the mechanism)

Inspected the seed in fitted space vs proportion space and the gradient there.

| label | algo | param-space seed min | **max \|x0\| (fitted)** | \|grad(seed)\| |
|---|---|---|---|---|
| B0S-BR | BRU (sigmoid) | 1e-5 | **15,900** | 3.5e6 |
| Lambda_c | BRU (sigmoid) | 5.8e-6 | **3,180** | 1.3e5 |
| omega(782) | BR (softmax) | 1.0e-5 | 11.5 | 3.5e3 |

The sigmoid map is `param = 0.5 + atan(x)/π`, inverse `x = tan(π·(param − 0.5))`. A branching
fraction of `1e-5` inverts to `x = tan(π·(1e-5 − 0.5)) ≈ −1/(π·1e-5) ≈ −3.2e4`. So a *physically
ordinary* tiny branching fraction places the optimizer **deep in the flat saturated tail** of
the reparametrization, where `dparam/dx ∝ 1/(1+x²) → 0`. Gradient/Newton methods are dead on
arrival there; simplex (minuit) is not.

The softmax (single-particle BR) version is far milder (`|x0|` ~ 10, not ~10⁴) because the
sum-to-1 normalization keeps coordinates bounded — consistent with the softmax fits being the
*least* severe failures in the first-pass report.

## Experiment D/E — a fix that works

Two cheap, principled levers, validated across all 22 reproducible disagreements:

1. **Clip the fitted-space seed out of saturation** (`x0 = clip(x0, −10, 10)`). A proportion of
   `0.5 + atan(−10)/π ≈ 0.03` is a perfectly good *starting* point for a fraction the optimizer
   will drive down — and now the gradient there is alive.
2. **Use a trust-region method with the explicit JAX Hessian** (`method='trust-ncg'`,
   `hess=jax.hessian(chi2)`) instead of bare `Newton-CG`. The trust region handles the
   indefinite curvature that the plain CG line-search cannot. (Note: clipping the seed but
   keeping Newton-CG sometimes makes things *worse* — the Hessian/trust-region is the load-bearing
   part.)
3. A **Nelder-Mead escape + re-solve** rescues the few genuine model-blowup cases (K_L eta phase).

Driver = `clip → trust-ncg → (Nelder-Mead → trust-ncg) escape if improving`, guarded against
non-finite Hessians:

**Recovered 20/22 of the disagreements to minuit's chi2 exactly**, including:
- all four BRU/sigmoid blowups (B0S-BR, Lambda_c, D_s, D+-, D0)
- the softmax/BR cases (omega, eta, eta'(958), Sigma+, chi_c012, …)
- the spectacular K_L eta+-,00 phase/CPT/param 1e27–1e30 blowups (→ 16.46 / 19.99 / 9.07)

The 2 that still resist (**K_L** and **K+-**, both softmax BR) actually got *worse* than bare
scipy because the Nelder-Mead escape overshot — the driver needs to also keep the plain
clip+trust-ncg result and return the best of all attempts. Minor bookkeeping, not a new mechanism.

## Experiment F — the two remaining softmax BR fits are width-scaling failures

Follow-up diagnostic on the two fits above showed that the "minor bookkeeping" hypothesis was
wrong/incomplete. **K_L** and **K+-** are not primarily stuck because of the softmax branching
fractions. Their branching fractions are already very close to the Minuit values. The chi2 gap is
mostly carried by the lifetime-derived width parameter:

- `K+-`: SciPy chi2 56.58 vs Minuit 53.47. Replacing only `S010W` in the SciPy result with the
  Minuit value drops chi2 to 53.53.
- `K_L`: SciPy chi2 247.85 vs Minuit 37.40. Replacing only `S013W` in the SciPy result with the
  Minuit value drops chi2 to 44.36; replacing all non-decay parameters gives 44.38.
- Replacing only the nuisance parameter in `K_L` (`nuisance_S009K3P`) leaves chi2 essentially
  unchanged at 247.88, so the br-adjust nuisance is not the failure mechanism.

Both fits have lifetime nodes converted during preprocessing to `lifetime = 1 / width`
(`S010T`, `S013T`). The raw width coordinates are large:

- `S010W ≈ 8.1e7`
- `S013W ≈ 1.9e7`

In raw coordinates, the lifetime derivative is tiny:

```text
d(1 / W) / dW = -1 / W^2
```

So the objective can be meaningfully sensitive to lifetime while appearing nearly flat in the raw
`W` coordinate. A physically important width move is O(1e5-1e6), while the other fitted
coordinates (softmax logits) are O(1-10). Plain SciPy `Newton-CG` is coordinate-scale sensitive
enough to declare success while leaving `W` essentially at the seed.

An ablation separated scaling from the explicit-Hessian/trust-region changes. **Width scaling
alone fixes both fits under production-style `Newton-CG`**:

| fit | start | raw Newton-CG | linear-width Newton-CG | log-width Newton-CG | Minuit |
|---|---:|---:|---:|---:|---:|
| `K+-` | seed | 56.5816259723 | 53.4749782752 | 53.4749782543 | 53.4749686136 |
| `K+-` | bad SciPy point | 56.5816251758 | 53.4749756466 | 53.4749756156 | 53.4749686136 |
| `K_L` | seed | 247.853481408 | 37.4027492762 | 37.4028336519 | 37.4027489274 |
| `K_L` | bad SciPy point | 247.853470489 | 37.4027501762 | 37.4027493566 | 37.4027489274 |

The chain-rule reason this works is simple. If `W = scale * z`, then

```text
dχ²/dz  = scale * dχ²/dW
d²χ²/dz² = scale² * d²χ²/dW²
```

The optimum is unchanged, but the width direction is no longer numerically microscopic to the
optimizer. A `log(width)` coordinate works for the same reason and also enforces positivity.

Refined diagnosis:

1. Most disagreements are caused by saturated sigmoid/softmax seeds for tiny branching fractions.
2. The two remaining softmax BR disagreements (`K+-`, `K_L`) are primarily raw-coordinate
   scaling failures in lifetime-derived width parameters.
3. The softmax gauge redundancy is real, but fixing one softmax logit alone did not solve these
   two cases.

## Recommended changes (not yet applied)

In rough priority order:

1. **Clip the fitted-space seed out of saturation before optimizing**, for both the sigmoid and
   softmax paths (right after `fitted_param_init = params_to_fitted_params(param_init)` in
   `fit.py`). This is the root-cause fix and is optimizer-agnostic. Consider whether the
   reparametrization itself should be re-scaled (e.g. a gentler map, or seeding decay params at a
   floor like 1e-3 rather than 1e-5) so the saturated region is never the start.
2. **Scale lifetime-derived width coordinates before passing them to scipy** (or optimize them as
   `log(width)`). This is the specific fix for `K+-` and `K_L`, and probably a good general
   preconditioning rule for any parameter that enters a `1 / width` relation.
3. **Replace the scipy `Newton-CG` default with a trust-region + explicit Hessian** (`trust-ncg`
   or `trust-constr`), and add a small restart loop (mirroring minuit's migrad/simplex
   alternation) with best-of-attempts bookkeeping. JAX already gives us the exact Hessian cheaply.
4. **Surface a warning in `run_fits.py`** when scipy `success=False` *or* when the obtained chi2
   exceeds minuit's / the PDG chi2 by more than a small margin — the "silent" cat-2 failures
   (`success=True` but wrong) are the dangerous ones.
5. The **K_L eta+-,00 phase 1e27 blowup** confirms the first-pass hypothesis that the model
   diverges in some region (the lifetime `1/width` relation, `S012T`). Worth bounding or
   reformulating that equation type so the objective can't reach 1e27 in the first place.

## Caveats

- The clip bound (`±10`) and the escape schedule are tuned-by-hand defaults from this run, not
  derived constants — revisit if the reparametrization changes.
- The width-scaling numbers above are diagnostic, not production-code changes. The temporary
  scripts used for them lived in `/tmp` and were not committed.
- All numbers are from the 2026-06-22 DB state; re-run if the DB changes.
- minuit remains the safer production default today; these changes are about making the scipy
  path (and, via lever 1, the model) robust enough to be a trustworthy cross-check.
