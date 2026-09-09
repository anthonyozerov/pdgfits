# minuit vs scipy optimizer comparison

**Date:** 2026-06-22
**Author:** Claude (Opus 4.8), for follow-up in a later session
**Status:** First-pass observational comparison. No code changed. Root causes are hypotheses, not confirmed.

## What I did

Ran the full fit suite both ways and compared per-fit results:

```bash
conda run -n pdg python -m pdgfits.run_fits --optimizer minuit > minuit.out 2>&1
conda run -n pdg python -m pdgfits.run_fits --optimizer scipy  > scipy.out  2>&1
```

- Default `--fit_space unconstrained` for both.
- Both runs completed cleanly (exit 0) and produced **81 fits** each (the suite skips `IGNORE` algorithm fits and explicitly skips `tauhflav`).
- **Raw logs are saved in `outputs/`** (committed alongside the repo, not ephemeral):
  - `outputs/minuit_2026-06-22.out`
  - `outputs/scipy_2026-06-22.out`
  - `outputs/compare.py` — the parser/comparison script used below. Run it with `conda run -n pdg python outputs/compare.py outputs/minuit_2026-06-22.out outputs/scipy_2026-06-22.out`.

  The comparison script extracts per-fit `chi2 obtained`, minuit `fit valid`/`hesse accurate`, and scipy `success`/`message` by splitting logs into blocks on the 100-`=` separator and regex-matching the printed lines. To regenerate fresh logs, re-run the two commands above.

Comparison metric: relative chi2 difference `|chi2_minuit - chi2_scipy| / max(|chi2_minuit|, |chi2_scipy|)`, flagged when `> 1e-3`.

## Optimizer configuration (from `src/pdgfits/fit.py`, ~lines 153-218)

The two backends are **not** configured symmetrically — keep this in mind when interpreting differences:

- **minuit** (`optimizer='minuit'`): iminuit with JAX gradient, `errordef=1`, `strategy=0`. Crucially it does a **robust alternation**: `migrad(); simplex(); migrad(); simplex(); migrad(); simplex(); migrad(); hesse()` — four migrad passes interleaved with simplex restarts. This multi-restart pattern is almost certainly why minuit escapes bad regions that trap scipy.
- **scipy** (`optimizer='scipy'`): a **single** `scipy.optimize.minimize` call. `method='Newton-CG'` in the unconstrained case (used here), `jac=True` via JAX value_and_grad, no bounds/constraints. `method='SLSQP'` only in the constrained path (not exercised in this run). Covariance from `2 * pinv(hessian)`.

So this comparison is really "4×(migrad+simplex) vs 1×Newton-CG", not a pure algorithm A/B. That asymmetry is a candidate explanation for most disagreements.

## Headline results

| | scipy chi2 matches minuit | scipy chi2 differs | total |
|---|---|---|---|
| scipy `success: True`  | 52 | 10 | 62 |
| scipy `success: False` |  6 | 13 | 19 |
| total | 58 | 23 | 81 |

- **58/81 agree** (rel diff ≤ 1e-3).
- **minuit reported all 81 fits valid** (`fit valid: True`, hessian accurate). minuit never matched-or-beat scipy's worst cases — in every disagreement, **scipy's chi2 is higher (worse minimum)** except `D0-topological` where both are ~0.
- scipy printed **19 `success: False`**. Two distinct failure messages:
  - `Warning: CG iterations didn't converge. The Hessian is not positive definite.`
  - `Warning: Desired error not necessarily achieved due to precision loss.`

## The three interesting categories

### 1. scipy `success: False` AND worse minimum (13 fits) — genuine failures
Some blow up by many orders of magnitude:

| fit | minuit chi2 | scipy chi2 | scipy message (abbrev) |
|---|---|---|---|
| K_L eta+-,00 phase | 16.5 | 1.1e27 | precision loss |
| K_L eta+-,00 ph CPT | 20 | 1.8e30 | precision loss |
| K_L eta+-,00 param. | 9.07 | 4.5e5 | CG / Hessian not PD |
| Lambda_c | 58.8 | 6120 | CG / Hessian not PD |
| D_s | 12.1 | 3340 | CG / Hessian not PD |
| chi_c012 psi(2S) | 394 | 4490 | CG / Hessian not PD |
| eta | 46.4 | 558 | CG / Hessian not PD |
| D+- | 65 | 568 | CG / Hessian not PD |
| omega(782) | 48.9 | 191 | CG / Hessian not PD |
| D0 | 150 | 360 | CG / Hessian not PD |
| psi(3770) | 20.1 | 46.8 | CG / Hessian not PD |
| rho_3(1690) | 14.7 | 28.3 | CG / Hessian not PD |
| Sigma+ decay param. | 5.17 | 59.6 | CG / Hessian not PD |

### 2. scipy `success: True` but DIFFERENT (worse) minimum (10 fits) — silent disagreements
These are the most concerning: scipy reports no problem yet is wrong.

| fit | minuit chi2 | scipy chi2 | rel diff |
|---|---|---|---|
| B0S-BR | 25.8 | 77900 | 1.0 |
| B+J/psi | 73.7 | 10300 | 0.99 |
| B0 | 82.9 | 4260 | 0.98 |
| Lam-b-0 | 14.3 | 632 | 0.98 |
| K_L | 37.4 | 248 | 0.85 |
| eta^'(958) | 69.6 | 418 | 0.83 |
| eta_c J/psi psi(2S) | 186 | 302 | 0.38 |
| K+- | 53.5 | 56.6 | 0.055 |
| Upsilon(2S) | 11.8 | 12.1 | 0.025 |
| D0-topological | 1.4e-7 | 5.0e-16 | (both ≈0, ignore) |

Note several of these are `BR`-type fits (B0, B+J/psi, B0S-BR, K_L, eta'(958)) — worth checking whether the softmax/sigmoid reparametrization interacts badly with Newton-CG.

### 3. scipy `success: False` but chi2 still correct (6 fits) — false alarms
scipy's warning fired but it landed on the right minimum: `K0l3 form factors`, `B masses`, `K+l3 lin form fact`, `Xibar+`, `eta_c(2S)`, `Xi_c`. (All rel diff = 0.) These are "Hessian not positive definite at the optimum" / "precision loss" warnings that don't reflect a wrong answer — possibly flat/degenerate directions or nuisance parameters.

## Working hypotheses (unconfirmed)

1. **Single Newton-CG vs multi-restart migrad/simplex** is the dominant factor. A fairer test: give scipy restarts, or try `method='trust-ncg'`/`'L-BFGS-B'`/`'trust-constr'`.
2. **Reparametrization sensitivity.** Several disagreements are BR fits using softmax/sigmoid maps (`param_maps.py`). Newton-CG may be sensitive to the curvature these introduce.
3. **"Hessian not positive definite"** suggests scipy stalls in saddle/indefinite regions that simplex restarts walk minuit out of.
4. The huge blow-ups (1e27–1e30 on the K_L eta phase fits) smell like scipy stepping into a region where the chi2 model diverges (e.g. a `1/width` lifetime relation, or a division equation type) — check those fits' equation types.

## Suggested further investigation (for the next session)

- The logs from this run are in `outputs/minuit_2026-06-22.out` / `outputs/scipy_2026-06-22.out` (re-run the two commands to generate a fresh pair if the DB has changed).
- Pick 2-3 representative failures across categories — e.g. `B0S-BR` (cat 2, BR), `Lambda_c` (cat 1, CG fail), `K_L eta+-,00 phase` (cat 1, blow-up) — and run each single fit with `--fit_label`, instrument `fit.py` to dump scipy's `result` (nit, status, x) and compare the converged parameter vectors against minuit's.
- Try alternate scipy methods / add restarts in the scipy branch and see how many of the 23 close.
- Check whether the category-2 silent failures correlate with `algorithm in {BR, BRU}` and which `fit_space`.
- Consider whether `run_fits.py` should surface a warning when scipy `success: False` OR when the obtained chi2 exceeds the PDG chi2 by some margin (currently it just prints both).

## Reproduction notes

- Must use the **`pdg` conda env** (`conda run -n pdg ...`); base python can't import `pdgfits`.
- Requires the SSH tunnel to `127.0.0.1:5433` to be up (it was for this run).
- `run_fits.py` prints `chi2 obtained` and `chi2 obtained by PDG` per fit; minuit also prints `fit valid` / `errors from hessian accurate`; scipy prints `scipy success` / `message`.
