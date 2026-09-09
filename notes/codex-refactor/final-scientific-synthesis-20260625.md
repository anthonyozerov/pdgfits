# Final scientific synthesis: pdgfits asymmetric-error / fitting refactor

Date: 2026-06-25  
Branch: `codex/asym-refactor-simplify-20260624-221852`  
Final completed round: Round 21, commit `6fce292` (`Add round21 guarded refit comparator`)

## Executive conclusion

The overnight work gives fairly strong evidence that the current scientific method for asymmetric errors is basically the right one: profile-likelihood/Wilks endpoints, found by bracketed search and accepted only after verifying the profiled chi-squared equation `chi2 = chi2_min + 1`.

The investigations did **not** find a substantive endpoint/profile-root correctness failure. They also did not justify changing `build_chi2.py`, replacing the endpoint search with MINOS/VM/Newton-style endpoint equations, or globally changing BR/BRU charts. The right scientific conclusion is boring but valuable: the profile-endpoint machinery appears valid for the tested PDG problem classes, and the main improvements should be about diagnostics, certificates, and modest robustness/performance — not mathematical replacement.

The one real issue found late in the loop is different: in synthetic, severely chart-saturated `B0::S042B95` toys, the **unconstrained baseline refit** can miss slightly lower local minima even when Minuit reports a valid/accurate fit. The largest Round 21 lowering was about `0.035` in chi-squared; `12/1000` B0 toys lowered by more than `0.01`. Given Anthony's point that `~0.01` chi-squared differences are not scientifically important by themselves, this is best interpreted as a numerical hygiene/globality warning for saturated BR/BRU fits, **not** as evidence that the reported asymmetric-error method is scientifically invalid. Actual snapshot `B0::S042B95` was stable under the same checks, and calibration threshold conclusions did not change.

## What changed in the code

The actual production-ish changes are limited and mostly sensible:

1. `src/pdgfits/asym_errors.py`
   - streamlined profile solving;
   - exposed profile-root / profile-search diagnostics;
   - kept the core invariant: accept endpoints only after profiled chi-squared residual verification.

2. `src/pdgfits/avg.py`
   - removed redundant average-side profile solves;
   - preserved average-side profiling semantics.

3. `tests/test_asym_errors.py` and `src/pdgfits/CLAUDE.md`
   - added/updated behavior-preserving diagnostic expectations and method notes.

Everything after the main refactor/perf commits was mostly notes/harness work under `notes/codex-refactor/`. `src/pdgfits/build_chi2.py` was not edited in the method-rethink rounds.

The most important code-level lesson is negative: do not complicate the production solver unless there is a hard scientific reason. The evidence supports keeping the bracketed/profile-verified endpoint approach, with diagnostics and possibly one narrow fit-side guard.

## Evidence about the asymmetric-error method

The core scientific object is a profile-likelihood endpoint: for a target scalar, hold the target at a candidate value, profile over nuisance/fitted parameters, and find where the profiled chi-squared rises by 1. The validity requirement is not "optimizer says success"; it is that the endpoint verifies the profile equation and remains feasible/stationary enough to trust.

Across the rounds:

- descent/multistart checks found no materially lower constrained-profile optima;
- profile-curve scans did not reveal nonmonotone or disconnected endpoint pathologies;
- structured globality searches found no meaningful lower constrained profiles;
- broad calibration endpoint subsets verified endpoint residuals within the existing tolerance;
- Round 18 had `100` endpoint-side rows, `0` endpoint-root failures, and max endpoint residual `0.0049933978` within the `5e-3` tolerance.

So the endpoint/profile-root mechanism held up. This matters more than tiny chi-squared differences in unconstrained toy refits: the scientific asymmetric-error claim is about the profiled `chi2_min + 1` crossing, and those crossings were not shown to be wrong.

## What MINOS taught us

MINOS was useful as a conceptual and diagnostic comparator, not as a drop-in replacement.

For direct-coordinate cases where the comparison is fair, iminuit/MINOS agreed closely with pdgfits endpoints; the largest reported shift was about `2e-4`, below the current endpoint residual scale. That supports the pdgfits profile-likelihood interpretation.

But MINOS is not a general oracle for pdgfits targets:

- many pdgfits endpoints are scalar functions or chart-mapped quantities, not direct Minuit coordinates;
- BR/BRU constraints and internal coordinate maps create conditioning issues MINOS does not magically solve;
- wholesale MINOS replacement would not address arbitrary target profiling cleanly.

The useful borrowed ideas are: covariance/HESSE-based starts as diagnostics, structured crossing statuses, and explicit risk labels. The experiments rejected universal covariance starts: on chart-saturated B0 they were expensive and not generally better.

## Rejected method changes

The loop tried several plausible "big rethink" directions. Most were useful precisely because they failed cleanly.

- **HESSE/parabolic endpoints:** good for risk stratification, not a replacement for profile endpoints. Nonlinearity/chart saturation is real.
- **Universal MINOS-style covariance starts:** rejected; sometimes more expensive, especially for saturated B0.
- **BR/BRU chart replacement:** blanket chart replacement rejected. Direct physical charts helped one well-conditioned certificate case but did not generalize and were worse/slower for saturated B0.
- **Venzon--Moolgavkar / endpoint-equation methods:** converged on small/direct cases and can polish diagnostics, but unbracketed endpoint equations are weaker than verified bisection/profile search as production logic.
- **Globality/disconnected-component fears in endpoint profiles:** tested but not supported by evidence.

This is a good outcome. The space of tempting clever solver changes is large; the evidence says most of them are not worth the complexity.

## Statistical calibration / Wilks regularity

The calibration rounds shifted the live scientific question from "are endpoint roots numerically wrong?" to "are the profile-likelihood q statistics well calibrated in delicate PDG regimes?"

Round 18 ran `5000` toys over five cases and two DGPs:

- `B0::S042B95`: boundary/tiny-parameter and always chart-saturated;
- `B0S-BR::S086.37`: asymmetric/profile-shape;
- `eta_c J/psi psi(2S)::M026G01`: asymmetric/profile-shape;
- `G(2000),G(1800)::K002M/K003M`: clean controls.

Main calibration conclusion:

- B0S's earlier high-q concern did **not** survive as a robust DGP-invariant tail. Under local-Gaussian toys there was a mild center/CDF shift; under split-normal it mostly disappeared.
- Eta remained profile-shape volatile, with some outliers, but tail counts were near nominal.
- B0 stayed chart/boundary saturated in all toys, but aggregate q calibration was close to chi-square-1 expectations.
- Clean controls behaved plausibly at campaign scale.

Thus there is no evidence here for a broad scientific failure of Wilks/profile-likelihood asymmetric errors. There are regularity caveats for boundary/tiny-parameter and asymmetric-profile cases, but they are caveats to document and stratify, not an argument to replace the method.

## The late B0 refit/globality issue

Rounds 19--21 followed up one Round 18 negative-q row. The fixed-truth profile itself was stable; the issue was that the reported unconstrained toy refit was not quite the lowest point found by restarts.

Round 21 compared guarded refit policies over all `1000` B0 calibration toys plus controls:

- `22/1000` B0 toys lowered by more than `1e-4` under the full guarded saturated-chart policy;
- `20/1000` lowered by more than `1e-3`;
- `12/1000` lowered by more than `1e-2`;
- max lowering: `0.0347463256`;
- all material rows were in extreme saturated chart diagnostics;
- actual snapshot `B0::S042B95` fired the guard but improved only `7.5e-09`;
- clean controls had no material lowerings;
- B0S/eta representative rows did not fire the saturation guard, and their returned-MLE lowerings were borderline (`~1e-4`, below `1e-3`);
- calibration threshold counts did not materially change.

My interpretation: this is a real numerical robustness issue in synthetic saturated BR/BRU B0-like fits, but not a scientific crisis. Since the affected chi-squared differences are tiny relative to scientific interpretability, and the actual snapshot is stable, I would not let this distract from the main conclusion. If production robustness is desired, implement a narrow guarded refit policy; if scientific reporting is the priority, document it as a caveat.

## Recommended next steps

1. **Do not change `build_chi2.py`.** Nothing in the evidence points to the chi-squared construction as the problem.

2. **Keep the bracketed, profile-verified asymmetric-error endpoint method.** It survived the strongest tests run overnight.

3. **Do not replace the method with MINOS, HESSE endpoints, alternate BR/BRU charts, or VM endpoint equations.** Use their ideas for diagnostics, not as the core solver.

4. **Optional production patch:** if you want stronger numerical hygiene for saturated BR/BRU fits, implement a narrow `fit.run_fit()` guard triggered only by extreme chart saturation, with deterministic decay-floor/tiny-parameter starts and explicit diagnostics. Treat it as robustness instrumentation, not a correction to the scientific endpoint method. It should be tested on actual snapshots, B0 toys, B0S/eta profile-shape controls, clean controls, and runtime.

5. **For scientific writeup/use:** state that the profile-likelihood asymmetric-error method appears valid on tested cases, with documented regularity risks near boundaries and in nonlinear asymmetric profiles. The largest remaining uncertainty is not endpoint solving but statistical calibration under realistic PDG asymmetric-error DGPs.

## One-sentence version

The overnight loop mostly failed to find a reason to rethink the asymmetric-error mathematics — which is good news: profile-verified `chi2_min + 1` endpoints look scientifically sound here; the only real finding is a narrow, low-stakes unconstrained-refit stability issue in synthetic saturated B0/BR-BRU toys.
