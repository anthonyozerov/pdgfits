# Minuit/MINOS notes for pdgfits asymmetric-error redesign

Purpose: give future supervised Codex rounds a concrete HEP-standard target to compare against, instead of continuing only with micro-optimizations or wrapper cleanup.

## Sources read

Primary/reference sources:

- ROOT Minuit2 guide: https://root.cern.ch/root/htmldoc/guides/minuit2/Minuit2.html
- ROOT `MnMinos` class docs: https://root.cern.ch/doc/v640/classROOT_1_1Minuit2_1_1MnMinos.html
- ROOT `MnMinos.cxx` source docs: https://root.cern.ch/doc/v640/MnMinos_8cxx_source.html
- ROOT `MnFunctionCross.cxx` source docs: https://root.cern/doc/master/MnFunctionCross_8cxx_source.html
- iminuit HESSE/MINOS notebook: https://scikit-hep.org/iminuit/notebooks/hesse_and_minos.html
- iminuit reference: https://iminuit.readthedocs.io/en/latest/reference.html
- iminuit FAQ: https://scikit-hep.org/iminuit/faq.html
- Venzon & Moolgavkar (1988), “A Method for Computing Profile-Likelihood-Based Confidence Intervals”: https://ideas.repec.org/a/bla/jorssc/v37y1988i1p87-94.html
- Fischer & Lewis (2021), “A robust and efficient algorithm to find profile likelihood confidence intervals”: https://link.springer.com/article/10.1007/s11222-021-10012-y

## What Minuit is doing

Minuit separates three ideas that are tangled in many homegrown fitters:

1. **Minimum finding**: `MIGRAD` (`MnMigrad`) is the main local minimizer, a variable-metric/quasi-Newton method. Minuit stops on an estimated-distance-to-minimum criterion (EDM), not on raw parameter or function-value deltas. The ROOT guide says `MnMigrad::operator()` stops when EDM is below `0.001 * tolerance * up`.
2. **Parabolic covariance/errors**: `HESSE` computes the second-derivative matrix at the minimum and inverts it. The Minuit2 guide describes the error matrix as twice the inverse Hessian, transformed to external coordinates if needed, multiplied by `FCNBase::up()`. This includes parameter correlations but assumes local quadratic/parabolic behavior, so it cannot represent nonlinear asymmetric errors.
3. **Profile-likelihood asymmetric errors**: `MINOS` computes one-parameter profile-likelihood intervals. It fixes/scans one parameter, re-minimizes all the others, and finds where the profiled FCN rises by `up` from the global minimum.

For least-squares / chi-squared fits, `up = 1` gives the one-sigma Wilks/profile endpoint. For negative log-likelihood fits, `up = 0.5`. This exactly matches the pdgfits asymmetric-error invariant in chi-squared form:

```text
profile_chi2(endpoint) = chi2_min + 1
```

## MINOS definition

The Minuit2 guide defines the MINOS error for parameter `n` as the change in that parameter that causes `F'` to increase by `FCNBase::up()`, where `F'` is the minimum of the FCN with respect to all other free parameters. The algorithm varies parameter `n`, repeatedly minimizes over the other `npar - 1` parameters, and finds the two values where the profiled FCN equals `Fmin + up`.

iminuit’s explanation is the same in likelihood language: MINOS scans the likelihood along one parameter `theta_i`, while minimizing with respect to all other parameters `theta_k`, producing the profile likelihood. The interval endpoints are where the profile rises by `errordef` above the minimum.

Important caveat: iminuit explicitly warns that MINOS intervals are not “exact” finite-sample intervals. They have the advertised coverage only asymptotically via Wilks/profile-likelihood theory. That matters for pdgfits: MINOS can inspire the numerical method, but it does not magically solve the statistical modeling/coverage question.

## What MINOS uses computationally

`MnMinos` requires a valid minimum first. In normal Minuit workflow, that means `MIGRAD` and an error matrix/HESSE-like covariance are already available.

From the ROOT `MnMinos.cxx` and `MnFunctionCross.cxx` docs:

- `MnMinos::FindCrossValue(direction, par, ...)` handles one side: `direction=+1` upper, `direction=-1` lower.
- If `maxcalls == 0`, Minuit sets a large default based on the number of variable parameters:
  `2 * (nvar + 1) * (200 + 100*nvar + 5*nvar*nvar)`.
- Initial target-parameter step is roughly one parabolic error: `value(par) + direction * error(par)`, clamped to parameter limits.
- Initial guesses for correlated parameters are predicted from the error matrix/covariance, so MINOS starts each conditional minimization near the expected sub-minimum along the error ellipse.
- `MnFunctionCross` then searches for the step `a` where the profiled function equals `Fmin + up`:
  - fix selected parameter(s) at `pmid + a * pdir`;
  - run `MIGRAD` over all remaining free parameters;
  - check for new lower minimum, call-limit hit, invalid minimum, or parameter limit;
  - use function values and parabolic/linear interpolation to update `a`;
  - converge when function value and step changes meet tolerance.
- It returns a structured `MnCross` status: valid crossing, new minimum, parameter limit, call limit, invalid/failure.

This is the most relevant implementation lesson for pdgfits: MINOS is not “just bisection plus minimization.” It uses the covariance/error matrix to predict conditional sub-minima, and it treats crossing status as first-class evidence.

## Parameter transformations and limits

Minuit distinguishes external user parameters from internal variable parameters. Box limits are transformed away internally:

- two-sided limits use an arcsin transform;
- one-sided limits use square-root style transforms.

The Minuit2 guide warns that nonlinear limit transformations can create numerical issues and recommends avoiding limits when possible, especially for final error analysis. iminuit’s FAQ similarly says MINUIT only handles independent box constraints; dependent constraints require transformations, another minimizer, or barrier/interior methods, and hard discontinuous penalties break MIGRAD.

This maps directly onto pdgfits risks:

- pdgfits uses fitted-coordinate charts / parameter maps (`arctan`, softmax/simplex-like maps, BR/BRU transformations) to enforce physical constraints.
- Those charts are useful, but stationarity and covariance in chart coordinates may be misleading near saturation or boundaries.
- A MINOS-inspired comparison should therefore record whether discrepancies appear near limits/chart saturation, not just endpoint residuals.

## How pdgfits currently compares to MINOS

Similarities:

- Both target profile-likelihood/Wilks endpoints (`chi2_min + 1`).
- Both re-minimize nuisance/free coordinates at fixed target values.
- Both rely on a local minimizer and local certificates rather than a global proof.
- Both need explicit failure/status reporting when a profile solve finds a new minimum, hits a boundary/call limit, or cannot certify the crossing.

Differences:

- MINOS traditionally profiles one *parameter coordinate* at a time. pdgfits often profiles an arbitrary scalar target function of physical parameters, e.g. relationship-derived observables or nuisance-like targets. That is a harder constraint: `target_func(params(fp)) = value`.
- MINOS uses covariance/error-matrix predictions for conditional sub-minima. pdgfits currently uses continuation/MLE-projected starts, SLSQP/trust-constr, KKT polish, and descent checks; covariance-predicted nuisance starts are not the core organizing principle.
- MINOS’ `MnFunctionCross` combines root/crossing search with repeated conditional `MIGRAD` minimizations. pdgfits currently exposes a clearer `ProfileRoot`/bisection interface after round 5, which is more conservative but may do more profile calls.
- Minuit handles box limits via internal coordinate transforms. pdgfits handles physical constraints via custom parameter maps. These are not interchangeable.

## What not to conclude

- Do **not** conclude “just use Minuit/MINOS everywhere.” pdgfits has arbitrary scalar targets, JAX-derived chi2, custom physical maps, and PDG-specific interpolation/error behavior. A blind replacement would be dumb.
- Do **not** treat MINOS as a global optimizer. It is still local and can fail; ROOT has explicit statuses for new minima, call limits, limits, and invalid crossings.
- Do **not** treat MINOS intervals as exact finite-sample uncertainty statements. iminuit says that exactness claim is wrong; the coverage is asymptotic.
- Do **not** replace bracketed residual verification with unbracketed Newton/secant steps unless hard-case evidence proves it no less robust.

## Promising pdgfits follow-up experiments

### 1. iminuit/MINOS independent comparator where applicable

Use `iminuit` as an independent HEP-standard comparator on cases where the target is a direct parameter coordinate or can be represented as such without changing the model.

Suggested Codex task:

- Check whether `iminuit` is available in the `pdg` environment; if not, do not install globally without approval, but write the exact isolated install/venv command or use an existing dependency if present.
- Build a notes-only comparator for a small set:
  - one simple average target where the primary value is a direct coordinate;
  - one low-dimensional fit target that is close to a fitted coordinate;
  - one hard B0/B0S/eta target only if it can be represented without abusing the model.
- Run `MIGRAD`, `HESSE`, and `MINOS` with `errordef=1` for chi2.
- Compare Minos endpoints to pdgfits endpoints and fresh profiled chi2 residuals.
- Record HESSE-vs-MINOS disagreement as a nonlinearity diagnostic.

This would be the cleanest “Are we reinventing MINOS badly?” test.

### 2. MINOS-style covariance-predicted starts inside current pdgfits profiles

Even if we do not use Minuit, we can test one MINOS design idea: using the covariance/Hessian to predict conditional sub-minima when a target is displaced.

Suggested Codex task:

- Derive a local linearized predictor for nuisance/free fitted coordinates at fixed scalar target value.
- Compare current start strategy vs. covariance-predicted starts on hard endpoints.
- Metrics: profile calls, function evaluations, success/certificate methods, endpoint residuals, and any lower fixed-target chi2 found.
- Keep it notes-only or optional until a hard-case win appears.

This is likely the highest-value algorithmic borrowing from MINOS.

### 3. VM/RVM-style endpoint equation solver spike

Venzon & Moolgavkar frame endpoint finding as solving a system:

1. target/profile level condition (`chi2 = chi2_min + 1`), and
2. nuisance stationarity at the constrained optimum.

Fischer & Lewis (2021) describe a robust VM variant that adds trust-region checks because plain VM/Newton can fail on nonconvex, discontinuous, or unidentifiable likelihoods.

Suggested Codex task:

- For a low-dimensional smooth hard case, prototype a notes-only equation/trust-region solver using JAX gradient/Hessian information.
- Compare against current constrained-profile + bisection endpoints.
- Do not ship unless it is simpler or catches real failures.

This is a bigger mathematical rethink than another optimizer fallback, but also riskier.

### 4. Profile-curve diagnostics copied from iminuit practice

iminuit encourages plotting/checking the likelihood profile if minimization/error reliability is uncertain. pdgfits round 4 started this.

Suggested Codex task:

- Generalize profile-curve scans to all descent-check endpoints or a stratified hard set.
- Store method/certificate changes along the profile curve.
- Add HESSE/parabolic prediction overlays where covariance exists.
- Use this to choose between HESSE-like, MINOS-like, and current profile endpoints.

## Recommended immediate supervisor direction

After the current round finishes, the next supervised Codex prompt should explicitly read this note and prioritize a **MINOS comparator / MINOS-style start-prediction spike**, not more interface cleanup.

Best next prompt shape:

1. Read this note and the round 5/6 reports.
2. Decide whether `iminuit` is available and suitable for a notes-only comparator.
3. If yes, implement the smallest honest `iminuit`/MINOS comparison on direct-coordinate cases and one hard case if representable.
4. If no, implement the MINOS-style covariance-predicted start diagnostic inside the existing pdgfits profile machinery.
5. Quantitatively compare endpoints, residuals, HESSE/parabolic estimates, MINOS/profile estimates, method/certificate status, and function/profile-call counts.
6. Reject the idea explicitly if it cannot fairly represent pdgfits targets.

The key question for Codex should be: **which parts of MINOS are methodological improvements we should borrow, and which parts are incompatible with pdgfits’ arbitrary-target / mapped-parameter setting?**
