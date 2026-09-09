# Asymmetric Error Algorithm and Evidence Report

Generated: 2026-06-24 14:06 PDT

Branch context: `codex-cloud-asym-avg-local-experiments` through commit `ccb5f65`.

This report summarizes the current asymmetric-error algorithm for averages and fits, why it is better than the previous version, what evidence we have that the returned endpoints correspond to Wilks/profile-likelihood endpoints, and what evidence or method improvements would be worth pursuing next.

## Executive summary

The current asymmetric-error calculation is a profile-likelihood procedure. For a scalar reported target value `t`, it tries to find the lower and upper values `t_-`, `t_+` such that

```text
chi2_profile(t_±) = chi2_min + 1,
```

where `chi2_profile(t)` means chi2 minimized over all other fitted/nuisance degrees of freedom while holding the target fixed.

Current evidence is strong for the snapshot cases tested, but it is empirical/numerical evidence, not a theorem:

- Averages: full snapshot sweep passed `2647/2647` average nodes and verified `5294/5294` endpoints. Max fresh endpoint residual was `0.0049997`, i.e. at the chosen binary-search tolerance.
- Fits: staged broad snapshot sweep passed `227/227` target rows and verified `454/454` endpoints. This is the same target key set as the previous broad sweep, where only `214/227` rows passed and `13` failed.
- All previous 13 fit-side failures are now `ok`, with no regressions among the previous 214 successful rows.
- The current fit-side solver is simpler than an intermediate overbuilt version: BFGS-Hessian `trust-constr` and reduced/nullspace Powell fallbacks were removed because focused hard-case ablation showed they were unnecessary.

The strongest correctness evidence is direct endpoint residual verification: the code evaluates the profiled chi2 at each reported endpoint and checks it is within tolerance of `chi2_min + 1`. The main remaining uncertainty is globality of the constrained profile solve in hard nonlinear fit geometries. The KKT/projected-gradient/descent checks provide local certificates, not global proofs.

## 1. Current algorithm

### 1.1 Shared endpoint search: `binary_search_error`

Both averages and fits ultimately call `binary_search_error(profile_chi2, val, chi2_min, lb, ub, residual_tol=5e-3, contract_brackets=True)`.

For each side:

1. Set the Wilks target:

   ```text
   target = chi2_min + 1
   ```

2. Start from an inner point at the fitted value and an outer bracket guess.

3. Evaluate the outer bracket. If the profile evaluation fails and `contract_brackets=True`, contract the outer point halfway back toward the MLE until a usable point is found or the contraction width becomes too small.

4. If the usable outer point has `profile_chi2 < target`, expand outward by doubling the displacement from the MLE until `profile_chi2 >= target` or expansion fails.

5. Bisection on that bracket until the profiled chi2 residual is within `residual_tol`:

   ```text
   abs(profile_chi2(mid) - (chi2_min + 1)) <= 5e-3
   ```

6. Return errors as distances from the fitted value to the upper/lower endpoint.

7. Store diagnostics in `binary_search_error.last_diagnostics`, including:
   - upper/lower endpoint coordinates;
   - upper/lower chi2 values;
   - upper/lower residuals;
   - endpoint search method (`bisection`);
   - evaluation counters and cache hits.

Recent small optimization: the final bisection point already has a computed profile value, so final verification reuses that value instead of solving the same profile point a second time. This saves exactly two profile calls per target/node in the benchmark. The per-search cache is present but had zero hits in the hard benchmark; bisection rarely requests the exact same floating-point coordinate twice.

### 1.2 Average-side profile algorithm

Averages use `avg.run_avg()` and a direct fixed-coordinate profile.

For a single average node:

1. Build a minimal synthetic fit-like problem for the node and its nuisance terms through the standard preprocessing / `build_funcs` / `build_chi2` machinery.

2. Minimize the average chi2 to obtain the central value and `chi2_min`.

3. Choose initial endpoint brackets from measurement-scale spans:

   ```text
   ub = val + 4 * max(primary_error_p)
   lb = val - 4 * max(primary_error_n)
   ```

4. Define `profile_chi2(x_primary)` by fixing the primary coordinate explicitly.
   - If there is only one parameter, evaluate chi2 directly.
   - If there are nuisance parameters, minimize over the nuisance coordinates only.

5. Nuisance profiling tries:
   - the previous nuisance optimum as a warm start;
   - the base nuisance vector if distinct;
   - BFGS with gradient;
   - Nelder-Mead fallback from the best available candidate.

6. Accept the average profile point if the optimizer succeeded or the nuisance-gradient norm is small enough:

   ```text
   nuisance_grad_norm <= 1e-5 * max(abs(best_fun), 1.0)
   ```

7. Call `binary_search_error()` to locate upper/lower endpoints.

Why averages are simpler: the target is a direct primary coordinate. Fixing that coordinate and optimizing only nuisance coordinates avoids a nonlinear equality-constraint solve. This is both conceptually cleaner and empirically robust for the full average snapshot.

### 1.3 Fit-side profile algorithm

Fits use `calc_asym_errors()`. For each node/parameter target:

1. Select `target_func` from either `node_funcs` or `parameter_funcs`.

2. Compute the central target value at the best fit.

3. Estimate a target scale/standard deviation from the fit covariance if available:

   ```text
   J = jacobian(fitted_params_to_params)(fitted_values)
   param_cov = J @ covariance @ J.T
   target_std = sqrt(J_target @ param_cov @ J_target.T)
   ```

   If unavailable or invalid, fallback to a small relative scale.

4. Build a constrained-profile callable with `build_constrained_profile_chi2()`. For a fixed target value `v`, it minimizes chi2 in fitted-parameter space subject to:

   ```text
   target_func(fitted_params_to_params(fp)) = v
   ```

5. Verify the base profile at the best-fit target is not above `chi2_min` by more than `1e-5`.

6. Call `binary_search_error()` with initial bracket `target_value ± 2 * target_std`.

For each fixed target value, the constrained-profile solver does the following:

1. Scale the constraint by `target_scale` for numerical conditioning.

2. Project candidate starts onto the scalar constraint using repeated gradient projection.

3. Try starts from:
   - the previous accepted constrained optimum (`last_x`), for continuation/warm-start behavior;
   - the original best-fit point projected to the current target, if distinct.

4. For each start:
   - try SLSQP with gradient and nonlinear equality constraint;
   - if not accepted, try exact-Hessian `trust-constr`;
   - KKT-polish exact-Hessian candidates by solving the stationarity + constraint system.

5. Accept a constrained profile point only if all are true:
   - chi2 is finite;
   - scaled equality-constraint violation is <= `1e-7`;
   - either projected/KKT stationarity is small enough, or a local projected feasible-descent probe cannot reduce chi2 by more than `1e-4`.

6. If no candidate satisfies these checks, fail loudly with diagnostics rather than returning a silent bogus profile value.

7. Record a `ProfilePoint` with chi2, constraint violation, scaled violation, projected gradient norm, objective gradient norm, method string, runtime, and descent-improvement diagnostics.

Current fit-side method strings include e.g. `SLSQP`, `SLSQP+descent-check`, `trust-constr-exact-hess+KKT`, and related combinations.

## 2. Why this works better than the previous version

### 2.1 Empirical improvement

Previous staged fit broad sweep (`notes/asym-errors-broad-sweep.md`):

```text
Target rows: 227
Successful rows: 214
Error rows: 13
Successful endpoints verified: 428
Max abs residual among successful endpoints: 0.004978
```

Current simplified broad sweep (`notes/asym-fit-simplified-broad-sweep.md`):

```text
Target rows: 227
Successful rows: 227
Error rows: 0
Successful endpoints verified: 454
Max abs residual among successful endpoints: 0.004978
```

The target key set is identical between the old and current simplified sweeps. All 13 previous failures are now successful, and no previous successes regressed.

Average-side evidence was already strong:

```text
Average nodes: 2647/2647 ok
Fresh endpoint checks: 5294/5294 ok
Max abs fresh residual: 0.0049997
```

Endpoint-search micro-optimization evidence:

```text
Focused hard fits:     37/37 ok, profile calls 627 -> 553
Focused hard averages: 12/12 ok, profile calls 322 -> 298
Max residuals unchanged
```

That optimization is computationally modest but clean: it avoids a redundant final profile solve.

### 2.2 Computational reasons for improvement

The important improvements are not “more tests pass, therefore good.” They address observed failure modes.

#### A. Feasible-but-nonstationary fit profiles are now rejected or certified

The earlier hard failures were often feasible constrained points whose target equality violation was tiny, but whose projected/KKT gradient showed the optimizer had not actually minimized chi2 on the constrained slice. That is scientifically bad: feasibility alone does not define the profile likelihood.

Current code requires either:

- projected/KKT stationarity; or
- a no-meaningful-feasible-descent certificate.

This directly targets the previous failure class. It does not prove global optimality, but it is much stronger than accepting optimizer success or small constraint violation.

#### B. Exact-Hessian `trust-constr` fallback is needed for some hard fits

Focused ablation showed `slsqp_kkt` failed known `B0` hard targets, with projected gradients still around `4.96e3`. Keeping exact-Hessian `trust-constr` fallback is therefore justified by a concrete failure case.

The computational reason is plausible: exact Hessian information can help equality-constrained optimization in ill-conditioned nonlinear fitted-parameter coordinates, especially BR/BRU-like fits. That is a numerical reason, not a theorem that it will always find the global constrained minimum.

#### C. KKT polishing improves local constrained optimality

KKT polishing solves the first-order stationarity plus equality-constraint equations near a candidate. This is mathematically aligned with the local optimality conditions for an equality-constrained minimum.

Again: KKT stationarity is local. It does not rule out a distant lower constrained basin.

#### D. Descent check handles high-curvature coordinate artifacts

Some BR/BRU failures had large projected gradients in fitted coordinates, while direct local feasible descent probes lowered chi2 by only tiny amounts compared with endpoint tolerance. The descent check accepts such points only when no local feasible descent larger than `1e-4` is found.

This is a numerical local certificate. It is not as clean as exact KKT stationarity, and the report should be honest about that. It is retained because removing it immediately failed known `eta_c J/psi psi(2S)` hard targets.

#### E. Bracket contraction fixes unreachable initial endpoint guesses

For `eta_c(2S) / M059.4`, removing bracket contraction made the upper bracket fail with scaled constraint violation around `80.3`. The current bracket contraction pulls failed outward brackets back toward the MLE while preserving same-side search and loud failure if no usable point exists.

This is computationally sensible for BR/BRU/domain-constrained targets where covariance-based `±2σ` guesses can be outside the usable target range.

#### F. Averages use the right simpler problem

For averages, fixing the primary coordinate explicitly and minimizing only nuisance coordinates avoids a more fragile generic nonlinear equality-constraint formulation. This is not just a software simplification; it matches the mathematical structure of single-node averages.

The average full-snapshot sweep supports that this direct profile path is robust in practice.

### 2.3 What is empirical rather than theoretically proven

We do not have a theoretical proof that every fit-side constrained profile solve is the global minimum over the whole nonlinear constrained manifold.

What we have:

- feasibility checks;
- local stationarity/KKT checks;
- local descent checks;
- endpoint residual verification;
- broad and focused empirical sweeps over hard cases.

That is good numerical evidence. It is not a global optimization proof.

## 3. Evidence that endpoints correspond to Wilks' theorem

Wilks' theorem motivates the one-parameter profile-likelihood rule:

```text
Delta chi2 = chi2_profile(t) - chi2_min = 1
```

for an approximate 1-sigma interval under the usual regularity/asymptotic assumptions.

The code's endpoint correctness criterion is exactly this numerical condition. Evidence comes from several layers.

### 3.1 Direct endpoint residual verification

Every returned endpoint stores/checks residuals:

```text
upper_residual = chi2_profile(upper_endpoint) - (chi2_min + 1)
lower_residual = chi2_profile(lower_endpoint) - (chi2_min + 1)
```

Current staged fit sweep:

```text
Endpoint checks: 454
Median abs residual: 0.000919
Mean abs residual: 0.001384
95% abs residual: 0.004249
99% abs residual: 0.004914
Max abs residual: 0.004978
```

Current full average sweep:

```text
Endpoint checks: 5294
Median abs residual: 0.001984
Mean abs residual: 0.002173
95% abs residual: 0.004632
99% abs residual: 0.004924
Max abs residual: 0.0049997
```

Those maxima sit at the configured bisection tolerance `5e-3`; they are not outlying failures.

### 3.2 Profile minimization checks

Endpoint residuals matter only if `chi2_profile(t)` was actually profiled. The current algorithm checks constrained profile quality:

- scaled equality constraint <= `1e-7`;
- finite chi2;
- projected/KKT stationarity or local no-meaningful-descent certificate;
- loud error if these fail.

This addresses the main scientific failure mode found earlier: endpoints with the right fixed target coordinate but not actually minimized over nuisance/fitted directions.

### 3.3 Regression against known hard cases

Known hard labels/targets now pass:

- `Lam-b-0`: 20/20 staged broad rows ok, max runtime around 87.6 s in the simplified sweep.
- `Upsilon(2S)`: 6/6 ok.
- `B0`: 9/9 ok.
- `B0S-BR`: 9/9 ok.
- `eta_c J/psi psi(2S)`: 8/8 ok.
- `K_3^*(1780)`: 2/2 ok.
- `eta_c(2S)`: 2/2 ok.

The 13 previous broad-sweep failures all became `ok` in the current sweep.

### 3.4 What this does not prove

Wilks' theorem itself has assumptions: approximate regularity, sufficient data/asymptotics, no problematic boundaries, and effectively one parameter of interest. The code verifies the numerical `Delta chi2 = 1` profile endpoints; it does not prove Wilks coverage in finite samples or at physical boundaries.

The constrained optimizers also give local certificates. They do not guarantee a lower disconnected constrained basin does not exist.

So the precise statement should be:

> The current implementation numerically finds and verifies profile-chi2 endpoints satisfying the Wilks `Delta chi2 = 1` criterion on the tested snapshot targets, with local constrained-optimality checks. It is not a proof of global profile optimality or frequentist coverage for every PDG fit.

## 4. How to improve the evidence if the method is working

Good next evidence should stress globality, coverage, and reproducibility rather than just more happy-path rows.

### 4.1 Expanded all-target sweeps for high-risk labels

The current fit sweep is staged representative + capped follow-up, not every possible node/parameter in every fit. Next high-value sweep:

- all targets for high-risk labels:
  - `chi_c012 psi(2S)`;
  - `Lam-b-0`;
  - `B0`;
  - `B0S-BR`;
  - `eta_c J/psi psi(2S)`;
  - `D0`;
  - `Lambda_c`;
  - `B+J/psi`;
- record residuals, KKT/descent usage, method strings, runtimes, and failure diagnostics.

### 4.2 Multi-start checks on hard endpoints

For endpoints accepted via `+descent-check` or large projected gradients, rerun constrained profiles from multiple starts:

- MLE-projected start;
- previous continuation start;
- random perturbations projected to the constraint;
- covariance-direction perturbations;
- starts near parameter-map boundaries.

Compare the best found constrained chi2. If no start improves chi2 by more than e.g. `1e-3` to `5e-3`, confidence increases substantially.

### 4.3 Independent solver cross-checks

For a curated hard set, compare current solver against independent formulations:

- direct reduced-coordinate parameterization where possible;
- scipy `trust-constr` with different Hessian handling;
- Minuit/MINOS-style profile where applicable;
- brute-force grid or dense one-dimensional scans for low-dimensional examples;
- local tangent-space minimization checks.

The point is not to ship multiple solvers, but to detect cases where the current solver lands in the wrong constrained basin.

### 4.4 Tighten endpoint residual tolerance on a subset

Run hard endpoints at `residual_tol=1e-3` or `1e-4` and compare errors/runtimes. This would test whether results are stable under tighter endpoint solving. It will cost extra profile calls, so do it on a hard subset first.

### 4.5 Track profile curves, not just endpoints

For hard targets, evaluate profile chi2 on a grid along each side and plot/record:

- monotonicity away from MLE;
- bracket behavior;
- endpoint location;
- method used at each fixed target;
- KKT/descent diagnostics.

This can reveal nonmonotone profiles or disconnected feasible regions that bisection alone could miss.

### 4.6 Coverage experiments where simulation is feasible

For simplified synthetic problems and maybe selected real-fit local approximations, simulate datasets from known parameters and check whether the `Delta chi2 = 1` intervals have the expected coverage. This tests Wilks applicability, not just numerical endpoint solving.

### 4.7 Reproducibility runs

Repeat hard sweeps under:

- fresh process/JAX compilation;
- different target order;
- no resume cache;
- possibly small optimizer tolerance perturbations.

The outputs should be stable to within the endpoint tolerance. This is useful because continuation/warm-start behavior can hide order sensitivity.

## 5. Possible directions for speed, robustness, and correctness improvements

### 5.1 Endpoint search speed

#### A. Remove or keep the evaluation cache deliberately

The cache had zero hits in the focused endpoint-search benchmark. It is tiny and useful for diagnostics, but it is not currently buying speed. If minimalism is the priority, remove it and keep only counters plus final bisection-value reuse.

#### B. Tolerance/runtime tradeoff modes

Expose named modes:

- `fast`: current `5e-3` residual tolerance;
- `strict`: `1e-3` or `1e-4` for publication or final tables;
- `diagnostic`: tighter tolerance plus detailed profile diagnostics.

Need benchmark first; tighter tolerance may cost several extra profile solves per endpoint.

#### C. Safeguarded interpolation only if it beats bisection cleanly

A Brent-style bracketed root solver might reduce calls on smooth profiles, but it introduces a coordinate-space tolerance and still needs residual verification. Previous secant-like exploration was not compelling. Only revisit this with a hard-set benchmark and keep bisection fallback.

#### D. Smarter initial brackets

Current fits use covariance-derived `±2 * target_std`; averages use measurement-span brackets. Possible improvements:

- use previous successful endpoint scale for related node/parameter targets;
- estimate local curvature from profile evaluations near the MLE;
- choose brackets to reduce expansion/contraction in BR/BRU targets.

This should be benchmarked carefully because bad clever brackets can be worse than boring robust brackets.

### 5.2 Profile optimization speed

#### A. Reduce calls to exact-Hessian `trust-constr`

Exact-Hessian fallback is necessary for some endpoints but expensive. Possible improvement: better pre-screening so it is called only when SLSQP really lacks acceptable local optimality.

Need care: SLSQP+KKT-only failed known `B0` targets, so don't remove fallback blindly.

#### B. Reuse compiled JAX functions aggressively

Hard fits spend time in repeated profile calls. Ensure chi2, gradient, Hessian, target gradients, and constraint Hessians are compiled once per target where possible. Avoid shape/pytree changes that trigger recompilation.

#### C. Parallelize at target-row granularity, conservatively

The broad sweep ran sequentially to protect the VPS. For future sweeps, process-level parallelism with 2 workers might reduce wall time while keeping RAM safe. JAX-heavy 3+ workers may be risky on this VPS.

Important: use resumable JSONL output and per-target timeouts.

#### D. Separate slow labels into dedicated jobs

`chi_c012 psi(2S)` and `Lam-b-0` dominate runtime. Run them as separate resumable batches so ordinary labels finish quickly and failures are easier to isolate.

### 5.3 Robustness/correctness

#### A. Stronger multistart certification for `+descent-check` endpoints

`+descent-check` is the least theorem-like acceptance path. Add optional diagnostic mode:

- run N projected starts;
- record best alternative chi2;
- fail or warn if an alternative improves profile chi2 by more than tolerance.

This would materially improve evidence for the hardest endpoints.

#### B. Scale-aware KKT criteria

The current projected-gradient test uses a mix of absolute and relative thresholds; high-curvature fitted coordinates motivated the descent check. A more principled scale-aware KKT norm could reduce reliance on descent probes.

This is nontrivial; do not tune thresholds to pass cases without a clear interpretation.

#### C. Reduced-coordinate formulations for specific parameter maps

For BR/BRU/simplex-like structures, a chart-aware reduced parameterization might produce cleaner constrained profiles than generic equality constraints in fitted space. This could improve both correctness and runtime, but might add complexity.

Worth exploring only if the current solver fails expanded hard-target sweeps or multistart checks.

#### D. Boundary-aware Wilks diagnostics

BR/BRU targets near physical boundaries may violate Wilks regularity. Add diagnostics for endpoints near parameter-map boundaries, saturated sigmoid/arctan regions, or simplex edges. Numerical `Delta chi2 = 1` can be correct while Wilks coverage is questionable.

#### E. Store richer provenance for accepted endpoints

For every reported endpoint, retain:

- endpoint coordinate;
- residual;
- final method;
- scaled constraint violation;
- projected gradient norm;
- descent improvement;
- number of profile evaluations;
- runtime;
- whether endpoint was accepted by KKT or descent check.

This is already partly available; make sure final output tables preserve it consistently.

## 6. Recommended next steps

1. Do not change the core solver immediately. The current simplified solver has good staged evidence and every retained component has a known reason.

2. Run an expanded all-target sweep for the highest-risk labels, not the entire universe first. Use resumable outputs and conservative parallelism.

3. Add a diagnostic multistart mode for endpoints accepted via `+descent-check`, then run it on the hardest endpoints from the broad sweep.

4. Consider removing the endpoint eval cache if minimalism matters more than diagnostics; keep final bisection-value reuse, which has measured value.

5. If multistart or expanded sweeps find failures, investigate scale-aware KKT norms or chart-aware reduced-coordinate profiling. Otherwise, prefer evidence-building over more algorithmic cleverness.

## 7. Bottom line

The current asymmetric-error implementation is much better than the previous version in the practical sense that it passes the same staged fit target set with zero errors instead of 13, passes the full average snapshot sweep, verifies every reported endpoint against `chi2_min + 1`, and fails loudly when profile minimization is not certified.

The mathematical story is solid at the local numerical-optimization level: constrained profiles are checked for feasibility and stationarity/descent, and endpoints are checked against the Wilks `Delta chi2 = 1` criterion. The unproven part is global constrained optimality and finite-sample Wilks coverage in pathological nonlinear/boundary cases. The next best work is to improve evidence for those, not to add more solver machinery by default.
