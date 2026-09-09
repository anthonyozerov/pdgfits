# Codex Cloud asymmetric-error review

Date: 2026-06-23

Context:

- GitHub issue: `anthonyozerov/pdgfits-private#1`, "asymmetric errors in fits and averages"
- Codex Cloud task: `task_e_6a39dfd8863c833387d0a8e1cd4101de`, "GitHub Mention: asymmetric errors in fits and averages"
- Review worktree: `/tmp/pdgfits-asym-cloud-task-e-6a39dfd`
- Review branch: `codex-cloud-asym-errors-review`
- Snapshot used for real runs: `/tmp/pdgfits-asym-cloud-task-e-6a39dfd/data/pdg-snapshot`

## Issue Requirement

The issue asks for rigorous asymmetric errors from Wilks' theorem for:

- parameters
- nuisance parameters
- nodes

It specifically notes that asymmetric errors in fits and/or averages had convergence issues.

## Cloud Agent Changes

The cloud agent changed three files:

- `src/pdgfits/asym_errors.py`
- `src/pdgfits/build_chi2.py`
- `tests/test_asym_errors.py`

Main behavioral change:

- Replaced the fit-side custom Newton/projected-gradient profile optimizer with an SLSQP equality-constrained profile likelihood solve.
- Hardened `binary_search_error`.
- Added `build_chi2` optional `translate_dep` and `adjust` support.
- Added toy tests for quadratic profile errors and parameter/node profiling.

## Verification Run

All commands below were run from the isolated worktree with:

```bash
PYTHONPATH=/tmp/pdgfits-asym-cloud-task-e-6a39dfd/src
PDGFITS_DATA_BACKEND=snapshot
PDGFITS_SNAPSHOT_DIR=/tmp/pdgfits-asym-cloud-task-e-6a39dfd/data/pdg-snapshot
```

Offline test suite:

```bash
conda run -n pdg env PYTHONPATH=/tmp/pdgfits-asym-cloud-task-e-6a39dfd/src \
  python -m pytest tests/ -q
```

Result:

```text
157 passed, 2 skipped in 7.72s
```

Real fit-side profile optimization checks:

```bash
python -m pdgfits.run_fits --fit_label 'G(2000),G(1800)' --calc_asym_errors
python -m pdgfits.run_fits --fit_label 'Upsilon(2S)' --calc_asym_errors
python -m pdgfits.run_fits --fit_label 'Lam-b-0' --calc_asym_errors
```

Observed:

- `G(2000),G(1800)` completed all profile optimizations for two parameters and one node.
- `Upsilon(2S)` completed parameters, nodes, and `nuisance_M048.8`.
- `Lam-b-0` completed all reported profile optimizations, including four nuisance parameters: `nuisance_S042.10`, `nuisance_S042.30`, `nuisance_S051.2`, `nuisance_S051.4`.

Average-side profile checks:

```bash
python -m pdgfits.run_avgs --node M026R08 --output /tmp/pdgfits-asym-cloud-task-e-6a39dfd/avg_m026r08.csv
python -m pdgfits.run_avgs --interesting --output /tmp/pdgfits-asym-cloud-task-e-6a39dfd/avg_interesting.csv
```

Observed:

- `M026R08` completed and included nuisance parameters from `br_adjust`.
- The curated `--interesting` average set completed without profile-search exceptions.
- One average printed a `nan` Birge diagnostic from `birge.py`; the profile error search itself still returned finite errors.

## Assessment

The cloud change is a real improvement for fit-side asymmetric errors. It implements the Wilks/profile-likelihood construction intended by the scientific context: find the points where the profiled chi-squared reaches `chi2_min + 1` while profiling over fitted parameters subject to the target constraint. The PDG asymmetric-error interpolation in `build_chi2` is not silently replaced by a different likelihood.

The real snapshot runs show that the new SLSQP profile optimizations converge on representative fits, including nuisance-parameter targets.

I would not consider issue #1 fully closed yet for averages. The average path still has its own profile minimization in `avg.py` using Nelder-Mead over nuisance parameters, so the cloud change mostly hardens the shared binary search rather than moving averages to the same constrained-profile implementation used by `calc_asym_errors`.

Open caveats:

- `profile_chi2` in `asym_errors.py` checks feasibility and finite objective, but does not check `result.success`.
- The new tests are toy tests; they do not cover real asymmetric-error likelihoods, constrained branching-fraction fits, boundary behavior, or nuisance correlations.
- `calc_asym_errors` relies on Hessian covariance to choose initial brackets; this is a pragmatic initialization, not a guarantee.
- The average-side implementation should be reviewed separately if "fits and averages" is meant literally.

## Unified Diff

```diff
diff --git a/src/pdgfits/asym_errors.py b/src/pdgfits/asym_errors.py
index 69addad..f891578 100644
--- a/src/pdgfits/asym_errors.py
+++ b/src/pdgfits/asym_errors.py
@@ -1,9 +1,11 @@
+from decimal import Decimal
+
 import jax
 from jax import numpy as jnp
-from decimal import Decimal
-import nlopt
+import numpy as np
 from scipy.optimize import NonlinearConstraint, minimize
 
+
 def symmetrize(y, val, error_n, error_p):
     resid = y - val
     error_min = jnp.minimum(error_n, error_p)
@@ -15,6 +17,7 @@ def symmetrize(y, val, error_n, error_p):
     error_between = (prod_e2 - resid * diff_e) / sum_e
     return jnp.clip(error_between, error_min, error_max)
 
+
 def binary_search_error(profile_chi2, val, chi2_min, lb, ub, verbose=False):
     """
     Find 1-sigma asymmetric errors by binary search on a profile chi-squared.
@@ -22,74 +25,72 @@ def binary_search_error(profile_chi2, val, chi2_min, lb, ub, verbose=False):
     profile_chi2: callable(scalar) -> chi2 profiled over all other parameters
     val: optimal value of the target quantity
     chi2_min: minimum chi2 at the optimum
-    err_scale: scale for convergence tolerance (search stops when bracket < 1e-3 * err_scale)
     lb: initial lower bound (lb < val); expanded outward if chi2 doesn't reach chi2_min+1
     ub: initial upper bound (ub > val); expanded outward if chi2 doesn't reach chi2_min+1
 
     Returns (upper_err, lower_err).
     """
     target = chi2_min + 1
-    chi2 = jnp.inf
-
     max_iter = 60
-    iter = 0
-    while profile_chi2(ub) < target and iter < max_iter:
-        iter += 1
+
+    def as_float(x):
+        return float(np.asarray(x))
+
+    if not lb < val < ub:
+        raise ValueError(f"Expected lb < val < ub, got lb={lb}, val={val}, ub={ub}")
+
+    iter_count = 0
+    while as_float(profile_chi2(ub)) < target and iter_count < max_iter:
+        iter_count += 1
         ub = val + 2 * (ub - val)
     if verbose:
-        print(f'expanded ub {iter} times')
+        print(f'expanded ub {iter_count} times')
         print(profile_chi2(ub))
         print(profile_chi2(val))
-    if iter == max_iter:
+    if iter_count == max_iter:
         raise ValueError(f'Expanding ub failed to converge after {max_iter} iterations')
+
     lo, hi = val, ub
-    lo_init = lo
-    hi_init = hi
-    iter = 0
-    while jnp.abs(chi2-target)>0.005 and iter < max_iter:
-        iter += 1
+    width_tol = max(abs(ub - val) * 1e-6, np.finfo(float).eps * max(abs(val), abs(ub), 1.0))
+    for _ in range(max_iter):
         mid = (lo + hi) / 2
-        chi2 = profile_chi2(mid)
-        if chi2 < target:
-            if verbose: print(f'at {(mid-lo_init)/(hi_init-lo_init):.2f} get {chi2:.2f} -> increase')
+        chi2_mid = as_float(profile_chi2(mid))
+        if chi2_mid < target:
             lo = mid
         else:
-            if verbose: print(f'at {(mid-lo_init)/(hi_init-lo_init):.2f} get {chi2:.2f} -> decrease')
             hi = mid
-    if iter == max_iter:
-        print(f'chi2 achieved: {chi2:.2f}, target: {target:.2f}')
+        if abs(hi - lo) <= width_tol or abs(chi2_mid - target) <= 0.005:
+            break
+    else:
+        print(f'chi2 achieved: {chi2_mid:.2f}, target: {target:.2f}')
         raise ValueError(f'Binary search failed to converge after {max_iter} iterations')
-    assert jnp.abs(chi2 - target) < 0.1, f'Expected chi2 ~ {target}, got {chi2} at mid={mid}, lo={lo}, hi={hi}'
-
     upper_err = float(hi - val)
 
-    chi2 = jnp.inf
-    iter = 0
-    while profile_chi2(lb) < target and iter < max_iter:
-        iter += 1
+    iter_count = 0
+    while as_float(profile_chi2(lb)) < target and iter_count < max_iter:
+        iter_count += 1
         lb = val - 2 * (val - lb)
     if verbose:
-        print(f'expanded lb {iter} times')
+        print(f'expanded lb {iter_count} times')
         print(profile_chi2(lb))
         print(profile_chi2(val))
-    if iter == max_iter:
+    if iter_count == max_iter:
         raise ValueError(f'Expanding lb failed to converge after {max_iter} iterations')
+
     lo, hi = lb, val
-    iter = 0
-    while jnp.abs(chi2-target)>0.005 and iter < max_iter:
-        iter += 1
+    width_tol = max(abs(val - lb) * 1e-6, np.finfo(float).eps * max(abs(lb), abs(val), 1.0))
+    for _ in range(max_iter):
         mid = (lo + hi) / 2
-        chi2 = profile_chi2(mid)
-        if chi2 < target:
-            if verbose: print(f'at {(mid-lo_init)/(hi_init-lo_init):.2f} get {chi2:.2f} -> decrease')
+        chi2_mid = as_float(profile_chi2(mid))
+        if chi2_mid < target:
             hi = mid
-        else:   
-            if verbose: print(f'at {(mid-lo_init)/(hi_init-lo_init):.2f} get {chi2:.2f} -> increase')
+        else:
             lo = mid
-    if iter == max_iter:
-        print(f'chi2 achieved: {chi2}')
+        if abs(hi - lo) <= width_tol or abs(chi2_mid - target) <= 0.005:
+            break
+    else:
+        print(f'chi2 achieved: {chi2_mid:.2f}, target: {target:.2f}')
         raise ValueError(f'Binary search failed to converge after {max_iter} iterations')
-    assert jnp.abs(chi2 - target) < 0.1, f'Expected chi2 ~ {target}, got {chi2} at mid={mid}, lo={lo}, hi={hi}'
     lower_err = float(val - lo)
 
     return upper_err, lower_err
@@ -97,31 +98,31 @@ def binary_search_error(profile_chi2, val, chi2_min, lb, ub, verbose=False):
 
 def calc_asym_errors(fit, targets=None):
     """
-    Calculate asymmetric errors via binary search for a subset of nodes or parameters.
+    Calculate Wilks/profile-likelihood asymmetric errors for nodes or parameters.
 
-    fit: dict returned by run_fit
-    targets: list of node or parameter names, or None to use all nodes
+    The returned mapping is keyed by target name.  Each value contains the MLE
+    value plus the positive and negative 1-sigma errors, found from
+    profile_chi2(target) = chi2_min + 1 while profiling over all fitted
+    parameters subject to the target constraint.
     """
     nodes = fit['nodes']
     parameters = fit['parameters']
     node_funcs = fit['node_funcs']
     parameter_funcs = fit['parameter_funcs']
     fitted_params_to_params = fit['fitted_params_to_params']
-    fitted_values = fit['fitted_values']
+    fitted_values = np.array(fit['fitted_values'], dtype=np.float64)
     param_values = fit['param_values']
     chi2 = fit['chi2']
     chi2_grad = fit['chi2_grad']
-    chi2_min = fit['chi2_min']
+    chi2_min = float(fit['chi2_min'])
     covariance = fit['covariance']
-    hess = jax.hessian(chi2)
-    chi2_grad_jax = jax.jit(jax.grad(chi2))
-    assert chi2(fitted_values) == chi2_min, f'chi2 at optimum: {chi2(fitted_values)} != {chi2_min}'
 
     if targets is None:
-        targets = sorted(list(set(parameters) | set(nodes)))
+        targets = sorted(set(parameters) | set(nodes))
 
-    J = jax.jacobian(fitted_params_to_params)(fitted_values)
+    J = jax.jacobian(fitted_params_to_params)(jnp.array(fitted_values))
     param_cov = J @ covariance @ J.T
+    results = {}
 
     for target in targets:
         print(target)
@@ -133,84 +134,71 @@ def calc_asym_errors(fit, targets=None):
             print(f'Warning: {target} not found in nodes or parameters, skipping.')
             continue
 
+        def constraint_jax(fp):
+            return target_func(fitted_params_to_params(fp))
+
+        constraint_grad_jax = jax.jit(jax.grad(constraint_jax))
         target_value = float(target_func(param_values))
-        J_target = jax.jacobian(target_func)(param_values)
-        target_std = float(jnp.sqrt(J_target @ param_cov @ J_target.T))
+        target_grad_param = jax.jacobian(target_func)(param_values)
+        target_var = float(target_grad_param @ param_cov @ target_grad_param.T)
+        target_std = float(np.sqrt(max(target_var, 0.0)))
+        if not np.isfinite(target_std) or target_std <= 0:
+            print(f'Warning: {target} has non-positive propagated std ({target_std}), skipping.')
+            continue
 
-        def make_constraint(func):
-            @jax.jit
-            def constraint_jax(fp):
-                return func(fitted_params_to_params(fp))
-            constraint_grad_jax = jax.jit(jax.grad(constraint_jax))
-            return constraint_jax, constraint_grad_jax
+        def objective(x):
+            return float(chi2(jnp.array(x, dtype=jnp.float64)))
 
-        constraint_jax, constraint_grad_jax = make_constraint(target_func)
-        assert jnp.isclose(constraint_jax(fitted_values), target_value, atol=1e-6*target_std), f'Constraint violated for {target}: {constraint_jax(fitted_values)} != {target_value}'
+        def objective_grad(x):
+            return np.array(chi2_grad(jnp.array(x, dtype=jnp.float64)), dtype=np.float64)
 
-        fp0 = jnp.array(fitted_values)
+        def feasible_start(v):
+            g = np.array(constraint_grad_jax(jnp.array(fitted_values, dtype=jnp.float64)), dtype=np.float64)
+            denom = float(g @ g)
+            if denom == 0.0 or not np.isfinite(denom):
+                return fitted_values.copy()
+            return fitted_values + ((float(v) - target_value) / denom) * g
 
         def profile_chi2(v):
-            def newton_cond(carry):
-                x, c, i = carry
-                return (jnp.abs(c) >= 1e-8 * target_std) & (i < 200)
-
-            def newton_body(carry):
-                x, c, i = carry
-                gc = constraint_grad_jax(x)
-                denom = jnp.dot(gc, gc)
-                step = (c / denom) * gc
-                return x - step, constraint_jax(x - step) - v, i + 1
-
-            x, _, _ = jax.lax.while_loop(
-                newton_cond, newton_body,
-                (fp0, constraint_jax(fp0) - v, jnp.int32(0))
+            v = float(v)
+            scale = max(abs(target_std), abs(target_value) * 1e-12, 1e-300)
+
+            def con_fun(x):
+                residual = constraint_jax(jnp.array(x, dtype=jnp.float64)) - v
+                return np.array([float(residual / scale)])
+
+            def con_jac(x):
+                grad = constraint_grad_jax(jnp.array(x, dtype=jnp.float64)) / scale
+                return np.array([grad], dtype=np.float64)
+
+            constraint = NonlinearConstraint(con_fun, 0.0, 0.0, jac=con_jac)
+            x0 = feasible_start(v)
+            result = minimize(
+                objective,
+                x0,
+                method='SLSQP',
+                jac=objective_grad,
+                constraints=[constraint],
+                options={'ftol': 1e-10, 'maxiter': 1000},
             )
+            violation = abs(con_fun(result.x)[0])
+            if violation > 1e-5 or not np.isfinite(result.fun):
+                return np.inf
+            return float(result.fun)
 
-            lr = jnp.float64(1e-3)
-
-            def pgd_cond(carry):
-                x, t, g_norm = carry
-                return (g_norm >= 1e-7 * target_std) & (t < 5000)
-
-            def pgd_body(carry):
-                x, t, _ = carry
-                g_obj = chi2_grad_jax(x)
-                g_con = constraint_grad_jax(x)
-                jax.debug.print("g_con = {g_con}", g_con=g_con)
-                denom = jnp.dot(g_con, g_con)
-                jax.debug.print("denom = {denom}", denom=denom)
-                g_proj = g_obj - (jnp.dot(g_obj, g_con) / denom) * g_con
-                x = x - lr * g_proj
-                jax.debug.print("x = {x}", x=x)
-                g_con = constraint_grad_jax(x)
-                jax.debug.print("g_con = {g_con}", g_con=g_con)
-                denom = jnp.dot(g_con, g_con)
-                x = x - (constraint_jax(x) - v) / denom * g_con
-                return x, t + 1, jnp.linalg.norm(g_proj)
-
-            x, _, _ = jax.lax.while_loop(
-                pgd_cond, pgd_body,
-                (x, jnp.int32(0), jnp.array(jnp.inf))
-            )
+        prof_at_mle = profile_chi2(target_value)
+        if prof_at_mle > chi2_min + max(1e-4, 1e-8 * max(abs(chi2_min), 1.0)):
+            raise AssertionError(f'profile chi2 at target value: {prof_at_mle} > {chi2_min}')
 
-            return chi2(x)
-        # print('CONSTRAINT GRAD TEST:')
-        # print(constraint_grad_jax(jnp.array([1.62735912e+03, 1.74248989e+03, -5.94038873e+03, 1.92734985e+02,
-#   1.34481746e+01, 2.29959757e+03, 1.88719903e-01, 2.24521997e+01,
-#   1.32776714e+01, 2.80500000e-01, 7.19500000e-01, 3.91782322e-01,
-#  -5.99434543e+02, -5.67476965e+06])))
-        # JIT once; wrap so callers can pass Python floats and still get cache hits
-        # (passing a raw Python float would retrace on every call).
-        # _profile_chi2_jit = jax.jit(_profile_chi2)
-        # profile_chi2 = lambda v: _profile_chi2_jit(jnp.float64(v))
-        assert profile_chi2(target_value) <= chi2_min, f'profile chi2 at target value: {profile_chi2(target_value)} > {chi2_min}'
-
-        print(target_std)
         lb = target_value - 2 * target_std
         ub = target_value + 2 * target_std
+        upper_err, lower_err = binary_search_error(profile_chi2, target_value, chi2_min, lb, ub)
 
-        upper_err, lower_err = binary_search_error(
-            profile_chi2, target_value, chi2_min, lb, ub
-        )
-
+        results[target] = {
+            'value': target_value,
+            'error_p': upper_err,
+            'error_n': lower_err,
+        }
         print(f'{target}: {Decimal(target_value):.5E} + {Decimal(upper_err):.2E} - {Decimal(lower_err):.2E}')
+
+    return results
diff --git a/src/pdgfits/build_chi2.py b/src/pdgfits/build_chi2.py
index 0c5a50f..c58a76b 100644
--- a/src/pdgfits/build_chi2.py
+++ b/src/pdgfits/build_chi2.py
@@ -3,7 +3,11 @@ import jax.numpy as jnp
 import jax
 
 
-def build_chi2(y, mu, error_n, error_p, corr_mat_inv, fitted_params_to_params, use_jit=True):
+def build_chi2(y, mu, error_n, error_p, corr_mat_inv, fitted_params_to_params, translate_dep=None, adjust=None, use_jit=True):
+    if translate_dep is None:
+        translate_dep = lambda x: 0
+    if adjust is None:
+        adjust = lambda x: 1
 
     error_min = jnp.minimum(error_n, error_p)
     error_max = jnp.maximum(error_n, error_p)
@@ -11,17 +15,19 @@ def build_chi2(y, mu, error_n, error_p, corr_mat_inv, fitted_params_to_params, u
     diff_e = error_p - error_n
     prod_e2 = 2 * error_n * error_p
 
-    def get_sigma(resid):
-        error_between = (prod_e2 - resid * diff_e) / sum_e
-        return jnp.clip(error_between, error_min, error_max)
+    def get_sigma(resid, adjustment):
+        resid_ua = resid / adjustment
+        error_between = (prod_e2 - resid_ua * diff_e) / sum_e
+        return jnp.clip(error_between, error_min, error_max) * adjustment
 
     maybe_jit = jax.jit if use_jit else (lambda f: f)
 
     @maybe_jit
     def chi2_open(fitted_params, y_arg):
         params = fitted_params_to_params(fitted_params)
-        resid = y_arg - mu(params)
-        error = get_sigma(resid)
+        adjustment = adjust(params)
+        resid = y_arg * adjustment + translate_dep(params) - mu(params)
+        error = get_sigma(resid, adjustment)
         normed_resid = resid / error
         return normed_resid @ corr_mat_inv @ normed_resid
 
diff --git a/tests/test_asym_errors.py b/tests/test_asym_errors.py
new file mode 100644
index 0000000..c5d2f9f
--- /dev/null
+++ b/tests/test_asym_errors.py
@@ -0,0 +1,56 @@
+import jax
+import jax.numpy as jnp
+import numpy as np
+import pytest
+
+jax.config.update("jax_enable_x64", True)
+
+from pdgfits.asym_errors import binary_search_error, calc_asym_errors
+
+
+def test_binary_search_error_quadratic():
+    upper, lower = binary_search_error(
+        lambda x: (x - 2.0) ** 2,
+        val=2.0,
+        chi2_min=0.0,
+        lb=1.5,
+        ub=2.5,
+    )
+
+    assert upper == pytest.approx(1.0, abs=5e-3)
+    assert lower == pytest.approx(1.0, abs=5e-3)
+
+
+def test_calc_asym_errors_profiles_parameter_and_node():
+    def fitted_params_to_params(fp):
+        return fp
+
+    def chi2(fp):
+        return ((fp[0] - 1.0) / 2.0) ** 2 + ((fp[1] + 2.0) / 3.0) ** 2
+
+    chi2_grad = jax.jit(jax.grad(chi2))
+    param_funcs = [lambda p: p[0], lambda p: p[1]]
+    node_funcs = [lambda p: p[0] + p[1]]
+
+    fit = {
+        'nodes': ['sum'],
+        'parameters': ['a', 'b'],
+        'node_funcs': node_funcs,
+        'parameter_funcs': param_funcs,
+        'fitted_params_to_params': fitted_params_to_params,
+        'fitted_values': jnp.array([1.0, -2.0], dtype=jnp.float64),
+        'param_values': jnp.array([1.0, -2.0], dtype=jnp.float64),
+        'chi2': chi2,
+        'chi2_grad': lambda x: np.array(chi2_grad(x)),
+        'chi2_min': 0.0,
+        'covariance': jnp.diag(jnp.array([2.0**2, 3.0**2], dtype=jnp.float64)),
+    }
+
+    result = calc_asym_errors(fit, targets=['a', 'sum'])
+
+    assert result['a']['value'] == pytest.approx(1.0)
+    assert result['a']['error_p'] == pytest.approx(2.0, abs=1e-2)
+    assert result['a']['error_n'] == pytest.approx(2.0, abs=1e-2)
+    assert result['sum']['value'] == pytest.approx(-1.0)
+    assert result['sum']['error_p'] == pytest.approx(np.sqrt(13.0), abs=1e-2)
+    assert result['sum']['error_n'] == pytest.approx(np.sqrt(13.0), abs=1e-2)
```
