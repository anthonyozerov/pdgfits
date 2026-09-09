"""Fixed-target minimization for joint fits and direct-coordinate averages."""

from dataclasses import dataclass
import time

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import NonlinearConstraint, OptimizeResult, minimize, root


@dataclass
class ProfilePoint:
    value: float
    chi2: float
    success: bool
    constraint_violation: float
    message: str = ""
    scaled_constraint_violation: float | None = None
    projected_grad_norm: float | None = None
    objective_grad_norm: float | None = None
    method: str = ""
    nfev: int | None = None
    nit: int | None = None
    runtime_sec: float | None = None
    descent_improvement: float | None = None


@dataclass(frozen=True)
class ProfileSolverOptions:
    use_slsqp: bool = True
    polish_slsqp: bool = False
    use_exact_hessian: bool = True
    use_kkt_polish: bool = True
    use_descent_check: bool = True


def _as_float(x):
    return float(np.asarray(x, dtype=np.float64))


def build_constrained_profile_chi2(
    chi2,
    fitted_params_to_params,
    target_func,
    fitted_values,
    *,
    target_scale=1.0,
    target_name="target",
    optimizer_options=None,
    solver_options=None,
    chi2_grad_jax=None,
    chi2_hess_jax=None,
):
    """Return a checked profile-chi2 callable for fixing target_func(params)=v.

    The returned function minimizes the same chi2 in fitted-parameter space under a
    scalar nonlinear equality constraint. It raises RuntimeError rather than
    returning a silently non-converged objective value.
    """
    solver_options = solver_options or ProfileSolverOptions()
    fp_best = np.asarray(fitted_values, dtype=np.float64)
    target_scale = abs(float(target_scale))
    if not np.isfinite(target_scale) or target_scale == 0.0:
        target_scale = abs(_as_float(target_func(fitted_params_to_params(fp_best))))
    if not np.isfinite(target_scale) or target_scale == 0.0:
        target_scale = 1.0
    cons_tol = 1e-7
    # Endpoint searches verify profile_chi2 = chi2_min + 1 to 5e-3.  This
    # tighter tolerance is only for certifying high-curvature constrained
    # solves where first-order residuals are dominated by coordinate scaling.
    stationarity_fun_tol = 1e-4

    def chi2_np(x):
        val = _as_float(chi2(jnp.asarray(x, dtype=jnp.float64)))
        if not np.isfinite(val):
            return np.inf
        return val

    if chi2_grad_jax is None:
        chi2_grad_jax = jax.jit(jax.grad(chi2))

    def chi2_grad_np(x):
        return np.asarray(chi2_grad_jax(jnp.asarray(x, dtype=jnp.float64)), dtype=np.float64)

    if chi2_hess_jax is None:
        chi2_hess_jax = jax.jit(jax.hessian(chi2))

    def chi2_hess_np(x):
        return np.asarray(chi2_hess_jax(jnp.asarray(x, dtype=jnp.float64)), dtype=np.float64)

    @jax.jit
    def constraint_value_jax(fp, fixed_value):
        return target_func(fitted_params_to_params(fp)) - fixed_value

    @jax.jit
    def scaled_constraint_value_jax(fp, fixed_value):
        return constraint_value_jax(fp, fixed_value) / target_scale

    scaled_constraint_grad_jax = jax.jit(jax.grad(lambda fp, fixed_value: scaled_constraint_value_jax(fp, fixed_value)))
    scaled_constraint_hess_jax = jax.jit(
        jax.hessian(lambda fp, fixed_value: scaled_constraint_value_jax(fp, fixed_value))
    )

    last_x = fp_best.copy()
    diagnostics = []

    def project_start(start, v, tol=cons_tol):
        x = np.asarray(start, dtype=np.float64).copy()
        for _ in range(20):
            if not np.all(np.isfinite(x)):
                break
            c = _as_float(scaled_constraint_value_jax(jnp.asarray(x), v))
            if not np.isfinite(c):
                break
            if abs(c) <= tol:
                return x
            g = np.asarray(scaled_constraint_grad_jax(jnp.asarray(x), v), dtype=np.float64)
            denom = float(np.dot(g, g))
            if not np.all(np.isfinite(g)) or not np.isfinite(denom) or denom == 0.0:
                break
            with np.errstate(over="ignore", invalid="ignore"):
                step = (c / denom) * g
            if not np.all(np.isfinite(step)):
                break
            x_next = x - step
            if not np.all(np.isfinite(x_next)):
                break
            x = x_next
        return x

    def profile(v):
        nonlocal last_x
        v = float(v)
        t_profile_start = time.perf_counter()
        def make_constraint(exact_hess=False):
            kwargs = {}
            if exact_hess:
                kwargs["hess"] = lambda x, multiplier: np.asarray(
                    multiplier[0] * scaled_constraint_hess_jax(jnp.asarray(x), v),
                    dtype=np.float64,
                )
            return NonlinearConstraint(
                fun=lambda x: _as_float(scaled_constraint_value_jax(jnp.asarray(x), v)),
                lb=0.0,
                ub=0.0,
                jac=lambda x: np.asarray(scaled_constraint_grad_jax(jnp.asarray(x), v), dtype=np.float64),
                **kwargs,
            )

        def scaled_violation(x):
            return abs(_as_float(constraint_value_jax(jnp.asarray(x), v))) / target_scale

        def projected_grad_norm(x):
            gradient = _projected_gradient(x)
            return np.inf if gradient is None else float(np.linalg.norm(gradient))

        def _projected_gradient(x):
            x = np.asarray(x, dtype=np.float64)
            g_obj = chi2_grad_np(x)
            g_con = np.asarray(scaled_constraint_grad_jax(jnp.asarray(x), v), dtype=np.float64)
            denom = float(np.dot(g_con, g_con))
            if not np.isfinite(denom) or denom == 0.0:
                return None
            return g_obj - (float(np.dot(g_obj, g_con)) / denom) * g_con

        def projected_descent_improvement(x):
            """Return the best local feasible chi2 decrease found from x.

            This is deliberately a fallback certificate, not a replacement for
            KKT stationarity. It guards the nonsmooth/high-curvature BR/BRU
            cases where a large fitted-coordinate projected gradient can imply
            only sub-tolerance movement in chi2.
            """
            x = np.asarray(x, dtype=np.float64)
            if not np.all(np.isfinite(x)):
                return np.inf
            base_fun = chi2_np(x)
            if not np.isfinite(base_fun) or scaled_violation(x) > cons_tol:
                return np.inf
            g_proj = _projected_gradient(x)
            if g_proj is None:
                return np.inf
            g_proj_norm = float(np.linalg.norm(g_proj))
            if not np.isfinite(g_proj_norm):
                return np.inf
            if g_proj_norm == 0.0:
                return 0.0

            direction = -g_proj / g_proj_norm
            coord_scale = max(float(np.linalg.norm(x)), 1.0)
            best_decrease = 0.0
            for exponent in range(-14, -3):
                step = coord_scale * (10.0 ** exponent)
                trial = project_start(x + step * direction, v, tol=min(cons_tol, 1e-10))
                if not np.all(np.isfinite(trial)) or scaled_violation(trial) > cons_tol:
                    continue
                trial_fun = chi2_np(trial)
                if np.isfinite(trial_fun):
                    best_decrease = max(best_decrease, base_fun - trial_fun)
            return float(best_decrease)

        def result_ok(opt_result):
            cached = getattr(opt_result, "profile_result_ok", None)
            if cached is not None:
                return bool(cached)
            scaled_cviol = scaled_violation(opt_result.x)
            finite_result = np.isfinite(opt_result.fun)
            if not finite_result or scaled_cviol > cons_tol:
                opt_result.profile_result_ok = False
                return False
            g_norm = float(np.linalg.norm(chi2_grad_np(opt_result.x)))
            pg_norm = projected_grad_norm(opt_result.x)
            if pg_norm <= max(1e-3, 1e-6 * g_norm):
                opt_result.profile_descent_improvement = 0.0
                opt_result.profile_result_ok = True
                return True
            if not solver_options.use_descent_check:
                opt_result.profile_result_ok = False
                return False
            descent_improvement = projected_descent_improvement(opt_result.x)
            opt_result.profile_descent_improvement = descent_improvement
            if descent_improvement <= stationarity_fun_tol:
                method = str(getattr(opt_result, "profile_method", ""))
                if "descent-check" not in method:
                    opt_result.profile_method = f"{method}+descent-check" if method else "descent-check"
                opt_result.profile_result_ok = True
                return True
            opt_result.profile_result_ok = False
            return False

        def kkt_polish(opt_result):
            if not np.isfinite(opt_result.fun) or not np.all(np.isfinite(opt_result.x)):
                return opt_result
            x0 = project_start(opt_result.x, v)
            g_obj = chi2_grad_np(x0)
            g_con = np.asarray(scaled_constraint_grad_jax(jnp.asarray(x0), v), dtype=np.float64)
            denom = float(np.dot(g_con, g_con))
            if not np.isfinite(denom) or denom == 0.0:
                return opt_result
            lambda0 = -float(np.dot(g_obj, g_con)) / denom

            def residual(z):
                x = np.asarray(z[:-1], dtype=np.float64)
                lam = float(z[-1])
                stationarity = (
                    chi2_grad_np(x)
                    + lam * np.asarray(scaled_constraint_grad_jax(jnp.asarray(x), v), dtype=np.float64)
                )
                return np.r_[stationarity, _as_float(scaled_constraint_value_jax(jnp.asarray(x), v))]

            try:
                polished = root(residual, np.r_[x0, lambda0], method="hybr", options={"maxfev": 5000})
            except (FloatingPointError, ValueError, np.linalg.LinAlgError):
                return opt_result

            x_polished = np.asarray(polished.x[:-1], dtype=np.float64)
            if not np.all(np.isfinite(x_polished)) or scaled_violation(x_polished) > cons_tol:
                return opt_result
            fun_polished = chi2_np(x_polished)
            if not np.isfinite(fun_polished):
                return opt_result
            if fun_polished > opt_result.fun + max(1e-7, 1e-9 * abs(opt_result.fun)):
                return opt_result
            profile_method = getattr(opt_result, "profile_method", "")
            if profile_method:
                profile_method = f"{profile_method}+KKT"
            else:
                profile_method = "KKT"
            return OptimizeResult(
                x=x_polished,
                fun=fun_polished,
                success=bool(polished.success),
                status=getattr(polished, "status", None),
                message=f"KKT polish: {polished.message}",
                nfev=getattr(polished, "nfev", None),
                profile_method=profile_method,
            )

        def maybe_keep_feasible_start(opt_result, x0):
            x0_fun = chi2_np(x0)
            if not np.isfinite(x0_fun) or scaled_violation(x0) > cons_tol:
                return opt_result
            if (not np.isfinite(opt_result.fun)) or (
                x0_fun <= opt_result.fun + max(1e-8, 1e-10 * abs(opt_result.fun))
            ):
                is_best_fit_start = (
                    np.linalg.norm(np.asarray(x0, dtype=np.float64) - fp_best)
                    <= 1e-10 * max(np.linalg.norm(fp_best), 1.0)
                )
                return OptimizeResult(
                    x=np.asarray(x0, dtype=np.float64),
                    fun=x0_fun,
                    success=is_best_fit_start,
                    message="retained lower-chi2 feasible projected start",
                    profile_method="projected-start",
                )
            return opt_result

        def solve_from(x0):
            candidates = []
            # Keep the tested order: SLSQP first, exact-Hessian fallback second.
            for method, enabled, polish in [
                ("SLSQP", solver_options.use_slsqp, solver_options.polish_slsqp),
                ("trust-constr", solver_options.use_exact_hessian, True),
            ]:
                if method == "trust-constr" and not enabled:
                    continue
                result = OptimizeResult(x=np.asarray(x0), fun=chi2_np(x0), success=False,
                                        message="SLSQP disabled", profile_method="projected-start")
                if enabled:
                    exact = method == "trust-constr"
                    kwargs = {"hess": chi2_hess_np} if exact else {}
                    options = ({"gtol": 1e-10, "xtol": 1e-10, "maxiter": 2000} if exact else
                               {"ftol": 1e-10, "maxiter": 2000, **(optimizer_options or {})})
                    try:
                        result = minimize(chi2_np, np.asarray(x0), method=method, jac=chi2_grad_np,
                                          constraints=[make_constraint(exact_hess=exact)],
                                          options=options, **kwargs)
                    except (FloatingPointError, ValueError, np.linalg.LinAlgError) as exc:
                        result = OptimizeResult(x=np.asarray(x0), fun=chi2_np(x0), success=False,
                                                message=f"{method} failed: {exc}")
                    result = maybe_keep_feasible_start(result, x0)
                    if not getattr(result, "profile_method", ""):
                        result.profile_method = "trust-constr-exact-hess" if exact else method
                    if polish and solver_options.use_kkt_polish:
                        result = kkt_polish(result)
                candidates.append(result)
                if result_ok(result):
                    return result
            return min(candidates, key=lambda candidate: candidate.fun if np.isfinite(candidate.fun) else np.inf)

        starts = [project_start(last_x, v)]
        best_start = project_start(fp_best, v)
        if np.linalg.norm(best_start - starts[0]) > 1e-12 * max(np.linalg.norm(best_start), np.linalg.norm(starts[0]), 1.0):
            starts.append(best_start)

        candidates = [solve_from(start) for start in starts]
        ok_candidates = [candidate for candidate in candidates if result_ok(candidate)]
        if ok_candidates:
            result = min(ok_candidates, key=lambda candidate: candidate.fun)
        else:
            result = min(candidates, key=lambda candidate: candidate.fun if np.isfinite(candidate.fun) else np.inf)

        cviol = abs(_as_float(constraint_value_jax(jnp.asarray(result.x), v)))
        scaled_cviol = cviol / target_scale
        finite = np.isfinite(result.fun)
        ok = result_ok(result)
        g_norm = float(np.linalg.norm(chi2_grad_np(result.x))) if finite else np.inf
        pg_norm = projected_grad_norm(result.x) if finite else np.inf
        point = ProfilePoint(
            v,
            float(result.fun) if finite else np.inf,
            ok,
            cviol,
            str(result.message),
            scaled_constraint_violation=scaled_cviol,
            projected_grad_norm=pg_norm,
            objective_grad_norm=g_norm,
            method=str(getattr(result, "profile_method", "")),
            nfev=getattr(result, "nfev", None),
            nit=getattr(result, "nit", None),
            runtime_sec=time.perf_counter() - t_profile_start,
            descent_improvement=getattr(result, "profile_descent_improvement", None),
        )
        diagnostics.append(point)
        profile.last_point = point
        if not ok:
            raise RuntimeError(
                f"Profile minimization failed for {target_name}={v}: success={result.success}, "
                f"finite={finite}, constraint_violation={cviol:.3g}, "
                f"scaled_constraint_violation={scaled_cviol:.3g}, "
                f"projected_grad_norm={pg_norm:.3g}, message={result.message}"
            )
        last_x = np.asarray(result.x, dtype=np.float64)
        return point.chi2

    profile.diagnostics = diagnostics
    profile.last_point = None
    return profile


def build_coordinate_profile_chi2(chi2, param_values, primary_idx, target_name="target"):
    """Fix one physical coordinate and minimize over the remaining coordinates."""
    base = np.delete(np.asarray(param_values, dtype=float), primary_idx)
    last = base.copy()
    value_and_grad = jax.jit(jax.value_and_grad(chi2))
    points = []

    def profile(value):
        nonlocal last
        value = float(value)
        if not len(base):
            result = float(chi2(jnp.array([value])))
            point = ProfilePoint(value, result, np.isfinite(result), 0.0, "direct")
        else:
            def full(nuisance):
                return jnp.asarray(np.insert(nuisance, primary_idx, value))

            def objective(nuisance):
                return float(chi2(full(nuisance)))

            def objective_gradient(nuisance):
                val, grad = value_and_grad(full(nuisance))
                return float(val), np.delete(np.asarray(grad), primary_idx)

            def stationary(fun, x):
                return np.linalg.norm(objective_gradient(x)[1]) <= 1e-5 * max(abs(fun), 1.0)

            starts = [last]
            if np.linalg.norm(base-last) > 1e-12 * max(np.linalg.norm(base), np.linalg.norm(last), 1.0):
                starts.append(base)
            candidates, messages = [], []
            for start in starts:
                initial = objective(start)
                if np.isfinite(initial):
                    # Feasibility is not an optimization-success certificate.
                    candidates.append((initial, start, False, "feasible start"))
                use_simplex = True
                try:
                    fit = minimize(objective_gradient, start, jac=True, method="BFGS",
                                   options={"maxiter": max(1000, 500*len(start)), "gtol": 1e-8})
                    messages.append(f"BFGS: {fit.message}")
                    if np.isfinite(fit.fun):
                        candidates.append((float(fit.fun), np.asarray(fit.x), bool(fit.success), str(fit.message)))
                        use_simplex = not (fit.success or stationary(fit.fun, fit.x))
                except (FloatingPointError, ValueError, np.linalg.LinAlgError) as exc:
                    messages.append(f"BFGS: {exc}")
                if use_simplex:
                    nm_start = candidates[-1][1] if candidates else start
                    try:
                        fit = minimize(objective, nm_start, method="Nelder-Mead", options={
                            "maxiter": max(1000, 500*len(start)), "fatol": 1e-8, "adaptive": True,
                            "xatol": max(np.linalg.norm(nm_start)*1e-10, np.finfo(float).eps)})
                        messages.append(f"Nelder-Mead: {fit.message}")
                        if np.isfinite(fit.fun):
                            candidates.append((float(fit.fun), np.asarray(fit.x), bool(fit.success), str(fit.message)))
                    except (FloatingPointError, ValueError, np.linalg.LinAlgError) as exc:
                        messages.append(f"Nelder-Mead: {exc}")
            if not candidates:
                raise RuntimeError(f"No finite profile candidate for {target_name}: {'; '.join(messages)}")
            minimum = min(c[0] for c in candidates)
            tolerance = max(1e-6, 1e-8*abs(minimum))
            successful = [c for c in candidates if c[2] and c[0] <= minimum+tolerance]
            fun, x, success, message = min(successful or candidates, key=lambda c: c[0])
            grad_norm = float(np.linalg.norm(objective_gradient(x)[1]))
            ok = success or grad_norm <= 1e-5*max(abs(fun), 1.0)
            point = ProfilePoint(value, fun, bool(ok), 0.0,
                                 f"{message}; nuisance_grad_norm={grad_norm:.3g}")
            if ok:
                last = x
        points.append(point)
        profile.last_point = point
        if not point.success:
            raise RuntimeError(f"Profile minimization failed for {target_name}={value}: {point.message}")
        return point.chi2

    profile.diagnostics = points
    profile.last_point = None
    return profile

