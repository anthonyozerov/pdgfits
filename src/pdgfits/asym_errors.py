from decimal import Decimal
from dataclasses import dataclass
import time
from typing import Callable, Protocol

import jax
from jax import numpy as jnp
import numpy as np
from scipy.optimize import NonlinearConstraint, OptimizeResult, minimize, root


def symmetrize(y, val, error_n, error_p):
    resid = y - val
    error_min = jnp.minimum(error_n, error_p)
    error_max = jnp.maximum(error_n, error_p)
    sum_e = error_n + error_p
    diff_e = error_p - error_n
    prod_e2 = 2 * error_n * error_p

    error_between = (prod_e2 - resid * diff_e) / sum_e
    return jnp.clip(error_between, error_min, error_max)


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


class ProfileProblem(Protocol):
    target_name: str
    target_value: float | None
    chi2_min: float | None
    lower_initial: float | None
    upper_initial: float | None

    def evaluate(self, value: float) -> ProfilePoint:
        ...


def _profile_point_dict(point: ProfilePoint | None) -> dict:
    if point is None:
        return {}

    def finite_or_none(value):
        if value is None:
            return None
        out = float(value)
        return out if np.isfinite(out) else None

    return {
        "value": finite_or_none(point.value),
        "chi2": finite_or_none(point.chi2),
        "success": point.success,
        "constraint_violation": finite_or_none(point.constraint_violation),
        "scaled_constraint_violation": finite_or_none(point.scaled_constraint_violation),
        "projected_grad_norm": finite_or_none(point.projected_grad_norm),
        "objective_grad_norm": finite_or_none(point.objective_grad_norm),
        "method": point.method,
        "nfev": point.nfev,
        "nit": point.nit,
        "runtime_sec": finite_or_none(point.runtime_sec),
        "descent_improvement": finite_or_none(point.descent_improvement),
        "message": point.message,
    }


def _matches_profile_value(point: ProfilePoint, value: float, chi2: float) -> bool:
    value_scale = max(abs(float(value)), 1.0)
    chi2_scale = max(abs(float(chi2)), 1.0)
    return (
        abs(float(point.value) - float(value)) <= 1e-10 * value_scale
        and abs(float(point.chi2) - float(chi2)) <= 1e-8 * chi2_scale
    )


@dataclass
class CallableProfileProblem:
    profile_chi2: Callable[[float], float]
    target_name: str = "target"
    target_value: float | None = None
    chi2_min: float | None = None
    lower_initial: float | None = None
    upper_initial: float | None = None

    def evaluate(self, value: float) -> ProfilePoint:
        value = float(value)
        chi2 = _as_float(self.profile_chi2(value))
        point = getattr(self.profile_chi2, "last_point", None)
        if isinstance(point, ProfilePoint) and _matches_profile_value(point, value, chi2):
            return point
        diagnostics = getattr(self.profile_chi2, "diagnostics", None)
        if diagnostics:
            point = diagnostics[-1]
            if isinstance(point, ProfilePoint) and _matches_profile_value(point, value, chi2):
                return point
        return ProfilePoint(
            value=value,
            chi2=chi2,
            success=True,
            constraint_violation=np.nan,
            message="callable profile returned a finite chi2 without structured point diagnostics",
            method="callable",
        )


@dataclass
class ProfileEndpoint:
    side: str
    endpoint: float
    error: float
    chi2: float
    residual: float
    point: ProfilePoint
    counts: dict

    def diagnostics(self) -> dict:
        return {
            f"{self.side}_endpoint": self.endpoint,
            f"{self.side}_chi2": self.chi2,
            f"{self.side}_residual": self.residual,
            f"{self.side}_error": self.error,
            f"{self.side}_profile_point": _profile_point_dict(self.point),
        }


@dataclass
class ProfileRoot:
    target_value: float
    target_chi2: float
    upper: ProfileEndpoint
    lower: ProfileEndpoint
    search_method: str
    function_evals: int
    cache_hits: int
    side_counts: dict
    runtime_sec: float

    def diagnostics(self) -> dict:
        diagnostics = {
            "target": self.target_chi2,
            "search_method": self.search_method,
            "function_evals": self.function_evals,
            "cache_hits": self.cache_hits,
            "side_counts": self.side_counts,
            "root_runtime_sec": self.runtime_sec,
            "target_value": self.target_value,
        }
        diagnostics.update(self.upper.diagnostics())
        diagnostics.update(self.lower.diagnostics())
        return diagnostics


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

    constraint_grad_jax = jax.jit(jax.grad(lambda fp, fixed_value: constraint_value_jax(fp, fixed_value)))
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
            x = np.asarray(x, dtype=np.float64)
            g_obj = chi2_grad_np(x)
            g_con = np.asarray(scaled_constraint_grad_jax(jnp.asarray(x), v), dtype=np.float64)
            denom = float(np.dot(g_con, g_con))
            if not np.isfinite(denom) or denom == 0.0:
                return np.inf
            g_proj = g_obj - (float(np.dot(g_obj, g_con)) / denom) * g_con
            return float(np.linalg.norm(g_proj))

        def kkt_stationarity_norm(x):
            return projected_grad_norm(x)

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
            if not np.isfinite(g_proj_norm) or g_proj_norm == 0.0:
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
            result = OptimizeResult(
                x=np.asarray(x0, dtype=np.float64),
                fun=chi2_np(x0),
                success=False,
                message="projected start only",
                profile_method="projected-start",
            )
            if solver_options.use_slsqp:
                try:
                    result = minimize(
                        chi2_np,
                        x0,
                        method="SLSQP",
                        jac=chi2_grad_np,
                        constraints=[make_constraint()],
                        options={"ftol": 1e-10, "maxiter": 2000, **(optimizer_options or {})},
                    )
                except (FloatingPointError, ValueError, np.linalg.LinAlgError) as exc:
                    result = OptimizeResult(
                        x=np.asarray(x0, dtype=np.float64),
                        fun=chi2_np(x0),
                        success=False,
                        message=f"SLSQP failed: {exc}",
                    )
                result = maybe_keep_feasible_start(result, x0)
                if not getattr(result, "profile_method", ""):
                    result.profile_method = "SLSQP"
                if solver_options.polish_slsqp and solver_options.use_kkt_polish:
                    result = kkt_polish(result)
            else:
                result = OptimizeResult(
                    x=np.asarray(x0, dtype=np.float64),
                    fun=chi2_np(x0),
                    success=False,
                    message="SLSQP disabled",
                    profile_method="projected-start",
                )
            candidates.append(result)
            if result_ok(result):
                return result

            if solver_options.use_exact_hessian:
                try:
                    exact_hess_result = minimize(
                        chi2_np,
                        np.asarray(x0, dtype=np.float64),
                        method="trust-constr",
                        jac=chi2_grad_np,
                        hess=chi2_hess_np,
                        constraints=[make_constraint(exact_hess=True)],
                        options={"gtol": 1e-10, "xtol": 1e-10, "maxiter": 2000},
                    )
                except (FloatingPointError, ValueError, np.linalg.LinAlgError) as exc:
                    exact_hess_result = OptimizeResult(
                        x=np.asarray(x0, dtype=np.float64),
                        fun=chi2_np(x0),
                        success=False,
                        message=f"trust-constr exact Hessian failed: {exc}",
                    )
                exact_hess_result = maybe_keep_feasible_start(exact_hess_result, x0)
                if not getattr(exact_hess_result, "profile_method", ""):
                    exact_hess_result.profile_method = "trust-constr-exact-hess"
                if solver_options.use_kkt_polish:
                    exact_hess_result = kkt_polish(exact_hess_result)
                candidates.append(exact_hess_result)
                if result_ok(exact_hess_result):
                    return exact_hess_result

            ok_candidates = [candidate for candidate in candidates if result_ok(candidate)]
            if ok_candidates:
                return min(ok_candidates, key=lambda candidate: candidate.fun)
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
        pg_norm = kkt_stationarity_norm(result.x) if finite else np.inf
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


def _coerce_profile_problem(
    profile_chi2,
    *,
    target_name="target",
    target_value=None,
    chi2_min=None,
    lower_initial=None,
    upper_initial=None,
) -> ProfileProblem:
    if hasattr(profile_chi2, "evaluate"):
        return profile_chi2
    return CallableProfileProblem(
        profile_chi2=profile_chi2,
        target_name=target_name,
        target_value=target_value,
        chi2_min=chi2_min,
        lower_initial=lower_initial,
        upper_initial=upper_initial,
    )


def find_profile_root(
    profile_problem,
    val=None,
    chi2_min=None,
    lb=None,
    ub=None,
    verbose=False,
    residual_tol=5e-3,
    contract_brackets=True,
):
    """Find and verify profile-likelihood endpoints for an explicit problem."""
    t_root_start = time.perf_counter()
    problem = _coerce_profile_problem(profile_problem)
    if val is None:
        val = getattr(problem, "target_value", None)
    if chi2_min is None:
        chi2_min = getattr(problem, "chi2_min", None)
    if lb is None:
        lb = getattr(problem, "lower_initial", None)
    if ub is None:
        ub = getattr(problem, "upper_initial", None)
    if val is None or chi2_min is None or lb is None or ub is None:
        raise TypeError("find_profile_root requires val, chi2_min, lb, and ub, either as arguments or problem attributes")

    target = float(chi2_min) + 1.0
    val = float(val); lb = float(lb); ub = float(ub)
    if not lb < val < ub:
        raise ValueError(f"Expected lb < val < ub, got lb={lb}, val={val}, ub={ub}")
    max_iter = 80
    eval_cache = {}
    eval_counts = {"function_evals": 0, "cache_hits": 0}
    side_counts = {
        "upper": {"function_evals": 0, "cache_hits": 0, "bracket_evals": 0, "bisect_evals": 0},
        "lower": {"function_evals": 0, "cache_hits": 0, "bracket_evals": 0, "bisect_evals": 0},
    }

    def checked_point(x, *, side, phase):
        x = float(x)
        side_count = side_counts[side]
        if x in eval_cache:
            eval_counts["cache_hits"] += 1
            side_count["cache_hits"] += 1
            return eval_cache[x]
        eval_counts["function_evals"] += 1
        side_count["function_evals"] += 1
        if phase in ("bracket", "bisect"):
            side_count[f"{phase}_evals"] += 1
        point = problem.evaluate(x)
        if not isinstance(point, ProfilePoint):
            raise TypeError(f"profile problem returned {type(point).__name__}, expected ProfilePoint")
        if not np.isfinite(point.chi2):
            raise RuntimeError(f"profile_chi2 returned non-finite value {point.chi2} at {x}")
        eval_cache[x] = point
        return point

    def solve_side(lo, hi, upper):
        side = "upper" if upper else "lower"
        fixed = lo if upper else hi
        moving = hi if upper else lo

        def bracket_eval(inner, outer):
            """Evaluate an outward bracket point, contracting if it is unreachable.

            BR/BRU maps can make a naive covariance-based +/-2 sigma bracket
            either hard to reach in one optimizer continuation step or outside
            the feasible target range.  Contracting toward the MLE keeps the
            search on the same side while preserving loud failure if no usable
            bracket exists.
            """
            last_exc = None
            candidate = float(outer)
            if not contract_brackets:
                return candidate, checked_point(candidate, side=side, phase="bracket")
            width_floor = max(
                abs(outer - inner) * 1e-12,
                np.finfo(float).eps * max(abs(inner), abs(outer), 1.0),
            )
            for _ in range(max_iter):
                try:
                    point = checked_point(candidate, side=side, phase="bracket")
                    return candidate, point
                except Exception as exc:  # noqa: BLE001 - report after contraction attempts
                    last_exc = exc
                    candidate = 0.5 * (float(inner) + candidate)
                    if abs(candidate - inner) <= width_floor:
                        break
            raise RuntimeError(
                f"Could not evaluate profile bracket on {'upper' if upper else 'lower'} side "
                f"between {inner} and {outer}"
            ) from last_exc

        n_expand = 0
        moving, moving_point = bracket_eval(fixed, moving)
        while moving_point.chi2 < target and n_expand < max_iter:
            n_expand += 1
            inner = moving
            moving = val + 2.0 * (moving - val)
            moving, moving_point = bracket_eval(inner, moving)
        if n_expand == max_iter:
            raise ValueError(f"Expanding {'ub' if upper else 'lb'} failed after {max_iter} iterations")
        lo2, hi2 = (fixed, moving) if upper else (moving, fixed)
        mid = None
        mid_point = None
        for _ in range(max_iter):
            mid = 0.5 * (lo2 + hi2)
            mid_point = checked_point(mid, side=side, phase="bisect")
            if mid_point.chi2 < target:
                if upper: lo2 = mid
                else: hi2 = mid
            else:
                if upper: hi2 = mid
                else: lo2 = mid
            if abs(mid_point.chi2 - target) <= residual_tol:
                break
        endpoint = mid
        # The endpoint is exactly the final bisection point, so this is the
        # profile value that verifies profile_chi2(endpoint) = chi2_min + 1.
        endpoint_point = mid_point
        endpoint_chi2 = endpoint_point.chi2
        resid = endpoint_chi2 - target
        if abs(resid) > max(2 * residual_tol, 1e-2):
            raise RuntimeError(
                f"Profile endpoint verification failed: endpoint={endpoint}, chi2={endpoint_chi2}, "
                f"target={target}, residual={resid}"
            )
        error = (endpoint - val) if upper else (val - endpoint)
        return ProfileEndpoint(
            side=side,
            endpoint=endpoint,
            error=error,
            chi2=endpoint_chi2,
            residual=resid,
            point=endpoint_point,
            counts=side_counts[side].copy(),
        )

    upper = solve_side(val, ub, True)
    lower = solve_side(lb, val, False)
    root_result = ProfileRoot(
        target_value=val,
        target_chi2=target,
        upper=upper,
        lower=lower,
        search_method="bisection",
        function_evals=eval_counts["function_evals"],
        cache_hits=eval_counts["cache_hits"],
        side_counts=side_counts,
        runtime_sec=time.perf_counter() - t_root_start,
    )
    if verbose:
        print(root_result.diagnostics())
    return root_result


def binary_search_error(
    profile_chi2,
    val,
    chi2_min,
    lb,
    ub,
    verbose=False,
    residual_tol=5e-3,
    contract_brackets=True,
):
    """Find profile-likelihood 1-sigma errors and verify endpoints hit Delta chi2=1."""
    problem = _coerce_profile_problem(
        profile_chi2,
        target_value=val,
        chi2_min=chi2_min,
        lower_initial=lb,
        upper_initial=ub,
    )
    root_result = find_profile_root(
        problem,
        val,
        chi2_min,
        lb,
        ub,
        verbose=verbose,
        residual_tol=residual_tol,
        contract_brackets=contract_brackets,
    )
    binary_search_error.last_root = root_result
    binary_search_error.last_diagnostics = root_result.diagnostics()
    return float(root_result.upper.error), float(root_result.lower.error)


binary_search_error.last_diagnostics = None
binary_search_error.last_root = None


def calc_asym_errors(fit, targets=None):
    nodes = fit['nodes']; parameters = fit['parameters']
    node_funcs = fit['node_funcs']; parameter_funcs = fit['parameter_funcs']
    fitted_params_to_params = fit['fitted_params_to_params']
    fitted_values = fit['fitted_values']; param_values = fit['param_values']
    chi2 = fit['chi2']; chi2_min = fit['chi2_min']; covariance = fit['covariance']
    if targets is None:
        targets = sorted(list(set(parameters) | set(nodes)))
    J = jax.jacobian(fitted_params_to_params)(fitted_values)
    param_cov = J @ covariance @ J.T if covariance is not None else None
    chi2_grad_jax = jax.jit(jax.grad(chi2))
    chi2_hess_jax = jax.jit(jax.hessian(chi2))
    base_chi2 = _as_float(chi2(jnp.asarray(fitted_values, dtype=jnp.float64)))
    if base_chi2 > chi2_min + 1e-5:
        raise RuntimeError(f'chi2 at fitted optimum {base_chi2} > chi2_min {chi2_min}')
    results = {}
    for target in targets:
        print(target)
        if target in nodes:
            target_func = node_funcs[nodes.index(target)]
        elif target in parameters:
            target_func = parameter_funcs[parameters.index(target)]
        else:
            print(f'Warning: {target} not found in nodes or parameters, skipping.')
            continue
        target_value = float(target_func(param_values))
        if param_cov is not None:
            J_target = jax.jacobian(target_func)(param_values)
            target_std = float(jnp.sqrt(J_target @ param_cov @ J_target.T))
        else:
            target_std = max(abs(target_value), 1.0) * 1e-3
        if not np.isfinite(target_std) or target_std <= 0:
            target_std = max(abs(target_value), 1.0) * 1e-3
        profile_chi2 = build_constrained_profile_chi2(
            chi2, fitted_params_to_params, target_func, fitted_values,
            target_scale=target_std, target_name=target,
            chi2_grad_jax=chi2_grad_jax,
            chi2_hess_jax=chi2_hess_jax,
        )
        base_value = float(target_func(fitted_params_to_params(fitted_values)))
        if abs(base_value - target_value) > 1e-8 * max(abs(target_value), 1.0):
            raise RuntimeError(f'target value mismatch at fitted optimum {base_value} != {target_value}')
        lb = target_value - 2 * target_std
        ub = target_value + 2 * target_std
        profile_problem = CallableProfileProblem(
            profile_chi2=profile_chi2,
            target_name=target,
            target_value=target_value,
            chi2_min=chi2_min,
            lower_initial=lb,
            upper_initial=ub,
        )
        upper_err, lower_err = binary_search_error(
            profile_problem, target_value, chi2_min, lb, ub,
        )
        diag = binary_search_error.last_diagnostics.copy()
        results[target] = {"value": target_value, "error_p": upper_err, "error_n": lower_err, **diag}
        print(f'{target}: {Decimal(target_value):.5E} + {Decimal(upper_err):.2E} - {Decimal(lower_err):.2E} residuals=({diag["upper_residual"]:.2g},{diag["lower_residual"]:.2g})')
    return results
