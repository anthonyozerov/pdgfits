from decimal import Decimal
from dataclasses import asdict, dataclass
import time
from typing import Callable

import jax
from jax import numpy as jnp
import numpy as np
from scipy.optimize import linprog
from pdgfits.build_chi2 import build_chi2
from pdgfits.param_maps import covariance_factor, get_decay_info, physical_coordinates
from pdgfits.profiles import (
    ProfilePoint, ProfileSolverOptions, build_constrained_profile_chi2,
    build_coordinate_profile_chi2,
)


def symmetrize(y, val, error_n, error_p):
    resid = y - val
    error_min = jnp.minimum(error_n, error_p)
    error_max = jnp.maximum(error_n, error_p)
    sum_e = error_n + error_p
    diff_e = error_p - error_n
    prod_e2 = 2 * error_n * error_p

    error_between = (prod_e2 - resid * diff_e) / sum_e
    return jnp.clip(error_between, error_min, error_max)


def _profile_point_dict(point):
    if point is None:
        return {}
    return {key: (float(value) if np.isfinite(value) else None)
            if isinstance(value, (float, np.floating)) else value
            for key, value in asdict(point).items()}


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
        chi2 = float(self.profile_chi2(value))
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
    is_bound: bool = False

    def diagnostics(self) -> dict:
        return {
            f"{self.side}_endpoint": self.endpoint,
            f"{self.side}_chi2": self.chi2,
            f"{self.side}_residual": self.residual,
            f"{self.side}_error": self.error,
            f"{self.side}_profile_point": _profile_point_dict(self.point),
            f"{self.side}_is_bound": self.is_bound,
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


def _coerce_profile_problem(
    profile_chi2,
    *,
    target_name="target",
    target_value=None,
    chi2_min=None,
    lower_initial=None,
    upper_initial=None,
):
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
    limits=(-np.inf, np.inf),
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
    if not np.isfinite([target, val, lb, ub, residual_tol]).all() or residual_tol <= 0:
        raise ValueError("Profile search inputs must be finite and residual_tol positive")
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
        if not point.success or not np.isfinite(point.chi2):
            raise RuntimeError(f"Unsuccessful profile at {x}: chi2={point.chi2}, {point.message}")
        eval_cache[x] = point
        return point

    def solve_side(lo, hi, upper):
        side = "upper" if upper else "lower"
        fixed = lo if upper else hi
        moving = hi if upper else lo
        boundary = limits[1] if upper else limits[0]
        moving = min(moving, boundary) if upper else max(moving, boundary)

        def boundary_endpoint(point):
            return ProfileEndpoint(side, boundary, abs(boundary-val), point.chi2,
                                   point.chi2-target, point, side_counts[side].copy(), True)

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
                except (RuntimeError, FloatingPointError) as exc:
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
            if moving == boundary:
                return boundary_endpoint(moving_point)
            n_expand += 1
            inner = moving
            moving = val + 2.0 * (moving - val)
            moving = min(moving, boundary) if upper else max(moving, boundary)
            moving, moving_point = bracket_eval(inner, moving)
        if n_expand == max_iter:
            raise ValueError(f"Expanding {'ub' if upper else 'lb'} failed after {max_iter} iterations")
        lo2, hi2 = (fixed, moving) if upper else (moving, fixed)
        qlo, qhi = (float(chi2_min), moving_point.chi2) if upper else (moving_point.chi2, float(chi2_min))
        mid = None
        mid_point = None
        previous_step = np.inf
        for _ in range(max_iter):
            # sqrt(Delta Q) is linear for a quadratic profile. Interpolate in
            # that coordinate, retaining the bracket and a bisection safeguard.
            left = np.sqrt(max(qlo-float(chi2_min), 0.))
            right = np.sqrt(max(qhi-float(chi2_min), 0.))
            fraction = (1-left)/(right-left) if right != left else .5
            # A steep bracket endpoint can make secant steps retain almost
            # the entire bracket indefinitely. Bisect when successive steps
            # fail to contract, preserving fast quadratic convergence.
            candidate = lo2 + fraction*(hi2-lo2)
            if (not .001 < fraction < .999
                    or (mid is not None and abs(candidate-mid) > .5*previous_step)):
                candidate = .5*(lo2+hi2)
            previous_step = hi2-lo2 if mid is None else abs(candidate-mid)
            mid = candidate
            mid_point = checked_point(mid, side=side, phase="bisect")
            if mid_point.chi2 < target:
                if upper: lo2, qlo = mid, mid_point.chi2
                else: hi2, qhi = mid, mid_point.chi2
            else:
                if upper: hi2, qhi = mid, mid_point.chi2
                else: lo2, qlo = mid, mid_point.chi2
            if abs(mid_point.chi2 - target) <= residual_tol:
                break
        endpoint = mid
        # The endpoint is exactly the final bisection point, so this is the
        # profile value that verifies profile_chi2(endpoint) = chi2_min + 1.
        endpoint_point = mid_point
        endpoint_chi2 = endpoint_point.chi2
        resid = endpoint_chi2 - target
        if abs(resid) > residual_tol:
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
        search_method="sqrt-secant",
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
    root_result = find_profile_root(
        profile_chi2, val, chi2_min, lb, ub, verbose=verbose,
        residual_tol=residual_tol, contract_brackets=contract_brackets,
    )
    binary_search_error.last_root = root_result
    binary_search_error.last_diagnostics = root_result.diagnostics()
    return float(root_result.upper.error), float(root_result.lower.error)


binary_search_error.last_diagnostics = None


def _cached_value_and_grad(function):
    """Share a solver's value and derivative call at exactly the same point."""
    last_x, last_result = None, None
    def evaluate(x):
        nonlocal last_x, last_result
        if last_x is None or not np.array_equal(x, last_x):
            last_result = tuple(np.asarray(v) for v in function(x))
            last_x = np.array(x, copy=True)
        return last_result
    return evaluate
binary_search_error.last_root = None


def _physical_profile_chart(fit):
    """Replace open sigmoid/softmax coordinates by their closed physical domain.

    A sum-to-one fit uses an affine basis with one dependent fraction. This
    reaches zero exactly and introduces no penalty approximation.
    """
    parameters = fit['parameters']
    particles, decay = get_decay_info(parameters)
    chart = physical_coordinates(fit)
    if chart is None:
        return None
    center, transform, bounds = chart
    sum_one = fit['algorithm'] != 'BRU' and len(particles) == 1
    mapping = jax.jit(lambda z: jnp.asarray(center) + jnp.asarray(transform) @ z)
    data = fit['meas_df']
    chi2, _, _, opened = build_chi2(jnp.asarray(data['value'].to_numpy()), fit['mu_adjust'],
                               jnp.asarray(data['error_n'].to_numpy()), jnp.asarray(data['error_p'].to_numpy()),
                               jnp.linalg.pinv(fit['corr_mat']), mapping)
    # Rebuilding a scaled fit must retain its input scales.
    if 'input_scales' in fit:
        chi2 = jax.jit(lambda z: opened(z, jnp.asarray(data['value'].to_numpy()), fit['input_scales']))
    limits = {parameters[i]: (0., 1.) for i in decay}
    bounded = set(limits)
    for node in fit['nodes']:
        rows = fit['rel_df'][fit['rel_df'].node == node]
        keys = set(rows['parameter_key'])
        if (not keys <= bounded or rows['coeff_parameter_key'].notna().any()
                or np.any(rows['coefficient'].astype(float) < 0)):
            continue
        limits[node] = (0., np.inf)
        kind = fit['fit_df'].set_index('node').loc[node, 'type']
        if kind == '/':
            terms = np.zeros((2, len(center)))
            for _, row in rows.iterrows():
                terms[int(row.summation)-1, parameters.index(row.parameter_key)] = float(row.coefficient)
            numerator, denominator = terms
            positive = denominator > 0
            # A ratio of nonnegative linear forms is a weighted average of
            # their coefficient ratios when the numerator has no extra terms.
            if positive.any() and np.all(numerator[~positive] == 0):
                ratios = numerator[positive]/denominator[positive]
                limits[node] = (float(ratios.min()), float(ratios.max()))
        if kind == '+':
            coefficients = np.zeros(len(center))
            for _, row in rows.iterrows():
                coefficients[parameters.index(row.parameter_key)] = float(row.coefficient)
            aeq = np.isin(np.arange(len(center)), decay)[None, :].astype(float) if sum_one else None
            kwargs = {'bounds': [(0, 1) if i in decay else (None, None) for i in range(len(center))],
                      'A_eq': aeq, 'b_eq': [1.] if sum_one else None}
            low = linprog(coefficients, **kwargs)
            high = linprog(-coefficients, **kwargs)
            if low.success and high.success:
                limits[node] = (float(low.fun), float(-high.fun))
    return chi2, mapping, np.zeros(transform.shape[1]), np.eye(transform.shape[1]), bounds, limits


def calc_asym_errors(fit, targets=None):
    nodes = fit['nodes']; parameters = fit['parameters']
    node_funcs = fit['node_funcs']; parameter_funcs = fit['parameter_funcs']
    fitted_params_to_params = fit['fitted_params_to_params']
    fitted_values = fit['fitted_values']; param_values = fit['param_values']
    chi2 = fit['chi2']; chi2_min = fit['chi2_min']; covariance = fit['covariance']
    chart = _physical_profile_chart(fit) if 'mu_adjust' in fit and covariance is not None else None
    linear_constraint, limits = None, {}
    if chart is not None:
        chi2, fitted_params_to_params, fitted_values, covariance, linear_constraint, limits = chart
    # Profile in approximately unit-uncertainty coordinates. The original
    # physical map and objective remain unchanged; only the optimizer's chart
    # changes. Zero-variance coordinates include Minuit's fixed softmax gauge.
    if covariance is not None and chart is None:
        transform = jnp.asarray(covariance_factor(covariance))
        center = jnp.asarray(fitted_values)
        original_chi2, original_map = chi2, fitted_params_to_params
        chi2 = jax.jit(lambda z: original_chi2(center + transform @ z))
        fitted_params_to_params = jax.jit(lambda z: original_map(center + transform @ z))
        fitted_values = jnp.zeros(transform.shape[1])
        covariance = np.eye(transform.shape[1])
    if targets is None:
        targets = sorted(list(set(parameters) | set(nodes)))
    functions = []
    valid_targets = []
    for target in targets:
        # A named physical parameter takes precedence over an auxiliary
        # relationship carrying the same ID (the eta width is such a case).
        if target in parameters:
            functions.append(parameter_funcs[parameters.index(target)])
        elif target in nodes:
            functions.append(node_funcs[nodes.index(target)])
        else:
            print(f'Warning: {target} not found in nodes or parameters, skipping.')
            continue
        valid_targets.append(target)
    if not functions:
        return {}
    # Compile values and derivatives together for the entire target list.
    # A runtime target index reuses the Hessian compilation on hard fallbacks.
    def target_vector(fp):
        params = fitted_params_to_params(fp)
        return jnp.stack([function(params) for function in functions])
    target_values = jax.jit(target_vector)
    target_jacobian = jax.jit(jax.jacfwd(target_vector))
    target_hessian = jax.jit(jax.hessian(lambda fp, index: target_vector(fp)[index]))
    target_value_grad = jax.jit(jax.value_and_grad(lambda fp, index: target_vector(fp)[index]))
    centers = np.asarray(target_values(fitted_values))
    target_jac = np.asarray(target_jacobian(fitted_values))
    chi2_grad_jax = jax.jit(jax.grad(chi2))
    chi2_hess_jax = jax.jit(jax.hessian(chi2))
    chi2_value_and_grad = _cached_value_and_grad(jax.jit(jax.value_and_grad(chi2)))
    base_chi2 = float(chi2(jnp.asarray(fitted_values, dtype=jnp.float64)))
    if base_chi2 > chi2_min + 1e-5:
        raise RuntimeError(f'chi2 at fitted optimum {base_chi2} > chi2_min {chi2_min}')
    mapped = np.asarray(fitted_params_to_params(fitted_values))
    if not np.allclose(mapped, np.asarray(param_values), rtol=1e-10, atol=1e-14):
        raise RuntimeError('Profile coordinate map changed the fitted parameters')
    if 'mu_adjust' in fit:
        check_prediction = jax.jit(fit['mu_adjust'])
        measurement_precision = np.linalg.pinv(fit['corr_mat'])
    results = {}
    for index, (target, target_func) in enumerate(zip(valid_targets, functions)):
        print(target)
        target_value = float(centers[index])
        if target in limits and limits[target][0] == limits[target][1]:
            results[target] = {'value': target_value, 'error_n': 0., 'error_p': 0.,
                               'lower_endpoint': target_value, 'upper_endpoint': target_value,
                               'lower_residual': -1., 'upper_residual': -1.,
                               'lower_is_bound': True, 'upper_is_bound': True,
                               'search_method': 'constant target'}
            continue
        if covariance is not None:
            target_std = float(np.sqrt(target_jac[index] @ covariance @ target_jac[index]))
        else:
            target_std = max(abs(target_value), 1.0) * 1e-3
        if not np.isfinite(target_std) or target_std <= 0:
            target_std = max(abs(target_value), 1.0) * 1e-3
        target_evaluate = _cached_value_and_grad(lambda fp, i=index: target_value_grad(fp, i))
        profile_chi2 = build_constrained_profile_chi2(
            chi2, fitted_params_to_params, target_func, fitted_values,
            target_scale=target_std, target_name=target,
            chi2_grad_jax=chi2_grad_jax,
            chi2_hess_jax=chi2_hess_jax,
            chi2_value_and_grad=chi2_value_and_grad,
            linear_constraint=linear_constraint,
            target_derivatives=(lambda fp, evaluate=target_evaluate: float(evaluate(fp)[0]),
                                lambda fp, evaluate=target_evaluate: evaluate(fp)[1],
                                lambda fp, i=index: np.asarray(target_hessian(fp, i))),
        )
        lb = target_value - 2 * target_std
        ub = target_value + 2 * target_std
        root_result = find_profile_root(profile_chi2, target_value, chi2_min, lb, ub,
                                        limits=limits.get(target, (-np.inf, np.inf)))
        upper_err, lower_err = root_result.upper.error, root_result.lower.error
        diag = root_result.diagnostics()
        for side, endpoint in [('upper', root_result.upper), ('lower', root_result.lower)]:
            if endpoint.point.fitted_values is not None:
                physical = np.asarray(fitted_params_to_params(jnp.asarray(endpoint.point.fitted_values)))
                diag[f'{side}_parameters'] = physical.tolist()
                if 'mu_adjust' in fit:
                    measurements = fit['meas_df']
                    scales = np.asarray(fit.get('input_scales', np.ones(len(measurements))))
                    residual = measurements['value'].to_numpy()-np.asarray(check_prediction(physical))
                    lower = measurements['error_n'].to_numpy()*scales
                    upper = measurements['error_p'].to_numpy()*scales
                    sigma = np.clip((2*lower*upper-residual*(upper-lower))/(lower+upper),
                                    np.minimum(lower, upper), np.maximum(lower, upper))
                    normalized = residual/sigma
                    checked_q = float(normalized @ measurement_precision @ normalized)
                    if abs(checked_q-endpoint.chi2) > 1e-7*max(abs(checked_q), 1.):
                        raise RuntimeError(f'Independent physical-coordinate objective check failed for {target}')
                    diag[f'{side}_checked_chi2'] = checked_q
        results[target] = {"value": target_value, "error_p": upper_err, "error_n": lower_err, **diag}
        print(f'{target}: {Decimal(target_value):.5E} + {Decimal(upper_err):.2E} - {Decimal(lower_err):.2E} residuals=({diag["upper_residual"]:.2g},{diag["lower_residual"]:.2g})')
    return results
