#!/usr/bin/env python
"""Round 11 notes-only BR/BRU chart-conditioning experiment.

This harness keeps the scientific chi2 model unchanged and changes only the
coordinates used for fixed-target profile minimization.  It compares production
arctan fitted coordinates against bounded physical, log-positive, and logit
charts for hard BR/BRU endpoints.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import gc
import json
import math
import signal
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import jax

jax.config.update("jax_enable_x64", True)

from jax import numpy as jnp
import numpy as np
from scipy.optimize import Bounds, NonlinearConstraint, OptimizeResult, minimize, root

from pdgfits.asym_errors import CallableProfileProblem, ProfilePoint, calc_asym_errors, find_profile_root
from pdgfits.fit import run_fit
from pdgfits.param_maps import get_decay_info


CONS_TOL = 1e-7
BOUND_TOL = 1e-10
STATIONARITY_FUN_TOL = 1e-4
MATERIAL_IMPROVEMENT = 1e-4
DECAY_LB = 1e-12
DECAY_UB = 1.0 - 1e-12


@dataclass(frozen=True)
class CaseSpec:
    case_id: str
    label: str
    target: str
    rationale: str


DEFAULT_CASES = [
    CaseSpec(
        case_id="fit_b0_s042b95",
        label="B0",
        target="S042B95",
        rationale="severely saturated BRU node target from rounds 8-10",
    ),
    CaseSpec(
        case_id="fit_b0s_s08637",
        label="B0S-BR",
        target="S086.37",
        rationale="less-saturated BRU physical-parameter comparator",
    ),
]

DEFAULT_CHARTS = ("logit", "log_positive", "direct_physical")


class ChartTimeout(TimeoutError):
    pass


def _alarm_handler(signum, frame):  # noqa: ARG001
    raise ChartTimeout("chart evaluation timed out")


def as_float(x: Any) -> float:
    return float(np.asarray(x, dtype=np.float64))


def finite_or_none(value: Any) -> Any:
    if value is None or isinstance(value, (bool, str, int)):
        return value
    try:
        out = float(value)
    except (TypeError, ValueError):
        return value
    return out if math.isfinite(out) else None


def json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [finite_or_none(x) for x in value.tolist()]
    if isinstance(value, np.generic):
        return finite_or_none(value.item())
    return finite_or_none(value)


def clean_message(message: Any) -> str:
    return " ".join(str(message).split())


def write_rows(path: Path, rows: list[dict[str, Any]], *, jsonl: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if jsonl:
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, default=json_default, sort_keys=True) + "\n")
        return
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: finite_or_none(row.get(key)) for key in fieldnames})


def target_func_for_fit(fit: dict[str, Any], target: str) -> Callable[[Any], Any]:
    if target in fit["nodes"]:
        return fit["node_funcs"][fit["nodes"].index(target)]
    if target in fit["parameters"]:
        return fit["parameter_funcs"][fit["parameters"].index(target)]
    raise KeyError(f"{target!r} not found in {fit['label']}")


def target_std_for_fit(fit: dict[str, Any], target_func: Callable[[Any], Any], target_value: float) -> float:
    covariance = fit.get("covariance")
    if covariance is None:
        return max(abs(float(target_value)), 1.0) * 1e-3
    try:
        fitted_values = fit["fitted_values"]
        param_values = fit["param_values"]
        jac_params = jax.jacobian(fit["fitted_params_to_params"])(fitted_values)
        param_cov = jac_params @ covariance @ jac_params.T
        jac_target = jax.jacobian(target_func)(param_values)
        target_var = as_float(jac_target @ param_cov @ jac_target.T)
        target_std = math.sqrt(target_var) if target_var >= 0 else math.nan
    except Exception:  # noqa: BLE001 - diagnostic fallback
        target_std = math.nan
    if not math.isfinite(target_std) or target_std <= 0:
        target_std = max(abs(float(target_value)), 1.0) * 1e-3
    return float(target_std)


def bounds_for_chart(n_params: int, decay_idxs: np.ndarray, chart: str) -> Bounds:
    lb = np.full(n_params, -np.inf, dtype=np.float64)
    ub = np.full(n_params, np.inf, dtype=np.float64)
    if len(decay_idxs) == 0:
        return Bounds(lb, ub)
    if chart == "direct_physical":
        lb[decay_idxs] = DECAY_LB
        ub[decay_idxs] = DECAY_UB
    elif chart == "log_positive":
        lb[decay_idxs] = math.log(DECAY_LB)
        ub[decay_idxs] = math.log(DECAY_UB)
    elif chart == "logit":
        pass
    else:
        raise ValueError(f"unknown chart {chart!r}")
    return Bounds(lb, ub)


def build_chart_maps(decay_idxs: np.ndarray, chart: str):
    decay_jidx = jnp.asarray(decay_idxs, dtype=jnp.int32)

    if chart == "direct_physical":

        @jax.jit
        def chart_to_params(z):
            return z

        @jax.jit
        def params_to_chart(params):
            return params

        return chart_to_params, params_to_chart

    if chart == "log_positive":

        @jax.jit
        def chart_to_params(z):
            return z.at[decay_jidx].set(jnp.exp(z[decay_jidx]))

        @jax.jit
        def params_to_chart(params):
            decay = jnp.clip(params[decay_jidx], DECAY_LB, DECAY_UB)
            return params.at[decay_jidx].set(jnp.log(decay))

        return chart_to_params, params_to_chart

    if chart == "logit":

        @jax.jit
        def chart_to_params(z):
            return z.at[decay_jidx].set(jax.nn.sigmoid(z[decay_jidx]))

        @jax.jit
        def params_to_chart(params):
            decay = jnp.clip(params[decay_jidx], DECAY_LB, DECAY_UB)
            logit = jnp.log(decay) - jnp.log1p(-decay)
            return params.at[decay_jidx].set(logit)

        return chart_to_params, params_to_chart

    raise ValueError(f"unknown chart {chart!r}")


def clip_to_bounds(x: np.ndarray, bounds: Bounds) -> np.ndarray:
    out = np.asarray(x, dtype=np.float64).copy()
    finite_lb = np.isfinite(bounds.lb)
    finite_ub = np.isfinite(bounds.ub)
    out[finite_lb] = np.maximum(out[finite_lb], bounds.lb[finite_lb])
    out[finite_ub] = np.minimum(out[finite_ub], bounds.ub[finite_ub])
    return out


def bound_violation(x: np.ndarray, bounds: Bounds) -> float:
    below = np.where(np.isfinite(bounds.lb), np.maximum(bounds.lb - x, 0.0), 0.0)
    above = np.where(np.isfinite(bounds.ub), np.maximum(x - bounds.ub, 0.0), 0.0)
    return float(max(np.max(below), np.max(above), 0.0))


def project_out_norm(grad: np.ndarray, rows: list[np.ndarray]) -> float:
    mat = np.vstack(rows)
    try:
        gram = mat @ mat.T
        coeff = np.linalg.pinv(gram) @ (mat @ grad)
        projected = grad - mat.T @ coeff
    except np.linalg.LinAlgError:
        return math.inf
    return float(np.linalg.norm(projected))


def distinct_start(starts: list[tuple[str, np.ndarray]], x: np.ndarray) -> bool:
    x = np.asarray(x, dtype=np.float64)
    for _, existing in starts:
        scale = max(float(np.linalg.norm(existing)), float(np.linalg.norm(x)), 1.0)
        if float(np.linalg.norm(existing - x)) <= 1e-12 * scale:
            return False
    return True


def chart_diagnostics(
    fit: dict[str, Any],
    decay_idxs: np.ndarray,
    chart_to_params: Callable[[Any], Any],
    z: np.ndarray | None,
) -> dict[str, Any]:
    if z is None:
        return {}
    z = np.asarray(z, dtype=np.float64)
    out: dict[str, Any] = {}
    try:
        params = np.asarray(chart_to_params(jnp.asarray(z, dtype=jnp.float64)), dtype=np.float64)
    except Exception:  # noqa: BLE001
        out["chart_eval_failed"] = True
        return out
    out["n_decay_params"] = int(len(decay_idxs))
    if len(decay_idxs) == 0:
        return out
    decay_params = params[decay_idxs]
    chart_decay = z[decay_idxs]
    out.update(
        {
            "min_decay_param": float(np.nanmin(decay_params)),
            "max_decay_param": float(np.nanmax(decay_params)),
            "max_abs_chart_decay_coord": float(np.nanmax(np.abs(chart_decay))),
            "decay_active_lower_1e8": int(np.sum(decay_params <= DECAY_LB + 1e-8)),
            "decay_active_upper_1e8": int(np.sum(decay_params >= DECAY_UB - 1e-8)),
        }
    )
    try:
        fitted = np.asarray(
            fit["params_to_fitted_params"](jnp.asarray(params, dtype=jnp.float64)),
            dtype=np.float64,
        )
        out["max_abs_original_fitted_decay_coord"] = float(np.nanmax(np.abs(fitted[decay_idxs])))
    except Exception:  # noqa: BLE001
        out["max_abs_original_fitted_decay_coord"] = None
    out["chart_saturation_warning"] = bool(
        (out["max_abs_original_fitted_decay_coord"] or 0.0) > 10.0
        or out["min_decay_param"] < 1e-6
        or out["max_decay_param"] > 1 - 1e-6
    )
    return out


def build_chart_profile(
    fit: dict[str, Any],
    target_func: Callable[[Any], Any],
    target_value: float,
    target_scale: float,
    *,
    chart: str,
    target_name: str,
) -> Callable[[float], float]:
    parameters = list(fit["parameters"])
    _, decay_idxs_raw = get_decay_info(parameters)
    decay_idxs = np.asarray(decay_idxs_raw, dtype=int)
    n_params = len(parameters)
    bounds = bounds_for_chart(n_params, decay_idxs, chart)
    chart_to_params, params_to_chart = build_chart_maps(decay_idxs, chart)
    z_best = np.asarray(params_to_chart(fit["param_values"]), dtype=np.float64)
    target_scale = abs(float(target_scale))
    if not math.isfinite(target_scale) or target_scale <= 0:
        target_scale = max(abs(float(target_value)), 1.0)

    @jax.jit
    def objective_jax(z):
        params = chart_to_params(z)
        return fit["chi2"](fit["params_to_fitted_params"](params))

    @jax.jit
    def target_chart_jax(z):
        return target_func(chart_to_params(z))

    @jax.jit
    def constraint_value_jax(z, fixed_value):
        return target_chart_jax(z) - fixed_value

    @jax.jit
    def scaled_constraint_value_jax(z, fixed_value):
        return constraint_value_jax(z, fixed_value) / target_scale

    value_grad_jax = jax.jit(jax.value_and_grad(objective_jax))
    hess_jax = jax.jit(jax.hessian(objective_jax))
    scaled_constraint_grad_jax = jax.jit(
        jax.grad(lambda z, fixed_value: scaled_constraint_value_jax(z, fixed_value))
    )
    scaled_constraint_hess_jax = jax.jit(
        jax.hessian(lambda z, fixed_value: scaled_constraint_value_jax(z, fixed_value))
    )

    def objective_np(z: np.ndarray) -> float:
        if bound_violation(np.asarray(z, dtype=np.float64), bounds) > 1e-8:
            return math.inf
        try:
            value = as_float(objective_jax(jnp.asarray(z, dtype=jnp.float64)))
        except Exception:  # noqa: BLE001
            return math.inf
        return value if math.isfinite(value) else math.inf

    def objective_grad_np(z: np.ndarray) -> np.ndarray:
        value, grad = value_grad_jax(jnp.asarray(z, dtype=jnp.float64))
        if not math.isfinite(float(value)):
            raise FloatingPointError("non-finite chart objective")
        grad = np.asarray(grad, dtype=np.float64)
        if not np.all(np.isfinite(grad)):
            raise FloatingPointError("non-finite chart objective gradient")
        return grad

    def objective_hess_np(z: np.ndarray) -> np.ndarray:
        hess = np.asarray(hess_jax(jnp.asarray(z, dtype=jnp.float64)), dtype=np.float64)
        if not np.all(np.isfinite(hess)):
            raise FloatingPointError("non-finite chart objective Hessian")
        return hess

    def constraint_grad_np(z: np.ndarray, v: float) -> np.ndarray:
        grad = np.asarray(
            scaled_constraint_grad_jax(jnp.asarray(z, dtype=jnp.float64), float(v)),
            dtype=np.float64,
        )
        if not np.all(np.isfinite(grad)):
            raise FloatingPointError("non-finite chart constraint gradient")
        return grad

    last_z = z_best.copy()
    diagnostics: list[ProfilePoint] = []
    details: list[dict[str, Any]] = []

    def scaled_violation(z: np.ndarray, v: float) -> float:
        return abs(as_float(constraint_value_jax(jnp.asarray(z, dtype=jnp.float64), float(v)))) / target_scale

    def raw_violation(z: np.ndarray, v: float) -> float:
        return abs(as_float(constraint_value_jax(jnp.asarray(z, dtype=jnp.float64), float(v))))

    def project_start(start: np.ndarray, v: float, tol: float = CONS_TOL) -> np.ndarray:
        z = clip_to_bounds(start, bounds)
        for _ in range(40):
            if not np.all(np.isfinite(z)):
                break
            try:
                c = as_float(scaled_constraint_value_jax(jnp.asarray(z, dtype=jnp.float64), float(v)))
            except Exception:  # noqa: BLE001
                break
            if not math.isfinite(c) or abs(c) <= tol:
                return z
            try:
                g = constraint_grad_np(z, v)
            except Exception:  # noqa: BLE001
                break
            denom = float(np.dot(g, g))
            if not math.isfinite(denom) or denom == 0.0:
                break
            step = (c / denom) * g
            base_abs_c = abs(c)
            accepted = False
            for damping in (1.0, 0.5, 0.25, 0.1, 0.05, 0.01):
                trial = clip_to_bounds(z - damping * step, bounds)
                try:
                    trial_abs_c = abs(
                        as_float(scaled_constraint_value_jax(jnp.asarray(trial, dtype=jnp.float64), float(v)))
                    )
                except Exception:  # noqa: BLE001
                    continue
                if math.isfinite(trial_abs_c) and trial_abs_c <= max(base_abs_c * 0.95, tol):
                    z = trial
                    accepted = True
                    break
            if not accepted:
                break
        return z

    def eq_projected_grad_norm(z: np.ndarray, v: float) -> float:
        grad = objective_grad_np(z)
        cgrad = constraint_grad_np(z, v)
        denom = float(np.dot(cgrad, cgrad))
        if not math.isfinite(denom) or denom == 0.0:
            return math.inf
        projected = grad - (float(np.dot(grad, cgrad)) / denom) * cgrad
        return float(np.linalg.norm(projected))

    def active_projected_grad_norm(z: np.ndarray, v: float, active_tol: float = 1e-8) -> float:
        grad = objective_grad_np(z)
        rows = [constraint_grad_np(z, v)]
        for idx, (value, lb, ub) in enumerate(zip(z, bounds.lb, bounds.ub, strict=True)):
            if np.isfinite(lb) and value <= lb + active_tol:
                row = np.zeros_like(z)
                row[idx] = 1.0
                rows.append(row)
            elif np.isfinite(ub) and value >= ub - active_tol:
                row = np.zeros_like(z)
                row[idx] = 1.0
                rows.append(row)
        return project_out_norm(grad, rows)

    def projected_gradient(z: np.ndarray, v: float) -> np.ndarray | None:
        try:
            grad = objective_grad_np(z)
            cgrad = constraint_grad_np(z, v)
        except Exception:  # noqa: BLE001
            return None
        denom = float(np.dot(cgrad, cgrad))
        if not math.isfinite(denom) or denom == 0.0:
            return None
        return grad - (float(np.dot(grad, cgrad)) / denom) * cgrad

    def projected_descent_improvement(z: np.ndarray, v: float) -> float:
        z = np.asarray(z, dtype=np.float64)
        if not np.all(np.isfinite(z)):
            return math.inf
        base_fun = objective_np(z)
        if not math.isfinite(base_fun) or scaled_violation(z, v) > CONS_TOL:
            return math.inf
        g_proj = projected_gradient(z, v)
        if g_proj is None:
            return math.inf
        g_proj_norm = float(np.linalg.norm(g_proj))
        if not math.isfinite(g_proj_norm) or g_proj_norm == 0.0:
            return 0.0
        direction = -g_proj / g_proj_norm
        coord_scale = max(float(np.linalg.norm(z)), 1.0)
        best_decrease = 0.0
        for exponent in range(-14, -3):
            step = coord_scale * (10.0 ** exponent)
            trial = project_start(z + step * direction, v, tol=min(CONS_TOL, 1e-10))
            if (
                not np.all(np.isfinite(trial))
                or bound_violation(trial, bounds) > BOUND_TOL
                or scaled_violation(trial, v) > CONS_TOL
            ):
                continue
            trial_fun = objective_np(trial)
            if math.isfinite(trial_fun):
                best_decrease = max(best_decrease, base_fun - trial_fun)
        return float(best_decrease)

    def result_ok(opt_result: OptimizeResult, v: float) -> bool:
        cached = getattr(opt_result, "profile_result_ok", None)
        if cached is not None:
            return bool(cached)
        z = np.asarray(opt_result.x, dtype=np.float64)
        fun = float(opt_result.fun) if math.isfinite(float(opt_result.fun)) else objective_np(z)
        if not math.isfinite(fun):
            opt_result.profile_result_ok = False
            return False
        if bound_violation(z, bounds) > BOUND_TOL or scaled_violation(z, v) > CONS_TOL:
            opt_result.profile_result_ok = False
            return False
        g_norm = float(np.linalg.norm(objective_grad_np(z)))
        eq_pg = eq_projected_grad_norm(z, v)
        active_pg = active_projected_grad_norm(z, v)
        stationarity = min(eq_pg, active_pg)
        opt_result.profile_projected_grad_norm = eq_pg
        opt_result.profile_active_projected_grad_norm = active_pg
        if stationarity <= max(1e-3, 1e-6 * g_norm):
            opt_result.profile_descent_improvement = 0.0
            opt_result.profile_certificate = "projected-gradient"
            opt_result.profile_result_ok = True
            return True
        descent_improvement = projected_descent_improvement(z, v)
        opt_result.profile_descent_improvement = descent_improvement
        if descent_improvement <= STATIONARITY_FUN_TOL:
            method = str(getattr(opt_result, "profile_method", ""))
            if "descent-check" not in method:
                opt_result.profile_method = f"{method}+descent-check" if method else "descent-check"
            opt_result.profile_certificate = "descent-check"
            opt_result.profile_result_ok = True
            return True
        opt_result.profile_certificate = "failed-stationarity"
        opt_result.profile_result_ok = False
        return False

    def kkt_polish(opt_result: OptimizeResult, v: float) -> OptimizeResult:
        z0 = np.asarray(getattr(opt_result, "x", z_best), dtype=np.float64)
        if not math.isfinite(float(getattr(opt_result, "fun", math.inf))) or not np.all(np.isfinite(z0)):
            return opt_result
        z0 = project_start(z0, v)
        if bound_violation(z0, bounds) > BOUND_TOL:
            return opt_result
        try:
            g_obj = objective_grad_np(z0)
            g_con = constraint_grad_np(z0, v)
        except Exception:  # noqa: BLE001
            return opt_result
        denom = float(np.dot(g_con, g_con))
        if not math.isfinite(denom) or denom == 0.0:
            return opt_result
        lambda0 = -float(np.dot(g_obj, g_con)) / denom

        def residual(w):
            z = np.asarray(w[:-1], dtype=np.float64)
            lam = float(w[-1])
            try:
                stationarity = objective_grad_np(z) + lam * constraint_grad_np(z, v)
                constraint = as_float(scaled_constraint_value_jax(jnp.asarray(z, dtype=jnp.float64), float(v)))
            except Exception:  # noqa: BLE001
                return np.full(len(w), np.inf, dtype=np.float64)
            return np.r_[stationarity, constraint]

        try:
            polished = root(residual, np.r_[z0, lambda0], method="hybr", options={"maxfev": 5000})
        except (FloatingPointError, ValueError, np.linalg.LinAlgError):
            return opt_result
        z_polished = np.asarray(polished.x[:-1], dtype=np.float64)
        if (
            not np.all(np.isfinite(z_polished))
            or bound_violation(z_polished, bounds) > BOUND_TOL
            or scaled_violation(z_polished, v) > CONS_TOL
        ):
            return opt_result
        fun_polished = objective_np(z_polished)
        if not math.isfinite(fun_polished):
            return opt_result
        if fun_polished > opt_result.fun + max(1e-7, 1e-9 * abs(float(opt_result.fun))):
            return opt_result
        method = str(getattr(opt_result, "profile_method", ""))
        method = f"{method}+KKT" if method else "KKT"
        return OptimizeResult(
            x=z_polished,
            fun=fun_polished,
            success=bool(polished.success),
            status=getattr(polished, "status", None),
            message=f"KKT polish: {polished.message}",
            nfev=getattr(polished, "nfev", None),
            profile_method=method,
        )

    def maybe_keep_feasible_start(opt_result: OptimizeResult, z0: np.ndarray, v: float) -> OptimizeResult:
        z0_fun = objective_np(z0)
        if (
            not math.isfinite(z0_fun)
            or bound_violation(z0, bounds) > BOUND_TOL
            or scaled_violation(z0, v) > CONS_TOL
        ):
            return opt_result
        result_fun = float(opt_result.fun) if math.isfinite(float(opt_result.fun)) else math.inf
        if z0_fun <= result_fun + max(1e-8, 1e-10 * abs(result_fun)):
            is_best_start = np.linalg.norm(z0 - z_best) <= 1e-10 * max(float(np.linalg.norm(z_best)), 1.0)
            return OptimizeResult(
                x=np.asarray(z0, dtype=np.float64),
                fun=z0_fun,
                success=is_best_start,
                message="retained lower-chi2 feasible projected start",
                profile_method="projected-start",
            )
        return opt_result

    def make_constraint(v: float, exact_hess: bool = False) -> NonlinearConstraint:
        kwargs = {}
        if exact_hess:
            kwargs["hess"] = lambda z, multiplier: np.asarray(
                multiplier[0] * scaled_constraint_hess_jax(jnp.asarray(z, dtype=jnp.float64), float(v)),
                dtype=np.float64,
            )
        return NonlinearConstraint(
            fun=lambda z: as_float(scaled_constraint_value_jax(jnp.asarray(z, dtype=jnp.float64), float(v))),
            lb=0.0,
            ub=0.0,
            jac=lambda z: constraint_grad_np(z, v),
            **kwargs,
        )

    def solve_from(z0: np.ndarray, start_name: str, v: float) -> OptimizeResult:
        candidates: list[OptimizeResult] = []
        try:
            slsqp = minimize(
                objective_np,
                z0,
                method="SLSQP",
                jac=objective_grad_np,
                bounds=bounds,
                constraints=[make_constraint(v)],
                options={"ftol": 1e-10, "maxiter": 2000},
            )
        except (FloatingPointError, ValueError, np.linalg.LinAlgError) as exc:
            slsqp = OptimizeResult(x=z0, fun=objective_np(z0), success=False, message=f"SLSQP failed: {exc}")
        slsqp = maybe_keep_feasible_start(slsqp, z0, v)
        if not getattr(slsqp, "profile_method", ""):
            slsqp.profile_method = "SLSQP"
        slsqp.profile_start_name = start_name
        candidates.append(slsqp)
        if result_ok(slsqp, v):
            return slsqp

        try:
            trust = minimize(
                objective_np,
                z0,
                method="trust-constr",
                jac=objective_grad_np,
                hess=objective_hess_np,
                bounds=bounds,
                constraints=[make_constraint(v, exact_hess=True)],
                options={"gtol": 1e-10, "xtol": 1e-10, "maxiter": 2000},
            )
        except (FloatingPointError, TypeError, ValueError, np.linalg.LinAlgError) as exc:
            trust = OptimizeResult(x=z0, fun=objective_np(z0), success=False, message=f"trust-constr failed: {exc}")
        trust = maybe_keep_feasible_start(trust, z0, v)
        if not getattr(trust, "profile_method", ""):
            trust.profile_method = "trust-constr-exact-hess"
        trust.profile_start_name = start_name
        trust = kkt_polish(trust, v)
        trust.profile_start_name = start_name
        candidates.append(trust)
        if result_ok(trust, v):
            return trust

        ok_candidates = [candidate for candidate in candidates if result_ok(candidate, v)]
        if ok_candidates:
            return min(ok_candidates, key=lambda candidate: candidate.fun)
        return min(candidates, key=lambda candidate: candidate.fun if math.isfinite(float(candidate.fun)) else math.inf)

    def profile(v: float) -> float:
        nonlocal last_z
        v = float(v)
        t0 = time.perf_counter()
        starts: list[tuple[str, np.ndarray]] = [("continuation_projected", project_start(last_z, v))]
        mle_start = project_start(z_best, v)
        if distinct_start(starts, mle_start):
            starts.append(("mle_projected", mle_start))

        candidates = []
        candidate_records = []
        for start_name, start in starts:
            result = solve_from(start, start_name, v)
            ok = result_ok(result, v)
            z = np.asarray(result.x, dtype=np.float64)
            fun = float(result.fun) if math.isfinite(float(result.fun)) else objective_np(z)
            record = {
                "chart": chart,
                "target": target_name,
                "fixed_value": v,
                "record_type": "candidate",
                "start_name": start_name,
                "start_fun": objective_np(start),
                "start_scaled_constraint_violation": scaled_violation(start, v),
                "candidate_chi2": fun if math.isfinite(fun) else None,
                "candidate_ok": ok,
                "candidate_success": bool(getattr(result, "success", False)),
                "candidate_method": str(getattr(result, "profile_method", "")),
                "candidate_certificate": str(getattr(result, "profile_certificate", "")),
                "candidate_message": clean_message(getattr(result, "message", "")),
                "candidate_nfev": getattr(result, "nfev", None),
                "candidate_nit": getattr(result, "nit", None),
                "candidate_scaled_constraint_violation": scaled_violation(z, v)
                if np.all(np.isfinite(z)) else None,
                "candidate_bound_violation": bound_violation(z, bounds) if np.all(np.isfinite(z)) else None,
                "candidate_projected_grad_norm": getattr(result, "profile_projected_grad_norm", None),
                "candidate_active_projected_grad_norm": getattr(result, "profile_active_projected_grad_norm", None),
                "candidate_objective_grad_norm": float(np.linalg.norm(objective_grad_np(z)))
                if math.isfinite(fun) and np.all(np.isfinite(z)) else None,
                "candidate_descent_improvement": getattr(result, "profile_descent_improvement", None),
            }
            record.update({f"candidate_{k}": val for k, val in chart_diagnostics(fit, decay_idxs, chart_to_params, z).items()})
            candidate_records.append(record)
            candidates.append(result)

        ok_candidates = [candidate for candidate in candidates if result_ok(candidate, v)]
        if ok_candidates:
            result = min(ok_candidates, key=lambda candidate: candidate.fun)
        else:
            result = min(candidates, key=lambda candidate: candidate.fun if math.isfinite(float(candidate.fun)) else math.inf)

        z = np.asarray(result.x, dtype=np.float64)
        fun = float(result.fun) if math.isfinite(float(result.fun)) else objective_np(z)
        ok = result_ok(result, v)
        cviol = raw_violation(z, v) if np.all(np.isfinite(z)) else math.inf
        scaled_cviol = cviol / target_scale
        g_norm = float(np.linalg.norm(objective_grad_np(z))) if math.isfinite(fun) else math.inf
        eq_pg = eq_projected_grad_norm(z, v) if math.isfinite(fun) else math.inf
        active_pg = active_projected_grad_norm(z, v) if math.isfinite(fun) else math.inf
        point = ProfilePoint(
            v,
            fun if math.isfinite(fun) else math.inf,
            ok,
            cviol,
            clean_message(getattr(result, "message", "")),
            scaled_constraint_violation=scaled_cviol,
            projected_grad_norm=eq_pg,
            objective_grad_norm=g_norm,
            method=str(getattr(result, "profile_method", "")),
            nfev=getattr(result, "nfev", None),
            nit=getattr(result, "nit", None),
            runtime_sec=time.perf_counter() - t0,
            descent_improvement=getattr(result, "profile_descent_improvement", None),
        )
        detail = {
            "chart": chart,
            "target": target_name,
            "fixed_value": v,
            "selected_start_name": str(getattr(result, "profile_start_name", "")),
            "selected_method": point.method,
            "selected_certificate": str(getattr(result, "profile_certificate", "")),
            "selected_chi2": point.chi2,
            "selected_ok": ok,
            "selected_nfev": point.nfev,
            "selected_nit": point.nit,
            "selected_runtime_sec": point.runtime_sec,
            "selected_eq_projected_grad_norm": eq_pg,
            "selected_active_projected_grad_norm": active_pg,
            "selected_bound_violation": bound_violation(z, bounds),
            "n_candidate_starts": len(candidate_records),
            "candidate_nfev_total": sum(int(r["candidate_nfev"] or 0) for r in candidate_records),
            "candidate_records": candidate_records,
        }
        detail.update({f"selected_{k}": val for k, val in chart_diagnostics(fit, decay_idxs, chart_to_params, z).items()})
        point.round11_detail = detail
        diagnostics.append(point)
        details.append(detail)
        profile.last_point = point
        if not ok:
            raise RuntimeError(
                f"{chart} profile failed for {target_name}={v}: chi2={fun}, "
                f"scaled_cviol={scaled_cviol:.3g}, eq_pg={eq_pg:.3g}, "
                f"active_pg={active_pg:.3g}, message={point.message}"
            )
        last_z = z.copy()
        return point.chi2

    profile.diagnostics = diagnostics
    profile.last_point = None
    profile.round11_details = details
    return profile


def endpoint_detail(root, side: str) -> dict[str, Any]:
    endpoint = root.lower if side == "lower" else root.upper
    point = endpoint.point
    detail = getattr(point, "round11_detail", {}) or {}
    out = {
        "chart_endpoint": endpoint.endpoint,
        "chart_error": endpoint.error,
        "chart_profile_chi2": endpoint.chi2,
        "chart_residual": endpoint.residual,
        "chart_profile_success": point.success,
        "chart_profile_method": point.method,
        "chart_profile_message": clean_message(point.message),
        "chart_profile_nfev": point.nfev,
        "chart_profile_nit": point.nit,
        "chart_profile_runtime_sec": point.runtime_sec,
        "chart_constraint_violation": point.constraint_violation,
        "chart_scaled_constraint_violation": point.scaled_constraint_violation,
        "chart_eq_projected_grad_norm": point.projected_grad_norm,
        "chart_objective_grad_norm": point.objective_grad_norm,
        "chart_descent_improvement": point.descent_improvement,
        "chart_root_function_evals": root.function_evals,
        "chart_side_function_evals": endpoint.counts.get("function_evals"),
        "chart_side_bracket_evals": endpoint.counts.get("bracket_evals"),
        "chart_side_bisect_evals": endpoint.counts.get("bisect_evals"),
    }
    for key, value in detail.items():
        if key == "candidate_records":
            continue
        out[f"detail_{key}"] = value
    return out


def production_side_fields(asym: dict[str, Any], side: str) -> dict[str, Any]:
    point = asym.get(f"{side}_profile_point", {}) or {}
    return {
        "production_endpoint": asym.get(f"{side}_endpoint"),
        "production_error": asym.get(f"{side}_error"),
        "production_profile_chi2": asym.get(f"{side}_chi2"),
        "production_residual": asym.get(f"{side}_residual"),
        "production_profile_success": point.get("success"),
        "production_profile_method": point.get("method"),
        "production_profile_message": clean_message(point.get("message")),
        "production_profile_nfev": point.get("nfev"),
        "production_profile_nit": point.get("nit"),
        "production_scaled_constraint_violation": point.get("scaled_constraint_violation"),
        "production_projected_grad_norm": point.get("projected_grad_norm"),
        "production_objective_grad_norm": point.get("objective_grad_norm"),
        "production_descent_improvement": point.get("descent_improvement"),
        "production_root_function_evals": asym.get("function_evals"),
    }


def flatten_profile_details(
    case: CaseSpec,
    chart: str,
    details: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for detail in details:
        base = {
            "case_id": case.case_id,
            "label": case.label,
            "target": case.target,
            "chart": chart,
            "record_type": "profile_selected",
        }
        for key, value in detail.items():
            if key != "candidate_records":
                base[key] = value
        rows.append(base)
        for candidate in detail.get("candidate_records", []):
            rows.append(
                {
                    "case_id": case.case_id,
                    "label": case.label,
                    "target": case.target,
                    "chart": chart,
                    **candidate,
                }
            )
    return rows


def fixed_check_row(
    case: CaseSpec,
    chart: str,
    side: str,
    profile: Callable[[float], float],
    fixed_value: float,
    production_chi2: float,
    target_chi2: float,
) -> dict[str, Any]:
    try:
        chart_chi2 = float(profile(fixed_value))
        point = getattr(profile, "last_point", None)
        detail = getattr(point, "round11_detail", {}) if point is not None else {}
        row = {
            "case_id": case.case_id,
            "label": case.label,
            "target": case.target,
            "chart": chart,
            "side": side,
            "fixed_value": fixed_value,
            "target_chi2": target_chi2,
            "production_profile_chi2": production_chi2,
            "chart_profile_chi2": chart_chi2,
            "chart_residual": chart_chi2 - target_chi2,
            "improvement_vs_production": production_chi2 - chart_chi2,
            "material_lower_constrained_profile": bool(production_chi2 - chart_chi2 > MATERIAL_IMPROVEMENT),
            "status": "ok",
        }
        if point is not None:
            row.update(
                {
                    "chart_profile_method": point.method,
                    "chart_profile_success": point.success,
                    "chart_profile_nfev": point.nfev,
                    "chart_scaled_constraint_violation": point.scaled_constraint_violation,
                    "chart_eq_projected_grad_norm": point.projected_grad_norm,
                    "chart_objective_grad_norm": point.objective_grad_norm,
                    "chart_descent_improvement": point.descent_improvement,
                }
            )
        for key, value in detail.items():
            if key != "candidate_records":
                row[f"detail_{key}"] = value
        return row
    except Exception as exc:  # noqa: BLE001 - diagnostic row records failure
        return {
            "case_id": case.case_id,
            "label": case.label,
            "target": case.target,
            "chart": chart,
            "side": side,
            "fixed_value": fixed_value,
            "target_chi2": target_chi2,
            "production_profile_chi2": production_chi2,
            "status": "error",
            "exception_type": type(exc).__name__,
            "exception": str(exc),
        }


def run_case(
    case: CaseSpec,
    charts: list[str],
    *,
    residual_tol: float,
    stdout_log: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    with stdout_log.open("a", encoding="utf-8") as log:
        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            t_fit = time.perf_counter()
            fit = run_fit(case.label, verbose=False)
            fit_runtime_sec = time.perf_counter() - t_fit
            if fit is None:
                raise RuntimeError(f"run_fit returned None for {case.label}")
            target_func = target_func_for_fit(fit, case.target)
            asym = calc_asym_errors(fit, targets=[case.target])[case.target]

    target_value = float(asym["value"])
    target_std = target_std_for_fit(fit, target_func, target_value)
    chi2_min = float(fit["chi2_min"])
    target_chi2 = chi2_min + 1.0
    lb = target_value - 2.0 * target_std
    ub = target_value + 2.0 * target_std

    result_rows: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    fixed_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []

    for chart in charts:
        profile = build_chart_profile(
            fit,
            target_func,
            target_value,
            target_std,
            chart=chart,
            target_name=case.target,
        )
        problem = CallableProfileProblem(
            profile_chi2=profile,
            target_name=f"{case.target}:{chart}",
            target_value=target_value,
            chi2_min=chi2_min,
            lower_initial=lb,
            upper_initial=ub,
        )
        try:
            with stdout_log.open("a", encoding="utf-8") as log:
                log.write(f"\n===== {case.case_id} {chart} root =====\n")
                log.flush()
            t_root = time.perf_counter()
            root_result = find_profile_root(
                problem,
                target_value,
                chi2_min,
                lb,
                ub,
                residual_tol=residual_tol,
            )
            root_runtime_sec = time.perf_counter() - t_root
        except Exception as exc:  # noqa: BLE001
            failure_rows.append(
                {
                    "case_id": case.case_id,
                    "label": case.label,
                    "target": case.target,
                    "chart": chart,
                    "stage": "root",
                    "status": "error",
                    "exception_type": type(exc).__name__,
                    "exception": str(exc),
                }
            )
            eval_rows.extend(flatten_profile_details(case, chart, getattr(profile, "round11_details", [])))
            continue

        eval_rows.extend(flatten_profile_details(case, chart, profile.round11_details))
        n_details_before_fixed = len(profile.round11_details)
        for side in ("lower", "upper"):
            prod = production_side_fields(asym, side)
            row = {
                "case_id": case.case_id,
                "label": case.label,
                "algorithm": fit["algorithm"],
                "target": case.target,
                "rationale": case.rationale,
                "chart": chart,
                "side": side,
                "n_parameters": len(fit["parameters"]),
                "chi2_min": chi2_min,
                "target_chi2": target_chi2,
                "target_value": target_value,
                "target_std": target_std,
                "initial_lb": lb,
                "initial_ub": ub,
                "fit_runtime_sec": fit_runtime_sec,
                "root_runtime_sec": root_runtime_sec,
                "status": "ok",
                **prod,
                **endpoint_detail(root_result, side),
            }
            row["endpoint_delta_vs_production"] = row["chart_endpoint"] - row["production_endpoint"]
            row["profile_chi2_delta_vs_production_endpoint"] = (
                row["chart_profile_chi2"] - row["production_profile_chi2"]
            )
            row["same_endpoint_within_5e3_residual_scale"] = bool(
                abs(row["endpoint_delta_vs_production"]) <= max(1e-12, 0.01 * target_std)
            )
            row["chart_verified_within_tol"] = bool(abs(row["chart_residual"]) <= residual_tol)
            result_rows.append(row)

        # Fresh fixed-target checks at the production endpoints answer whether
        # the alternate chart finds a lower constrained profile at the same
        # reported endpoint.
        for side in ("lower", "upper"):
            fixed_rows.append(
                fixed_check_row(
                    case,
                    chart,
                    side,
                    profile,
                    float(asym[f"{side}_endpoint"]),
                    float(asym[f"{side}_chi2"]),
                    target_chi2,
                )
            )
        eval_rows.extend(flatten_profile_details(case, chart, profile.round11_details[n_details_before_fixed:]))

    return result_rows, eval_rows, fixed_rows, failure_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", action="append", default=None, help="case_id to run")
    parser.add_argument("--chart", action="append", default=None, choices=DEFAULT_CHARTS)
    parser.add_argument("--results-csv", default="notes/codex-refactor/round11_brbru_chart_conditioning_results.csv")
    parser.add_argument("--results-jsonl", default="notes/codex-refactor/round11_brbru_chart_conditioning_results.jsonl")
    parser.add_argument("--evaluations-csv", default="notes/codex-refactor/round11_brbru_chart_conditioning_evaluations.csv")
    parser.add_argument("--evaluations-jsonl", default="notes/codex-refactor/round11_brbru_chart_conditioning_evaluations.jsonl")
    parser.add_argument("--fixed-checks-csv", default="notes/codex-refactor/round11_brbru_chart_conditioning_fixed_checks.csv")
    parser.add_argument("--fixed-checks-jsonl", default="notes/codex-refactor/round11_brbru_chart_conditioning_fixed_checks.jsonl")
    parser.add_argument("--failures-csv", default="notes/codex-refactor/round11_brbru_chart_conditioning_failures.csv")
    parser.add_argument("--failures-jsonl", default="notes/codex-refactor/round11_brbru_chart_conditioning_failures.jsonl")
    parser.add_argument("--stdout-log", default="notes/logs/round11_brbru_chart_conditioning_stdout.log")
    parser.add_argument("--residual-tol", type=float, default=5e-3)
    parser.add_argument("--chart-timeout-sec", type=int, default=420)
    args = parser.parse_args()

    selected_cases = [case for case in DEFAULT_CASES if args.case is None or case.case_id in set(args.case)]
    if not selected_cases:
        raise SystemExit(f"No cases selected by {args.case}")
    charts = args.chart or list(DEFAULT_CHARTS)

    paths = [
        Path(args.results_csv),
        Path(args.results_jsonl),
        Path(args.evaluations_csv),
        Path(args.evaluations_jsonl),
        Path(args.fixed_checks_csv),
        Path(args.fixed_checks_jsonl),
        Path(args.failures_csv),
        Path(args.failures_jsonl),
    ]
    for path in paths:
        if path.exists():
            path.unlink()
    stdout_log = Path(args.stdout_log)
    stdout_log.parent.mkdir(parents=True, exist_ok=True)
    if stdout_log.exists():
        stdout_log.unlink()

    all_results: list[dict[str, Any]] = []
    all_evals: list[dict[str, Any]] = []
    all_fixed: list[dict[str, Any]] = []
    all_failures: list[dict[str, Any]] = []

    old_handler = signal.signal(signal.SIGALRM, _alarm_handler)
    try:
        for idx, case in enumerate(selected_cases, start=1):
            print(f"[{idx}/{len(selected_cases)}] {case.case_id}")
            try:
                signal.alarm(args.chart_timeout_sec)
                results, evals, fixed, failures = run_case(
                    case,
                    charts,
                    residual_tol=args.residual_tol,
                    stdout_log=stdout_log,
                )
                signal.alarm(0)
            except Exception as exc:  # noqa: BLE001
                signal.alarm(0)
                results, evals, fixed = [], [], []
                failures = [
                    {
                        "case_id": case.case_id,
                        "label": case.label,
                        "target": case.target,
                        "stage": "case",
                        "status": "error",
                        "exception_type": type(exc).__name__,
                        "exception": str(exc),
                    }
                ]
            all_results.extend(results)
            all_evals.extend(evals)
            all_fixed.extend(fixed)
            all_failures.extend(failures)
            print(
                f"  results={len(results)} fixed={len(fixed)} failures={len(failures)} "
                f"eval_rows={len(evals)}"
            )
            gc.collect()
            try:
                jax.clear_caches()
            except Exception:
                pass
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

    write_rows(Path(args.results_csv), all_results)
    write_rows(Path(args.results_jsonl), all_results, jsonl=True)
    write_rows(Path(args.evaluations_csv), all_evals)
    write_rows(Path(args.evaluations_jsonl), all_evals, jsonl=True)
    write_rows(Path(args.fixed_checks_csv), all_fixed)
    write_rows(Path(args.fixed_checks_jsonl), all_fixed, jsonl=True)
    write_rows(Path(args.failures_csv), all_failures)
    write_rows(Path(args.failures_jsonl), all_failures, jsonl=True)

    print(
        "wrote "
        f"results={len(all_results)} fixed={len(all_fixed)} "
        f"eval_rows={len(all_evals)} failures={len(all_failures)}"
    )


if __name__ == "__main__":
    main()
