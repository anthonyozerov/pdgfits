#!/usr/bin/env python
"""Physical-space fixed-target profile comparator for hard asym endpoints.

This is notes-only diagnostic tooling.  It compares the production fitted-space
constrained profile chi2 at selected endpoints against an independent direct
physical-parameter solve with explicit branching-fraction bounds.
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

import jax

jax.config.update("jax_enable_x64", True)

from jax import numpy as jnp
import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, NonlinearConstraint, OptimizeResult, minimize

from pdgfits.fit import run_fit
from pdgfits.param_maps import get_decay_info


CONS_TOL = 1e-7
BOUND_TOL = 1e-10
MATERIAL_IMPROVEMENT = 1e-3
DECAY_LB = 1e-12
DECAY_UB = 1.0 - 1e-12

DEFAULT_ENDPOINTS = [
    "B0::S042B95::upper",
    "B0::S042B95::lower",
    "B0S-BR::S086.37::upper",
    "B0S-BR::S086.37::lower",
    "eta_c J/psi psi(2S)::M026W::upper",
    "eta_c J/psi psi(2S)::M026W::lower",
]

SUMMARY_FIELDS = [
    "label",
    "algorithm",
    "target",
    "target_kind",
    "side",
    "endpoint",
    "target_value",
    "target_std",
    "chi2_min",
    "target_chi2",
    "current_profile_chi2",
    "current_residual",
    "current_method",
    "best_comparator_chi2",
    "best_comparator_residual",
    "improvement_vs_current",
    "material_improvement",
    "best_start_name",
    "best_solver",
    "best_success",
    "best_constraint_violation",
    "best_scaled_constraint_violation",
    "best_bound_violation",
    "best_simplex_violation",
    "best_eq_projected_grad_norm",
    "best_active_fixed_grad_norm",
    "best_objective_grad_norm",
    "best_min_decay_param",
    "best_max_decay_param",
    "best_max_abs_decay_fitted_coord",
    "best_decay_active_lower_1e8",
    "best_decay_active_upper_1e8",
    "n_decay_params",
    "simplex_groups",
    "n_starts",
    "n_attempts",
    "n_feasible",
    "n_success",
    "n_failed",
    "fit_runtime_sec",
    "endpoint_runtime_sec",
    "status",
    "exception_type",
    "exception",
]

ATTEMPT_FIELDS = [
    "label",
    "target",
    "side",
    "endpoint",
    "start_name",
    "start_kind",
    "solver",
    "success",
    "feasible",
    "chi2",
    "residual",
    "improvement_vs_current",
    "constraint_violation",
    "scaled_constraint_violation",
    "bound_violation",
    "simplex_violation",
    "eq_projected_grad_norm",
    "active_fixed_grad_norm",
    "objective_grad_norm",
    "min_decay_param",
    "max_decay_param",
    "max_abs_decay_fitted_coord",
    "decay_active_lower_1e8",
    "decay_active_upper_1e8",
    "nfev",
    "nit",
    "runtime_sec",
    "message",
]


class EndpointTimeout(TimeoutError):
    pass


@dataclass
class PhysicalProfileContext:
    fit: dict
    target: str
    target_kind: str
    target_func: object
    endpoint: float
    target_scale: float
    bounds: Bounds
    decay_idxs: np.ndarray
    simplex_groups: list[np.ndarray]
    simplex_constraints: list[LinearConstraint]
    p_best: np.ndarray
    param_cov: np.ndarray | None


def _alarm_handler(signum, frame):  # noqa: ARG001
    raise EndpointTimeout("endpoint timed out")


def _as_float(x) -> float:
    return float(np.asarray(x, dtype=np.float64))


def clean(obj):
    if isinstance(obj, dict):
        return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return clean(obj.tolist())
    if isinstance(obj, np.generic):
        return clean(obj.item())
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else str(obj)
    return obj


def append_csv(path: Path, row: dict, fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow({field: clean(row.get(field)) for field in fieldnames})


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(clean(row), sort_keys=True) + "\n")


def parse_endpoint_spec(spec: str) -> tuple[str, str, str]:
    parts = spec.split("::")
    if len(parts) != 3 or parts[2] not in {"upper", "lower"}:
        raise ValueError(f"Expected LABEL::TARGET::upper|lower, got {spec!r}")
    return parts[0], parts[1], parts[2]


def target_function(fit: dict, target: str):
    if target in fit["nodes"]:
        return "node", fit["node_funcs"][fit["nodes"].index(target)]
    if target in fit["parameters"]:
        return "parameter", fit["parameter_funcs"][fit["parameters"].index(target)]
    raise KeyError(f"{target!r} is not a node or parameter in {fit['label']}")


def particles_for_fit(fit: dict) -> list[str]:
    return list(
        np.unique(
            [
                name[:4]
                for name in list(fit["nodes"]) + list(fit["parameters"])
                if not str(name).startswith("nuisance_")
            ]
        )
    )


def is_simplex_br(fit: dict) -> bool:
    algorithm = fit["algorithm"]
    return algorithm in ["BR", "BR (NO MATRIX)", "BR PRINT"] and len(particles_for_fit(fit)) == 1


def simplex_groups_for_fit(fit: dict, decay_idxs: np.ndarray) -> list[np.ndarray]:
    if not is_simplex_br(fit) or len(decay_idxs) == 0:
        return []
    parameters = np.asarray(fit["parameters"], dtype=object)
    groups: list[np.ndarray] = []
    for particle in np.unique([str(parameters[i])[:4] for i in decay_idxs]):
        group = np.array([int(i) for i in decay_idxs if str(parameters[int(i)]).startswith(f"{particle}.")])
        if len(group) > 0:
            groups.append(group)
    return groups


def build_bounds(n_params: int, decay_idxs: np.ndarray) -> Bounds:
    lb = np.full(n_params, -np.inf, dtype=np.float64)
    ub = np.full(n_params, np.inf, dtype=np.float64)
    if len(decay_idxs) > 0:
        lb[decay_idxs] = DECAY_LB
        ub[decay_idxs] = DECAY_UB
    return Bounds(lb, ub)


def build_simplex_constraints(n_params: int, groups: list[np.ndarray]) -> list[LinearConstraint]:
    constraints: list[LinearConstraint] = []
    for group in groups:
        mat = np.zeros((1, n_params), dtype=np.float64)
        mat[0, group] = 1.0
        constraints.append(LinearConstraint(mat, 1.0, 1.0))
    return constraints


def clip_to_bounds(x: np.ndarray, bounds: Bounds) -> np.ndarray:
    out = np.asarray(x, dtype=np.float64).copy()
    finite_lb = np.isfinite(bounds.lb)
    finite_ub = np.isfinite(bounds.ub)
    out[finite_lb] = np.maximum(out[finite_lb], bounds.lb[finite_lb])
    out[finite_ub] = np.minimum(out[finite_ub], bounds.ub[finite_ub])
    return out


def covariance_sqrt(covariance, dim: int) -> tuple[np.ndarray | None, np.ndarray, np.ndarray]:
    if covariance is None:
        return None, np.array([], dtype=np.float64), np.empty((dim, 0), dtype=np.float64)
    cov = np.asarray(covariance, dtype=np.float64)
    if cov.shape != (dim, dim) or not np.all(np.isfinite(cov)):
        return None, np.array([], dtype=np.float64), np.empty((dim, 0), dtype=np.float64)
    cov = 0.5 * (cov + cov.T)
    try:
        evals, evecs = np.linalg.eigh(cov)
    except np.linalg.LinAlgError:
        return None, np.array([], dtype=np.float64), np.empty((dim, 0), dtype=np.float64)
    keep = np.isfinite(evals) & (evals > 0)
    if not np.any(keep):
        return None, np.array([], dtype=np.float64), np.empty((dim, 0), dtype=np.float64)
    evals = evals[keep]
    evecs = evecs[:, keep]
    order = np.argsort(evals)[::-1]
    evals = evals[order]
    evecs = evecs[:, order]
    return evecs @ np.diag(np.sqrt(evals)), evals, evecs


def physical_param_cov(fit: dict) -> np.ndarray | None:
    covariance = fit["covariance"]
    if covariance is None:
        return None
    try:
        jac_params = jax.jacobian(fit["fitted_params_to_params"])(fit["fitted_values"])
        param_cov = np.asarray(jac_params @ covariance @ jac_params.T, dtype=np.float64)
    except Exception:  # noqa: BLE001 - diagnostic start generation can fall back
        return None
    if param_cov.shape != (len(fit["parameters"]), len(fit["parameters"])):
        return None
    return 0.5 * (param_cov + param_cov.T)


def dedupe_starts(starts: list[tuple[str, str, np.ndarray]]) -> list[tuple[str, str, np.ndarray]]:
    deduped: list[tuple[str, str, np.ndarray]] = []
    for name, kind, start in starts:
        start = np.asarray(start, dtype=np.float64)
        if not np.all(np.isfinite(start)):
            continue
        duplicate = False
        for _, _, existing in deduped:
            scale = max(float(np.linalg.norm(start)), float(np.linalg.norm(existing)), 1.0)
            if float(np.linalg.norm(start - existing)) <= 1e-12 * scale:
                duplicate = True
                break
        if not duplicate:
            deduped.append((name, kind, start))
    return deduped


def build_context(fit: dict, target: str, endpoint: float, target_scale: float) -> PhysicalProfileContext:
    target_kind, target_func = target_function(fit, target)
    _, decay_idxs = get_decay_info(fit["parameters"])
    decay_idxs = np.asarray(decay_idxs, dtype=int)
    p_best = np.asarray(fit["param_values"], dtype=np.float64)
    bounds = build_bounds(len(p_best), decay_idxs)
    simplex_groups = simplex_groups_for_fit(fit, decay_idxs)
    simplex_constraints = build_simplex_constraints(len(p_best), simplex_groups)
    target_scale = abs(float(target_scale))
    if not np.isfinite(target_scale) or target_scale <= 0:
        target_scale = 1.0
    return PhysicalProfileContext(
        fit=fit,
        target=target,
        target_kind=target_kind,
        target_func=target_func,
        endpoint=float(endpoint),
        target_scale=target_scale,
        bounds=bounds,
        decay_idxs=decay_idxs,
        simplex_groups=simplex_groups,
        simplex_constraints=simplex_constraints,
        p_best=clip_to_bounds(p_best, bounds),
        param_cov=physical_param_cov(fit),
    )


def build_physical_functions(ctx: PhysicalProfileContext):
    fit = ctx.fit
    endpoint = float(ctx.endpoint)
    target_scale = float(ctx.target_scale) if np.isfinite(ctx.target_scale) and ctx.target_scale > 0 else 1.0

    @jax.jit
    def objective_jax(params):
        fitted = fit["params_to_fitted_params"](params)
        return fit["chi2"](fitted)

    @jax.jit
    def constraint_raw_jax(params):
        return ctx.target_func(params) - endpoint

    @jax.jit
    def constraint_scaled_jax(params):
        return constraint_raw_jax(params) / target_scale

    obj_value_grad_jax = jax.jit(jax.value_and_grad(objective_jax))
    obj_hess_jax = jax.jit(jax.hessian(objective_jax))
    cons_grad_jax = jax.jit(jax.grad(constraint_scaled_jax))
    cons_hess_jax = jax.jit(jax.hessian(constraint_scaled_jax))

    def obj_value(x):
        value = _as_float(objective_jax(jnp.asarray(x, dtype=jnp.float64)))
        return value if np.isfinite(value) else np.inf

    def obj_grad(x):
        val, grad = obj_value_grad_jax(jnp.asarray(x, dtype=jnp.float64))
        val = float(val)
        grad = np.asarray(grad, dtype=np.float64)
        if not np.isfinite(val):
            val = np.inf
        if not np.all(np.isfinite(grad)):
            raise FloatingPointError("non-finite physical objective gradient")
        return val, grad

    def obj_jac(x):
        return obj_grad(x)[1]

    def obj_hess(x):
        hess = np.asarray(obj_hess_jax(jnp.asarray(x, dtype=jnp.float64)), dtype=np.float64)
        if not np.all(np.isfinite(hess)):
            raise FloatingPointError("non-finite physical objective Hessian")
        return hess

    def cons_raw(x):
        return _as_float(constraint_raw_jax(jnp.asarray(x, dtype=jnp.float64)))

    def cons_scaled(x):
        return _as_float(constraint_scaled_jax(jnp.asarray(x, dtype=jnp.float64)))

    def cons_grad(x):
        grad = np.asarray(cons_grad_jax(jnp.asarray(x, dtype=jnp.float64)), dtype=np.float64)
        if not np.all(np.isfinite(grad)):
            raise FloatingPointError("non-finite physical constraint gradient")
        return grad

    def cons_hess(x):
        hess = np.asarray(cons_hess_jax(jnp.asarray(x, dtype=jnp.float64)), dtype=np.float64)
        if not np.all(np.isfinite(hess)):
            raise FloatingPointError("non-finite physical constraint Hessian")
        return hess

    return obj_value, obj_grad, obj_jac, obj_hess, cons_raw, cons_scaled, cons_grad, cons_hess


def bound_violation(x: np.ndarray, bounds: Bounds) -> float:
    below = np.where(np.isfinite(bounds.lb), np.maximum(bounds.lb - x, 0.0), 0.0)
    above = np.where(np.isfinite(bounds.ub), np.maximum(x - bounds.ub, 0.0), 0.0)
    return float(max(np.max(below), np.max(above), 0.0))


def simplex_violation(x: np.ndarray, groups: list[np.ndarray]) -> float:
    if not groups:
        return 0.0
    return float(max(abs(float(np.sum(x[group])) - 1.0) for group in groups))


def project_start(
    start: np.ndarray,
    ctx: PhysicalProfileContext,
    cons_scaled,
    cons_grad,
    *,
    tol: float = CONS_TOL,
) -> np.ndarray:
    x = clip_to_bounds(start, ctx.bounds)
    # Exact simplex projection is only needed for future single-particle BR
    # experiments.  The default hard targets in this round are BRU.
    for group in ctx.simplex_groups:
        group_sum = float(np.sum(x[group]))
        if np.isfinite(group_sum) and group_sum > 0:
            x[group] = x[group] / group_sum
    for _ in range(40):
        if not np.all(np.isfinite(x)):
            break
        try:
            c = float(cons_scaled(x))
        except (FloatingPointError, ValueError):
            break
        if not np.isfinite(c) or abs(c) <= tol:
            return x
        try:
            g = cons_grad(x)
        except (FloatingPointError, ValueError):
            break
        denom = float(np.dot(g, g))
        if not np.isfinite(denom) or denom == 0.0:
            break
        base_abs_c = abs(c)
        accepted = False
        step = (c / denom) * g
        for damping in (1.0, 0.5, 0.25, 0.1, 0.05, 0.01):
            trial = clip_to_bounds(x - damping * step, ctx.bounds)
            for group in ctx.simplex_groups:
                group_sum = float(np.sum(trial[group]))
                if np.isfinite(group_sum) and group_sum > 0:
                    trial[group] = trial[group] / group_sum
            try:
                trial_abs_c = abs(float(cons_scaled(trial)))
            except (FloatingPointError, ValueError):
                continue
            if np.isfinite(trial_abs_c) and trial_abs_c <= max(base_abs_c * 0.95, tol):
                x = trial
                accepted = True
                break
        if not accepted:
            break
    return x


def build_starts(
    ctx: PhysicalProfileContext,
    cons_scaled,
    cons_grad,
    *,
    rng: np.random.Generator,
    cov_dirs: int,
    random_starts: int,
    cov_scale: float,
    random_scale: float,
) -> list[tuple[str, str, np.ndarray]]:
    dim = len(ctx.p_best)
    starts: list[tuple[str, str, np.ndarray]] = [("mle_projected", "mle", ctx.p_best)]

    sqrt_cov, evals, evecs = covariance_sqrt(ctx.param_cov, dim)
    for i in range(min(cov_dirs, len(evals))):
        step = cov_scale * math.sqrt(float(evals[i])) * evecs[:, i]
        starts.append((f"param_cov_eig{i + 1}_plus", "param_covariance_direction", ctx.p_best + step))
        starts.append((f"param_cov_eig{i + 1}_minus", "param_covariance_direction", ctx.p_best - step))

    for i in range(random_starts):
        z = rng.normal(size=dim)
        if sqrt_cov is not None:
            step = random_scale * (sqrt_cov @ z) / max(math.sqrt(dim), 1.0)
            kind = "random_param_covariance"
        else:
            scale = random_scale * max(float(np.linalg.norm(ctx.p_best)), 1.0) / max(math.sqrt(dim), 1.0)
            step = scale * z
            kind = "random_coordinate"
        starts.append((f"random_mle_{i + 1}", kind, ctx.p_best + step))

    projected = [
        (name, kind, project_start(start, ctx, cons_scaled, cons_grad))
        for name, kind, start in starts
    ]
    return dedupe_starts(projected)


def projected_grad_norm(grad: np.ndarray, constraint_grad: np.ndarray) -> float:
    denom = float(np.dot(constraint_grad, constraint_grad))
    if not np.isfinite(denom) or denom == 0.0:
        return np.inf
    projected = grad - (float(np.dot(grad, constraint_grad)) / denom) * constraint_grad
    return float(np.linalg.norm(projected))


def active_fixed_projected_grad_norm(
    x: np.ndarray,
    grad: np.ndarray,
    constraint_grad: np.ndarray,
    bounds: Bounds,
    *,
    active_tol: float = 1e-8,
) -> float:
    rows = [constraint_grad]
    for idx, (value, lb, ub) in enumerate(zip(x, bounds.lb, bounds.ub, strict=True)):
        if np.isfinite(lb) and value <= lb + active_tol:
            row = np.zeros_like(x)
            row[idx] = 1.0
            rows.append(row)
        elif np.isfinite(ub) and value >= ub - active_tol:
            row = np.zeros_like(x)
            row[idx] = 1.0
            rows.append(row)
    mat = np.vstack(rows)
    try:
        gram = mat @ mat.T
        coeff = np.linalg.pinv(gram) @ (mat @ grad)
        projected = grad - mat.T @ coeff
    except np.linalg.LinAlgError:
        return np.inf
    return float(np.linalg.norm(projected))


def chart_diagnostics(ctx: PhysicalProfileContext, x: np.ndarray) -> dict:
    diagnostics = {
        "min_decay_param": None,
        "max_decay_param": None,
        "max_abs_decay_fitted_coord": None,
        "decay_active_lower_1e8": 0,
        "decay_active_upper_1e8": 0,
    }
    if len(ctx.decay_idxs) == 0:
        return diagnostics
    decay_vals = np.asarray(x[ctx.decay_idxs], dtype=np.float64)
    diagnostics["min_decay_param"] = float(np.min(decay_vals))
    diagnostics["max_decay_param"] = float(np.max(decay_vals))
    diagnostics["decay_active_lower_1e8"] = int(np.sum(decay_vals <= DECAY_LB + 1e-8))
    diagnostics["decay_active_upper_1e8"] = int(np.sum(decay_vals >= DECAY_UB - 1e-8))
    try:
        fitted = np.asarray(ctx.fit["params_to_fitted_params"](jnp.asarray(x, dtype=jnp.float64)), dtype=np.float64)
        diagnostics["max_abs_decay_fitted_coord"] = float(np.max(np.abs(fitted[ctx.decay_idxs])))
    except Exception:  # noqa: BLE001 - diagnostic only
        diagnostics["max_abs_decay_fitted_coord"] = None
    return diagnostics


def attempt_row(
    ctx: PhysicalProfileContext,
    label: str,
    target: str,
    side: str,
    endpoint: float,
    start_name: str,
    start_kind: str,
    solver: str,
    result: OptimizeResult,
    current_profile_chi2: float,
    target_chi2: float,
    obj_value,
    obj_grad,
    cons_raw,
    cons_scaled,
    cons_grad,
) -> dict:
    x = np.asarray(result.x, dtype=np.float64) if hasattr(result, "x") else np.full(len(ctx.p_best), np.nan)
    finite_x = np.all(np.isfinite(x))
    chi2 = float(result.fun) if hasattr(result, "fun") and np.isfinite(result.fun) else (obj_value(x) if finite_x else np.inf)
    try:
        cviol = abs(float(cons_raw(x))) if finite_x else np.inf
    except Exception:  # noqa: BLE001 - row diagnostic should survive failures
        cviol = np.inf
    try:
        scaled_cviol = abs(float(cons_scaled(x))) if finite_x else np.inf
    except Exception:  # noqa: BLE001 - row diagnostic should survive failures
        scaled_cviol = np.inf
    bviol = bound_violation(x, ctx.bounds) if finite_x else np.inf
    sviol = simplex_violation(x, ctx.simplex_groups) if finite_x else np.inf
    feasible = (
        np.isfinite(chi2)
        and scaled_cviol <= CONS_TOL
        and bviol <= BOUND_TOL
        and sviol <= CONS_TOL
    )
    row = {
        "label": label,
        "target": target,
        "side": side,
        "endpoint": endpoint,
        "start_name": start_name,
        "start_kind": start_kind,
        "solver": solver,
        "success": bool(getattr(result, "success", False)),
        "feasible": bool(feasible),
        "chi2": chi2 if np.isfinite(chi2) else math.inf,
        "residual": chi2 - target_chi2 if np.isfinite(chi2) else math.inf,
        "improvement_vs_current": current_profile_chi2 - chi2 if np.isfinite(chi2) else -math.inf,
        "constraint_violation": cviol,
        "scaled_constraint_violation": scaled_cviol,
        "bound_violation": bviol,
        "simplex_violation": sviol,
        "nfev": getattr(result, "nfev", None),
        "nit": getattr(result, "nit", None),
        "runtime_sec": getattr(result, "profile_runtime_sec", None),
        "message": str(getattr(result, "message", "")),
    }
    if finite_x and np.isfinite(chi2):
        try:
            grad = result.profile_grad
        except AttributeError:
            try:
                _, grad = obj_grad(x)
            except Exception:  # noqa: BLE001 - gradient diagnostics are optional
                grad = None
        try:
            cgrad = cons_grad(x)
        except Exception:  # noqa: BLE001
            cgrad = None
        if grad is not None and cgrad is not None:
            row["objective_grad_norm"] = float(np.linalg.norm(grad))
            row["eq_projected_grad_norm"] = projected_grad_norm(grad, cgrad)
            row["active_fixed_grad_norm"] = active_fixed_projected_grad_norm(x, grad, cgrad, ctx.bounds)
        row.update({f"{key}": value for key, value in chart_diagnostics(ctx, x).items()})
    return row


def solve_from_start(
    ctx: PhysicalProfileContext,
    start: np.ndarray,
    solver: str,
    obj_value,
    obj_grad,
    obj_jac,
    obj_hess,
    cons_scaled,
    cons_grad,
    cons_hess,
) -> OptimizeResult:
    x0 = clip_to_bounds(start, ctx.bounds)
    constraints = list(ctx.simplex_constraints)
    if solver == "slsqp":
        constraints = [
            {
                "type": "eq",
                "fun": cons_scaled,
                "jac": cons_grad,
            },
            *constraints,
        ]
        t0 = time.perf_counter()
        try:
            result = minimize(
                obj_grad,
                x0,
                method="SLSQP",
                jac=True,
                bounds=ctx.bounds,
                constraints=constraints,
                options={"ftol": 1e-10, "maxiter": 2000},
            )
        except (FloatingPointError, ValueError, np.linalg.LinAlgError) as exc:
            result = OptimizeResult(x=x0, fun=obj_value(x0), success=False, message=f"SLSQP failed: {exc}")
        result.profile_runtime_sec = time.perf_counter() - t0
    elif solver == "trust-constr":
        nonlinear_constraint = NonlinearConstraint(
            fun=cons_scaled,
            lb=0.0,
            ub=0.0,
            jac=cons_grad,
            hess=lambda x, multiplier: multiplier[0] * cons_hess(x),
        )
        t0 = time.perf_counter()
        try:
            result = minimize(
                obj_value,
                x0,
                method="trust-constr",
                jac=obj_jac,
                hess=obj_hess,
                bounds=ctx.bounds,
                constraints=[nonlinear_constraint, *constraints],
                options={"gtol": 1e-10, "xtol": 1e-10, "maxiter": 2000},
            )
        except (FloatingPointError, TypeError, ValueError, np.linalg.LinAlgError) as exc:
            result = OptimizeResult(x=x0, fun=obj_value(x0), success=False, message=f"trust-constr failed: {exc}")
        result.profile_runtime_sec = time.perf_counter() - t0
    else:
        raise ValueError(f"unknown solver {solver!r}")
    if hasattr(result, "x") and np.all(np.isfinite(result.x)):
        try:
            value, grad = obj_grad(result.x)
            result.fun = value
            result.profile_grad = grad
        except Exception:  # noqa: BLE001 - row diagnostics still record solver output
            pass
    return result


def run_endpoint(
    fit: dict,
    source_row: dict,
    side: str,
    *,
    rng: np.random.Generator,
    cov_dirs: int,
    random_starts: int,
    cov_scale: float,
    random_scale: float,
    solvers: list[str],
    fit_runtime_sec: float,
    stdout_log: Path,
) -> tuple[dict, list[dict]]:
    label = str(source_row["label"])
    target = str(source_row["target"])
    endpoint = float(source_row[f"{side}_endpoint"])
    current_profile_chi2 = float(source_row[f"{side}_chi2"])
    current_residual = float(source_row[f"{side}_residual"])
    current_method = str(source_row[f"{side}_method"])
    chi2_min = float(source_row["chi2_min"])
    target_chi2 = chi2_min + 1.0
    ctx = build_context(fit, target, endpoint, float(source_row["target_std"]))
    funcs = build_physical_functions(ctx)
    obj_value, obj_grad, obj_jac, obj_hess, cons_raw, cons_scaled, cons_grad, cons_hess = funcs
    starts = build_starts(
        ctx,
        cons_scaled,
        cons_grad,
        rng=rng,
        cov_dirs=cov_dirs,
        random_starts=random_starts,
        cov_scale=cov_scale,
        random_scale=random_scale,
    )

    attempt_rows: list[dict] = []
    t_endpoint = time.perf_counter()
    for start_name, start_kind, start in starts:
        projected_cviol = abs(float(cons_scaled(start))) if np.all(np.isfinite(start)) else math.inf
        with stdout_log.open("a") as log:
            log.write(
                f"start {label} / {target} / {side} {start_name} "
                f"scaled_cviol={projected_cviol:.3g}\n"
            )
            log.flush()
        for solver in solvers:
            result = solve_from_start(
                ctx,
                start,
                solver,
                obj_value,
                obj_grad,
                obj_jac,
                obj_hess,
                cons_scaled,
                cons_grad,
                cons_hess,
            )
            row = attempt_row(
                ctx,
                label,
                target,
                side,
                endpoint,
                start_name,
                start_kind,
                solver,
                result,
                current_profile_chi2,
                target_chi2,
                obj_value,
                obj_grad,
                cons_raw,
                cons_scaled,
                cons_grad,
            )
            attempt_rows.append(row)
            with stdout_log.open("a") as log:
                log.write(
                    f"  {solver} success={row['success']} feasible={row['feasible']} "
                    f"chi2={row['chi2']} improvement={row['improvement_vs_current']} "
                    f"scaled_cviol={row['scaled_constraint_violation']} "
                    f"message={row['message']}\n"
                )

    feasible_rows = [
        row for row in attempt_rows
        if row.get("feasible") and np.isfinite(float(row.get("chi2", math.inf)))
    ]
    best = min(feasible_rows, key=lambda row: float(row["chi2"])) if feasible_rows else None
    endpoint_runtime_sec = time.perf_counter() - t_endpoint
    if best is None:
        status = "no_feasible_comparator"
        best_values = {}
    else:
        improvement = current_profile_chi2 - float(best["chi2"])
        status = "material_improvement" if improvement > MATERIAL_IMPROVEMENT else "ok"
        best_values = {
            "best_comparator_chi2": float(best["chi2"]),
            "best_comparator_residual": float(best["residual"]),
            "improvement_vs_current": improvement,
            "material_improvement": bool(improvement > MATERIAL_IMPROVEMENT),
            "best_start_name": best["start_name"],
            "best_solver": best["solver"],
            "best_success": best["success"],
            "best_constraint_violation": best["constraint_violation"],
            "best_scaled_constraint_violation": best["scaled_constraint_violation"],
            "best_bound_violation": best["bound_violation"],
            "best_simplex_violation": best["simplex_violation"],
            "best_eq_projected_grad_norm": best.get("eq_projected_grad_norm"),
            "best_active_fixed_grad_norm": best.get("active_fixed_grad_norm"),
            "best_objective_grad_norm": best.get("objective_grad_norm"),
            "best_min_decay_param": best.get("min_decay_param"),
            "best_max_decay_param": best.get("max_decay_param"),
            "best_max_abs_decay_fitted_coord": best.get("max_abs_decay_fitted_coord"),
            "best_decay_active_lower_1e8": best.get("decay_active_lower_1e8"),
            "best_decay_active_upper_1e8": best.get("decay_active_upper_1e8"),
        }

    summary = {
        "label": label,
        "algorithm": fit["algorithm"],
        "target": target,
        "target_kind": ctx.target_kind,
        "side": side,
        "endpoint": endpoint,
        "target_value": float(source_row["target_value"]),
        "target_std": float(source_row["target_std"]),
        "chi2_min": chi2_min,
        "target_chi2": target_chi2,
        "current_profile_chi2": current_profile_chi2,
        "current_residual": current_residual,
        "current_method": current_method,
        "n_decay_params": int(len(ctx.decay_idxs)),
        "simplex_groups": int(len(ctx.simplex_groups)),
        "n_starts": int(len(starts)),
        "n_attempts": int(len(attempt_rows)),
        "n_feasible": int(len(feasible_rows)),
        "n_success": int(sum(bool(row.get("success")) for row in attempt_rows)),
        "n_failed": int(sum(not bool(row.get("feasible")) for row in attempt_rows)),
        "fit_runtime_sec": fit_runtime_sec,
        "endpoint_runtime_sec": endpoint_runtime_sec,
        "status": status,
        **best_values,
    }
    return summary, attempt_rows


def load_source_rows(path: Path) -> dict[tuple[str, str], dict]:
    df = pd.read_csv(path)
    rows: dict[tuple[str, str], dict] = {}
    for _, row in df.iterrows():
        if row.get("status") != "ok":
            continue
        rows[(str(row["label"]), str(row["target"]))] = row.to_dict()
    return rows


def read_completed(summary_csv: Path) -> set[tuple[str, str, str]]:
    if not summary_csv.exists():
        return set()
    completed = set()
    with summary_csv.open(newline="") as f:
        for row in csv.DictReader(f):
            if row.get("status"):
                completed.add((row.get("label", ""), row.get("target", ""), row.get("side", "")))
    return completed


def prepare_outputs(paths: list[Path], resume: bool) -> None:
    if resume:
        return
    for path in paths:
        if path.exists():
            path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-csv", default="notes/asym_fit_sweep_results_simplified.csv")
    parser.add_argument("--endpoint", action="append", default=None, help="LABEL::TARGET::upper|lower")
    parser.add_argument("--summary-csv", default="notes/codex-refactor/round06_physical_profile_summary.csv")
    parser.add_argument("--summary-jsonl", default="notes/codex-refactor/round06_physical_profile_summary.jsonl")
    parser.add_argument("--attempts-csv", default="notes/codex-refactor/round06_physical_profile_attempts.csv")
    parser.add_argument("--attempts-jsonl", default="notes/codex-refactor/round06_physical_profile_attempts.jsonl")
    parser.add_argument("--stdout-log", default="notes/logs/round06_physical_profile_comparator_stdout.log")
    parser.add_argument("--target-timeout-sec", type=int, default=300)
    parser.add_argument("--cov-dirs", type=int, default=1)
    parser.add_argument("--random-starts", type=int, default=2)
    parser.add_argument("--cov-scale", type=float, default=0.5)
    parser.add_argument("--random-scale", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=20260625)
    parser.add_argument("--solvers", default="slsqp,trust-constr")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    summary_csv = Path(args.summary_csv)
    summary_jsonl = Path(args.summary_jsonl)
    attempts_csv = Path(args.attempts_csv)
    attempts_jsonl = Path(args.attempts_jsonl)
    stdout_log = Path(args.stdout_log)
    stdout_log.parent.mkdir(parents=True, exist_ok=True)
    prepare_outputs([summary_csv, summary_jsonl, attempts_csv, attempts_jsonl], args.resume)

    endpoints = args.endpoint or DEFAULT_ENDPOINTS
    source_rows = load_source_rows(Path(args.source_csv))
    completed = read_completed(summary_csv) if args.resume else set()
    solvers = [solver.strip() for solver in args.solvers.split(",") if solver.strip()]
    rng = np.random.default_rng(args.seed)

    old_handler = signal.signal(signal.SIGALRM, _alarm_handler)
    fit_cache: dict[str, tuple[dict, float]] = {}
    try:
        for idx, spec in enumerate(endpoints, start=1):
            label, target, side = parse_endpoint_spec(spec)
            if (label, target, side) in completed:
                print(f"[{idx}/{len(endpoints)}] {spec} already complete")
                continue
            source_row = source_rows.get((label, target))
            if source_row is None:
                row = {
                    "label": label,
                    "target": target,
                    "side": side,
                    "status": "missing_source_row",
                    "exception": f"No ok source row for {label}::{target} in {args.source_csv}",
                }
                append_csv(summary_csv, row, SUMMARY_FIELDS)
                append_jsonl(summary_jsonl, row)
                print(f"[{idx}/{len(endpoints)}] {spec} missing source row")
                continue

            with stdout_log.open("a") as log:
                log.write(f"\n===== {idx}/{len(endpoints)} {spec} =====\n")
                log.flush()
            try:
                signal.alarm(args.target_timeout_sec)
                if label not in fit_cache:
                    t_fit = time.perf_counter()
                    with stdout_log.open("a") as log:
                        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                            fit = run_fit(label, verbose=False)
                    if fit is None:
                        raise RuntimeError(f"fit {label!r} was skipped")
                    fit_cache[label] = (fit, time.perf_counter() - t_fit)
                fit, fit_runtime_sec = fit_cache[label]
                summary, attempts = run_endpoint(
                    fit,
                    source_row,
                    side,
                    rng=rng,
                    cov_dirs=args.cov_dirs,
                    random_starts=args.random_starts,
                    cov_scale=args.cov_scale,
                    random_scale=args.random_scale,
                    solvers=solvers,
                    fit_runtime_sec=fit_runtime_sec,
                    stdout_log=stdout_log,
                )
                signal.alarm(0)
            except Exception as exc:  # noqa: BLE001 - diagnostic row records failure
                signal.alarm(0)
                summary = {
                    "label": label,
                    "target": target,
                    "side": side,
                    "status": "error",
                    "exception_type": type(exc).__name__,
                    "exception": str(exc),
                }
                attempts = []
                with stdout_log.open("a") as log:
                    log.write(f"EXCEPTION {type(exc).__name__}: {exc}\n")

            append_csv(summary_csv, summary, SUMMARY_FIELDS)
            append_jsonl(summary_jsonl, summary)
            for attempt in attempts:
                append_csv(attempts_csv, attempt, ATTEMPT_FIELDS)
                append_jsonl(attempts_jsonl, attempt)
            print(
                f"[{idx}/{len(endpoints)}] {spec} {summary.get('status')} "
                f"best={summary.get('best_comparator_chi2')} "
                f"improvement={summary.get('improvement_vs_current')} "
                f"feasible={summary.get('n_feasible')}/{summary.get('n_attempts')}"
            )
            gc.collect()
            try:
                jax.clear_caches()
            except Exception:
                pass
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


if __name__ == "__main__":
    main()
