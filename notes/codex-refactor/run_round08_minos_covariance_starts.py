#!/usr/bin/env python
"""Round 08 notes-only MINOS-style covariance-start diagnostics.

This harness tests the local quadratic/MINOS predictor

    dx = C grad(t) (target - t0) / (grad(t)^T C grad(t))

inside the existing pdgfits fixed-target profile machinery.  It deliberately
does not alter build_chi2.py or replace bracketed endpoint bisection.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import jax
from jax import numpy as jnp
import numpy as np
from scipy.optimize import NonlinearConstraint, OptimizeResult, minimize, root

from pdgfits.asym_errors import (
    CallableProfileProblem,
    ProfilePoint,
    calc_asym_errors,
    find_profile_root,
)
from pdgfits.avg import run_avg
from pdgfits.fit import run_fit
from pdgfits.param_maps import get_decay_info
from pdgfits.query import avg_queries


jax.config.update("jax_enable_x64", True)


@dataclass(frozen=True)
class CaseSpec:
    case_id: str
    kind: str
    label_or_node: str
    target: str
    rationale: str


DEFAULT_CASES = [
    CaseSpec(
        case_id="fit_b0_s042b95",
        kind="fit",
        label_or_node="B0",
        target="S042B95",
        rationale="hard BRU node ratio target with severe fitted-chart saturation",
    ),
    CaseSpec(
        case_id="fit_b0s_s08637",
        kind="fit",
        label_or_node="B0S-BR",
        target="S086.37",
        rationale="hard BRU physical parameter target; round 06 physical chart matched production",
    ),
    CaseSpec(
        case_id="fit_eta_m026w",
        kind="fit",
        label_or_node="eta_c J/psi psi(2S)",
        target="M026W",
        rationale="hard direct non-decay coordinate in a BRU fit",
    ),
    CaseSpec(
        case_id="avg_m026r08",
        kind="avg",
        label_or_node="M026R08",
        target="M026R08",
        rationale="direct-coordinate average with nuisance adjustments from round 07",
    ),
]


POLICIES = ("covariance_only", "current_plus_covariance")


def finite_or_none(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (bool, str, int)):
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


def as_float(x: Any) -> float:
    return float(np.asarray(x, dtype=np.float64))


def clean_message(message: Any) -> str:
    return " ".join(str(message).split())


def norm_or_none(x: Any) -> float | None:
    if x is None:
        return None
    arr = np.asarray(x, dtype=np.float64)
    if not np.all(np.isfinite(arr)):
        return None
    return float(np.linalg.norm(arr))


def safe_chi2(chi2: Callable[[Any], Any], x: np.ndarray) -> float:
    try:
        value = as_float(chi2(jnp.asarray(x, dtype=jnp.float64)))
    except Exception:  # noqa: BLE001 - diagnostics-only guard
        return math.inf
    return value if math.isfinite(value) else math.inf


def covariance_prediction(
    x0: np.ndarray,
    covariance: np.ndarray | None,
    target_grad: np.ndarray,
    displacement: float,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    info: dict[str, Any] = {
        "predictor_source": "covariance",
        "predictor_available": False,
        "predictor_displacement": displacement,
    }
    if covariance is None:
        info["predictor_failure"] = "missing_covariance"
        return None, info
    cov = np.asarray(covariance, dtype=np.float64)
    grad = np.asarray(target_grad, dtype=np.float64)
    x0 = np.asarray(x0, dtype=np.float64)
    if cov.shape != (len(x0), len(x0)) or grad.shape != (len(x0),):
        info["predictor_failure"] = f"shape_mismatch_cov={cov.shape}_grad={grad.shape}_n={len(x0)}"
        return None, info
    if not (np.all(np.isfinite(cov)) and np.all(np.isfinite(grad))):
        info["predictor_failure"] = "nonfinite_cov_or_grad"
        return None, info
    cov_grad = cov @ grad
    denom = float(grad @ cov_grad)
    info.update(
        {
            "predictor_grad_norm": float(np.linalg.norm(grad)),
            "predictor_cov_grad_norm": float(np.linalg.norm(cov_grad)),
            "predictor_denom": denom,
        }
    )
    if not math.isfinite(denom) or abs(denom) <= 1e-30:
        info["predictor_failure"] = "singular_or_zero_target_variance"
        return None, info
    dx = cov_grad * (float(displacement) / denom)
    x_pred = x0 + dx
    info.update(
        {
            "predictor_available": bool(np.all(np.isfinite(x_pred))),
            "predictor_dx_norm": norm_or_none(dx),
        }
    )
    if not np.all(np.isfinite(x_pred)):
        info["predictor_failure"] = "nonfinite_prediction"
        return None, info
    return x_pred, info


def distinct_start(starts: list[tuple[str, str, np.ndarray, dict[str, Any]]], x: np.ndarray) -> bool:
    x = np.asarray(x, dtype=np.float64)
    for _, _, existing, _ in starts:
        scale = max(float(np.linalg.norm(existing)), float(np.linalg.norm(x)), 1.0)
        if np.linalg.norm(existing - x) <= 1e-12 * scale:
            return False
    return True


def chart_summary(parameters: list[str], fitted_to_params: Callable[[Any], Any], x: np.ndarray | None) -> dict[str, Any]:
    if x is None:
        return {}
    out: dict[str, Any] = {}
    try:
        params = np.asarray(fitted_to_params(jnp.asarray(x, dtype=jnp.float64)), dtype=np.float64)
    except Exception:  # noqa: BLE001 - diagnostics-only guard
        out["chart_eval_failed"] = True
        return out
    x = np.asarray(x, dtype=np.float64)
    _, decay_idxs = get_decay_info(parameters)
    decay_idxs = np.asarray(decay_idxs, dtype=int)
    out["n_decay_params"] = int(len(decay_idxs))
    if len(decay_idxs) == 0:
        return out
    decay_params = params[decay_idxs]
    decay_fitted = x[decay_idxs]
    out.update(
        {
            "min_decay_param": float(np.nanmin(decay_params)),
            "max_decay_param": float(np.nanmax(decay_params)),
            "max_abs_decay_fitted_coord": float(np.nanmax(np.abs(decay_fitted))),
            "decay_active_lower_1e8": int(np.sum(decay_params <= 1e-8)),
            "decay_active_upper_1e8": int(np.sum(decay_params >= 1 - 1e-8)),
            "chart_saturation_warning": bool(
                np.nanmax(np.abs(decay_fitted)) > 10
                or np.nanmin(decay_params) < 1e-6
                or np.nanmax(decay_params) > 1 - 1e-6
            ),
        }
    )
    return out


def build_constrained_profile_for_policy(
    fit: dict[str, Any],
    target_func: Callable[[Any], Any],
    target_value: float,
    target_scale: float,
    *,
    policy: str,
    target_name: str,
) -> Callable[[float], float]:
    """Local copy of the production constrained profile with start diagnostics."""
    chi2 = fit["chi2"]
    fitted_to_params = fit["fitted_params_to_params"]
    fp_best = np.asarray(fit["fitted_values"], dtype=np.float64)
    covariance = np.asarray(fit["covariance"], dtype=np.float64) if fit.get("covariance") is not None else None
    parameters = list(fit["parameters"])

    if target_scale <= 0 or not math.isfinite(target_scale):
        target_scale = max(abs(float(target_value)), 1.0)
    cons_tol = 1e-7
    stationarity_fun_tol = 1e-4

    chi2_grad_jax = jax.jit(jax.grad(chi2))
    chi2_hess_jax = jax.jit(jax.hessian(chi2))

    def chi2_np(x: np.ndarray) -> float:
        return safe_chi2(chi2, np.asarray(x, dtype=np.float64))

    def chi2_grad_np(x: np.ndarray) -> np.ndarray:
        return np.asarray(chi2_grad_jax(jnp.asarray(x, dtype=jnp.float64)), dtype=np.float64)

    def chi2_hess_np(x: np.ndarray) -> np.ndarray:
        return np.asarray(chi2_hess_jax(jnp.asarray(x, dtype=jnp.float64)), dtype=np.float64)

    @jax.jit
    def target_fitted_jax(fp):
        return target_func(fitted_to_params(fp))

    @jax.jit
    def constraint_value_jax(fp, fixed_value):
        return target_fitted_jax(fp) - fixed_value

    @jax.jit
    def scaled_constraint_value_jax(fp, fixed_value):
        return constraint_value_jax(fp, fixed_value) / target_scale

    constraint_grad_jax = jax.jit(jax.grad(lambda fp, fixed_value: constraint_value_jax(fp, fixed_value)))
    scaled_constraint_grad_jax = jax.jit(jax.grad(lambda fp, fixed_value: scaled_constraint_value_jax(fp, fixed_value)))
    scaled_constraint_hess_jax = jax.jit(
        jax.hessian(lambda fp, fixed_value: scaled_constraint_value_jax(fp, fixed_value))
    )
    target_grad_best = np.asarray(jax.grad(target_fitted_jax)(jnp.asarray(fp_best, dtype=jnp.float64)), dtype=np.float64)

    last_x = fp_best.copy()
    diagnostics: list[ProfilePoint] = []
    details: list[dict[str, Any]] = []

    def project_start(start: np.ndarray, v: float, tol: float = cons_tol) -> np.ndarray:
        x = np.asarray(start, dtype=np.float64).copy()
        for _ in range(20):
            if not np.all(np.isfinite(x)):
                break
            c = as_float(scaled_constraint_value_jax(jnp.asarray(x, dtype=jnp.float64), v))
            if not math.isfinite(c):
                break
            if abs(c) <= tol:
                return x
            g = np.asarray(scaled_constraint_grad_jax(jnp.asarray(x, dtype=jnp.float64), v), dtype=np.float64)
            denom = float(np.dot(g, g))
            if not np.all(np.isfinite(g)) or not math.isfinite(denom) or denom == 0.0:
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

    def profile(v: float) -> float:
        nonlocal last_x
        v = float(v)
        t_profile_start = time.perf_counter()

        def make_constraint(exact_hess: bool = False) -> NonlinearConstraint:
            kwargs = {}
            if exact_hess:
                kwargs["hess"] = lambda x, multiplier: np.asarray(
                    multiplier[0] * scaled_constraint_hess_jax(jnp.asarray(x, dtype=jnp.float64), v),
                    dtype=np.float64,
                )
            return NonlinearConstraint(
                fun=lambda x: as_float(scaled_constraint_value_jax(jnp.asarray(x, dtype=jnp.float64), v)),
                lb=0.0,
                ub=0.0,
                jac=lambda x: np.asarray(scaled_constraint_grad_jax(jnp.asarray(x, dtype=jnp.float64), v), dtype=np.float64),
                **kwargs,
            )

        def scaled_violation(x: np.ndarray) -> float:
            return abs(as_float(constraint_value_jax(jnp.asarray(x, dtype=jnp.float64), v))) / target_scale

        def projected_grad_norm(x: np.ndarray) -> float:
            x = np.asarray(x, dtype=np.float64)
            g_obj = chi2_grad_np(x)
            g_con = np.asarray(scaled_constraint_grad_jax(jnp.asarray(x, dtype=jnp.float64), v), dtype=np.float64)
            denom = float(np.dot(g_con, g_con))
            if not math.isfinite(denom) or denom == 0.0:
                return math.inf
            g_proj = g_obj - (float(np.dot(g_obj, g_con)) / denom) * g_con
            return float(np.linalg.norm(g_proj))

        def projected_gradient(x: np.ndarray) -> np.ndarray | None:
            x = np.asarray(x, dtype=np.float64)
            g_obj = chi2_grad_np(x)
            g_con = np.asarray(scaled_constraint_grad_jax(jnp.asarray(x, dtype=jnp.float64), v), dtype=np.float64)
            denom = float(np.dot(g_con, g_con))
            if not math.isfinite(denom) or denom == 0.0:
                return None
            return g_obj - (float(np.dot(g_obj, g_con)) / denom) * g_con

        def projected_descent_improvement(x: np.ndarray) -> float:
            x = np.asarray(x, dtype=np.float64)
            if not np.all(np.isfinite(x)):
                return math.inf
            base_fun = chi2_np(x)
            if not math.isfinite(base_fun) or scaled_violation(x) > cons_tol:
                return math.inf
            g_proj = projected_gradient(x)
            if g_proj is None:
                return math.inf
            g_proj_norm = float(np.linalg.norm(g_proj))
            if not math.isfinite(g_proj_norm) or g_proj_norm == 0.0:
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
                if math.isfinite(trial_fun):
                    best_decrease = max(best_decrease, base_fun - trial_fun)
            return float(best_decrease)

        def result_ok(opt_result: OptimizeResult) -> bool:
            cached = getattr(opt_result, "profile_result_ok", None)
            if cached is not None:
                return bool(cached)
            scaled_cviol = scaled_violation(opt_result.x)
            finite_result = math.isfinite(float(opt_result.fun))
            if not finite_result or scaled_cviol > cons_tol:
                opt_result.profile_result_ok = False
                return False
            g_norm = float(np.linalg.norm(chi2_grad_np(opt_result.x)))
            pg_norm = projected_grad_norm(opt_result.x)
            if pg_norm <= max(1e-3, 1e-6 * g_norm):
                opt_result.profile_descent_improvement = 0.0
                opt_result.profile_result_ok = True
                return True
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

        def kkt_polish(opt_result: OptimizeResult) -> OptimizeResult:
            if not math.isfinite(float(opt_result.fun)) or not np.all(np.isfinite(opt_result.x)):
                return opt_result
            x0 = project_start(opt_result.x, v)
            g_obj = chi2_grad_np(x0)
            g_con = np.asarray(scaled_constraint_grad_jax(jnp.asarray(x0, dtype=jnp.float64), v), dtype=np.float64)
            denom = float(np.dot(g_con, g_con))
            if not math.isfinite(denom) or denom == 0.0:
                return opt_result
            lambda0 = -float(np.dot(g_obj, g_con)) / denom

            def residual(z):
                x = np.asarray(z[:-1], dtype=np.float64)
                lam = float(z[-1])
                stationarity = (
                    chi2_grad_np(x)
                    + lam * np.asarray(scaled_constraint_grad_jax(jnp.asarray(x, dtype=jnp.float64), v), dtype=np.float64)
                )
                return np.r_[stationarity, as_float(scaled_constraint_value_jax(jnp.asarray(x, dtype=jnp.float64), v))]

            try:
                polished = root(residual, np.r_[x0, lambda0], method="hybr", options={"maxfev": 5000})
            except (FloatingPointError, ValueError, np.linalg.LinAlgError):
                return opt_result
            x_polished = np.asarray(polished.x[:-1], dtype=np.float64)
            if not np.all(np.isfinite(x_polished)) or scaled_violation(x_polished) > cons_tol:
                return opt_result
            fun_polished = chi2_np(x_polished)
            if not math.isfinite(fun_polished):
                return opt_result
            if fun_polished > opt_result.fun + max(1e-7, 1e-9 * abs(float(opt_result.fun))):
                return opt_result
            profile_method = getattr(opt_result, "profile_method", "")
            profile_method = f"{profile_method}+KKT" if profile_method else "KKT"
            return OptimizeResult(
                x=x_polished,
                fun=fun_polished,
                success=bool(polished.success),
                status=getattr(polished, "status", None),
                message=f"KKT polish: {polished.message}",
                nfev=getattr(polished, "nfev", None),
                profile_method=profile_method,
            )

        def maybe_keep_feasible_start(opt_result: OptimizeResult, x0: np.ndarray) -> OptimizeResult:
            x0_fun = chi2_np(x0)
            if not math.isfinite(x0_fun) or scaled_violation(x0) > cons_tol:
                return opt_result
            if (not math.isfinite(float(opt_result.fun))) or (
                x0_fun <= opt_result.fun + max(1e-8, 1e-10 * abs(float(opt_result.fun)))
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

        def solve_from(x0: np.ndarray, start_name: str, start_kind: str) -> OptimizeResult:
            candidates = []
            result = OptimizeResult(
                x=np.asarray(x0, dtype=np.float64),
                fun=chi2_np(x0),
                success=False,
                message="projected start only",
                profile_method="projected-start",
            )
            try:
                result = minimize(
                    chi2_np,
                    x0,
                    method="SLSQP",
                    jac=chi2_grad_np,
                    constraints=[make_constraint()],
                    options={"ftol": 1e-10, "maxiter": 2000},
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
            result.profile_start_name = start_name
            result.profile_start_kind = start_kind
            candidates.append(result)
            if result_ok(result):
                return result

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
            exact_hess_result.profile_start_name = start_name
            exact_hess_result.profile_start_kind = start_kind
            exact_hess_result = kkt_polish(exact_hess_result)
            exact_hess_result.profile_start_name = start_name
            exact_hess_result.profile_start_kind = start_kind
            candidates.append(exact_hess_result)
            if result_ok(exact_hess_result):
                return exact_hess_result

            ok_candidates = [candidate for candidate in candidates if result_ok(candidate)]
            if ok_candidates:
                return min(ok_candidates, key=lambda candidate: candidate.fun)
            return min(candidates, key=lambda candidate: candidate.fun if math.isfinite(float(candidate.fun)) else math.inf)

        starts: list[tuple[str, str, np.ndarray, dict[str, Any]]] = []
        if policy == "current_plus_covariance":
            starts.append(("continuation_projected", "current", project_start(last_x, v), {}))
            best_start = project_start(fp_best, v)
            if distinct_start(starts, best_start):
                starts.append(("mle_projected", "current", best_start, {}))

        if policy in ("covariance_only", "current_plus_covariance"):
            raw_pred, pred_info = covariance_prediction(
                fp_best,
                covariance,
                target_grad_best,
                v - float(target_value),
            )
            if raw_pred is not None:
                pred_info["covariance_raw_target"] = finite_or_none(as_float(target_fitted_jax(jnp.asarray(raw_pred, dtype=jnp.float64))))
                pred_info["covariance_raw_scaled_violation"] = (
                    (pred_info["covariance_raw_target"] - v) / target_scale
                    if pred_info["covariance_raw_target"] is not None else None
                )
                pred_projected = project_start(raw_pred, v)
                pred_info["covariance_projected_target"] = finite_or_none(
                    as_float(target_fitted_jax(jnp.asarray(pred_projected, dtype=jnp.float64)))
                )
                pred_info["covariance_projection_norm"] = norm_or_none(pred_projected - raw_pred)
                pred_info.update({f"cov_raw_{k}": val for k, val in chart_summary(parameters, fitted_to_params, raw_pred).items()})
                pred_info.update({f"cov_projected_{k}": val for k, val in chart_summary(parameters, fitted_to_params, pred_projected).items()})
                if distinct_start(starts, pred_projected):
                    starts.append(("covariance_projected", "covariance", pred_projected, pred_info))
                elif policy == "covariance_only":
                    starts.append(("covariance_projected_duplicate", "covariance", pred_projected, pred_info))

        if not starts:
            starts.append(("fallback_mle_projected", "fallback", project_start(fp_best, v), {}))

        candidates = []
        candidate_records = []
        for start_name, start_kind, start, start_meta in starts:
            result = solve_from(start, start_name, start_kind)
            ok = result_ok(result)
            record = {
                "policy": policy,
                "target": target_name,
                "value": v,
                "start_name": start_name,
                "start_kind": start_kind,
                "start_fun": chi2_np(start),
                "start_scaled_constraint_violation": scaled_violation(start),
                "candidate_chi2": float(result.fun) if math.isfinite(float(result.fun)) else None,
                "candidate_ok": ok,
                "candidate_success": bool(getattr(result, "success", False)),
                "candidate_method": str(getattr(result, "profile_method", "")),
                "candidate_message": clean_message(getattr(result, "message", "")),
                "candidate_nfev": getattr(result, "nfev", None),
                "candidate_nit": getattr(result, "nit", None),
                "candidate_scaled_constraint_violation": scaled_violation(result.x),
                "candidate_projected_grad_norm": projected_grad_norm(result.x)
                if math.isfinite(float(result.fun)) else None,
                "candidate_objective_grad_norm": float(np.linalg.norm(chi2_grad_np(result.x)))
                if math.isfinite(float(result.fun)) else None,
                "candidate_descent_improvement": getattr(result, "profile_descent_improvement", None),
                **start_meta,
            }
            record.update({f"candidate_{k}": val for k, val in chart_summary(parameters, fitted_to_params, result.x).items()})
            candidate_records.append(record)
            candidates.append(result)

        ok_candidates = [candidate for candidate in candidates if result_ok(candidate)]
        if ok_candidates:
            result = min(ok_candidates, key=lambda candidate: candidate.fun)
        else:
            result = min(candidates, key=lambda candidate: candidate.fun if math.isfinite(float(candidate.fun)) else math.inf)

        current_records = [r for r in candidate_records if r["start_kind"] == "current" and r["candidate_chi2"] is not None]
        covariance_records = [r for r in candidate_records if r["start_kind"] == "covariance" and r["candidate_chi2"] is not None]
        current_best = min((r["candidate_chi2"] for r in current_records), default=None)
        covariance_best = min((r["candidate_chi2"] for r in covariance_records), default=None)

        cviol = abs(as_float(constraint_value_jax(jnp.asarray(result.x, dtype=jnp.float64), v)))
        scaled_cviol = cviol / target_scale
        finite = math.isfinite(float(result.fun))
        ok = result_ok(result)
        g_norm = float(np.linalg.norm(chi2_grad_np(result.x))) if finite else math.inf
        pg_norm = projected_grad_norm(result.x) if finite else math.inf
        point = ProfilePoint(
            v,
            float(result.fun) if finite else math.inf,
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
        detail = {
            "policy": policy,
            "target": target_name,
            "value": v,
            "selected_start_name": str(getattr(result, "profile_start_name", "")),
            "selected_start_kind": str(getattr(result, "profile_start_kind", "")),
            "selected_method": point.method,
            "selected_chi2": point.chi2,
            "selected_ok": ok,
            "selected_nfev": point.nfev,
            "selected_nit": point.nit,
            "selected_runtime_sec": point.runtime_sec,
            "n_candidate_starts": len(candidate_records),
            "candidate_nfev_total": sum(int(r["candidate_nfev"] or 0) for r in candidate_records),
            "current_best_chi2": current_best,
            "covariance_best_chi2": covariance_best,
            "covariance_minus_current_chi2": (
                covariance_best - current_best if covariance_best is not None and current_best is not None else None
            ),
            "covariance_found_lower_than_current": (
                bool(covariance_best < current_best - 1e-8)
                if covariance_best is not None and current_best is not None else None
            ),
            "candidate_records": candidate_records,
        }
        detail.update({f"selected_{k}": val for k, val in chart_summary(parameters, fitted_to_params, result.x).items()})
        point.round08_detail = detail
        diagnostics.append(point)
        details.append(detail)
        profile.last_point = point
        if not ok:
            raise RuntimeError(
                f"Profile minimization failed for {target_name}={v}: success={result.success}, "
                f"finite={finite}, constraint_violation={cviol:.3g}, "
                f"scaled_constraint_violation={scaled_cviol:.3g}, projected_grad_norm={pg_norm:.3g}, "
                f"message={result.message}"
            )
        last_x = np.asarray(result.x, dtype=np.float64)
        return point.chi2

    profile.diagnostics = diagnostics
    profile.last_point = None
    profile.round08_details = details
    return profile


def build_avg_profile_for_policy(
    avg_result: dict[str, Any],
    covariance: np.ndarray,
    *,
    policy: str,
    target_name: str,
) -> Callable[[float], float]:
    """Average-side fixed-coordinate profile with optional covariance nuisance start."""
    parameters = list(avg_result["parameters"])
    primary_idx = parameters.index(target_name)
    param_values = np.asarray(avg_result["param_values"], dtype=np.float64)
    base_nuisance = np.delete(param_values, primary_idx)
    last_nuisance = base_nuisance.copy()
    chi2 = avg_result["chi2"]
    chi2_val_and_grad = jax.jit(jax.value_and_grad(chi2))
    details: list[dict[str, Any]] = []
    diagnostics: list[ProfilePoint] = []

    def full_params(x_primary: float, x_nuisance: np.ndarray) -> np.ndarray:
        return np.insert(np.asarray(x_nuisance, dtype=np.float64), primary_idx, x_primary)

    def profile_chi2(x_primary: float) -> float:
        nonlocal last_nuisance
        x_primary = float(x_primary)
        t0 = time.perf_counter()
        if len(parameters) == 1:
            x = np.array([x_primary], dtype=np.float64)
            prof_chi2 = safe_chi2(chi2, x)
            point = ProfilePoint(x_primary, prof_chi2, True, 0.0, "direct", method="direct")
            detail = {
                "policy": policy,
                "target": target_name,
                "value": x_primary,
                "selected_start_name": "direct",
                "selected_start_kind": "direct",
                "selected_method": "direct",
                "selected_chi2": prof_chi2,
                "selected_ok": True,
                "n_candidate_starts": 0,
                "candidate_nfev_total": 0,
            }
            point.round08_detail = detail
            diagnostics.append(point)
            details.append(detail)
            profile_chi2.last_point = point
            return prof_chi2

        def obj(x_nuisance: np.ndarray) -> float:
            return safe_chi2(chi2, full_params(x_primary, x_nuisance))

        def obj_and_grad(x_nuisance: np.ndarray) -> tuple[float, np.ndarray]:
            val_jax, grad_jax = chi2_val_and_grad(
                jnp.asarray(full_params(x_primary, x_nuisance), dtype=jnp.float64)
            )
            return float(val_jax), np.delete(np.asarray(grad_jax, dtype=np.float64), primary_idx)

        def grad(x_nuisance: np.ndarray) -> np.ndarray:
            return obj_and_grad(x_nuisance)[1]

        starts: list[tuple[str, str, np.ndarray, dict[str, Any]]] = []
        if policy == "current_plus_covariance":
            starts.append(("continuation_nuisance", "current", last_nuisance.copy(), {}))
            if distinct_start(starts, base_nuisance):
                starts.append(("mle_nuisance", "current", base_nuisance.copy(), {}))

        if policy in ("covariance_only", "current_plus_covariance"):
            cov = np.asarray(covariance, dtype=np.float64)
            denom = float(cov[primary_idx, primary_idx]) if cov.shape == (len(parameters), len(parameters)) else math.nan
            pred_meta = {
                "predictor_source": "hessian_covariance",
                "predictor_available": False,
                "predictor_displacement": x_primary - float(param_values[primary_idx]),
                "predictor_denom": denom,
            }
            if math.isfinite(denom) and abs(denom) > 1e-30:
                pred_full = param_values + cov[:, primary_idx] * ((x_primary - param_values[primary_idx]) / denom)
                pred_full[primary_idx] = x_primary
                pred_nuisance = np.delete(pred_full, primary_idx)
                pred_meta.update(
                    {
                        "predictor_available": bool(np.all(np.isfinite(pred_nuisance))),
                        "predictor_dx_norm": norm_or_none(pred_full - param_values),
                    }
                )
                if np.all(np.isfinite(pred_nuisance)):
                    if distinct_start(starts, pred_nuisance):
                        starts.append(("covariance_nuisance", "covariance", pred_nuisance, pred_meta))
                    elif policy == "covariance_only":
                        starts.append(("covariance_nuisance_duplicate", "covariance", pred_nuisance, pred_meta))
            else:
                pred_meta["predictor_failure"] = "singular_or_zero_primary_variance"

        if not starts:
            starts.append(("fallback_mle_nuisance", "fallback", base_nuisance.copy(), {}))

        candidates = []
        candidate_records = []
        messages = []
        for start_name, start_kind, start, start_meta in starts:
            start = np.asarray(start, dtype=np.float64)
            start_fun = obj(start)
            local_candidates: list[tuple[float, np.ndarray, bool, str, str, int | None, int | None]] = []
            if math.isfinite(start_fun):
                local_candidates.append((start_fun, start, True, "feasible start", "feasible-start", 0, 0))
            run_nelder_mead = True
            try:
                bfgs = minimize(
                    fun=obj_and_grad,
                    x0=start,
                    jac=True,
                    method="BFGS",
                    options={"maxiter": max(1000, 500 * len(start)), "gtol": 1e-8},
                )
                messages.append(f"{start_name} BFGS success={bfgs.success}: {bfgs.message}")
                if math.isfinite(float(bfgs.fun)):
                    bfgs_x = np.asarray(bfgs.x, dtype=np.float64)
                    local_candidates.append(
                        (float(bfgs.fun), bfgs_x, bool(bfgs.success), str(bfgs.message), "BFGS", bfgs.nfev, bfgs.nit)
                    )
                    if bfgs.success:
                        run_nelder_mead = False
                    else:
                        bfgs_grad_norm = float(np.linalg.norm(grad(bfgs_x)))
                        if bfgs_grad_norm <= 1e-5 * max(abs(float(bfgs.fun)), 1.0):
                            run_nelder_mead = False
            except Exception as exc:  # noqa: BLE001 - notes-only fallback records failure
                messages.append(f"{start_name} BFGS exception: {type(exc).__name__}: {exc}")

            if run_nelder_mead:
                nm_start = local_candidates[-1][1] if local_candidates else start
                xatol = max(np.linalg.norm(nm_start) * 1e-10, np.finfo(float).eps)
                try:
                    nm = minimize(
                        fun=obj,
                        x0=nm_start,
                        method="Nelder-Mead",
                        options={
                            "maxiter": max(1000, 500 * len(start)),
                            "xatol": xatol,
                            "fatol": 1e-8,
                            "adaptive": True,
                        },
                    )
                    messages.append(f"{start_name} Nelder-Mead success={nm.success}: {nm.message}")
                    if math.isfinite(float(nm.fun)):
                        local_candidates.append(
                            (float(nm.fun), np.asarray(nm.x, dtype=np.float64), bool(nm.success), str(nm.message), "Nelder-Mead", nm.nfev, nm.nit)
                        )
                except Exception as exc:  # noqa: BLE001 - notes-only fallback records failure
                    messages.append(f"{start_name} Nelder-Mead exception: {type(exc).__name__}: {exc}")

            if not local_candidates:
                continue
            min_fun = min(c[0] for c in local_candidates)
            fun_tol = max(1e-6, 1e-8 * abs(min_fun))
            successful_ties = [c for c in local_candidates if c[2] and c[0] <= min_fun + fun_tol]
            if successful_ties:
                best_fun, best_x, best_success, best_message, method, nfev, nit = min(successful_ties, key=lambda c: c[0])
            else:
                best_fun, best_x, best_success, best_message, method, nfev, nit = min(local_candidates, key=lambda c: c[0])
            nuisance_grad_norm = float(np.linalg.norm(grad(best_x)))
            ok = bool(best_success or nuisance_grad_norm <= 1e-5 * max(abs(best_fun), 1.0))
            candidates.append((best_fun, best_x, ok, best_message, method, start_name, start_kind, nfev, nit))
            candidate_records.append(
                {
                    "policy": policy,
                    "target": target_name,
                    "value": x_primary,
                    "start_name": start_name,
                    "start_kind": start_kind,
                    "start_fun": start_fun,
                    "candidate_chi2": best_fun,
                    "candidate_ok": ok,
                    "candidate_success": bool(best_success),
                    "candidate_method": method,
                    "candidate_message": clean_message(best_message),
                    "candidate_nfev": nfev,
                    "candidate_nit": nit,
                    "candidate_projected_grad_norm": nuisance_grad_norm,
                    **start_meta,
                }
            )

        if not candidates:
            raise RuntimeError(f"Profile chi2 minimization failed for node {target_name}: {'; '.join(messages)}")
        ok_candidates = [c for c in candidates if c[2]]
        if ok_candidates:
            best_fun, best_x, best_ok, best_message, method, start_name, start_kind, nfev, nit = min(ok_candidates, key=lambda c: c[0])
        else:
            best_fun, best_x, best_ok, best_message, method, start_name, start_kind, nfev, nit = min(candidates, key=lambda c: c[0])
        nuisance_grad_norm = float(np.linalg.norm(grad(best_x)))
        ok = bool(best_ok or nuisance_grad_norm <= 1e-5 * max(abs(best_fun), 1.0))
        point = ProfilePoint(
            x_primary,
            best_fun,
            ok,
            0.0,
            f"{best_message}; nuisance_grad_norm={nuisance_grad_norm:.3g}",
            projected_grad_norm=nuisance_grad_norm,
            method=method,
            nfev=nfev,
            nit=nit,
            runtime_sec=time.perf_counter() - t0,
        )
        current_records = [r for r in candidate_records if r["start_kind"] == "current"]
        covariance_records = [r for r in candidate_records if r["start_kind"] == "covariance"]
        current_best = min((r["candidate_chi2"] for r in current_records), default=None)
        covariance_best = min((r["candidate_chi2"] for r in covariance_records), default=None)
        detail = {
            "policy": policy,
            "target": target_name,
            "value": x_primary,
            "selected_start_name": start_name,
            "selected_start_kind": start_kind,
            "selected_method": method,
            "selected_chi2": best_fun,
            "selected_ok": ok,
            "selected_nfev": nfev,
            "selected_nit": nit,
            "selected_runtime_sec": point.runtime_sec,
            "n_candidate_starts": len(candidate_records),
            "candidate_nfev_total": sum(int(r["candidate_nfev"] or 0) for r in candidate_records),
            "current_best_chi2": current_best,
            "covariance_best_chi2": covariance_best,
            "covariance_minus_current_chi2": (
                covariance_best - current_best if covariance_best is not None and current_best is not None else None
            ),
            "covariance_found_lower_than_current": (
                bool(covariance_best < current_best - 1e-8)
                if covariance_best is not None and current_best is not None else None
            ),
            "candidate_records": candidate_records,
        }
        point.round08_detail = detail
        diagnostics.append(point)
        details.append(detail)
        profile_chi2.last_point = point
        if not ok:
            raise RuntimeError(
                f"Profile chi2 minimization failed for node {target_name}: "
                f"best_fun={best_fun}, nuisance_grad_norm={nuisance_grad_norm}, messages={'; '.join(messages)}"
            )
        last_nuisance = best_x
        return best_fun

    profile_chi2.diagnostics = diagnostics
    profile_chi2.last_point = None
    profile_chi2.round08_details = details
    return profile_chi2


def target_func_for_fit(fit: dict[str, Any], target: str) -> Callable[[Any], Any]:
    if target in fit["nodes"]:
        return fit["node_funcs"][fit["nodes"].index(target)]
    if target in fit["parameters"]:
        return fit["parameter_funcs"][fit["parameters"].index(target)]
    raise KeyError(f"{target} not found in fit nodes or parameters")


def target_std_for_fit(fit: dict[str, Any], target_func: Callable[[Any], Any], target_value: float) -> float:
    covariance = fit.get("covariance")
    if covariance is None:
        return max(abs(float(target_value)), 1.0) * 1e-3
    fitted_values = fit["fitted_values"]
    param_values = fit["param_values"]
    fitted_to_params = fit["fitted_params_to_params"]
    try:
        jac_params = jax.jacobian(fitted_to_params)(fitted_values)
        param_cov = jac_params @ covariance @ jac_params.T
        jac_target = jax.jacobian(target_func)(param_values)
        target_var = as_float(jac_target @ param_cov @ jac_target.T)
        target_std = math.sqrt(target_var) if target_var >= 0 else math.nan
    except Exception:  # noqa: BLE001 - diagnostic fallback
        target_std = math.nan
    if not math.isfinite(target_std) or target_std <= 0:
        target_std = max(abs(float(target_value)), 1.0) * 1e-3
    return float(target_std)


def endpoint_detail(root, side: str) -> dict[str, Any]:
    endpoint = root.lower if side == "lower" else root.upper
    detail = getattr(endpoint.point, "round08_detail", {}) or {}
    out = {
        "endpoint": endpoint.endpoint,
        "error": endpoint.error,
        "profile_chi2": endpoint.chi2,
        "residual": endpoint.residual,
        "profile_success": endpoint.point.success,
        "profile_method": endpoint.point.method,
        "profile_message": clean_message(endpoint.point.message),
        "profile_nfev": endpoint.point.nfev,
        "profile_nit": endpoint.point.nit,
        "profile_runtime_sec": endpoint.point.runtime_sec,
        "constraint_violation": endpoint.point.constraint_violation,
        "scaled_constraint_violation": endpoint.point.scaled_constraint_violation,
        "projected_grad_norm": endpoint.point.projected_grad_norm,
        "objective_grad_norm": endpoint.point.objective_grad_norm,
        "descent_improvement": endpoint.point.descent_improvement,
        "root_function_evals": root.function_evals,
        "root_cache_hits": root.cache_hits,
        "side_function_evals": endpoint.counts.get("function_evals"),
        "side_bracket_evals": endpoint.counts.get("bracket_evals"),
        "side_bisect_evals": endpoint.counts.get("bisect_evals"),
    }
    for key, value in detail.items():
        if key == "candidate_records":
            continue
        out[f"detail_{key}"] = value
    return out


def current_row_fields(asym: dict[str, Any], side: str) -> dict[str, Any]:
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
        "production_root_cache_hits": asym.get("cache_hits"),
    }


def run_fit_case(case: CaseSpec, policies: list[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    fit = run_fit(case.label_or_node, verbose=False)
    if fit is None:
        raise RuntimeError(f"run_fit returned None for {case.label_or_node}")
    target_func = target_func_for_fit(fit, case.target)
    asym = calc_asym_errors(fit, targets=[case.target])[case.target]
    target_value = float(asym["value"])
    target_std = target_std_for_fit(fit, target_func, target_value)
    lb = target_value - 2.0 * target_std
    ub = target_value + 2.0 * target_std
    rows: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    for policy in policies:
        profile = build_constrained_profile_for_policy(
            fit,
            target_func,
            target_value,
            target_std,
            policy=policy,
            target_name=case.target,
        )
        problem = CallableProfileProblem(
            profile_chi2=profile,
            target_name=case.target,
            target_value=target_value,
            chi2_min=float(fit["chi2_min"]),
            lower_initial=lb,
            upper_initial=ub,
        )
        root_result = find_profile_root(problem, target_value, float(fit["chi2_min"]), lb, ub)
        for detail in profile.round08_details:
            flat = {
                key: value
                for key, value in detail.items()
                if key != "candidate_records"
            }
            flat.update(
                {
                    "case_id": case.case_id,
                    "kind": case.kind,
                    "label_or_node": case.label_or_node,
                    "target": case.target,
                    "policy": policy,
                }
            )
            eval_rows.append(flat)
            for candidate in detail.get("candidate_records", []):
                candidate_flat = {
                    **candidate,
                    "case_id": case.case_id,
                    "kind": case.kind,
                    "label_or_node": case.label_or_node,
                    "target": case.target,
                    "policy": policy,
                    "profile_value": detail["value"],
                    "record_type": "candidate",
                }
                eval_rows.append(candidate_flat)
        for side in ("lower", "upper"):
            row = {
                "case_id": case.case_id,
                "kind": case.kind,
                "label_or_node": case.label_or_node,
                "target": case.target,
                "rationale": case.rationale,
                "side": side,
                "policy": policy,
                "n_parameters": len(fit["parameters"]),
                "chi2_min": float(fit["chi2_min"]),
                "target_chi2": float(fit["chi2_min"]) + 1.0,
                "target_value": target_value,
                "target_std": target_std,
                "initial_lb": lb,
                "initial_ub": ub,
                **current_row_fields(asym, side),
                **{f"policy_{k}": v for k, v in endpoint_detail(root_result, side).items()},
            }
            row["endpoint_delta_vs_production"] = row["policy_endpoint"] - row["production_endpoint"]
            row["profile_chi2_delta_vs_production"] = row["policy_profile_chi2"] - row["production_profile_chi2"]
            rows.append(row)
    return rows, eval_rows


def covariance_for_avg(avg_result: dict[str, Any]) -> np.ndarray:
    chi2 = avg_result["chi2"]
    x = jnp.asarray(avg_result["param_values"], dtype=jnp.float64)
    hess = np.asarray(jax.hessian(chi2)(x), dtype=np.float64)
    return 2.0 * np.linalg.pinv(hess)


def run_avg_case(case: CaseSpec, policies: list[str], avg_cache: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not avg_cache:
        avg_df, corr_df_dict = avg_queries(verbose=False)
        avg_cache["avg_df"] = avg_df
        avg_cache["corr_df_dict"] = corr_df_dict
    avg_df = avg_cache["avg_df"]
    corr_df_dict = avg_cache["corr_df_dict"]
    node = case.label_or_node
    avg_result = run_avg(node, avg_df[avg_df["node"] == node], corr_df_dict[node])
    if avg_result is None:
        raise RuntimeError(f"run_avg returned None for {node}")
    asym = {
        "value": float(avg_result["param_values"][avg_result["parameters"].index(case.target)]),
        "error_p": float(avg_result["error_p"]),
        "error_n": float(avg_result["error_n"]),
        **avg_result["asym_error_diagnostics"],
    }
    covariance = covariance_for_avg(avg_result)
    target_value = float(asym["value"])
    # A deliberately generous bracket; production used measurement spans that
    # are not retained in the result object with the adjustment closure.
    lb = target_value - 2.5 * max(float(asym["lower_error"]), abs(float(asym["error_n"])), 1e-12)
    ub = target_value + 2.5 * max(float(asym["upper_error"]), abs(float(asym["error_p"])), 1e-12)
    rows: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    for policy in policies:
        profile = build_avg_profile_for_policy(avg_result, covariance, policy=policy, target_name=case.target)
        problem = CallableProfileProblem(
            profile_chi2=profile,
            target_name=case.target,
            target_value=target_value,
            chi2_min=float(avg_result["chi2_min"]),
            lower_initial=lb,
            upper_initial=ub,
        )
        root_result = find_profile_root(problem, target_value, float(avg_result["chi2_min"]), lb, ub)
        for detail in profile.round08_details:
            flat = {
                key: value
                for key, value in detail.items()
                if key != "candidate_records"
            }
            flat.update(
                {
                    "case_id": case.case_id,
                    "kind": case.kind,
                    "label_or_node": case.label_or_node,
                    "target": case.target,
                    "policy": policy,
                }
            )
            eval_rows.append(flat)
            for candidate in detail.get("candidate_records", []):
                eval_rows.append(
                    {
                        **candidate,
                        "case_id": case.case_id,
                        "kind": case.kind,
                        "label_or_node": case.label_or_node,
                        "target": case.target,
                        "policy": policy,
                        "profile_value": detail["value"],
                        "record_type": "candidate",
                    }
                )
        for side in ("lower", "upper"):
            row = {
                "case_id": case.case_id,
                "kind": case.kind,
                "label_or_node": case.label_or_node,
                "target": case.target,
                "rationale": case.rationale,
                "side": side,
                "policy": policy,
                "n_parameters": len(avg_result["parameters"]),
                "chi2_min": float(avg_result["chi2_min"]),
                "target_chi2": float(avg_result["chi2_min"]) + 1.0,
                "target_value": target_value,
                "target_std": math.sqrt(float(covariance[avg_result["parameters"].index(case.target), avg_result["parameters"].index(case.target)])),
                "initial_lb": lb,
                "initial_ub": ub,
                **current_row_fields(asym, side),
                **{f"policy_{k}": v for k, v in endpoint_detail(root_result, side).items()},
            }
            row["endpoint_delta_vs_production"] = row["policy_endpoint"] - row["production_endpoint"]
            row["profile_chi2_delta_vs_production"] = row["policy_profile_chi2"] - row["production_profile_chi2"]
            rows.append(row)
    return rows, eval_rows


def parse_cases(selected: list[str] | None) -> list[CaseSpec]:
    if not selected:
        return DEFAULT_CASES
    by_id = {case.case_id: case for case in DEFAULT_CASES}
    cases = []
    for case_id in selected:
        if case_id not in by_id:
            raise SystemExit(f"unknown case id {case_id}; available: {', '.join(by_id)}")
        cases.append(by_id[case_id])
    return cases


def parse_policies(raw: str) -> list[str]:
    policies = [item.strip() for item in raw.split(",") if item.strip()]
    unknown = [policy for policy in policies if policy not in POLICIES]
    if unknown:
        raise SystemExit(f"unknown policies {unknown}; available: {', '.join(POLICIES)}")
    return policies


def main_impl(args: argparse.Namespace) -> int:
    cases = parse_cases(args.case)
    policies = parse_policies(args.policies)
    rows: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    avg_cache: dict[str, Any] = {}
    for case in cases:
        print(f"\n=== {case.case_id}: {case.kind} {case.label_or_node} / {case.target} ===")
        t0 = time.perf_counter()
        try:
            if case.kind == "fit":
                case_rows, case_eval_rows = run_fit_case(case, policies)
            elif case.kind == "avg":
                case_rows, case_eval_rows = run_avg_case(case, policies, avg_cache)
            else:
                raise ValueError(f"unknown case kind {case.kind}")
            rows.extend(case_rows)
            eval_rows.extend(case_eval_rows)
            print(f"completed {case.case_id} in {time.perf_counter() - t0:.3f}s")
            for row in case_rows:
                print(
                    row["policy"],
                    row["side"],
                    "prod",
                    row["production_endpoint"],
                    "policy",
                    row["policy_endpoint"],
                    "resid",
                    row["policy_residual"],
                    "method",
                    row["policy_profile_method"],
                    "cov-current",
                    row.get("policy_detail_covariance_minus_current_chi2"),
                )
        except Exception as exc:  # noqa: BLE001 - notes-only diagnostic records failure
            failure = {
                "case_id": case.case_id,
                "kind": case.kind,
                "label_or_node": case.label_or_node,
                "target": case.target,
                "exception": f"{type(exc).__name__}: {exc}",
            }
            failures.append(failure)
            print(f"FAILED {case.case_id}: {failure['exception']}")
            if not args.keep_going:
                break
    write_rows(Path(args.results_csv), rows)
    write_rows(Path(args.results_jsonl), rows, jsonl=True)
    write_rows(Path(args.evaluations_csv), eval_rows)
    write_rows(Path(args.evaluations_jsonl), eval_rows, jsonl=True)
    write_rows(Path(args.failures_csv), failures)
    write_rows(Path(args.failures_jsonl), failures, jsonl=True)
    print(f"\nwrote {len(rows)} result rows")
    print(f"wrote {len(eval_rows)} evaluation rows")
    print(f"wrote {len(failures)} failure rows")
    return 0 if not failures else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append", help="Case id to run; repeatable. Defaults to all cases.")
    parser.add_argument("--policies", default="covariance_only,current_plus_covariance")
    parser.add_argument("--results-csv", default="notes/codex-refactor/round08_minos_covariance_starts_results.csv")
    parser.add_argument("--results-jsonl", default="notes/codex-refactor/round08_minos_covariance_starts_results.jsonl")
    parser.add_argument("--evaluations-csv", default="notes/codex-refactor/round08_minos_covariance_starts_evaluations.csv")
    parser.add_argument("--evaluations-jsonl", default="notes/codex-refactor/round08_minos_covariance_starts_evaluations.jsonl")
    parser.add_argument("--failures-csv", default="notes/codex-refactor/round08_minos_covariance_starts_failures.csv")
    parser.add_argument("--failures-jsonl", default="notes/codex-refactor/round08_minos_covariance_starts_failures.jsonl")
    parser.add_argument("--stdout-log", default=None)
    parser.add_argument("--keep-going", action="store_true", default=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.stdout_log:
        path = Path(args.stdout_log)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f, contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
            return main_impl(args)
    return main_impl(args)


if __name__ == "__main__":
    sys.exit(main())
