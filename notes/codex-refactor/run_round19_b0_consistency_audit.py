#!/usr/bin/env python
"""Round 19 targeted audit for the B0 split-normal toy-84 q<0 row.

This is a notes-only diagnostic harness.  It intentionally reuses the round-16
toy generator and production chi2/refit/profile primitives, but records the
actual fixed-truth profile coordinates so that the q<0 mechanism can be checked
directly.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import math
import time
from pathlib import Path
from typing import Any, Callable

import jax
from jax import numpy as jnp
import numpy as np
from scipy.optimize import NonlinearConstraint, OptimizeResult, minimize, root

import run_round15_wilks_calibration_smoke as round15
import run_round16_calibration_pilot as round16

from pdgfits.asym_errors import CallableProfileProblem, build_constrained_profile_chi2, find_profile_root
from pdgfits.fit import _run_minuit_candidate, _run_unconstrained_scipy, run_fit
from pdgfits.param_maps import get_decay_info


jax.config.update("jax_enable_x64", True)


CASE = round15.CaseSpec(
    "fit_b0_s042b95",
    "boundary_tiny_parameter",
    "B0",
    "S042B95",
    "round-18 q-profile consistency warning",
)

REPRO_FIELDNAMES = [
    "toy_index",
    "seed",
    "status",
    "truth_target",
    "toy_target_mle",
    "toy_target_minus_truth",
    "target_std_baseline",
    "toy_chi2_min_reported",
    "toy_chi2_at_truth_params",
    "profile_true_chi2_production",
    "profile_true_chi2_audit_best",
    "profile_true_delta_chi2_production",
    "profile_true_delta_chi2_audit_best",
    "profile_minus_audit_best",
    "best_unconstrained_chi2",
    "best_unconstrained_start",
    "best_unconstrained_optimizer",
    "best_unconstrained_improvement_vs_reported",
    "q_using_best_unconstrained",
    "profile_consistency_failure_reported",
    "profile_consistency_failure_after_globality_check",
    "min_decay_param_mle",
    "max_decay_param_mle",
    "max_abs_decay_fitted_coord_mle",
    "profile_method_production",
    "profile_method_audit_best",
    "profile_projected_grad_norm_production",
    "profile_projected_grad_norm_audit_best",
    "profile_scaled_constraint_violation_production",
    "profile_scaled_constraint_violation_audit_best",
    "runtime_sec",
]

UNCONSTRAINED_FIELDNAMES = [
    "toy_index",
    "start_name",
    "optimizer",
    "start_chi2",
    "result_chi2",
    "improvement_vs_reported_refit",
    "improvement_vs_best_seen",
    "target_value",
    "target_minus_truth",
    "success",
    "valid",
    "accurate",
    "edm",
    "nfcn",
    "ngrad",
    "nfev",
    "njev",
    "gradient_norm",
    "min_decay_param",
    "max_decay_param",
    "max_abs_decay_fitted_coord",
    "message",
]

PROFILE_FIELDNAMES = [
    "toy_index",
    "start_name",
    "stage",
    "method",
    "start_chi2",
    "projected_start_chi2",
    "result_chi2",
    "improvement_vs_reported_refit",
    "improvement_vs_production_profile",
    "improvement_vs_best_profile",
    "constraint_abs",
    "scaled_constraint_violation",
    "target_value",
    "target_minus_truth",
    "success",
    "profile_ok",
    "nfev",
    "nit",
    "objective_grad_norm",
    "projected_grad_norm",
    "descent_improvement",
    "min_decay_param",
    "max_decay_param",
    "max_abs_decay_fitted_coord",
    "message",
]

ENDPOINT_FIELDNAMES = [
    "toy_index",
    "seed",
    "side",
    "endpoint",
    "error",
    "profile_chi2",
    "target_chi2_reported_refit",
    "target_chi2_best_unconstrained",
    "residual_vs_reported_refit",
    "residual_vs_best_unconstrained",
    "profile_success",
    "profile_method",
    "profile_nfev",
    "projected_grad_norm",
    "scaled_constraint_violation",
    "function_evals",
]


def clean_value(value: Any) -> Any:
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return value if math.isfinite(value) else None
    if value is None:
        return None
    return value


def json_default(obj: Any) -> Any:
    cleaned = clean_value(obj)
    if cleaned is not obj:
        return cleaned
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def write_rows(csv_path: Path, jsonl_path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: clean_value(row.get(key)) for key in fieldnames})
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    {key: clean_value(row.get(key)) for key in fieldnames},
                    default=json_default,
                    sort_keys=True,
                )
                + "\n"
            )


def parse_toy_indices(text: str) -> list[int]:
    out = []
    for part in text.split(","):
        part = part.strip()
        if part:
            out.append(int(part))
    return out


def finite_or_none(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def norm_or_none(values: np.ndarray) -> float | None:
    if not np.all(np.isfinite(values)):
        return None
    return float(np.linalg.norm(values))


class ToyAudit:
    def __init__(self, fit: dict[str, Any], target_func: Callable[[Any], Any], target_scale: float, y_toy: np.ndarray):
        self.fit = fit
        self.target_func = target_func
        self.target_scale = abs(float(target_scale)) if abs(float(target_scale)) > 0 else 1.0
        y_jax = jnp.asarray(y_toy, dtype=jnp.float64)
        chi2_open = fit["chi2_open"]

        def chi2(fp):
            return chi2_open(fp, y_jax)

        self.chi2 = chi2
        self.chi2_grad_jax = jax.jit(jax.grad(chi2))
        self.chi2_hess_jax = jax.jit(jax.hessian(chi2))
        self.chi2_value_and_grad_jax = jax.jit(jax.value_and_grad(chi2))
        self.chi2_hessp_jax = jax.jit(lambda x, p: jax.jvp(jax.grad(chi2), (x,), (p,))[1])

        @jax.jit
        def constraint_value_jax(fp, fixed_value):
            return target_func(fit["fitted_params_to_params"](fp)) - fixed_value

        @jax.jit
        def scaled_constraint_value_jax(fp, fixed_value):
            return constraint_value_jax(fp, fixed_value) / self.target_scale

        self.constraint_value_jax = constraint_value_jax
        self.scaled_constraint_value_jax = scaled_constraint_value_jax
        self.scaled_constraint_grad_jax = jax.jit(
            jax.grad(lambda fp, fixed_value: scaled_constraint_value_jax(fp, fixed_value))
        )
        self.scaled_constraint_hess_jax = jax.jit(
            jax.hessian(lambda fp, fixed_value: scaled_constraint_value_jax(fp, fixed_value))
        )
        self.decay_idxs = np.asarray(get_decay_info(list(fit["parameters"]))[1], dtype=int)

    def chi2_np(self, x: np.ndarray) -> float:
        value = float(self.chi2(jnp.asarray(x, dtype=jnp.float64)))
        return value if math.isfinite(value) else math.inf

    def chi2_grad_np(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(self.chi2_grad_jax(jnp.asarray(x, dtype=jnp.float64)), dtype=np.float64)

    def chi2_hess_np(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(self.chi2_hess_jax(jnp.asarray(x, dtype=jnp.float64)), dtype=np.float64)

    def scipy_obj(self, x: np.ndarray) -> tuple[float, np.ndarray]:
        val, grad = self.chi2_value_and_grad_jax(jnp.asarray(x, dtype=jnp.float64))
        return float(val), np.asarray(grad, dtype=np.float64)

    def scipy_hessp(self, x: np.ndarray, p: np.ndarray) -> np.ndarray:
        return np.asarray(
            self.chi2_hessp_jax(jnp.asarray(x, dtype=jnp.float64), jnp.asarray(p, dtype=jnp.float64)),
            dtype=np.float64,
        )

    def params(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(
            self.fit["fitted_params_to_params"](jnp.asarray(x, dtype=jnp.float64)),
            dtype=np.float64,
        )

    def target(self, x: np.ndarray) -> float:
        return float(self.target_func(jnp.asarray(self.params(x), dtype=jnp.float64)))

    def chart_summary(self, x: np.ndarray) -> dict[str, Any]:
        params = self.params(x)
        out = {
            "min_decay_param": None,
            "max_decay_param": None,
            "max_abs_decay_fitted_coord": None,
        }
        if len(self.decay_idxs) == 0:
            return out
        decay_params = params[self.decay_idxs]
        decay_fitted = np.asarray(x, dtype=np.float64)[self.decay_idxs]
        out.update(
            {
                "min_decay_param": finite_or_none(np.nanmin(decay_params)),
                "max_decay_param": finite_or_none(np.nanmax(decay_params)),
                "max_abs_decay_fitted_coord": finite_or_none(np.nanmax(np.abs(decay_fitted))),
            }
        )
        return out

    def physical_to_fitted(self, params: np.ndarray) -> np.ndarray:
        return np.asarray(
            self.fit["params_to_fitted_params"](jnp.asarray(params, dtype=jnp.float64)),
            dtype=np.float64,
        )

    def project_start(self, start: np.ndarray, fixed_value: float, tol: float = 1e-7) -> np.ndarray:
        x = np.asarray(start, dtype=np.float64).copy()
        for _ in range(20):
            if not np.all(np.isfinite(x)):
                break
            c = float(self.scaled_constraint_value_jax(jnp.asarray(x), fixed_value))
            if not math.isfinite(c):
                break
            if abs(c) <= tol:
                return x
            g = np.asarray(self.scaled_constraint_grad_jax(jnp.asarray(x), fixed_value), dtype=np.float64)
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

    def constraint_abs(self, x: np.ndarray, fixed_value: float) -> float:
        return abs(float(self.constraint_value_jax(jnp.asarray(x, dtype=jnp.float64), fixed_value)))

    def scaled_violation(self, x: np.ndarray, fixed_value: float) -> float:
        return self.constraint_abs(x, fixed_value) / self.target_scale

    def projected_grad_norm(self, x: np.ndarray, fixed_value: float) -> float:
        g_obj = self.chi2_grad_np(x)
        g_con = np.asarray(self.scaled_constraint_grad_jax(jnp.asarray(x), fixed_value), dtype=np.float64)
        denom = float(np.dot(g_con, g_con))
        if not math.isfinite(denom) or denom == 0.0:
            return math.inf
        g_proj = g_obj - (float(np.dot(g_obj, g_con)) / denom) * g_con
        return float(np.linalg.norm(g_proj))

    def projected_descent_improvement(self, x: np.ndarray, fixed_value: float) -> float:
        x = np.asarray(x, dtype=np.float64)
        if not np.all(np.isfinite(x)):
            return math.inf
        base_fun = self.chi2_np(x)
        if not math.isfinite(base_fun) or self.scaled_violation(x, fixed_value) > 1e-7:
            return math.inf
        g_obj = self.chi2_grad_np(x)
        g_con = np.asarray(self.scaled_constraint_grad_jax(jnp.asarray(x), fixed_value), dtype=np.float64)
        denom = float(np.dot(g_con, g_con))
        if not math.isfinite(denom) or denom == 0.0:
            return math.inf
        g_proj = g_obj - (float(np.dot(g_obj, g_con)) / denom) * g_con
        g_proj_norm = float(np.linalg.norm(g_proj))
        if not math.isfinite(g_proj_norm) or g_proj_norm == 0.0:
            return 0.0
        direction = -g_proj / g_proj_norm
        coord_scale = max(float(np.linalg.norm(x)), 1.0)
        best_decrease = 0.0
        for exponent in range(-14, -3):
            step = coord_scale * (10.0**exponent)
            trial = self.project_start(x + step * direction, fixed_value, tol=1e-10)
            if not np.all(np.isfinite(trial)) or self.scaled_violation(trial, fixed_value) > 1e-7:
                continue
            trial_fun = self.chi2_np(trial)
            if math.isfinite(trial_fun):
                best_decrease = max(best_decrease, base_fun - trial_fun)
        return float(best_decrease)

    def profile_result_ok(self, result: OptimizeResult, fixed_value: float) -> tuple[bool, float | None]:
        if not np.isfinite(result.fun) or self.scaled_violation(result.x, fixed_value) > 1e-7:
            return False, None
        g_norm = float(np.linalg.norm(self.chi2_grad_np(result.x)))
        pg_norm = self.projected_grad_norm(result.x, fixed_value)
        if pg_norm <= max(1e-3, 1e-6 * g_norm):
            return True, 0.0
        descent_improvement = self.projected_descent_improvement(result.x, fixed_value)
        return descent_improvement <= 1e-4, descent_improvement

    def kkt_polish(self, result: OptimizeResult, fixed_value: float) -> OptimizeResult:
        if not np.isfinite(result.fun) or not np.all(np.isfinite(result.x)):
            return result
        x0 = self.project_start(result.x, fixed_value)
        g_obj = self.chi2_grad_np(x0)
        g_con = np.asarray(self.scaled_constraint_grad_jax(jnp.asarray(x0), fixed_value), dtype=np.float64)
        denom = float(np.dot(g_con, g_con))
        if not math.isfinite(denom) or denom == 0.0:
            return result
        lambda0 = -float(np.dot(g_obj, g_con)) / denom

        def residual(z):
            x = np.asarray(z[:-1], dtype=np.float64)
            lam = float(z[-1])
            stationarity = (
                self.chi2_grad_np(x)
                + lam * np.asarray(self.scaled_constraint_grad_jax(jnp.asarray(x), fixed_value), dtype=np.float64)
            )
            return np.r_[stationarity, float(self.scaled_constraint_value_jax(jnp.asarray(x), fixed_value))]

        try:
            polished = root(residual, np.r_[x0, lambda0], method="hybr", options={"maxfev": 5000})
        except (FloatingPointError, ValueError, np.linalg.LinAlgError):
            return result

        x_polished = np.asarray(polished.x[:-1], dtype=np.float64)
        if not np.all(np.isfinite(x_polished)) or self.scaled_violation(x_polished, fixed_value) > 1e-7:
            return result
        fun_polished = self.chi2_np(x_polished)
        if not math.isfinite(fun_polished):
            return result
        if fun_polished > result.fun + max(1e-7, 1e-9 * abs(result.fun)):
            return result
        method = str(getattr(result, "profile_method", "")) or "profile"
        return OptimizeResult(
            x=x_polished,
            fun=fun_polished,
            success=bool(polished.success),
            status=getattr(polished, "status", None),
            message=f"KKT polish: {polished.message}",
            nfev=getattr(polished, "nfev", None),
            profile_method=f"{method}+KKT",
        )

    def run_profile_from_start(self, start_name: str, start: np.ndarray, fixed_value: float) -> tuple[dict[str, Any], np.ndarray]:
        x0 = self.project_start(start, fixed_value)
        projected_start_chi2 = self.chi2_np(x0)

        def make_constraint(exact_hess: bool = False):
            kwargs = {}
            if exact_hess:
                kwargs["hess"] = lambda x, multiplier: np.asarray(
                    multiplier[0] * self.scaled_constraint_hess_jax(jnp.asarray(x), fixed_value),
                    dtype=np.float64,
                )
            return NonlinearConstraint(
                fun=lambda x: float(self.scaled_constraint_value_jax(jnp.asarray(x), fixed_value)),
                lb=0.0,
                ub=0.0,
                jac=lambda x: np.asarray(self.scaled_constraint_grad_jax(jnp.asarray(x), fixed_value), dtype=np.float64),
                **kwargs,
            )

        attempts: list[tuple[str, str, OptimizeResult]] = []
        try:
            slsqp = minimize(
                self.chi2_np,
                x0,
                method="SLSQP",
                jac=self.chi2_grad_np,
                constraints=[make_constraint()],
                options={"ftol": 1e-10, "maxiter": 2000},
            )
        except (FloatingPointError, ValueError, np.linalg.LinAlgError) as exc:
            slsqp = OptimizeResult(
                x=x0,
                fun=projected_start_chi2,
                success=False,
                message=f"SLSQP failed: {exc}",
                nfev=None,
            )
        slsqp.profile_method = "SLSQP"
        attempts.append(("slsqp", "SLSQP", slsqp))

        ok, descent = self.profile_result_ok(slsqp, fixed_value)
        if ok:
            slsqp.profile_descent_improvement = descent
            return self.profile_row(start_name, "slsqp", slsqp, fixed_value, start, projected_start_chi2, ok), np.asarray(slsqp.x)

        try:
            trust = minimize(
                self.chi2_np,
                x0,
                method="trust-constr",
                jac=self.chi2_grad_np,
                hess=self.chi2_hess_np,
                constraints=[make_constraint(exact_hess=True)],
                options={"gtol": 1e-10, "xtol": 1e-10, "maxiter": 2000},
            )
        except (FloatingPointError, ValueError, np.linalg.LinAlgError) as exc:
            trust = OptimizeResult(
                x=x0,
                fun=projected_start_chi2,
                success=False,
                message=f"trust-constr exact Hessian failed: {exc}",
                nfev=None,
            )
        trust.profile_method = "trust-constr-exact-hess"
        trust_polished = self.kkt_polish(trust, fixed_value)
        attempts.append(("trust_constr", str(getattr(trust_polished, "profile_method", "")), trust_polished))

        ok_rows: list[tuple[float, bool, float | None, str, OptimizeResult]] = []
        all_rows: list[tuple[float, bool, float | None, str, OptimizeResult]] = []
        for stage, _method, result in attempts:
            candidate_ok, candidate_descent = self.profile_result_ok(result, fixed_value)
            all_rows.append((float(result.fun), candidate_ok, candidate_descent, stage, result))
            if candidate_ok:
                ok_rows.append((float(result.fun), candidate_ok, candidate_descent, stage, result))
        best_fun, best_ok, best_descent, best_stage, best_result = min(
            ok_rows or all_rows,
            key=lambda item: item[0] if math.isfinite(item[0]) else math.inf,
        )
        best_result.profile_descent_improvement = best_descent
        return (
            self.profile_row(start_name, best_stage, best_result, fixed_value, start, projected_start_chi2, best_ok),
            np.asarray(best_result.x, dtype=np.float64),
        )

    def profile_row(
        self,
        start_name: str,
        stage: str,
        result: OptimizeResult,
        fixed_value: float,
        raw_start: np.ndarray,
        projected_start_chi2: float,
        profile_ok: bool,
    ) -> dict[str, Any]:
        x = np.asarray(result.x, dtype=np.float64)
        chart = self.chart_summary(x)
        return {
            "start_name": start_name,
            "stage": stage,
            "method": str(getattr(result, "profile_method", "")),
            "start_chi2": self.chi2_np(raw_start),
            "projected_start_chi2": projected_start_chi2,
            "result_chi2": finite_or_none(result.fun),
            "constraint_abs": self.constraint_abs(x, fixed_value),
            "scaled_constraint_violation": self.scaled_violation(x, fixed_value),
            "target_value": self.target(x),
            "target_minus_truth": self.target(x) - fixed_value,
            "success": bool(getattr(result, "success", False)),
            "profile_ok": bool(profile_ok),
            "nfev": getattr(result, "nfev", None),
            "nit": getattr(result, "nit", None),
            "objective_grad_norm": norm_or_none(self.chi2_grad_np(x)),
            "projected_grad_norm": self.projected_grad_norm(x, fixed_value),
            "descent_improvement": getattr(result, "profile_descent_improvement", None),
            **chart,
            "message": round15.clean_message(getattr(result, "message", "")),
        }


def make_toy(seed: int, fit: dict[str, Any], context: dict[str, Any], dgp: str) -> tuple[np.ndarray, dict[str, Any]]:
    rng = np.random.default_rng(seed)
    truth_params = np.asarray(fit["param_values"], dtype=np.float64)
    mean_y = np.asarray(
        context["mu_adjust"](jnp.asarray(truth_params, dtype=jnp.float64)),
        dtype=np.float64,
    )
    return round16.draw_toy(rng, dgp, mean_y, context)


def run_production_profile(
    audit: ToyAudit,
    fit: dict[str, Any],
    x_hat: np.ndarray,
    target_func: Callable[[Any], Any],
    target_std: float,
    truth_target: float,
    toy_target: float,
    toy_chi2_min: float,
) -> tuple[Any, CallableProfileProblem]:
    profile_chi2 = build_constrained_profile_chi2(
        audit.chi2,
        fit["fitted_params_to_params"],
        target_func,
        x_hat,
        target_scale=target_std,
        target_name=CASE.target,
    )
    problem = CallableProfileProblem(
        profile_chi2,
        target_name=CASE.target,
        target_value=toy_target,
        chi2_min=toy_chi2_min,
    )
    point = problem.evaluate(truth_target)
    return point, problem


def run_unconstrained_from_start(
    audit: ToyAudit,
    start_name: str,
    start: np.ndarray,
    truth_target: float,
    reported_chi2_min: float,
    best_seen_chi2: float,
    optimizers: tuple[str, ...],
) -> list[tuple[dict[str, Any], np.ndarray]]:
    rows = []
    start_chi2 = audit.chi2_np(start)
    for optimizer in optimizers:
        if optimizer == "minuit":
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    m = _run_minuit_candidate(audit.chi2_np, audit.chi2_grad_np, start, False, None, None)
                x = np.asarray(m.values, dtype=np.float64)
                chi2 = float(m.fval)
                row = {
                    "start_name": start_name,
                    "optimizer": optimizer,
                    "start_chi2": start_chi2,
                    "result_chi2": chi2,
                    "success": bool(m.valid),
                    "valid": bool(m.valid),
                    "accurate": bool(m.accurate),
                    "edm": finite_or_none(m.fmin.edm),
                    "nfcn": int(m.nfcn),
                    "ngrad": int(m.ngrad),
                    "nfev": None,
                    "njev": None,
                    "message": "",
                }
            except Exception as exc:  # noqa: BLE001 - diagnostic row
                x = np.asarray(start, dtype=np.float64)
                chi2 = math.inf
                row = {
                    "start_name": start_name,
                    "optimizer": optimizer,
                    "start_chi2": start_chi2,
                    "result_chi2": None,
                    "success": False,
                    "valid": None,
                    "accurate": None,
                    "edm": None,
                    "nfcn": None,
                    "ngrad": None,
                    "nfev": None,
                    "njev": None,
                    "message": f"{type(exc).__name__}: {exc}",
                }
        elif optimizer == "scipy":
            try:
                result = _run_unconstrained_scipy(audit.scipy_obj, audit.chi2_np, audit.scipy_hessp, start)
                x = np.asarray(result.x, dtype=np.float64)
                chi2 = float(result.fun)
                row = {
                    "start_name": start_name,
                    "optimizer": optimizer,
                    "start_chi2": start_chi2,
                    "result_chi2": chi2,
                    "success": bool(result.success),
                    "valid": None,
                    "accurate": None,
                    "edm": None,
                    "nfcn": None,
                    "ngrad": None,
                    "nfev": getattr(result, "nfev", None),
                    "njev": getattr(result, "njev", None),
                    "message": round15.clean_message(getattr(result, "message", "")),
                }
            except Exception as exc:  # noqa: BLE001 - diagnostic row
                x = np.asarray(start, dtype=np.float64)
                chi2 = math.inf
                row = {
                    "start_name": start_name,
                    "optimizer": optimizer,
                    "start_chi2": start_chi2,
                    "result_chi2": None,
                    "success": False,
                    "valid": None,
                    "accurate": None,
                    "edm": None,
                    "nfcn": None,
                    "ngrad": None,
                    "nfev": None,
                    "njev": None,
                    "message": f"{type(exc).__name__}: {exc}",
                }
        else:
            raise ValueError(f"unknown optimizer {optimizer}")

        if math.isfinite(chi2):
            row.update(
                {
                    "improvement_vs_reported_refit": reported_chi2_min - chi2,
                    "improvement_vs_best_seen": best_seen_chi2 - chi2,
                    "target_value": audit.target(x),
                    "target_minus_truth": audit.target(x) - truth_target,
                    "gradient_norm": norm_or_none(audit.chi2_grad_np(x)),
                    **audit.chart_summary(x),
                }
            )
        else:
            row.update(
                {
                    "improvement_vs_reported_refit": None,
                    "improvement_vs_best_seen": None,
                    "target_value": None,
                    "target_minus_truth": None,
                    "gradient_norm": None,
                    **{"min_decay_param": None, "max_decay_param": None, "max_abs_decay_fitted_coord": None},
                }
            )
        rows.append((row, x))
    return rows


def add_start(starts: list[tuple[str, np.ndarray]], name: str, x: np.ndarray) -> None:
    x = np.asarray(x, dtype=np.float64)
    if not np.all(np.isfinite(x)):
        return
    for _name, existing in starts:
        scale = max(np.linalg.norm(existing), np.linalg.norm(x), 1.0)
        if np.linalg.norm(existing - x) <= 1e-10 * scale:
            return
    starts.append((name, x))


def alternative_starts(audit: ToyAudit, fit: dict[str, Any], x_hat: np.ndarray, profile_x: np.ndarray) -> list[tuple[str, np.ndarray]]:
    starts: list[tuple[str, np.ndarray]] = []
    add_start(starts, "production_reference_fit_start", np.asarray(fit["fitted_values"], dtype=np.float64))
    add_start(starts, "reported_refit_mle", x_hat)
    add_start(starts, "profile_truth_solution", profile_x)

    mle_params = audit.params(x_hat)
    truth_params = np.asarray(fit["param_values"], dtype=np.float64)
    add_start(starts, "truth_physical_params", audit.physical_to_fitted(truth_params))
    add_start(starts, "halfway_truth_reported_mle_params", audit.physical_to_fitted(0.5 * truth_params + 0.5 * mle_params))

    if len(audit.decay_idxs) > 0:
        for floor in (1e-6, 3e-6, 1e-5, 1e-4):
            params = mle_params.copy()
            params[audit.decay_idxs] = np.clip(params[audit.decay_idxs], floor, 1.0 - floor)
            add_start(starts, f"clip_decay_floor_{floor:g}", audit.physical_to_fitted(params))
        for factor in (0.5, 2.0, 10.0):
            params = mle_params.copy()
            tiny = audit.decay_idxs[params[audit.decay_idxs] <= 1e-6]
            params[tiny] = np.clip(params[tiny] * factor, 1e-12, 1.0 - 1e-12)
            add_start(starts, f"tiny_decay_factor_{factor:g}", audit.physical_to_fitted(params))
    return starts


def run_endpoints_for_toy(
    audit: ToyAudit,
    fit: dict[str, Any],
    target_func: Callable[[Any], Any],
    target_std: float,
    x_hat: np.ndarray,
    toy_target: float,
    reported_chi2_min: float,
    best_unconstrained_chi2: float,
    seed: int,
    toy_index: int,
    residual_tol: float,
) -> list[dict[str, Any]]:
    profile_chi2 = build_constrained_profile_chi2(
        audit.chi2,
        fit["fitted_params_to_params"],
        target_func,
        x_hat,
        target_scale=target_std,
        target_name=CASE.target,
    )
    lb = toy_target - 3.0 * target_std
    ub = toy_target + 3.0 * target_std
    root_result = find_profile_root(
        CallableProfileProblem(
            profile_chi2,
            target_name=CASE.target,
            target_value=toy_target,
            chi2_min=reported_chi2_min,
            lower_initial=lb,
            upper_initial=ub,
        ),
        residual_tol=residual_tol,
    )
    rows = []
    for side_name, side in (("lower", root_result.lower), ("upper", root_result.upper)):
        rows.append(
            {
                "toy_index": toy_index,
                "seed": seed,
                "side": side_name,
                "endpoint": side.endpoint,
                "error": side.error,
                "profile_chi2": side.chi2,
                "target_chi2_reported_refit": reported_chi2_min + 1.0,
                "target_chi2_best_unconstrained": best_unconstrained_chi2 + 1.0,
                "residual_vs_reported_refit": side.chi2 - (reported_chi2_min + 1.0),
                "residual_vs_best_unconstrained": side.chi2 - (best_unconstrained_chi2 + 1.0),
                "profile_success": side.point.success,
                "profile_method": side.point.method,
                "profile_nfev": side.point.nfev,
                "projected_grad_norm": side.point.projected_grad_norm,
                "scaled_constraint_violation": side.point.scaled_constraint_violation,
                "function_evals": root_result.function_evals,
            }
        )
    return rows


def run_one_toy(
    fit: dict[str, Any],
    context: dict[str, Any],
    target_func: Callable[[Any], Any],
    target_std: float,
    toy_index: int,
    seed: int,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    t0 = time.perf_counter()
    y_toy, _sample_info = make_toy(seed, fit, context, args.dgp)
    refit, toy_chi2 = round15.refit_toy(fit, y_toy)
    x_hat = np.asarray(refit["x_hat"], dtype=np.float64)
    toy_chi2_min = float(refit["chi2_min"])
    audit = ToyAudit(fit, target_func, target_std, y_toy)

    truth_params = np.asarray(fit["param_values"], dtype=np.float64)
    truth_fp = np.asarray(fit["fitted_values"], dtype=np.float64)
    truth_target = float(target_func(jnp.asarray(truth_params, dtype=jnp.float64)))
    toy_params = audit.params(x_hat)
    toy_target = float(target_func(jnp.asarray(toy_params, dtype=jnp.float64)))
    toy_chi2_at_truth = audit.chi2_np(truth_fp)

    prod_point, _prod_problem = run_production_profile(
        audit,
        fit,
        x_hat,
        target_func,
        target_std,
        truth_target,
        toy_target,
        toy_chi2_min,
    )
    production_profile_chi2 = float(prod_point.chi2)

    profile_rows: list[dict[str, Any]] = []
    unconstrained_rows: list[dict[str, Any]] = []

    current_profile_row, current_profile_x = audit.run_profile_from_start(
        "reported_refit_mle", x_hat, truth_target
    )
    current_profile_row["toy_index"] = toy_index
    current_profile_row["improvement_vs_reported_refit"] = toy_chi2_min - float(current_profile_row["result_chi2"])
    current_profile_row["improvement_vs_production_profile"] = production_profile_chi2 - float(current_profile_row["result_chi2"])
    profile_rows.append(current_profile_row)

    starts = alternative_starts(audit, fit, x_hat, current_profile_x)
    best_profile_chi2 = float(current_profile_row["result_chi2"])
    profile_solutions: list[tuple[str, np.ndarray, float]] = [("reported_refit_mle", current_profile_x, best_profile_chi2)]
    if toy_index == args.audit_toy:
        for start_name, start in starts:
            if start_name == "reported_refit_mle":
                continue
            row, x_profile = audit.run_profile_from_start(start_name, start, truth_target)
            row["toy_index"] = toy_index
            chi2 = float(row["result_chi2"]) if row["result_chi2"] is not None else math.inf
            row["improvement_vs_reported_refit"] = toy_chi2_min - chi2
            row["improvement_vs_production_profile"] = production_profile_chi2 - chi2
            profile_rows.append(row)
            profile_solutions.append((start_name, x_profile, chi2))
            if chi2 < best_profile_chi2:
                best_profile_chi2 = chi2

    for row in profile_rows:
        chi2 = float(row["result_chi2"]) if row["result_chi2"] is not None else math.inf
        row["improvement_vs_best_profile"] = best_profile_chi2 - chi2

    profile_best_name, profile_best_x, profile_best_chi2 = min(profile_solutions, key=lambda item: item[2])
    start_for_unconstrained = starts
    if toy_index != args.audit_toy:
        start_for_unconstrained = [
            ("production_reference_fit_start", np.asarray(fit["fitted_values"], dtype=np.float64)),
            ("reported_refit_mle", x_hat),
            ("profile_truth_solution", current_profile_x),
        ]

    best_seen = min(toy_chi2_min, profile_best_chi2)
    unconstrained_solutions: list[tuple[str, str, np.ndarray, float]] = []
    for start_name, start in start_for_unconstrained:
        optimizers = ("minuit", "scipy") if start_name in {"reported_refit_mle", "profile_truth_solution"} else ("minuit",)
        for row, x_result in run_unconstrained_from_start(
            audit,
            start_name,
            start,
            truth_target,
            toy_chi2_min,
            best_seen,
            optimizers,
        ):
            row["toy_index"] = toy_index
            chi2 = float(row["result_chi2"]) if row["result_chi2"] is not None else math.inf
            unconstrained_rows.append(row)
            unconstrained_solutions.append((start_name, row["optimizer"], x_result, chi2))
            best_seen = min(best_seen, chi2)

    for row in unconstrained_rows:
        chi2 = float(row["result_chi2"]) if row["result_chi2"] is not None else math.inf
        row["improvement_vs_best_seen"] = best_seen - chi2

    best_unconstrained_name, best_unconstrained_optimizer, _best_unconstrained_x, best_unconstrained_chi2 = min(
        unconstrained_solutions,
        key=lambda item: item[3] if math.isfinite(item[3]) else math.inf,
    )

    row = {
        "toy_index": toy_index,
        "seed": seed,
        "status": "ok",
        "truth_target": truth_target,
        "toy_target_mle": toy_target,
        "toy_target_minus_truth": toy_target - truth_target,
        "target_std_baseline": target_std,
        "toy_chi2_min_reported": toy_chi2_min,
        "toy_chi2_at_truth_params": toy_chi2_at_truth,
        "profile_true_chi2_production": production_profile_chi2,
        "profile_true_chi2_audit_best": best_profile_chi2,
        "profile_true_delta_chi2_production": production_profile_chi2 - toy_chi2_min,
        "profile_true_delta_chi2_audit_best": best_profile_chi2 - toy_chi2_min,
        "profile_minus_audit_best": production_profile_chi2 - best_profile_chi2,
        "best_unconstrained_chi2": best_unconstrained_chi2,
        "best_unconstrained_start": best_unconstrained_name,
        "best_unconstrained_optimizer": best_unconstrained_optimizer,
        "best_unconstrained_improvement_vs_reported": toy_chi2_min - best_unconstrained_chi2,
        "q_using_best_unconstrained": production_profile_chi2 - best_unconstrained_chi2,
        "profile_consistency_failure_reported": bool(production_profile_chi2 - toy_chi2_min < -1e-4),
        "profile_consistency_failure_after_globality_check": bool(production_profile_chi2 - best_unconstrained_chi2 < -1e-4),
        **{
            "min_decay_param_mle": round15.chart_summary(fit, x_hat)["min_decay_param_mle"],
            "max_decay_param_mle": round15.chart_summary(fit, x_hat)["max_decay_param_mle"],
            "max_abs_decay_fitted_coord_mle": round15.chart_summary(fit, x_hat)["max_abs_decay_fitted_coord_mle"],
        },
        "profile_method_production": prod_point.method,
        "profile_method_audit_best": profile_best_name,
        "profile_projected_grad_norm_production": prod_point.projected_grad_norm,
        "profile_projected_grad_norm_audit_best": min(
            row.get("projected_grad_norm", math.inf)
            for row in profile_rows
            if row.get("result_chi2") == best_profile_chi2
        ),
        "profile_scaled_constraint_violation_production": prod_point.scaled_constraint_violation,
        "profile_scaled_constraint_violation_audit_best": min(
            row.get("scaled_constraint_violation", math.inf)
            for row in profile_rows
            if row.get("result_chi2") == best_profile_chi2
        ),
        "runtime_sec": time.perf_counter() - t0,
    }

    endpoint_rows: list[dict[str, Any]] = []
    if toy_index == args.audit_toy and args.run_endpoint_check:
        endpoint_rows = run_endpoints_for_toy(
            audit,
            fit,
            target_func,
            target_std,
            x_hat,
            toy_target,
            toy_chi2_min,
            best_unconstrained_chi2,
            seed,
            toy_index,
            args.endpoint_residual_tol,
        )

    return row, unconstrained_rows, profile_rows, endpoint_rows


def summarize(
    reproduction_rows: list[dict[str, Any]],
    unconstrained_rows: list[dict[str, Any]],
    profile_rows: list[dict[str, Any]],
    endpoint_rows: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    audit_rows = [row for row in reproduction_rows if int(row["toy_index"]) == args.audit_toy]
    audit_row = audit_rows[0] if audit_rows else {}
    audit_profile_rows = [row for row in profile_rows if int(row["toy_index"]) == args.audit_toy]
    audit_unconstrained_rows = [row for row in unconstrained_rows if int(row["toy_index"]) == args.audit_toy]
    audit_best_profile = min(
        (float(row["result_chi2"]) for row in audit_profile_rows if row.get("result_chi2") is not None),
        default=math.inf,
    )
    audit_best_unconstrained = min(
        (float(row["result_chi2"]) for row in audit_unconstrained_rows if row.get("result_chi2") is not None),
        default=math.inf,
    )
    overall_best_profile = min(
        (float(row["result_chi2"]) for row in profile_rows if row.get("result_chi2") is not None),
        default=math.inf,
    )
    overall_best_unconstrained = min(
        (float(row["result_chi2"]) for row in unconstrained_rows if row.get("result_chi2") is not None),
        default=math.inf,
    )
    return {
        "metadata": {
            "audit_toy": args.audit_toy,
            "base_seed": args.base_seed,
            "dgp": args.dgp,
            "nearby_toys": parse_toy_indices(args.toy_indices),
            "case": f"{CASE.label}::{CASE.target}",
            "endpoint_residual_tol": args.endpoint_residual_tol,
        },
        "row_counts": {
            "reproduction_rows": len(reproduction_rows),
            "unconstrained_rows": len(unconstrained_rows),
            "profile_rows": len(profile_rows),
            "endpoint_rows": len(endpoint_rows),
        },
        "audit_toy": audit_row,
        "audit_toy_best_profile_chi2": audit_best_profile if math.isfinite(audit_best_profile) else None,
        "audit_toy_best_unconstrained_chi2": audit_best_unconstrained if math.isfinite(audit_best_unconstrained) else None,
        "overall_best_profile_chi2": overall_best_profile if math.isfinite(overall_best_profile) else None,
        "overall_best_unconstrained_chi2": overall_best_unconstrained if math.isfinite(overall_best_unconstrained) else None,
        "endpoint_max_abs_residual_vs_reported_refit": max(
            (abs(float(row["residual_vs_reported_refit"])) for row in endpoint_rows),
            default=None,
        ),
        "endpoint_max_abs_residual_vs_best_unconstrained": max(
            (abs(float(row["residual_vs_best_unconstrained"])) for row in endpoint_rows),
            default=None,
        ),
        "negative_q_after_globality_check_count": int(
            sum(bool(row.get("profile_consistency_failure_after_globality_check")) for row in reproduction_rows)
        ),
        "negative_q_reported_count": int(
            sum(bool(row.get("profile_consistency_failure_reported")) for row in reproduction_rows)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-seed", type=int, default=2026062518)
    parser.add_argument("--dgp", choices=round16.DGP_CHOICES, default="split_normal_pdg_resid")
    parser.add_argument("--toy-indices", default="83,84,85")
    parser.add_argument("--audit-toy", type=int, default=84)
    parser.add_argument("--run-endpoint-check", action="store_true")
    parser.add_argument("--endpoint-residual-tol", type=float, default=5e-3)
    parser.add_argument("--output-prefix", default="notes/codex-refactor/round19_b0_consistency_audit")
    args = parser.parse_args()

    prefix = Path(args.output_prefix)
    toy_indices = parse_toy_indices(args.toy_indices)

    fit = run_fit(CASE.label, verbose=False)
    if fit is None:
        raise RuntimeError(f"fit skipped for {CASE.label}")
    context = round15.build_fit_context(CASE.label, fit)
    target_func = round15.target_func_for_fit(fit, CASE.target)
    truth_target = float(target_func(jnp.asarray(fit["param_values"], dtype=jnp.float64)))
    target_std = round15.target_std_for_fit(fit, target_func, truth_target)

    reproduction_rows: list[dict[str, Any]] = []
    unconstrained_rows: list[dict[str, Any]] = []
    profile_rows: list[dict[str, Any]] = []
    endpoint_rows: list[dict[str, Any]] = []

    for toy_index in toy_indices:
        # Round 16/18 seed convention:
        # base + 1_000_000 * case_index(B0=0) + 100_000 * dgp_index(split=1) + toy.
        dgp_index = round16.DGP_CHOICES.index(args.dgp)
        seed = args.base_seed + 100_000 * dgp_index + toy_index
        row, unconstrained, profiles, endpoints = run_one_toy(
            fit,
            context,
            target_func,
            target_std,
            toy_index,
            seed,
            args,
        )
        reproduction_rows.append(row)
        unconstrained_rows.extend(unconstrained)
        profile_rows.extend(profiles)
        endpoint_rows.extend(endpoints)
        print(
            f"toy {toy_index}: reported q={row['profile_true_delta_chi2_production']:.9g}, "
            f"best unconstrained improvement={row['best_unconstrained_improvement_vs_reported']:.9g}, "
            f"q after check={row['q_using_best_unconstrained']:.9g}",
            flush=True,
        )

    write_rows(prefix.with_name(prefix.name + "_reproduction.csv"), prefix.with_name(prefix.name + "_reproduction.jsonl"), reproduction_rows, REPRO_FIELDNAMES)
    write_rows(prefix.with_name(prefix.name + "_unconstrained.csv"), prefix.with_name(prefix.name + "_unconstrained.jsonl"), unconstrained_rows, UNCONSTRAINED_FIELDNAMES)
    write_rows(prefix.with_name(prefix.name + "_profiles.csv"), prefix.with_name(prefix.name + "_profiles.jsonl"), profile_rows, PROFILE_FIELDNAMES)
    write_rows(prefix.with_name(prefix.name + "_endpoints.csv"), prefix.with_name(prefix.name + "_endpoints.jsonl"), endpoint_rows, ENDPOINT_FIELDNAMES)

    summary = summarize(reproduction_rows, unconstrained_rows, profile_rows, endpoint_rows, args)
    summary_path = prefix.with_name(prefix.name + "_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True, default=json_default) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True, default=json_default), flush=True)


if __name__ == "__main__":
    main()
