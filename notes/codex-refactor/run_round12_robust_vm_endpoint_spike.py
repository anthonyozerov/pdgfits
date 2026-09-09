#!/usr/bin/env python
"""Round 12 notes-only robust Venzon-Moolgavkar endpoint-equation spike.

This harness solves endpoint equations directly in fitted-coordinate space:

    grad chi2(x) + lambda * grad target(x) = 0
    chi2(x) = chi2_min + 1

The target value is not fixed by an outer bisection.  Instead, the endpoint is
the target(x) value at the KKT/level solution.  This is a comparator only: the
production bracketed profile endpoint and final profile verification remain the
source of truth.
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

from pdgfits.asym_errors import build_constrained_profile_chi2, calc_asym_errors
from pdgfits.avg import run_avg
from pdgfits.fit import run_fit
from pdgfits.query import avg_queries


jax.config.update("jax_enable_x64", True)


@dataclass(frozen=True)
class CaseSpec:
    case_id: str
    kind: str
    label_or_node: str
    target: str
    default: bool
    rationale: str


CASES = [
    CaseSpec(
        "fit_g2000_k002m",
        "fit",
        "G(2000),G(1800)",
        "K002M",
        True,
        "required smooth low-dimensional direct-coordinate MASS fit",
    ),
    CaseSpec(
        "fit_g2000_k003m",
        "fit",
        "G(2000),G(1800)",
        "K003M",
        True,
        "required smooth low-dimensional direct-coordinate MASS fit",
    ),
    CaseSpec(
        "avg_m002w",
        "avg",
        "M002W",
        "M002W",
        True,
        "required direct-coordinate average; lower endpoint was at residual tolerance edge in round 9",
    ),
    CaseSpec(
        "avg_m026r08",
        "avg",
        "M026R08",
        "M026R08",
        False,
        "optional direct-coordinate average with nuisance adjustments",
    ),
    CaseSpec(
        "fit_b0s_s08637",
        "fit",
        "B0S-BR",
        "S086.37",
        False,
        "optional nonlinear mapped BRU parameter; only run when time remains",
    ),
]


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


def as_float(value: Any) -> float:
    return float(np.asarray(value, dtype=np.float64))


def clean_message(message: Any) -> str:
    return " ".join(str(message).split())


def target_func_for_result(result: dict[str, Any], target: str) -> Callable[[Any], Any]:
    if target in result["nodes"]:
        return result["node_funcs"][result["nodes"].index(target)]
    if target in result["parameters"]:
        return result["parameter_funcs"][result["parameters"].index(target)]
    raise KeyError(f"{target} not found in nodes or parameters")


def covariance_for_avg(avg_result: dict[str, Any]) -> np.ndarray:
    chi2 = avg_result["chi2"]
    x = jnp.asarray(avg_result["param_values"], dtype=jnp.float64)
    hess = np.asarray(jax.hessian(chi2)(x), dtype=np.float64)
    return 2.0 * np.linalg.pinv(hess)


def load_case(
    case: CaseSpec,
    *,
    avg_cache: dict[str, Any],
    fit_cache: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], Callable[[Any], Any], np.ndarray]:
    if case.kind == "fit":
        if case.label_or_node not in fit_cache:
            fit_cache[case.label_or_node] = run_fit(case.label_or_node, verbose=False)
        result = fit_cache[case.label_or_node]
        if result is None:
            raise RuntimeError(f"run_fit returned None for {case.label_or_node}")
        target_func = target_func_for_result(result, case.target)
        asym = calc_asym_errors(result, targets=[case.target])[case.target]
        covariance = np.asarray(result["covariance"], dtype=np.float64)
        return result, asym, target_func, covariance

    if case.kind == "avg":
        if not avg_cache:
            avg_df, corr_df_dict = avg_queries(verbose=False)
            avg_cache["avg_df"] = avg_df
            avg_cache["corr_df_dict"] = corr_df_dict
        avg_df = avg_cache["avg_df"]
        corr_df_dict = avg_cache["corr_df_dict"]
        result = run_avg(
            case.label_or_node,
            avg_df[avg_df["node"] == case.label_or_node],
            corr_df_dict[case.label_or_node],
        )
        if result is None:
            raise RuntimeError(f"run_avg returned None for {case.label_or_node}")
        target_func = target_func_for_result(result, case.target)
        asym = {
            "value": float(result["param_values"][result["parameters"].index(case.target)]),
            "error_p": float(result["error_p"]),
            "error_n": float(result["error_n"]),
            **result["asym_error_diagnostics"],
        }
        covariance = covariance_for_avg(result)
        return result, asym, target_func, covariance

    raise ValueError(f"unknown case kind {case.kind}")


def covariance_predictor(
    x0: np.ndarray,
    covariance: np.ndarray,
    target_grad: np.ndarray,
    displacement: float,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    info: dict[str, Any] = {"predictor_available": False, "predictor_displacement": displacement}
    cov = np.asarray(covariance, dtype=np.float64)
    grad = np.asarray(target_grad, dtype=np.float64)
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
            "predictor_variance": denom,
        }
    )
    if not math.isfinite(denom) or denom <= 0.0:
        info["predictor_failure"] = "nonpositive_target_variance"
        return None, info
    dx = cov_grad * (float(displacement) / denom)
    x = np.asarray(x0, dtype=np.float64) + dx
    info.update(
        {
            "predictor_available": bool(np.all(np.isfinite(x))),
            "predictor_dx_norm": float(np.linalg.norm(dx)) if np.all(np.isfinite(dx)) else None,
        }
    )
    if not np.all(np.isfinite(x)):
        info["predictor_failure"] = "nonfinite_prediction"
        return None, info
    return x, info


class VMSolver:
    def __init__(
        self,
        result: dict[str, Any],
        target_func: Callable[[Any], Any],
        *,
        target_value: float,
        target_chi2: float,
        side_sign: float,
        start_x: np.ndarray,
    ) -> None:
        self.chi2 = result["chi2"]
        self.fitted_to_params = result["fitted_params_to_params"]
        self.target_func = target_func
        self.target_value = float(target_value)
        self.target_chi2 = float(target_chi2)
        self.side_sign = float(side_sign)
        self.n = len(start_x)
        self.counts = {
            "residual_evals": 0,
            "jacobian_evals": 0,
            "chi2_evals": 0,
            "chi2_grad_evals": 0,
            "chi2_hess_evals": 0,
            "target_grad_evals": 0,
            "target_hess_evals": 0,
        }

        @jax.jit
        def target_fitted(fp):
            return target_func(self.fitted_to_params(fp))

        self.target_fitted = target_fitted
        self.chi2_grad = jax.jit(jax.grad(self.chi2))
        self.chi2_hess = jax.jit(jax.hessian(self.chi2))
        self.target_grad = jax.jit(jax.grad(self.target_fitted))
        self.target_hess = jax.jit(jax.hessian(self.target_fitted))

    def target_np(self, x: np.ndarray) -> float:
        return as_float(self.target_fitted(jnp.asarray(x, dtype=jnp.float64)))

    def chi2_np(self, x: np.ndarray) -> float:
        self.counts["chi2_evals"] += 1
        try:
            value = as_float(self.chi2(jnp.asarray(x, dtype=jnp.float64)))
        except Exception:  # noqa: BLE001 - notes-only diagnostic
            return math.inf
        return value if math.isfinite(value) else math.inf

    def grad_np(self, x: np.ndarray) -> np.ndarray:
        self.counts["chi2_grad_evals"] += 1
        return np.asarray(self.chi2_grad(jnp.asarray(x, dtype=jnp.float64)), dtype=np.float64)

    def hess_np(self, x: np.ndarray) -> np.ndarray:
        self.counts["chi2_hess_evals"] += 1
        return np.asarray(self.chi2_hess(jnp.asarray(x, dtype=jnp.float64)), dtype=np.float64)

    def target_grad_np(self, x: np.ndarray) -> np.ndarray:
        self.counts["target_grad_evals"] += 1
        return np.asarray(self.target_grad(jnp.asarray(x, dtype=jnp.float64)), dtype=np.float64)

    def target_hess_np(self, x: np.ndarray) -> np.ndarray:
        self.counts["target_hess_evals"] += 1
        return np.asarray(self.target_hess(jnp.asarray(x, dtype=jnp.float64)), dtype=np.float64)

    def lambda_for_x(self, x: np.ndarray) -> float:
        grad = self.grad_np(x)
        target_grad = self.target_grad_np(x)
        denom = float(target_grad @ target_grad)
        if not math.isfinite(denom) or denom <= 0.0:
            return 0.0
        return -float(grad @ target_grad) / denom

    def raw_metrics(self, z: np.ndarray) -> dict[str, Any]:
        x = np.asarray(z[:-1], dtype=np.float64)
        lam = float(z[-1])
        chi2_value = self.chi2_np(x)
        grad = self.grad_np(x)
        target_grad = self.target_grad_np(x)
        stationarity = grad + lam * target_grad
        target_endpoint = self.target_np(x)
        level_residual = chi2_value - self.target_chi2
        target_displacement = target_endpoint - self.target_value
        denom = float(target_grad @ target_grad)
        if math.isfinite(denom) and denom > 0:
            projected = grad - float(grad @ target_grad) / denom * target_grad
            projected_grad_norm = float(np.linalg.norm(projected))
        else:
            projected_grad_norm = math.inf
        return {
            "chi2": chi2_value,
            "target_endpoint": target_endpoint,
            "level_residual": level_residual,
            "target_displacement": target_displacement,
            "side_ok": bool(self.side_sign * target_displacement > 0),
            "objective_grad_norm": float(np.linalg.norm(grad)) if np.all(np.isfinite(grad)) else math.inf,
            "target_grad_norm": float(np.linalg.norm(target_grad)) if np.all(np.isfinite(target_grad)) else math.inf,
            "kkt_stationarity_norm": float(np.linalg.norm(stationarity))
            if np.all(np.isfinite(stationarity))
            else math.inf,
            "projected_grad_norm": projected_grad_norm,
        }

    def residual_and_jacobian(
        self,
        z: np.ndarray,
        *,
        stationarity_scale: float,
        need_jacobian: bool,
    ) -> tuple[np.ndarray, np.ndarray | None, dict[str, Any]]:
        self.counts["residual_evals"] += 1
        x = np.asarray(z[:-1], dtype=np.float64)
        lam = float(z[-1])
        chi2_value = self.chi2_np(x)
        grad = self.grad_np(x)
        target_grad = self.target_grad_np(x)
        target_endpoint = self.target_np(x)
        stationarity = grad + lam * target_grad
        level_residual = chi2_value - self.target_chi2
        residual = np.r_[stationarity / stationarity_scale, level_residual]
        metrics = {
            "chi2": chi2_value,
            "target_endpoint": target_endpoint,
            "level_residual": level_residual,
            "target_displacement": target_endpoint - self.target_value,
            "scaled_residual_norm": float(np.linalg.norm(residual))
            if np.all(np.isfinite(residual))
            else math.inf,
        }
        if not need_jacobian:
            return residual, None, metrics
        self.counts["jacobian_evals"] += 1
        hess = self.hess_np(x)
        target_hess = self.target_hess_np(x)
        jacobian = np.zeros((self.n + 1, self.n + 1), dtype=np.float64)
        jacobian[: self.n, : self.n] = (hess + lam * target_hess) / stationarity_scale
        jacobian[: self.n, self.n] = target_grad / stationarity_scale
        jacobian[self.n, : self.n] = grad
        return residual, jacobian, metrics


def solve_linear(jacobian: np.ndarray, residual: np.ndarray) -> tuple[np.ndarray, bool, str]:
    try:
        return np.linalg.solve(jacobian, -residual), False, "solve"
    except np.linalg.LinAlgError:
        step, *_ = np.linalg.lstsq(jacobian, -residual, rcond=None)
        return step, True, "lstsq_singular"


def convergence_status(metrics: dict[str, Any]) -> tuple[bool, str]:
    level_ok = abs(float(metrics["level_residual"])) <= 1e-6
    stationarity_scale = max(float(metrics["objective_grad_norm"]), 1.0)
    stationarity_ok = float(metrics["projected_grad_norm"]) <= 1e-5 * stationarity_scale
    side_ok = bool(metrics["side_ok"])
    if level_ok and stationarity_ok and side_ok:
        return True, "converged"
    if not side_ok:
        return False, "side_miss"
    if not level_ok:
        return False, "residual_miss"
    return False, "stationarity_miss"


def solve_vm(
    vm: VMSolver,
    start_x: np.ndarray,
    *,
    solver: str,
    max_iter: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    start_x = np.asarray(start_x, dtype=np.float64)
    lambda0 = vm.lambda_for_x(start_x)
    z = np.r_[start_x, lambda0]
    start_stationarity = vm.grad_np(start_x) + lambda0 * vm.target_grad_np(start_x)
    stationarity_scale = max(float(np.linalg.norm(start_stationarity)), 1.0)
    x_scale = np.maximum(np.maximum(np.abs(start_x), 1.0), 1e-12)
    z_scale = np.r_[x_scale, max(abs(lambda0), 1.0)]
    radius = 1.0
    damping_events = 0
    trust_region_clips = 0
    rejected_steps = 0
    singular_steps = 0
    step_source = ""
    iteration_rows: list[dict[str, Any]] = []
    status = "max_iter"
    t0 = time.perf_counter()

    for iteration in range(max_iter + 1):
        residual, jacobian, scaled_metrics = vm.residual_and_jacobian(
            z,
            stationarity_scale=stationarity_scale,
            need_jacobian=True,
        )
        raw = vm.raw_metrics(z)
        converged, current_status = convergence_status(raw)
        iteration_rows.append(
            {
                "solver": solver,
                "iteration": iteration,
                "chi2": raw["chi2"],
                "target_endpoint": raw["target_endpoint"],
                "level_residual": raw["level_residual"],
                "target_displacement": raw["target_displacement"],
                "kkt_stationarity_norm": raw["kkt_stationarity_norm"],
                "projected_grad_norm": raw["projected_grad_norm"],
                "scaled_residual_norm": scaled_metrics["scaled_residual_norm"],
                "radius": radius if solver == "robust_trust" else None,
                "damping_events_so_far": damping_events,
                "trust_region_clips_so_far": trust_region_clips,
                "rejected_steps_so_far": rejected_steps,
            }
        )
        if converged:
            status = current_status
            break
        if iteration == max_iter:
            status = current_status
            break
        if not np.all(np.isfinite(residual)) or jacobian is None or not np.all(np.isfinite(jacobian)):
            status = "nonfinite"
            break

        step, singular, step_source = solve_linear(jacobian, residual)
        singular_steps += int(singular)
        if not np.all(np.isfinite(step)):
            status = "singular_kkt"
            break

        if solver == "plain_newton":
            z = z + step
            if not np.all(np.isfinite(z)):
                status = "nonfinite"
                break
            continue

        step_norm = float(np.linalg.norm(step / z_scale))
        if step_norm > radius and step_norm > 0.0:
            step = step * (radius / step_norm)
            trust_region_clips += 1

        base_norm = float(scaled_metrics["scaled_residual_norm"])
        accepted = False
        alpha = 1.0
        while alpha >= 2.0**-20:
            trial_z = z + alpha * step
            if not np.all(np.isfinite(trial_z)):
                alpha *= 0.5
                continue
            trial_residual, _, trial_scaled = vm.residual_and_jacobian(
                trial_z,
                stationarity_scale=stationarity_scale,
                need_jacobian=False,
            )
            trial_raw = vm.raw_metrics(trial_z)
            side_ok = bool(vm.side_sign * float(trial_raw["target_displacement"]) > 0)
            trial_norm = float(trial_scaled["scaled_residual_norm"])
            if (
                np.all(np.isfinite(trial_residual))
                and math.isfinite(trial_norm)
                and trial_norm < base_norm
                and side_ok
            ):
                accepted = True
                z = trial_z
                if alpha < 1.0:
                    damping_events += 1
                if alpha == 1.0 and step_norm <= 0.8 * radius:
                    radius = min(4.0, 1.5 * radius)
                elif alpha < 1.0:
                    radius = max(1e-6, 0.7 * radius)
                break
            alpha *= 0.5
        if not accepted:
            rejected_steps += 1
            radius *= 0.5
            if radius < 1e-8:
                status = "trust_region_stalled"
                break

    final_raw = vm.raw_metrics(z)
    converged, final_status = convergence_status(final_raw)
    if converged:
        status = final_status
    elif status == "converged":
        status = final_status
    result = {
        "solver": solver,
        "status": status,
        "success": bool(converged),
        "iterations": len(iteration_rows) - 1,
        "runtime_sec": time.perf_counter() - t0,
        "lambda": float(z[-1]) if np.all(np.isfinite(z)) else None,
        "step_source_last": step_source,
        "singular_steps": singular_steps,
        "damping_events": damping_events,
        "trust_region_clips": trust_region_clips,
        "rejected_steps": rejected_steps,
        **{f"final_{k}": v for k, v in final_raw.items()},
        **{f"count_{k}": v for k, v in vm.counts.items()},
    }
    return result, iteration_rows


def verification_profile(
    result: dict[str, Any],
    target_func: Callable[[Any], Any],
    endpoint: float,
    target_std: float,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    profile = build_constrained_profile_chi2(
        result["chi2"],
        result["fitted_params_to_params"],
        target_func,
        result["fitted_values"],
        target_scale=target_std,
        target_name="round12_vm_endpoint",
    )
    chi2_value = float(profile(endpoint))
    point = getattr(profile, "last_point", None)
    return {
        "verification_profile_chi2": chi2_value,
        "verification_profile_runtime_sec": time.perf_counter() - t0,
        "verification_profile_success": getattr(point, "success", None),
        "verification_profile_method": getattr(point, "method", None),
        "verification_profile_nfev": getattr(point, "nfev", None),
        "verification_profile_projected_grad_norm": getattr(point, "projected_grad_norm", None),
        "verification_profile_message": clean_message(getattr(point, "message", "")),
    }


def run_case(
    case: CaseSpec,
    *,
    avg_cache: dict[str, Any],
    fit_cache: dict[str, Any],
    verify_profile: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    result, asym, target_func, covariance = load_case(case, avg_cache=avg_cache, fit_cache=fit_cache)
    x0 = np.asarray(result["fitted_values"], dtype=np.float64)
    chi2_min = float(result["chi2_min"])
    target_chi2 = chi2_min + 1.0
    target_value = float(asym["value"])
    fitted_to_params = result["fitted_params_to_params"]

    @jax.jit
    def target_fitted(fp):
        return target_func(fitted_to_params(fp))

    target_grad0 = np.asarray(jax.grad(target_fitted)(jnp.asarray(x0, dtype=jnp.float64)), dtype=np.float64)
    target_variance = float(target_grad0 @ np.asarray(covariance, dtype=np.float64) @ target_grad0)
    target_std = math.sqrt(target_variance) if target_variance > 0 and math.isfinite(target_variance) else math.nan
    if not math.isfinite(target_std) or target_std <= 0:
        target_std = max(abs(target_value), 1.0) * 1e-3

    rows: list[dict[str, Any]] = []
    iter_rows: list[dict[str, Any]] = []
    for side in ("lower", "upper"):
        sign = -1.0 if side == "lower" else 1.0
        production_endpoint = float(asym[f"{side}_endpoint"])
        production_error = float(asym[f"{side}_error"])
        production_profile_chi2 = float(asym[f"{side}_chi2"])
        production_residual = float(asym[f"{side}_residual"])
        starts = [
            ("hesse_one_sigma", sign * target_std),
            ("production_displacement_cov", production_endpoint - target_value),
        ]
        for start_mode, displacement in starts:
            start_x, predictor = covariance_predictor(x0, covariance, target_grad0, displacement)
            if start_x is None:
                rows.append(
                    {
                        "case_id": case.case_id,
                        "kind": case.kind,
                        "label_or_node": case.label_or_node,
                        "target": case.target,
                        "side": side,
                        "start_mode": start_mode,
                        "solver": "not_run",
                        "status": predictor.get("predictor_failure"),
                        "success": False,
                        **predictor,
                    }
                )
                continue
            for solver_name in ("plain_newton", "robust_trust"):
                vm = VMSolver(
                    result,
                    target_func,
                    target_value=target_value,
                    target_chi2=target_chi2,
                    side_sign=sign,
                    start_x=start_x,
                )
                solve_result, solver_iters = solve_vm(
                    vm,
                    start_x,
                    solver=solver_name,
                    max_iter=20 if solver_name == "plain_newton" else 40,
                )
                endpoint = solve_result.get("final_target_endpoint")
                endpoint_delta = (
                    float(endpoint) - production_endpoint
                    if endpoint is not None and math.isfinite(float(endpoint))
                    else None
                )
                row = {
                    "case_id": case.case_id,
                    "kind": case.kind,
                    "label_or_node": case.label_or_node,
                    "target": case.target,
                    "rationale": case.rationale,
                    "n_parameters": len(result["parameters"]),
                    "side": side,
                    "start_mode": start_mode,
                    "target_value": target_value,
                    "target_std": target_std,
                    "target_chi2": target_chi2,
                    "production_endpoint": production_endpoint,
                    "production_error": production_error,
                    "production_profile_chi2": production_profile_chi2,
                    "production_residual": production_residual,
                    "production_root_function_evals": asym.get("function_evals"),
                    "production_side_function_evals": asym.get("side_counts", {}).get(side, {}).get("function_evals"),
                    "production_profile_method": asym.get(f"{side}_profile_point", {}).get("method"),
                    "production_profile_nfev": asym.get(f"{side}_profile_point", {}).get("nfev"),
                    **predictor,
                    **solve_result,
                    "endpoint_delta_vs_production": endpoint_delta,
                    "abs_endpoint_delta_vs_production": abs(endpoint_delta) if endpoint_delta is not None else None,
                    "vm_chi2_delta_vs_target": solve_result.get("final_level_residual"),
                    "vm_chi2_delta_vs_production_profile": (
                        float(solve_result["final_chi2"]) - production_profile_chi2
                        if math.isfinite(float(solve_result.get("final_chi2", math.inf)))
                        else None
                    ),
                    "vm_profile_solves": 0,
                    "raw_newton_saved_by_robust": None,
                }
                if verify_profile and solver_name == "robust_trust" and row["success"] and endpoint is not None:
                    try:
                        verification = verification_profile(result, target_func, float(endpoint), target_std)
                        row.update(verification)
                        row["verification_residual_vs_target"] = row["verification_profile_chi2"] - target_chi2
                        row["verification_minus_vm_chi2"] = (
                            row["verification_profile_chi2"] - float(solve_result["final_chi2"])
                        )
                        row["verification_minus_production_profile_chi2"] = (
                            row["verification_profile_chi2"] - production_profile_chi2
                        )
                    except Exception as exc:  # noqa: BLE001 - diagnostic row records failure
                        row.update(
                            {
                                "verification_exception": f"{type(exc).__name__}: {exc}",
                                "verification_profile_success": False,
                            }
                        )
                rows.append(row)
                for iter_row in solver_iters:
                    iter_rows.append(
                        {
                            "case_id": case.case_id,
                            "kind": case.kind,
                            "label_or_node": case.label_or_node,
                            "target": case.target,
                            "side": side,
                            "start_mode": start_mode,
                            **iter_row,
                        }
                    )

    # Mark robust saves after both solver rows exist for each start.
    by_key: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
    for row in rows:
        if row.get("solver") in ("plain_newton", "robust_trust"):
            by_key.setdefault((str(row["side"]), str(row["start_mode"])), {})[str(row["solver"])] = row
    for pair in by_key.values():
        raw = pair.get("plain_newton")
        robust = pair.get("robust_trust")
        if raw is not None and robust is not None:
            saved = bool((not raw.get("success", False)) and robust.get("success", False))
            robust["raw_newton_saved_by_robust"] = saved
            raw["raw_newton_saved_by_robust"] = False
    return rows, iter_rows


def parse_cases(selected: list[str] | None, include_optional: bool) -> list[CaseSpec]:
    by_id = {case.case_id: case for case in CASES}
    if selected:
        out = []
        for case_id in selected:
            if case_id not in by_id:
                raise SystemExit(f"unknown case id {case_id}; available: {', '.join(sorted(by_id))}")
            out.append(by_id[case_id])
        return out
    return [case for case in CASES if case.default or include_optional]


def main_impl(args: argparse.Namespace) -> int:
    cases = parse_cases(args.case, args.include_optional)
    rows: list[dict[str, Any]] = []
    iter_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    avg_cache: dict[str, Any] = {}
    fit_cache: dict[str, Any] = {}
    for case in cases:
        print(f"\n=== {case.case_id}: {case.kind} {case.label_or_node} / {case.target} ===")
        t0 = time.perf_counter()
        try:
            case_rows, case_iter_rows = run_case(
                case,
                avg_cache=avg_cache,
                fit_cache=fit_cache,
                verify_profile=not args.skip_profile_verification,
            )
            rows.extend(case_rows)
            iter_rows.extend(case_iter_rows)
            print(f"completed {case.case_id} in {time.perf_counter() - t0:.3f}s")
            for row in case_rows:
                if row.get("solver") == "robust_trust" and row.get("start_mode") == "hesse_one_sigma":
                    print(
                        row["side"],
                        row["status"],
                        "endpoint_delta",
                        row.get("endpoint_delta_vs_production"),
                        "resid",
                        row.get("final_level_residual"),
                        "verify",
                        row.get("verification_residual_vs_target"),
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
    write_rows(Path(args.iterations_csv), iter_rows)
    write_rows(Path(args.iterations_jsonl), iter_rows, jsonl=True)
    write_rows(Path(args.failures_csv), failures)
    write_rows(Path(args.failures_jsonl), failures, jsonl=True)
    print(f"\nwrote {len(rows)} result rows")
    print(f"wrote {len(iter_rows)} iteration rows")
    print(f"wrote {len(failures)} failure rows")
    return 0 if not failures else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append", help="Case id to run; repeatable. Defaults to required cases.")
    parser.add_argument("--include-optional", action="store_true", help="Also run optional avg_m026r08 and B0S-BR cases.")
    parser.add_argument("--skip-profile-verification", action="store_true", help="Do not rerun production profile verifier at VM endpoints.")
    parser.add_argument("--keep-going", action="store_true", help="Continue after a case-level failure.")
    parser.add_argument("--results-csv", default="notes/codex-refactor/round12_robust_vm_endpoint_results.csv")
    parser.add_argument("--results-jsonl", default="notes/codex-refactor/round12_robust_vm_endpoint_results.jsonl")
    parser.add_argument("--iterations-csv", default="notes/codex-refactor/round12_robust_vm_endpoint_iterations.csv")
    parser.add_argument("--iterations-jsonl", default="notes/codex-refactor/round12_robust_vm_endpoint_iterations.jsonl")
    parser.add_argument("--failures-csv", default="notes/codex-refactor/round12_robust_vm_endpoint_failures.csv")
    parser.add_argument("--failures-jsonl", default="notes/codex-refactor/round12_robust_vm_endpoint_failures.jsonl")
    parser.add_argument("--stdout-log", default=None, help="Optional path to tee stdout/stderr into.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.stdout_log:
        log_path = Path(args.stdout_log)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as log_file:
            with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
                return main_impl(args)
    return main_impl(args)


if __name__ == "__main__":
    raise SystemExit(main())
