#!/usr/bin/env python
"""Fixed-endpoint multistart diagnostics for fit-side profile solves.

This is validation tooling, not production code.  It mirrors the current
fit-side constrained-profile acceptance checks but exposes explicit starts for
hard fixed targets whose accepted endpoints used the descent-check path.
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
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

from jax import numpy as jnp
import numpy as np
from scipy.optimize import NonlinearConstraint, OptimizeResult, minimize, root

from pdgfits.fit import run_fit


CONS_TOL = 1e-7
STATIONARITY_FUN_TOL = 1e-4


SUMMARY_FIELDNAMES = [
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
    "accepted_method",
    "accepted_profile_chi2",
    "accepted_residual",
    "accepted_projected_grad_norm",
    "accepted_objective_grad_norm",
    "best_multistart_chi2",
    "best_multistart_residual",
    "best_start_name",
    "best_method",
    "improvement_vs_accepted",
    "material_improvement",
    "n_starts",
    "n_ok",
    "n_failed",
    "fit_runtime_sec",
    "endpoint_runtime_sec",
    "status",
    "exception_type",
    "exception",
]

START_FIELDNAMES = [
    "label",
    "target",
    "side",
    "endpoint",
    "start_name",
    "start_kind",
    "ok",
    "chi2",
    "residual",
    "improvement_vs_accepted",
    "method",
    "constraint_violation",
    "scaled_constraint_violation",
    "projected_grad_norm",
    "objective_grad_norm",
    "descent_improvement",
    "projected_start_chi2",
    "projected_start_scaled_constraint_violation",
    "nfev",
    "nit",
    "runtime_sec",
    "message",
]


class EndpointTimeout(TimeoutError):
    pass


@dataclass
class SolveCandidate:
    start_name: str
    start_kind: str
    ok: bool
    chi2: float
    constraint_violation: float
    scaled_constraint_violation: float
    projected_grad_norm: float
    objective_grad_norm: float
    method: str
    message: str
    nfev: int | None
    nit: int | None
    runtime_sec: float
    descent_improvement: float | None
    projected_start_chi2: float
    projected_start_scaled_constraint_violation: float
    x: np.ndarray | None


def _alarm_handler(signum, frame):  # noqa: ARG001
    raise EndpointTimeout("endpoint timed out")


def _as_float(x) -> float:
    return float(np.asarray(x, dtype=np.float64))


def _finite_float(value, default=math.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


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
        writer.writerow({key: clean(row.get(key)) for key in fieldnames})


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(clean(row), sort_keys=True) + "\n")


def parse_endpoint_spec(spec: str) -> tuple[str, str, str]:
    parts = spec.split("::")
    if len(parts) != 3 or parts[2] not in {"upper", "lower"}:
        raise ValueError(f"Expected endpoint spec 'LABEL::TARGET::upper|lower', got {spec!r}")
    return parts[0], parts[1], parts[2]


def read_source_rows(path: Path) -> dict[tuple[str, str], dict]:
    rows: dict[tuple[str, str], dict] = {}
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("status") != "ok":
                continue
            rows[(row.get("label", ""), row.get("target", ""))] = row
    return rows


def target_function(fit: dict, target: str) -> tuple[str, Callable]:
    if target in fit["nodes"]:
        return "node", fit["node_funcs"][fit["nodes"].index(target)]
    if target in fit["parameters"]:
        return "parameter", fit["parameter_funcs"][fit["parameters"].index(target)]
    raise KeyError(f"{target!r} is not a node or parameter in {fit['label']}")


def target_std_from_cov(fit: dict, target_func: Callable, target_value: float) -> float:
    covariance = fit["covariance"]
    if covariance is not None:
        jac_params = jax.jacobian(fit["fitted_params_to_params"])(fit["fitted_values"])
        param_cov = jac_params @ covariance @ jac_params.T
        jac_target = jax.jacobian(target_func)(fit["param_values"])
        target_std = float(jnp.sqrt(jac_target @ param_cov @ jac_target.T))
    else:
        target_std = max(abs(target_value), 1.0) * 1e-3
    if not np.isfinite(target_std) or target_std <= 0:
        target_std = max(abs(target_value), 1.0) * 1e-3
    return float(target_std)


def _covariance_sqrt(covariance, dim: int) -> tuple[np.ndarray | None, np.ndarray, np.ndarray]:
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
    sqrt_cov = evecs @ np.diag(np.sqrt(evals))
    return sqrt_cov, evals, evecs


def _dedupe_starts(starts: list[tuple[str, str, np.ndarray]]) -> list[tuple[str, str, np.ndarray]]:
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


def build_start_points(
    fp_best: np.ndarray,
    covariance,
    *,
    rng: np.random.Generator,
    cov_dirs: int,
    cov_scale: float,
    random_starts: int,
    random_scale: float,
) -> tuple[list[tuple[str, str, np.ndarray]], np.ndarray | None]:
    dim = len(fp_best)
    starts: list[tuple[str, str, np.ndarray]] = [("mle", "mle", fp_best)]
    sqrt_cov, evals, evecs = _covariance_sqrt(covariance, dim)

    for i in range(min(cov_dirs, len(evals))):
        step = cov_scale * math.sqrt(float(evals[i])) * evecs[:, i]
        starts.append((f"cov_eig{i + 1}_plus", "covariance_direction", fp_best + step))
        starts.append((f"cov_eig{i + 1}_minus", "covariance_direction", fp_best - step))

    for i in range(random_starts):
        z = rng.normal(size=dim)
        if sqrt_cov is not None:
            step = random_scale * (sqrt_cov @ z) / max(math.sqrt(dim), 1.0)
            kind = "random_covariance"
        else:
            norm_scale = random_scale * max(float(np.linalg.norm(fp_best)), 1.0) / max(math.sqrt(dim), 1.0)
            step = norm_scale * z
            kind = "random_coordinate"
        starts.append((f"random_mle_{i + 1}", kind, fp_best + step))

    return _dedupe_starts(starts), sqrt_cov


def add_endpoint_perturbation_starts(
    starts: list[tuple[str, str, np.ndarray]],
    endpoint_x: np.ndarray | None,
    sqrt_cov: np.ndarray | None,
    *,
    rng: np.random.Generator,
    n_endpoint_random: int,
    endpoint_random_scale: float,
) -> list[tuple[str, str, np.ndarray]]:
    if endpoint_x is None or not np.all(np.isfinite(endpoint_x)):
        return starts
    dim = len(endpoint_x)
    for i in range(n_endpoint_random):
        z = rng.normal(size=dim)
        if sqrt_cov is not None:
            step = endpoint_random_scale * (sqrt_cov @ z) / max(math.sqrt(dim), 1.0)
            kind = "random_endpoint_covariance"
        else:
            norm_scale = endpoint_random_scale * max(float(np.linalg.norm(endpoint_x)), 1.0) / max(math.sqrt(dim), 1.0)
            step = norm_scale * z
            kind = "random_endpoint_coordinate"
        starts.append((f"random_endpoint_{i + 1}", kind, endpoint_x + step))
    return _dedupe_starts(starts)


def build_fixed_target_solver(
    fit: dict,
    target_func: Callable,
    endpoint_value: float,
    target_scale: float,
):
    chi2 = fit["chi2"]
    fitted_params_to_params = fit["fitted_params_to_params"]
    fp_best = np.asarray(fit["fitted_values"], dtype=np.float64)
    target_scale = abs(float(target_scale))
    if not np.isfinite(target_scale) or target_scale == 0.0:
        target_scale = 1.0

    def chi2_np(x):
        val = _as_float(chi2(jnp.asarray(x, dtype=jnp.float64)))
        if not np.isfinite(val):
            return np.inf
        return val

    chi2_grad_jax = jax.jit(jax.grad(chi2))

    def chi2_grad_np(x):
        return np.asarray(chi2_grad_jax(jnp.asarray(x, dtype=jnp.float64)), dtype=np.float64)

    chi2_hess_jax = jax.jit(jax.hessian(chi2))

    def chi2_hess_np(x):
        return np.asarray(chi2_hess_jax(jnp.asarray(x, dtype=jnp.float64)), dtype=np.float64)

    @jax.jit
    def constraint_value_jax(fp):
        return target_func(fitted_params_to_params(fp)) - endpoint_value

    @jax.jit
    def scaled_constraint_value_jax(fp):
        return constraint_value_jax(fp) / target_scale

    scaled_constraint_grad_jax = jax.jit(jax.grad(scaled_constraint_value_jax))
    scaled_constraint_hess_jax = jax.jit(jax.hessian(scaled_constraint_value_jax))

    def project_start(start, tol=CONS_TOL):
        x = np.asarray(start, dtype=np.float64).copy()
        for _ in range(20):
            if not np.all(np.isfinite(x)):
                break
            c = _as_float(scaled_constraint_value_jax(jnp.asarray(x)))
            if not np.isfinite(c):
                break
            if abs(c) <= tol:
                return x
            g = np.asarray(scaled_constraint_grad_jax(jnp.asarray(x)), dtype=np.float64)
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

    def scaled_violation(x):
        return abs(_as_float(constraint_value_jax(jnp.asarray(x)))) / target_scale

    def projected_grad_norm(x):
        x = np.asarray(x, dtype=np.float64)
        g_obj = chi2_grad_np(x)
        g_con = np.asarray(scaled_constraint_grad_jax(jnp.asarray(x)), dtype=np.float64)
        denom = float(np.dot(g_con, g_con))
        if not np.isfinite(denom) or denom == 0.0:
            return np.inf
        g_proj = g_obj - (float(np.dot(g_obj, g_con)) / denom) * g_con
        return float(np.linalg.norm(g_proj))

    def _projected_gradient(x):
        x = np.asarray(x, dtype=np.float64)
        g_obj = chi2_grad_np(x)
        g_con = np.asarray(scaled_constraint_grad_jax(jnp.asarray(x)), dtype=np.float64)
        denom = float(np.dot(g_con, g_con))
        if not np.isfinite(denom) or denom == 0.0:
            return None
        return g_obj - (float(np.dot(g_obj, g_con)) / denom) * g_con

    def projected_descent_improvement(x):
        x = np.asarray(x, dtype=np.float64)
        if not np.all(np.isfinite(x)):
            return np.inf
        base_fun = chi2_np(x)
        if not np.isfinite(base_fun) or scaled_violation(x) > CONS_TOL:
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
            trial = project_start(x + step * direction, tol=min(CONS_TOL, 1e-10))
            if not np.all(np.isfinite(trial)) or scaled_violation(trial) > CONS_TOL:
                continue
            trial_fun = chi2_np(trial)
            if np.isfinite(trial_fun):
                best_decrease = max(best_decrease, base_fun - trial_fun)
        return float(best_decrease)

    def make_constraint(exact_hess=False):
        kwargs = {}
        if exact_hess:
            kwargs["hess"] = lambda x, multiplier: np.asarray(
                multiplier[0] * scaled_constraint_hess_jax(jnp.asarray(x)),
                dtype=np.float64,
            )
        return NonlinearConstraint(
            fun=lambda x: _as_float(scaled_constraint_value_jax(jnp.asarray(x))),
            lb=0.0,
            ub=0.0,
            jac=lambda x: np.asarray(scaled_constraint_grad_jax(jnp.asarray(x)), dtype=np.float64),
            **kwargs,
        )

    def result_ok(opt_result):
        cached = getattr(opt_result, "profile_result_ok", None)
        if cached is not None:
            return bool(cached)
        scaled_cviol = scaled_violation(opt_result.x)
        finite_result = np.isfinite(opt_result.fun)
        if not finite_result or scaled_cviol > CONS_TOL:
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
        if descent_improvement <= STATIONARITY_FUN_TOL:
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
        x0 = project_start(opt_result.x)
        g_obj = chi2_grad_np(x0)
        g_con = np.asarray(scaled_constraint_grad_jax(jnp.asarray(x0)), dtype=np.float64)
        denom = float(np.dot(g_con, g_con))
        if not np.isfinite(denom) or denom == 0.0:
            return opt_result
        lambda0 = -float(np.dot(g_obj, g_con)) / denom

        def residual(z):
            x = np.asarray(z[:-1], dtype=np.float64)
            lam = float(z[-1])
            stationarity = chi2_grad_np(x) + lam * np.asarray(
                scaled_constraint_grad_jax(jnp.asarray(x)), dtype=np.float64
            )
            return np.r_[stationarity, _as_float(scaled_constraint_value_jax(jnp.asarray(x)))]

        try:
            polished = root(residual, np.r_[x0, lambda0], method="hybr", options={"maxfev": 5000})
        except (FloatingPointError, ValueError, np.linalg.LinAlgError):
            return opt_result

        x_polished = np.asarray(polished.x[:-1], dtype=np.float64)
        if not np.all(np.isfinite(x_polished)) or scaled_violation(x_polished) > CONS_TOL:
            return opt_result
        fun_polished = chi2_np(x_polished)
        if not np.isfinite(fun_polished):
            return opt_result
        if fun_polished > opt_result.fun + max(1e-7, 1e-9 * abs(opt_result.fun)):
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

    def maybe_keep_feasible_start(opt_result, x0):
        x0_fun = chi2_np(x0)
        if not np.isfinite(x0_fun) or scaled_violation(x0) > CONS_TOL:
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

    def solve_from(start_name: str, start_kind: str, start: np.ndarray) -> SolveCandidate:
        t0 = time.perf_counter()
        projected_start = project_start(start)
        projected_start_chi2 = chi2_np(projected_start)
        projected_start_scaled_cviol = scaled_violation(projected_start)

        candidates = []
        try:
            result = minimize(
                chi2_np,
                projected_start,
                method="SLSQP",
                jac=chi2_grad_np,
                constraints=[make_constraint()],
                options={"ftol": 1e-10, "maxiter": 2000},
            )
        except (FloatingPointError, ValueError, np.linalg.LinAlgError) as exc:
            result = OptimizeResult(
                x=np.asarray(projected_start, dtype=np.float64),
                fun=projected_start_chi2,
                success=False,
                message=f"SLSQP failed: {exc}",
            )
        result = maybe_keep_feasible_start(result, projected_start)
        if not getattr(result, "profile_method", ""):
            result.profile_method = "SLSQP"
        candidates.append(result)

        if not result_ok(result):
            try:
                exact_hess_result = minimize(
                    chi2_np,
                    projected_start,
                    method="trust-constr",
                    jac=chi2_grad_np,
                    hess=chi2_hess_np,
                    constraints=[make_constraint(exact_hess=True)],
                    options={"gtol": 1e-10, "xtol": 1e-10, "maxiter": 2000},
                )
            except (FloatingPointError, ValueError, np.linalg.LinAlgError) as exc:
                exact_hess_result = OptimizeResult(
                    x=np.asarray(projected_start, dtype=np.float64),
                    fun=projected_start_chi2,
                    success=False,
                    message=f"trust-constr exact Hessian failed: {exc}",
                )
            exact_hess_result = maybe_keep_feasible_start(exact_hess_result, projected_start)
            if not getattr(exact_hess_result, "profile_method", ""):
                exact_hess_result.profile_method = "trust-constr-exact-hess"
            exact_hess_result = kkt_polish(exact_hess_result)
            candidates.append(exact_hess_result)

        ok_candidates = [candidate for candidate in candidates if result_ok(candidate)]
        if ok_candidates:
            final = min(ok_candidates, key=lambda candidate: candidate.fun)
        else:
            final = min(candidates, key=lambda candidate: candidate.fun if np.isfinite(candidate.fun) else np.inf)

        finite = np.isfinite(final.fun) and np.all(np.isfinite(final.x))
        cviol = abs(_as_float(constraint_value_jax(jnp.asarray(final.x)))) if finite else math.inf
        scaled_cviol = cviol / target_scale if finite else math.inf
        ok = result_ok(final)
        g_norm = float(np.linalg.norm(chi2_grad_np(final.x))) if finite else math.inf
        pg_norm = projected_grad_norm(final.x) if finite else math.inf
        return SolveCandidate(
            start_name=start_name,
            start_kind=start_kind,
            ok=ok,
            chi2=float(final.fun) if finite else math.inf,
            constraint_violation=cviol,
            scaled_constraint_violation=scaled_cviol,
            projected_grad_norm=pg_norm,
            objective_grad_norm=g_norm,
            method=str(getattr(final, "profile_method", "")),
            message=str(getattr(final, "message", "")),
            nfev=getattr(final, "nfev", None),
            nit=getattr(final, "nit", None),
            runtime_sec=time.perf_counter() - t0,
            descent_improvement=getattr(final, "profile_descent_improvement", None),
            projected_start_chi2=projected_start_chi2,
            projected_start_scaled_constraint_violation=projected_start_scaled_cviol,
            x=np.asarray(final.x, dtype=np.float64) if finite else None,
        )

    return solve_from


def candidate_row(
    candidate: SolveCandidate,
    *,
    label: str,
    target: str,
    side: str,
    endpoint: float,
    chi2_min: float,
    accepted_chi2: float,
) -> dict:
    return {
        "label": label,
        "target": target,
        "side": side,
        "endpoint": endpoint,
        "start_name": candidate.start_name,
        "start_kind": candidate.start_kind,
        "ok": candidate.ok,
        "chi2": candidate.chi2,
        "residual": candidate.chi2 - (chi2_min + 1.0),
        "improvement_vs_accepted": accepted_chi2 - candidate.chi2,
        "method": candidate.method,
        "constraint_violation": candidate.constraint_violation,
        "scaled_constraint_violation": candidate.scaled_constraint_violation,
        "projected_grad_norm": candidate.projected_grad_norm,
        "objective_grad_norm": candidate.objective_grad_norm,
        "descent_improvement": candidate.descent_improvement,
        "projected_start_chi2": candidate.projected_start_chi2,
        "projected_start_scaled_constraint_violation": candidate.projected_start_scaled_constraint_violation,
        "nfev": candidate.nfev,
        "nit": candidate.nit,
        "runtime_sec": candidate.runtime_sec,
        "message": candidate.message,
    }


def run_endpoint(
    *,
    fit: dict,
    fit_runtime: float,
    source_row: dict,
    side: str,
    args,
    start_jsonl: Path,
) -> dict:
    t0 = time.perf_counter()
    target = source_row["target"]
    target_kind, target_func = target_function(fit, target)
    target_value = float(target_func(fit["param_values"]))
    target_std = target_std_from_cov(fit, target_func, target_value)
    endpoint = _finite_float(source_row[f"{side}_endpoint"])
    accepted_chi2 = _finite_float(source_row[f"{side}_chi2"])
    accepted_residual = _finite_float(source_row[f"{side}_residual"])
    accepted_method = source_row.get(f"{side}_method", "")

    solver = build_fixed_target_solver(fit, target_func, endpoint, target_std)
    rng = np.random.default_rng(args.seed)
    fp_best = np.asarray(fit["fitted_values"], dtype=np.float64)
    starts, sqrt_cov = build_start_points(
        fp_best,
        fit["covariance"],
        rng=rng,
        cov_dirs=args.cov_dirs,
        cov_scale=args.cov_scale,
        random_starts=args.random_starts,
        random_scale=args.random_scale,
    )

    candidates: list[SolveCandidate] = []
    for start_name, start_kind, start in starts[:1]:
        candidate = solver(start_name, start_kind, start)
        candidates.append(candidate)
        append_jsonl(start_jsonl, candidate_row(
            candidate,
            label=fit["label"],
            target=target,
            side=side,
            endpoint=endpoint,
            chi2_min=float(fit["chi2_min"]),
            accepted_chi2=accepted_chi2,
        ))

    endpoint_x = candidates[0].x if candidates else None
    starts = add_endpoint_perturbation_starts(
        starts,
        endpoint_x,
        sqrt_cov,
        rng=rng,
        n_endpoint_random=args.endpoint_random_starts,
        endpoint_random_scale=args.endpoint_random_scale,
    )

    for start_name, start_kind, start in starts[1:]:
        candidate = solver(start_name, start_kind, start)
        candidates.append(candidate)
        append_jsonl(start_jsonl, candidate_row(
            candidate,
            label=fit["label"],
            target=target,
            side=side,
            endpoint=endpoint,
            chi2_min=float(fit["chi2_min"]),
            accepted_chi2=accepted_chi2,
        ))

    ok_candidates = [candidate for candidate in candidates if candidate.ok and np.isfinite(candidate.chi2)]
    finite_candidates = [candidate for candidate in candidates if np.isfinite(candidate.chi2)]
    if ok_candidates:
        best = min(ok_candidates, key=lambda candidate: candidate.chi2)
    elif finite_candidates:
        best = min(finite_candidates, key=lambda candidate: candidate.chi2)
    else:
        best = None

    target_chi2 = float(fit["chi2_min"]) + 1.0
    best_chi2 = best.chi2 if best is not None else math.nan
    improvement = accepted_chi2 - best_chi2 if best is not None else math.nan
    return {
        "label": fit["label"],
        "algorithm": fit["algorithm"],
        "target": target,
        "target_kind": target_kind,
        "side": side,
        "endpoint": endpoint,
        "target_value": target_value,
        "target_std": target_std,
        "chi2_min": float(fit["chi2_min"]),
        "target_chi2": target_chi2,
        "accepted_method": accepted_method,
        "accepted_profile_chi2": accepted_chi2,
        "accepted_residual": accepted_residual,
        "accepted_projected_grad_norm": _finite_float(source_row.get(f"{side}_projected_grad_norm")),
        "accepted_objective_grad_norm": _finite_float(source_row.get(f"{side}_objective_grad_norm")),
        "best_multistart_chi2": best_chi2,
        "best_multistart_residual": best_chi2 - target_chi2 if best is not None else math.nan,
        "best_start_name": best.start_name if best is not None else "",
        "best_method": best.method if best is not None else "",
        "improvement_vs_accepted": improvement,
        "material_improvement": bool(best is not None and improvement > args.improvement_tol),
        "n_starts": len(candidates),
        "n_ok": sum(candidate.ok for candidate in candidates),
        "n_failed": sum(not candidate.ok for candidate in candidates),
        "fit_runtime_sec": fit_runtime,
        "endpoint_runtime_sec": time.perf_counter() - t0,
        "status": "ok",
        "exception_type": "",
        "exception": "",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-csv", default="notes/asym_fit_sweep_results_simplified.csv")
    parser.add_argument("--summary-csv", default="notes/codex-refactor/round03_descent_multistart_summary.csv")
    parser.add_argument("--summary-jsonl", default="notes/codex-refactor/round03_descent_multistart_summary.jsonl")
    parser.add_argument("--start-jsonl", default="notes/codex-refactor/round03_descent_multistart_starts.jsonl")
    parser.add_argument("--stdout-log", default="notes/logs/round03_descent_multistart_stdout.log")
    parser.add_argument("--endpoint", action="append", required=True)
    parser.add_argument("--seed", type=int, default=20260624)
    parser.add_argument("--cov-dirs", type=int, default=2)
    parser.add_argument("--cov-scale", type=float, default=0.5)
    parser.add_argument("--random-starts", type=int, default=3)
    parser.add_argument("--random-scale", type=float, default=0.5)
    parser.add_argument("--endpoint-random-starts", type=int, default=3)
    parser.add_argument("--endpoint-random-scale", type=float, default=0.25)
    parser.add_argument("--improvement-tol", type=float, default=1e-3)
    parser.add_argument("--target-timeout-sec", type=int, default=240)
    parser.add_argument("--optimizer", choices=["minuit", "scipy"], default="minuit")
    parser.add_argument("--fit-space", choices=["unconstrained", "constrained"], default="unconstrained")
    args = parser.parse_args()

    source_rows = read_source_rows(Path(args.source_csv))
    endpoint_specs = [parse_endpoint_spec(spec) for spec in args.endpoint]
    labels = sorted({label for label, _, _ in endpoint_specs})
    summary_csv = Path(args.summary_csv)
    summary_jsonl = Path(args.summary_jsonl)
    start_jsonl = Path(args.start_jsonl)
    stdout_log = Path(args.stdout_log)
    stdout_log.parent.mkdir(parents=True, exist_ok=True)

    old_handler = signal.signal(signal.SIGALRM, _alarm_handler)
    try:
        for label in labels:
            fit = None
            fit_runtime = math.nan
            with stdout_log.open("a") as log:
                log.write(f"\n===== fit {label} =====\n")
                log.flush()
                t_fit = time.perf_counter()
                with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                    fit = run_fit(label, verbose=False, optimizer=args.optimizer, fit_space=args.fit_space)
                fit_runtime = time.perf_counter() - t_fit
            if fit is None:
                raise RuntimeError(f"run_fit returned None for {label}")

            for spec_label, target, side in endpoint_specs:
                if spec_label != label:
                    continue
                source_row = source_rows.get((label, target))
                if source_row is None:
                    raise KeyError(f"No ok source row for {label} / {target} in {args.source_csv}")

                try:
                    signal.alarm(args.target_timeout_sec)
                    row = run_endpoint(
                        fit=fit,
                        fit_runtime=fit_runtime,
                        source_row=source_row,
                        side=side,
                        args=args,
                        start_jsonl=start_jsonl,
                    )
                    signal.alarm(0)
                except Exception as exc:  # noqa: BLE001 - diagnostic rows should survive failures
                    signal.alarm(0)
                    row = {
                        "label": label,
                        "algorithm": fit.get("algorithm"),
                        "target": target,
                        "side": side,
                        "status": "timeout" if isinstance(exc, EndpointTimeout) else "error",
                        "exception_type": type(exc).__name__,
                        "exception": str(exc),
                        "fit_runtime_sec": fit_runtime,
                    }
                append_csv(summary_csv, row, SUMMARY_FIELDNAMES)
                append_jsonl(summary_jsonl, row)
                print(
                    f"{label} / {target} / {side}: {row.get('status')} "
                    f"improvement={row.get('improvement_vs_accepted')} "
                    f"best={row.get('best_multistart_chi2')}"
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
