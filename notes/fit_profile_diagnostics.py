#!/usr/bin/env python
"""Diagnostics for fit-side constrained profile minimization.

Run with:

    env PYTHONPATH=/root/pdgfits-private/src \
      PDGFITS_DATA_BACKEND=snapshot \
      PDGFITS_SNAPSHOT_DIR=/root/pdgfits-private/data/pdg-snapshot \
      XLA_PYTHON_CLIENT_PREALLOCATE=false \
      /root/micromamba/micromamba run -r /root/micromamba-root -n pdg \
      python notes/fit_profile_diagnostics.py

The script intentionally lives under notes/ so it can be kept as durable
diagnostic evidence without changing package behavior.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

from jax import numpy as jnp
import numpy as np
from scipy.linalg import null_space
from scipy.optimize import NonlinearConstraint, OptimizeResult, minimize, root

from pdgfits.fit import run_fit


@dataclass
class Attempt:
    name: str
    start_name: str
    metrics: dict


class PathFailure(RuntimeError):
    def __init__(self, message, *, call, v, last_x_before, starts, candidates):
        super().__init__(message)
        self.call = call
        self.v = v
        self.last_x_before = last_x_before
        self.starts = starts
        self.candidates = candidates


def _as_float(x):
    return float(np.asarray(x, dtype=np.float64))


def _clean(obj):
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean(v) for v in obj]
    if isinstance(obj, tuple):
        return [_clean(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [_clean(v) for v in obj.tolist()]
    if isinstance(obj, np.generic):
        return _clean(obj.item())
    if isinstance(obj, float):
        if math.isfinite(obj):
            return obj
        return str(obj)
    return obj


class DiagnosticProfiler:
    def __init__(self, fit, target_name):
        self.fit = fit
        self.target_name = target_name
        self.nodes = fit["nodes"]
        self.parameters = fit["parameters"]
        self.chi2 = fit["chi2"]
        self.fitted_params_to_params = fit["fitted_params_to_params"]
        self.fp_best = np.asarray(fit["fitted_values"], dtype=np.float64)
        self.param_values = fit["param_values"]
        self.chi2_min = float(fit["chi2_min"])

        if target_name in self.nodes:
            self.target_kind = "node"
            self.target_func = fit["node_funcs"][self.nodes.index(target_name)]
        elif target_name in self.parameters:
            self.target_kind = "parameter"
            self.target_func = fit["parameter_funcs"][self.parameters.index(target_name)]
        else:
            raise KeyError(f"{target_name!r} is not a node or parameter")

        self.target_value = float(self.target_func(self.param_values))
        covariance = fit["covariance"]
        if covariance is not None:
            jac_params = jax.jacobian(self.fitted_params_to_params)(fit["fitted_values"])
            param_cov = jac_params @ covariance @ jac_params.T
            jac_target = jax.jacobian(self.target_func)(self.param_values)
            self.target_std = float(jnp.sqrt(jac_target @ param_cov @ jac_target.T))
        else:
            self.target_std = max(abs(self.target_value), 1.0) * 1e-3
        if not np.isfinite(self.target_std) or self.target_std <= 0:
            self.target_std = max(abs(self.target_value), 1.0) * 1e-3
        self.target_scale = abs(float(self.target_std)) or 1.0

        self.lb = self.target_value - 2.0 * self.target_std
        self.ub = self.target_value + 2.0 * self.target_std
        self.target_chi2 = self.chi2_min + 1.0

        self.chi2_grad_jax = jax.jit(jax.grad(self.chi2))
        self.chi2_hess_jax = jax.jit(jax.hessian(self.chi2))

        @jax.jit
        def constraint_value(fp, fixed_value):
            return self.target_func(self.fitted_params_to_params(fp)) - fixed_value

        @jax.jit
        def scaled_constraint_value(fp, fixed_value):
            return constraint_value(fp, fixed_value) / self.target_scale

        self.constraint_value_jax = constraint_value
        self.scaled_constraint_value_jax = scaled_constraint_value
        self.constraint_grad_jax = jax.jit(jax.grad(lambda fp, fixed_value: constraint_value(fp, fixed_value)))
        self.scaled_constraint_grad_jax = jax.jit(
            jax.grad(lambda fp, fixed_value: scaled_constraint_value(fp, fixed_value))
        )
        self.scaled_constraint_hess_jax = jax.jit(
            jax.hessian(lambda fp, fixed_value: scaled_constraint_value(fp, fixed_value))
        )

    def chi2_np(self, x):
        val = _as_float(self.chi2(jnp.asarray(x, dtype=jnp.float64)))
        return val if np.isfinite(val) else np.inf

    def chi2_grad_np(self, x):
        return np.asarray(self.chi2_grad_jax(jnp.asarray(x, dtype=jnp.float64)), dtype=np.float64)

    def chi2_hess_np(self, x):
        return np.asarray(self.chi2_hess_jax(jnp.asarray(x, dtype=jnp.float64)), dtype=np.float64)

    def raw_constraint(self, x, v):
        return _as_float(self.constraint_value_jax(jnp.asarray(x, dtype=jnp.float64), float(v)))

    def scaled_constraint(self, x, v):
        return _as_float(self.scaled_constraint_value_jax(jnp.asarray(x, dtype=jnp.float64), float(v)))

    def scaled_constraint_grad(self, x, v):
        return np.asarray(
            self.scaled_constraint_grad_jax(jnp.asarray(x, dtype=jnp.float64), float(v)),
            dtype=np.float64,
        )

    def project_start(self, start, v, tol=1e-7, max_iter=50):
        x = np.asarray(start, dtype=np.float64).copy()
        for _ in range(max_iter):
            c = self.scaled_constraint(x, v)
            if abs(c) <= tol:
                return x
            g = self.scaled_constraint_grad(x, v)
            denom = float(np.dot(g, g))
            if not np.isfinite(denom) or denom == 0.0:
                break
            x = x - (c / denom) * g
        return x

    def constraint(self, v, *, exact_hess=False):
        kwargs = {}
        if exact_hess:
            kwargs["hess"] = lambda x, multiplier: np.asarray(
                multiplier[0]
                * self.scaled_constraint_hess_jax(jnp.asarray(x, dtype=jnp.float64), float(v)),
                dtype=np.float64,
            )
        return NonlinearConstraint(
            fun=lambda x: self.scaled_constraint(x, v),
            lb=0.0,
            ub=0.0,
            jac=lambda x: self.scaled_constraint_grad(x, v),
            **kwargs,
        )

    def metrics(self, x, v, result=None, elapsed=None):
        x = np.asarray(x, dtype=np.float64)
        fun = self.chi2_np(x)
        grad = self.chi2_grad_np(x)
        cgrad = self.scaled_constraint_grad(x, v)
        denom = float(np.dot(cgrad, cgrad))
        if np.isfinite(denom) and denom > 0:
            lam = -float(np.dot(grad, cgrad)) / denom
            kkt_vec = grad + lam * cgrad
            projected_grad = grad - (float(np.dot(grad, cgrad)) / denom) * cgrad
            kkt_resid = float(np.linalg.norm(kkt_vec))
            projected_grad_norm = float(np.linalg.norm(projected_grad))
        else:
            lam = np.nan
            kkt_resid = np.inf
            projected_grad_norm = np.inf
        raw_c = abs(self.raw_constraint(x, v))
        scaled_c = abs(self.scaled_constraint(x, v))
        out = {
            "fun": fun,
            "delta_chi2": fun - self.chi2_min,
            "endpoint_residual": fun - self.target_chi2,
            "raw_constraint_violation": raw_c,
            "scaled_constraint_violation": scaled_c,
            "objective_grad_norm": float(np.linalg.norm(grad)),
            "constraint_grad_norm": float(np.linalg.norm(cgrad)),
            "projected_grad_norm": projected_grad_norm,
            "kkt_stationarity_norm": kkt_resid,
            "kkt_lambda_scaled_constraint": lam,
            "current_acceptance": self.current_accepts(x, v, result),
            "x_norm": float(np.linalg.norm(x)),
        }
        if elapsed is not None:
            out["elapsed_sec"] = elapsed
        if result is not None:
            out.update(
                {
                    "success": bool(getattr(result, "success", False)),
                    "status": int(getattr(result, "status", 9999))
                    if getattr(result, "status", None) is not None
                    else None,
                    "message": str(getattr(result, "message", "")),
                    "nit": int(getattr(result, "nit", -1)) if getattr(result, "nit", None) is not None else None,
                    "nfev": int(getattr(result, "nfev", -1)) if getattr(result, "nfev", None) is not None else None,
                    "njev": int(getattr(result, "njev", -1)) if getattr(result, "njev", None) is not None else None,
                    "optimality": float(getattr(result, "optimality", np.nan))
                    if getattr(result, "optimality", None) is not None
                    else None,
                    "constr_violation": float(getattr(result, "constr_violation", np.nan))
                    if getattr(result, "constr_violation", None) is not None
                    else None,
                }
            )
        return out

    def current_accepts(self, x, v, result=None):
        scaled_c = abs(self.scaled_constraint(x, v))
        fun = self.chi2_np(x)
        if not np.isfinite(fun) or scaled_c > 1e-7:
            return False
        if result is not None and bool(getattr(result, "success", False)):
            return True
        grad_norm = float(np.linalg.norm(self.chi2_grad_np(x)))
        pg_norm = self.metrics_without_acceptance(x, v)["projected_grad_norm"]
        return pg_norm <= max(1e-5, 1e-5 * grad_norm)

    def metrics_without_acceptance(self, x, v):
        x = np.asarray(x, dtype=np.float64)
        grad = self.chi2_grad_np(x)
        cgrad = self.scaled_constraint_grad(x, v)
        denom = float(np.dot(cgrad, cgrad))
        if np.isfinite(denom) and denom > 0:
            projected_grad = grad - (float(np.dot(grad, cgrad)) / denom) * cgrad
            projected_grad_norm = float(np.linalg.norm(projected_grad))
        else:
            projected_grad_norm = np.inf
        return {"projected_grad_norm": projected_grad_norm}

    def maybe_keep_feasible_start(self, result, x0, v):
        x0_fun = self.chi2_np(x0)
        if not np.isfinite(x0_fun) or abs(self.scaled_constraint(x0, v)) > 1e-7:
            return result
        if (not np.isfinite(result.fun)) or (
            x0_fun <= result.fun + max(1e-8, 1e-10 * abs(result.fun))
        ):
            is_best_fit_start = np.linalg.norm(np.asarray(x0) - self.fp_best) <= 1e-10 * max(
                np.linalg.norm(self.fp_best), 1.0
            )
            return OptimizeResult(
                x=np.asarray(x0, dtype=np.float64),
                fun=x0_fun,
                success=is_best_fit_start,
                message="retained lower-chi2 feasible projected start",
                status=-999,
            )
        return result

    def solve_slsqp(self, x0, v, *, maxiter=2000, ftol=1e-10):
        t0 = time.perf_counter()
        result = minimize(
            self.chi2_np,
            np.asarray(x0, dtype=np.float64),
            method="SLSQP",
            jac=self.chi2_grad_np,
            constraints=[self.constraint(v)],
            options={"ftol": ftol, "maxiter": maxiter, "disp": False},
        )
        elapsed = time.perf_counter() - t0
        return result, self.metrics(result.x, v, result, elapsed)

    def solve_trust_constr(
        self,
        x0,
        v,
        *,
        maxiter=2000,
        gtol=1e-10,
        xtol=1e-10,
        exact_hess=False,
    ):
        t0 = time.perf_counter()
        kwargs = {}
        if exact_hess:
            kwargs["hess"] = self.chi2_hess_np
        result = minimize(
            self.chi2_np,
            np.asarray(x0, dtype=np.float64),
            method="trust-constr",
            jac=self.chi2_grad_np,
            constraints=[self.constraint(v, exact_hess=exact_hess)],
            options={"gtol": gtol, "xtol": xtol, "maxiter": maxiter, "verbose": 0},
            **kwargs,
        )
        elapsed = time.perf_counter() - t0
        return result, self.metrics(result.x, v, result, elapsed)

    def solve_current_from(self, x0, v):
        slsqp, slsqp_metrics = self.solve_slsqp(x0, v, maxiter=2000, ftol=1e-10)
        slsqp = self.maybe_keep_feasible_start(slsqp, x0, v)
        slsqp_metrics = self.metrics(slsqp.x, v, slsqp, slsqp_metrics.get("elapsed_sec"))
        if self.current_accepts(slsqp.x, v, slsqp):
            return slsqp, {"slsqp": slsqp_metrics}

        trust_start = np.asarray(slsqp.x if np.all(np.isfinite(slsqp.x)) else x0, dtype=np.float64)
        trust, trust_metrics = self.solve_trust_constr(
            trust_start, v, maxiter=2000, gtol=1e-10, xtol=1e-10, exact_hess=False
        )
        trust = self.maybe_keep_feasible_start(trust, x0, v)
        trust_metrics = self.metrics(trust.x, v, trust, trust_metrics.get("elapsed_sec"))
        if self.current_accepts(trust.x, v, trust) or (
            np.isfinite(trust.fun) and trust.fun < slsqp.fun
        ):
            return trust, {"slsqp": slsqp_metrics, "trust_constr": trust_metrics}
        return slsqp, {"slsqp": slsqp_metrics, "trust_constr": trust_metrics}

    def trace_current_path(self, residual_tol=5e-3, max_iter=80):
        last_x = self.fp_best.copy()
        calls = []

        def profile(v, phase):
            nonlocal last_x
            starts = [("continuation", self.project_start(last_x, v))]
            best_start = self.project_start(self.fp_best, v)
            if np.linalg.norm(best_start - starts[0][1]) > 1e-12 * max(
                np.linalg.norm(best_start), np.linalg.norm(starts[0][1]), 1.0
            ):
                starts.append(("mle", best_start))
            candidates = []
            for start_name, start in starts:
                result, sub = self.solve_current_from(start, v)
                candidates.append(
                    {
                        "start_name": start_name,
                        "start_metrics": self.metrics(start, v),
                        "result": result,
                        "sub": sub,
                        "metrics": self.metrics(result.x, v, result),
                    }
                )
            ok = [c for c in candidates if self.current_accepts(c["result"].x, v, c["result"])]
            if ok:
                best = min(ok, key=lambda c: c["result"].fun)
            else:
                best = min(
                    candidates,
                    key=lambda c: c["result"].fun if np.isfinite(c["result"].fun) else np.inf,
                )
            call = {
                "phase": phase,
                "v": float(v),
                "start_names": [name for name, _ in starts],
                "best_start": best["start_name"],
                "best_metrics": best["metrics"],
                "candidate_metrics": [
                    {
                        "start_name": c["start_name"],
                        "start_metrics": c["start_metrics"],
                        "sub": c["sub"],
                        "metrics": c["metrics"],
                    }
                    for c in candidates
                ],
            }
            calls.append(call)
            if not self.current_accepts(best["result"].x, v, best["result"]):
                raise PathFailure(
                    f"current path failed at {phase} for {self.target_name}={v}",
                    call=call,
                    v=float(v),
                    last_x_before=last_x.copy(),
                    starts={name: x for name, x in starts},
                    candidates=candidates,
                )
            last_x = np.asarray(best["result"].x, dtype=np.float64)
            return float(best["result"].fun)

        profile(self.target_value, "base")

        def solve_side(lo, hi, upper):
            fixed = lo if upper else hi
            moving = hi if upper else lo
            n_expand = 0
            side = "upper" if upper else "lower"
            while profile(moving, f"{side}_expand_{n_expand}") < self.target_chi2 and n_expand < max_iter:
                n_expand += 1
                moving = self.target_value + 2.0 * (moving - self.target_value)
            lo2, hi2 = (fixed, moving) if upper else (moving, fixed)
            for i in range(max_iter):
                mid = 0.5 * (lo2 + hi2)
                chi2_mid = profile(mid, f"{side}_bisect_{i}")
                if chi2_mid < self.target_chi2:
                    if upper:
                        lo2 = mid
                    else:
                        hi2 = mid
                else:
                    if upper:
                        hi2 = mid
                    else:
                        lo2 = mid
                if abs(chi2_mid - self.target_chi2) <= residual_tol:
                    break
            endpoint = 0.5 * (lo2 + hi2)
            profile(endpoint, f"{side}_verify")

        solve_side(self.target_value, self.ub, True)
        solve_side(self.lb, self.target_value, False)
        return calls, None

    def solve_reduced_nullspace(self, x0, v, *, maxiter=1000):
        x_anchor = self.project_start(x0, v)
        g = self.scaled_constraint_grad(x_anchor, v)
        basis = null_space(g.reshape(1, -1))
        if basis.size == 0:
            result = OptimizeResult(x=x_anchor, fun=self.chi2_np(x_anchor), success=False, message="empty nullspace")
            return result, self.metrics(result.x, v, result)

        def unpack(y):
            return self.project_start(x_anchor + basis @ np.asarray(y, dtype=np.float64), v)

        def obj(y):
            return self.chi2_np(unpack(y))

        y0 = np.zeros(basis.shape[1], dtype=np.float64)
        t0 = time.perf_counter()
        red = minimize(
            obj,
            y0,
            method="Powell",
            options={"maxiter": maxiter, "xtol": 1e-10, "ftol": 1e-10, "disp": False},
        )
        elapsed = time.perf_counter() - t0
        x = unpack(red.x)
        result = OptimizeResult(
            x=x,
            fun=self.chi2_np(x),
            success=bool(red.success),
            status=getattr(red, "status", None),
            message=f"reduced Powell: {red.message}",
            nit=getattr(red, "nit", None),
            nfev=getattr(red, "nfev", None),
        )
        return result, self.metrics(result.x, v, result, elapsed)

    def solve_kkt_root(self, x0, v):
        x0 = self.project_start(x0, v)
        grad = self.chi2_grad_np(x0)
        cgrad = self.scaled_constraint_grad(x0, v)
        denom = float(np.dot(cgrad, cgrad))
        lam0 = -float(np.dot(grad, cgrad)) / denom if denom > 0 else 0.0

        def residual(z):
            x = np.asarray(z[:-1], dtype=np.float64)
            lam = float(z[-1])
            return np.r_[self.chi2_grad_np(x) + lam * self.scaled_constraint_grad(x, v), self.scaled_constraint(x, v)]

        t0 = time.perf_counter()
        sol = root(residual, np.r_[x0, lam0], method="hybr", options={"maxfev": 5000})
        elapsed = time.perf_counter() - t0
        result = OptimizeResult(
            x=np.asarray(sol.x[:-1], dtype=np.float64),
            fun=self.chi2_np(sol.x[:-1]),
            success=bool(sol.success),
            status=getattr(sol, "status", None),
            message=f"kkt root: {sol.message}",
            nfev=getattr(sol, "nfev", None),
        )
        metrics = self.metrics(result.x, v, result, elapsed)
        metrics["root_residual_norm"] = float(np.linalg.norm(residual(sol.x)))
        return result, metrics

    def compare_methods(self, v, starts, *, include_random=False):
        rows: list[Attempt] = []

        def add(name, start_name, func):
            try:
                _, metrics = func()
            except Exception as exc:
                metrics = {"exception": repr(exc)}
            rows.append(Attempt(name, start_name, metrics))

        for start_name, start in starts.items():
            projected = self.project_start(start, v)
            add("start_only_projected", start_name, lambda s=projected: (None, self.metrics(s, v)))
            add(
                "slsqp_2000_ftol1e-10",
                start_name,
                lambda s=projected: self.solve_slsqp(s, v, maxiter=2000, ftol=1e-10),
            )
            add(
                "slsqp_10000_ftol1e-12",
                start_name,
                lambda s=projected: self.solve_slsqp(s, v, maxiter=10000, ftol=1e-12),
            )
            add(
                "slsqp_10000_ftol1e-8",
                start_name,
                lambda s=projected: self.solve_slsqp(s, v, maxiter=10000, ftol=1e-8),
            )
            add(
                "trust_constr_bfgs_5000",
                start_name,
                lambda s=projected: self.solve_trust_constr(
                    s, v, maxiter=5000, gtol=1e-10, xtol=1e-10, exact_hess=False
                ),
            )
            add(
                "trust_constr_exact_hess_2000",
                start_name,
                lambda s=projected: self.solve_trust_constr(
                    s, v, maxiter=2000, gtol=1e-10, xtol=1e-10, exact_hess=True
                ),
            )

        finite_rows = [r for r in rows if "fun" in r.metrics and np.isfinite(r.metrics["fun"])]
        if finite_rows:
            best_row = min(finite_rows, key=lambda r: r.metrics["fun"])
            best_x_start = starts.get(best_row.start_name, self.fp_best)
            add(
                "reduced_nullspace_powell",
                f"{best_row.start_name}_from_best",
                lambda s=best_x_start: self.solve_reduced_nullspace(s, v, maxiter=1000),
            )
            add(
                "kkt_root",
                f"{best_row.start_name}_from_best",
                lambda s=best_x_start: self.solve_kkt_root(s, v),
            )

        if include_random:
            rng = np.random.default_rng(12345)
            base = self.project_start(starts.get("continuation", self.fp_best), v)
            scales = np.maximum(np.abs(base), 1.0) * 1e-4
            for i in range(3):
                perturbed = self.project_start(base + rng.normal(size=base.size) * scales, v)
                add(
                    "slsqp_10000_ftol1e-10",
                    f"random_{i}",
                    lambda s=perturbed: self.solve_slsqp(s, v, maxiter=10000, ftol=1e-10),
                )
        return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default="Lam-b-0")
    parser.add_argument("--target", default="S040.10")
    parser.add_argument("--output", default="notes/fit-side-profile-diagnostics.json")
    parser.add_argument("--include-random", action="store_true")
    args = parser.parse_args()

    fit = run_fit(args.label, verbose=True)
    profiler = DiagnosticProfiler(fit, args.target)

    summary = {
        "label": args.label,
        "target": args.target,
        "target_kind": profiler.target_kind,
        "chi2_min": profiler.chi2_min,
        "target_chi2": profiler.target_chi2,
        "target_value": profiler.target_value,
        "target_std": profiler.target_std,
        "initial_lb": profiler.lb,
        "initial_ub": profiler.ub,
        "n_fitted_params": len(profiler.fp_best),
    }
    print(json.dumps(_clean(summary), indent=2))

    path = []
    failure = None
    try:
        path, _ = profiler.trace_current_path()
    except PathFailure as exc:
        failure = exc
        path = []
        print(f"TRACE_FAILURE {exc}")

    if failure is not None:
        compare_starts = {
            "continuation": failure.starts["continuation"],
            "mle": profiler.project_start(profiler.fp_best, failure.v),
        }
        for idx, candidate in enumerate(failure.candidates):
            compare_starts[f"failed_candidate_{idx}_{candidate['start_name']}"] = np.asarray(
                candidate["result"].x, dtype=np.float64
            )
        method_rows = profiler.compare_methods(
            failure.v, compare_starts, include_random=args.include_random
        )
        failure_payload = {
            "phase": failure.call["phase"],
            "value": failure.v,
            "last_x_before_metrics": profiler.metrics(failure.last_x_before, failure.v),
            "call": failure.call,
            "method_comparison": [
                {"method": row.name, "start": row.start_name, **row.metrics}
                for row in method_rows
            ],
        }
    else:
        failure_payload = None

    payload = {
        "summary": summary,
        "current_path": path,
        "failure": failure_payload,
    }
    output = Path(args.output)
    output.write_text(json.dumps(_clean(payload), indent=2) + "\n")
    print(f"WROTE {output}")

    if failure_payload is not None:
        rows = failure_payload["method_comparison"]
        print("method,start,success,current_acceptance,fun,endpoint_residual,scaled_c,pg_norm,nit,nfev,message")
        for row in rows:
            print(
                ",".join(
                    [
                        str(row.get("method")),
                        str(row.get("start")),
                        str(row.get("success")),
                        str(row.get("current_acceptance")),
                        f"{row.get('fun', np.nan):.16g}" if isinstance(row.get("fun"), float) else str(row.get("fun")),
                        f"{row.get('endpoint_residual', np.nan):.6g}"
                        if isinstance(row.get("endpoint_residual"), float)
                        else str(row.get("endpoint_residual")),
                        f"{row.get('scaled_constraint_violation', np.nan):.6g}"
                        if isinstance(row.get("scaled_constraint_violation"), float)
                        else str(row.get("scaled_constraint_violation")),
                        f"{row.get('projected_grad_norm', np.nan):.6g}"
                        if isinstance(row.get("projected_grad_norm"), float)
                        else str(row.get("projected_grad_norm")),
                        str(row.get("nit")),
                        str(row.get("nfev")),
                        repr(row.get("message", row.get("exception", ""))),
                    ]
                )
            )


if __name__ == "__main__":
    main()
