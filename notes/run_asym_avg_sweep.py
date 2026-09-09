#!/usr/bin/env python
"""Full snapshot average sweep for asymmetric-error endpoint validation.

This script is intentionally kept under notes/ so it can serve as durable
validation evidence without becoming package API.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import gc
import io
import json
import math
import os
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

from jax import numpy as jnp
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from pdgfits.avg import run_avg
from pdgfits.query import avg_queries


FIELDNAMES = [
    "node",
    "status",
    "raw_n_meas",
    "post_n_meas",
    "n_params",
    "n_nuisance",
    "has_adjust",
    "has_dep",
    "input_corr_rows",
    "input_corr_nonzero",
    "has_input_corr",
    "value",
    "chi2_min",
    "error_p",
    "error_n",
    "upper_endpoint",
    "lower_endpoint",
    "upper_residual",
    "lower_residual",
    "fresh_upper_chi2",
    "fresh_lower_chi2",
    "fresh_upper_residual",
    "fresh_lower_residual",
    "fresh_upper_status",
    "fresh_lower_status",
    "fresh_upper_grad_norm",
    "fresh_lower_grad_norm",
    "profile_calls",
    "profile_successes",
    "profile_failures",
    "profile_methods",
    "profile_message_summary",
    "runtime_sec",
    "exception_type",
    "exception",
]


def clean(obj):
    if is_dataclass(obj):
        return clean(asdict(obj))
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


def read_completed(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done = set()
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            node = row.get("node")
            if node:
                done.add(node)
    return done


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(clean(row), sort_keys=True) + "\n")


def append_csv(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow({key: clean(row.get(key)) for key in FIELDNAMES})


def summarize_profile(points) -> tuple[int, int, int, str, str]:
    points = list(points or [])
    methods = sorted({getattr(p, "method", "") for p in points if getattr(p, "method", "")})
    messages = {}
    for point in points:
        msg = str(getattr(point, "message", ""))
        if len(msg) > 120:
            msg = msg[:117] + "..."
        messages[msg] = messages.get(msg, 0) + 1
    summary = "; ".join(f"{count}x {msg}" for msg, count in sorted(messages.items())[:6])
    return (
        len(points),
        sum(bool(getattr(p, "success", False)) for p in points),
        sum(not bool(getattr(p, "success", False)) for p in points),
        "|".join(methods),
        summary,
    )


def fresh_fixed_coordinate_profile(result: dict, fixed_value: float, *, max_nuisance: int) -> dict:
    parameters = list(result["parameters"])
    primary = result["node"]
    primary_idx = parameters.index(primary)
    n_nuisance = len(parameters) - 1
    chi2 = result["chi2"]
    chi2_grad = result["chi2_grad"]
    param_values = np.asarray(result["param_values"], dtype=np.float64)

    if n_nuisance == 0:
        x = np.array([fixed_value], dtype=np.float64)
        return {
            "status": "ok",
            "chi2": float(chi2(jnp.asarray(x, dtype=jnp.float64))),
            "grad_norm": 0.0,
            "message": "direct",
        }

    if n_nuisance > max_nuisance:
        return {
            "status": "skipped",
            "chi2": None,
            "grad_norm": None,
            "message": f"n_nuisance={n_nuisance} exceeds max_nuisance={max_nuisance}",
        }

    base_nuisance = np.delete(param_values, primary_idx)

    def full_params(x_nuisance):
        return np.insert(np.asarray(x_nuisance, dtype=np.float64), primary_idx, fixed_value)

    def obj(x_nuisance):
        return float(chi2(jnp.asarray(full_params(x_nuisance), dtype=jnp.float64)))

    def grad(x_nuisance):
        return np.delete(
            np.asarray(chi2_grad(jnp.asarray(full_params(x_nuisance), dtype=jnp.float64)), dtype=np.float64),
            primary_idx,
        )

    candidates = []
    messages = []
    start_fun = obj(base_nuisance)
    if np.isfinite(start_fun):
        candidates.append((start_fun, base_nuisance, True, "base nuisance"))

    try:
        bfgs = minimize(
            obj,
            base_nuisance,
            jac=grad,
            method="BFGS",
            options={"maxiter": max(1000, 500 * n_nuisance), "gtol": 1e-8},
        )
        messages.append(f"BFGS success={bfgs.success}: {bfgs.message}")
        if np.isfinite(bfgs.fun):
            candidates.append((float(bfgs.fun), np.asarray(bfgs.x, dtype=np.float64), bool(bfgs.success), str(bfgs.message)))
    except Exception as exc:  # noqa: BLE001 - diagnostic script
        messages.append(f"BFGS exception={type(exc).__name__}: {exc}")

    nm_start = min(candidates, key=lambda c: c[0])[1] if candidates else base_nuisance
    try:
        nm = minimize(
            obj,
            nm_start,
            method="Nelder-Mead",
            options={
                "maxiter": max(1000, 500 * n_nuisance),
                "xatol": max(np.linalg.norm(nm_start) * 1e-10, np.finfo(float).eps),
                "fatol": 1e-8,
                "adaptive": True,
            },
        )
        messages.append(f"Nelder-Mead success={nm.success}: {nm.message}")
        if np.isfinite(nm.fun):
            candidates.append((float(nm.fun), np.asarray(nm.x, dtype=np.float64), bool(nm.success), str(nm.message)))
    except Exception as exc:  # noqa: BLE001 - diagnostic script
        messages.append(f"Nelder-Mead exception={type(exc).__name__}: {exc}")

    if not candidates:
        return {"status": "error", "chi2": None, "grad_norm": None, "message": "; ".join(messages)}

    best_fun = min(c[0] for c in candidates)
    fun_tol = max(1e-6, 1e-8 * abs(best_fun))
    successful_ties = [c for c in candidates if c[2] and c[0] <= best_fun + fun_tol]
    if successful_ties:
        best_fun, best_x, success, message = min(successful_ties, key=lambda c: c[0])
    else:
        best_fun, best_x, success, message = min(candidates, key=lambda c: c[0])
    grad_norm = float(np.linalg.norm(grad(best_x)))
    ok = bool(success or grad_norm <= 1e-5 * max(abs(best_fun), 1.0))
    return {
        "status": "ok" if ok else "nonstationary",
        "chi2": float(best_fun),
        "grad_norm": grad_norm,
        "message": f"{message}; {'; '.join(messages)}",
    }


def build_row_for_result(node: str, result: dict, raw_n: int, corr_df_node: pd.DataFrame, runtime: float, max_nuisance: int) -> dict:
    params = list(result["parameters"])
    diag = dict(result.get("asym_error_diagnostics") or {})
    profile_calls, profile_successes, profile_failures, methods, msg_summary = summarize_profile(
        result.get("profile_diagnostics")
    )
    value = float(result["param_values"][params.index(node)])
    target = float(diag.get("target", result["chi2_min"] + 1.0))
    upper_endpoint = float(diag.get("upper_endpoint", value + result["error_p"]))
    lower_endpoint = float(diag.get("lower_endpoint", value - result["error_n"]))
    fresh_upper = fresh_fixed_coordinate_profile(result, upper_endpoint, max_nuisance=max_nuisance)
    fresh_lower = fresh_fixed_coordinate_profile(result, lower_endpoint, max_nuisance=max_nuisance)
    fresh_upper_chi2 = fresh_upper.get("chi2")
    fresh_lower_chi2 = fresh_lower.get("chi2")
    meas_df = result["meas_df"]
    return {
        "node": node,
        "status": "ok",
        "raw_n_meas": raw_n,
        "post_n_meas": int(len(meas_df)),
        "n_params": int(len(params)),
        "n_nuisance": int(len(params) - 1),
        "has_adjust": bool(any("br_adjust" in str(x) for x in meas_df.get("measurement", []))),
        "has_dep": bool(any("dep_meas" in str(x) for x in meas_df.get("measurement", []))),
        "input_corr_rows": int(len(corr_df_node)),
        "input_corr_nonzero": int((corr_df_node.get("correlation", pd.Series(dtype=float)).fillna(0) != 0).sum()),
        "has_input_corr": bool(len(corr_df_node) > 0),
        "value": value,
        "chi2_min": float(result["chi2_min"]),
        "error_p": float(result["error_p"]),
        "error_n": float(result["error_n"]),
        "upper_endpoint": upper_endpoint,
        "lower_endpoint": lower_endpoint,
        "upper_residual": float(diag.get("upper_residual", math.nan)),
        "lower_residual": float(diag.get("lower_residual", math.nan)),
        "fresh_upper_chi2": fresh_upper_chi2,
        "fresh_lower_chi2": fresh_lower_chi2,
        "fresh_upper_residual": None if fresh_upper_chi2 is None else float(fresh_upper_chi2 - target),
        "fresh_lower_residual": None if fresh_lower_chi2 is None else float(fresh_lower_chi2 - target),
        "fresh_upper_status": fresh_upper.get("status"),
        "fresh_lower_status": fresh_lower.get("status"),
        "fresh_upper_grad_norm": fresh_upper.get("grad_norm"),
        "fresh_lower_grad_norm": fresh_lower.get("grad_norm"),
        "profile_calls": profile_calls,
        "profile_successes": profile_successes,
        "profile_failures": profile_failures,
        "profile_methods": methods,
        "profile_message_summary": msg_summary,
        "runtime_sec": runtime,
        "exception_type": "",
        "exception": "",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", default="notes/asym_avg_sweep_results.jsonl")
    parser.add_argument("--csv", default="notes/asym_avg_sweep_results.csv")
    parser.add_argument("--stdout-log", default="notes/logs/asym_avg_sweep_stdout.log")
    parser.add_argument("--node", action="append", default=None)
    parser.add_argument("--start-from", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--fresh-profile-max-nuisance", type=int, default=20)
    args = parser.parse_args()

    jsonl_path = Path(args.jsonl)
    csv_path = Path(args.csv)
    stdout_log = Path(args.stdout_log)
    stdout_log.parent.mkdir(parents=True, exist_ok=True)
    completed = read_completed(jsonl_path) if args.resume else set()

    avg_df, corr_df_dict = avg_queries(verbose=False)
    nodes = list(avg_df["node"].unique())
    if args.node:
        requested = set(args.node)
        nodes = [node for node in nodes if node in requested]
    if args.start_from:
        nodes = nodes[nodes.index(args.start_from):]
    if args.limit is not None:
        nodes = nodes[: args.limit]

    print(f"average nodes available={len(avg_df['node'].unique())}, selected={len(nodes)}, completed={len(completed)}")

    for idx, node in enumerate(nodes, start=1):
        if node in completed:
            continue
        meas_df_node = avg_df[avg_df["node"] == node]
        corr_df_node = corr_df_dict[node]
        raw_n = int(len(meas_df_node))
        t0 = time.perf_counter()
        row = None
        with stdout_log.open("a") as log:
            log.write(f"\n===== {idx}/{len(nodes)} {node} =====\n")
            log.flush()
            try:
                with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                    result = run_avg(node, meas_df_node, corr_df_node)
                runtime = time.perf_counter() - t0
                if result is None:
                    row = {
                        "node": node,
                        "status": "skipped",
                        "raw_n_meas": raw_n,
                        "runtime_sec": runtime,
                        "exception_type": "",
                        "exception": "run_avg returned None",
                    }
                else:
                    row = build_row_for_result(
                        node,
                        result,
                        raw_n,
                        corr_df_node,
                        runtime,
                        max_nuisance=args.fresh_profile_max_nuisance,
                    )
            except Exception as exc:  # noqa: BLE001 - sweep must continue
                runtime = time.perf_counter() - t0
                row = {
                    "node": node,
                    "status": "error",
                    "raw_n_meas": raw_n,
                    "input_corr_rows": int(len(corr_df_node)),
                    "input_corr_nonzero": int((corr_df_node.get("correlation", pd.Series(dtype=float)).fillna(0) != 0).sum()),
                    "has_input_corr": bool(len(corr_df_node) > 0),
                    "has_adjust": bool(meas_df_node["measurement"].astype(str).str.contains("br_adjust", regex=False).any()),
                    "has_dep": bool(meas_df_node["measurement"].astype(str).str.contains("dep_meas", regex=False).any()),
                    "runtime_sec": runtime,
                    "exception_type": type(exc).__name__,
                    "exception": str(exc),
                }
                log.write(f"EXCEPTION {type(exc).__name__}: {exc}\n")

        append_jsonl(jsonl_path, row)
        append_csv(csv_path, row)
        print(
            f"[{idx}/{len(nodes)}] {node} {row['status']} "
            f"runtime={row.get('runtime_sec', 0):.2f}s "
            f"fresh_resid=({row.get('fresh_upper_residual')},{row.get('fresh_lower_residual')})"
        )
        if idx % 25 == 0:
            gc.collect()
            try:
                jax.clear_caches()
            except Exception:
                pass


if __name__ == "__main__":
    main()
