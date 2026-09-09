#!/usr/bin/env python
"""Profile-curve monotonicity scan for selected asymmetric-error targets.

This is notes-only diagnostic tooling. It samples fixed-target profile chi2
values between the MLE target and previously verified endpoints, plus one
modest beyond-endpoint point when reachable. The goal is to catch obvious
nonmonotone or disconnected-profile behavior that endpoint bisection alone
would not expose.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import math
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

from jax import numpy as jnp
import numpy as np
import pandas as pd

from pdgfits.asym_errors import build_constrained_profile_chi2
from pdgfits.fit import run_fit


DEFAULT_TARGETS = [
    "B0::S042B95",
    "B0S-BR::S086.37",
    "eta_c J/psi psi(2S)::M026W",
]

SUMMARY_FIELDS = [
    "label",
    "target",
    "side",
    "status",
    "n_points_ok",
    "n_points_error",
    "min_adjacent_step",
    "min_delta_chi2",
    "max_delta_chi2",
    "endpoint_delta_chi2",
    "endpoint_residual",
    "beyond_delta_chi2",
    "nonmonotone_steps",
    "profile_methods",
    "runtime_sec",
    "exception",
]

POINT_FIELDS = [
    "label",
    "target",
    "side",
    "fraction",
    "value",
    "chi2",
    "delta_chi2",
    "status",
    "method",
    "scaled_constraint_violation",
    "projected_grad_norm",
    "descent_improvement",
    "runtime_sec",
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


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: clean(row.get(field)) for field in fields})


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(clean(row), sort_keys=True) + "\n")


def target_function(fit: dict, target: str):
    if target in fit["nodes"]:
        return fit["node_funcs"][fit["nodes"].index(target)]
    if target in fit["parameters"]:
        return fit["parameter_funcs"][fit["parameters"].index(target)]
    raise KeyError(f"{target!r} is not a node or parameter in {fit['label']}")


def target_std_from_cov(fit: dict, target_func, target_value: float) -> float:
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


def parse_target(spec: str) -> tuple[str, str]:
    parts = spec.split("::")
    if len(parts) != 2:
        raise ValueError(f"target spec must be LABEL::TARGET, got {spec!r}")
    return parts[0], parts[1]


def load_sweep_rows(path: Path) -> dict[tuple[str, str], dict]:
    df = pd.read_csv(path)
    rows = {}
    for _, row in df.iterrows():
        if row.get("status") != "ok":
            continue
        rows[(str(row["label"]), str(row["target"]))] = row.to_dict()
    return rows


def point_row(label, target, side, fraction, value, chi2_min, profile, exc=None):
    point = getattr(profile, "last_point", None)
    if exc is not None:
        return {
            "label": label,
            "target": target,
            "side": side,
            "fraction": fraction,
            "value": value,
            "status": "error",
            "exception": f"{type(exc).__name__}: {exc}",
        }
    return {
        "label": label,
        "target": target,
        "side": side,
        "fraction": fraction,
        "value": value,
        "chi2": float(point.chi2),
        "delta_chi2": float(point.chi2 - chi2_min),
        "status": "ok" if point.success else "error",
        "method": point.method,
        "scaled_constraint_violation": point.scaled_constraint_violation,
        "projected_grad_norm": point.projected_grad_norm,
        "descent_improvement": point.descent_improvement,
        "runtime_sec": point.runtime_sec,
        "exception": "",
    }


def summarize_side(label, target, side, rows, chi2_min):
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    deltas = [float(row["delta_chi2"]) for row in ok_rows]
    ordered = [row for row in ok_rows if float(row["fraction"]) <= 1.0]
    ordered.sort(key=lambda row: float(row["fraction"]))
    steps = [
        float(ordered[i + 1]["delta_chi2"]) - float(ordered[i]["delta_chi2"])
        for i in range(len(ordered) - 1)
    ]
    endpoint = min(ok_rows, key=lambda row: abs(float(row["fraction"]) - 1.0)) if ok_rows else None
    beyond = min(ok_rows, key=lambda row: abs(float(row["fraction"]) - 1.2)) if ok_rows else None
    methods = {}
    for row in ok_rows:
        method = str(row.get("method") or "")
        if method:
            methods[method] = methods.get(method, 0) + 1
    exception = "; ".join(row.get("exception", "") for row in rows if row.get("status") != "ok")
    return {
        "label": label,
        "target": target,
        "side": side,
        "status": "ok" if ok_rows and not any(step < -1e-3 for step in steps) else "review",
        "n_points_ok": len(ok_rows),
        "n_points_error": len(rows) - len(ok_rows),
        "min_adjacent_step": min(steps) if steps else None,
        "min_delta_chi2": min(deltas) if deltas else None,
        "max_delta_chi2": max(deltas) if deltas else None,
        "endpoint_delta_chi2": float(endpoint["delta_chi2"]) if endpoint else None,
        "endpoint_residual": float(endpoint["delta_chi2"]) - 1.0 if endpoint else None,
        "beyond_delta_chi2": float(beyond["delta_chi2"]) if beyond else None,
        "nonmonotone_steps": sum(step < -1e-3 for step in steps),
        "profile_methods": "|".join(f"{method}:{count}" for method, count in sorted(methods.items())),
        "runtime_sec": sum(float(row.get("runtime_sec") or 0.0) for row in ok_rows),
        "exception": exception,
    }


def scan_target(label: str, target: str, sweep_row: dict, fractions: list[float]):
    fit = run_fit(label, verbose=False)
    if fit is None:
        raise RuntimeError(f"fit {label!r} was skipped")
    target_func = target_function(fit, target)
    target_value = float(target_func(fit["param_values"]))
    target_std = target_std_from_cov(fit, target_func, target_value)
    chi2_min = float(fit["chi2_min"])
    profile = build_constrained_profile_chi2(
        fit["chi2"],
        fit["fitted_params_to_params"],
        target_func,
        fit["fitted_values"],
        target_scale=target_std,
        target_name=target,
    )
    endpoints = {
        "lower": float(sweep_row["lower_endpoint"]),
        "upper": float(sweep_row["upper_endpoint"]),
    }
    all_point_rows = []
    summary_rows = []
    for side, endpoint in endpoints.items():
        side_rows = []
        displacement = endpoint - target_value
        for fraction in fractions:
            value = target_value + fraction * displacement
            try:
                profile(value)
                row = point_row(label, target, side, fraction, value, chi2_min, profile)
            except Exception as exc:  # noqa: BLE001 - diagnostic row records failure
                row = point_row(label, target, side, fraction, value, chi2_min, profile, exc=exc)
            side_rows.append(row)
            all_point_rows.append(row)
        summary_rows.append(summarize_side(label, target, side, side_rows, chi2_min))
    return summary_rows, all_point_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep-csv", default="notes/asym_fit_sweep_results_simplified.csv")
    parser.add_argument("--target", action="append", default=None, help="LABEL::TARGET")
    parser.add_argument("--fractions", default="0.0,0.25,0.5,0.75,1.0,1.2")
    parser.add_argument("--summary-csv", default="notes/codex-refactor/round04_profile_curve_summary.csv")
    parser.add_argument("--summary-jsonl", default="notes/codex-refactor/round04_profile_curve_summary.jsonl")
    parser.add_argument("--points-csv", default="notes/codex-refactor/round04_profile_curve_points.csv")
    parser.add_argument("--points-jsonl", default="notes/codex-refactor/round04_profile_curve_points.jsonl")
    parser.add_argument("--stdout-log", default="notes/logs/round04_profile_curve_scan_stdout.log")
    args = parser.parse_args()

    fractions = [float(x) for x in args.fractions.split(",") if x.strip()]
    sweep_rows = load_sweep_rows(Path(args.sweep_csv))
    target_specs = args.target or DEFAULT_TARGETS

    summary_rows = []
    point_rows = []
    stdout_log = Path(args.stdout_log)
    stdout_log.parent.mkdir(parents=True, exist_ok=True)
    with stdout_log.open("w") as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        for spec in target_specs:
            label, target = parse_target(spec)
            key = (label, target)
            if key not in sweep_rows:
                raise KeyError(f"{spec!r} not found in {args.sweep_csv}")
            t0 = time.perf_counter()
            print(f"scanning {label} / {target}")
            target_summary, target_points = scan_target(label, target, sweep_rows[key], fractions)
            elapsed = time.perf_counter() - t0
            print(f"finished {label} / {target} in {elapsed:.2f}s")
            summary_rows.extend(target_summary)
            point_rows.extend(target_points)

    write_csv(Path(args.summary_csv), summary_rows, SUMMARY_FIELDS)
    write_jsonl(Path(args.summary_jsonl), summary_rows)
    write_csv(Path(args.points_csv), point_rows, POINT_FIELDS)
    write_jsonl(Path(args.points_jsonl), point_rows)
    print(f"wrote {len(summary_rows)} summary rows and {len(point_rows)} point rows")


if __name__ == "__main__":
    main()
