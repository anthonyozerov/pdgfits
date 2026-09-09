#!/usr/bin/env python
"""Staged fit-side asymmetric-error sweep with profile diagnostics.

The script writes one row per fit target. It is resumable and intentionally
kept under notes/ as validation tooling rather than package API.
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
from collections import Counter
from dataclasses import asdict, is_dataclass
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

from jax import numpy as jnp
import numpy as np

from pdgfits.asym_errors import build_constrained_profile_chi2, binary_search_error
from pdgfits.fit import run_fit
from pdgfits.query import all_fits


KNOWN_HARD_FITS = {"G(2000),G(1800)", "Upsilon(2S)", "Lam-b-0"}

FIELDNAMES = [
    "stage",
    "label",
    "algorithm",
    "status",
    "target",
    "target_kind",
    "target_value",
    "target_std",
    "chi2_min",
    "error_p",
    "error_n",
    "upper_endpoint",
    "lower_endpoint",
    "upper_residual",
    "lower_residual",
    "upper_chi2",
    "lower_chi2",
    "upper_constraint_violation",
    "lower_constraint_violation",
    "upper_scaled_constraint_violation",
    "lower_scaled_constraint_violation",
    "upper_projected_grad_norm",
    "lower_projected_grad_norm",
    "upper_objective_grad_norm",
    "lower_objective_grad_norm",
    "upper_method",
    "lower_method",
    "upper_profile_runtime_sec",
    "lower_profile_runtime_sec",
    "profile_calls",
    "profile_methods",
    "kkt_polish_calls",
    "n_params",
    "n_nodes",
    "n_nuisance",
    "n_meas",
    "fit_valid",
    "hesse_accurate",
    "fit_runtime_sec",
    "target_runtime_sec",
    "exception_type",
    "exception",
]


class TargetTimeout(TimeoutError):
    pass


def _alarm_handler(signum, frame):  # noqa: ARG001
    raise TargetTimeout("target timed out")


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


def key_for(row: dict) -> tuple[str, str, str]:
    return (str(row.get("stage", "")), str(row.get("label", "")), str(row.get("target", "")))


def read_completed(path: Path) -> set[tuple[str, str, str]]:
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
            done.add(key_for(row))
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


def target_function(fit: dict, target: str):
    if target in fit["nodes"]:
        return "node", fit["node_funcs"][fit["nodes"].index(target)]
    if target in fit["parameters"]:
        return "parameter", fit["parameter_funcs"][fit["parameters"].index(target)]
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


def choose_targets(fit: dict, stage: str, *, max_nodes: int, max_params: int, max_nuisance: int) -> list[str]:
    nodes = list(fit["nodes"])
    params = list(fit["parameters"])
    nuisance = [p for p in params if p.startswith("nuisance_")]
    regular_params = [p for p in params if not p.startswith("nuisance_")]

    if stage == "all" or fit["label"] in KNOWN_HARD_FITS:
        return sorted(set(nodes + params))

    chosen = []
    chosen.extend(regular_params[:max_params])
    chosen.extend(nuisance[:max_nuisance])
    chosen.extend(nodes[:max_nodes])
    return sorted(dict.fromkeys(chosen))


def closest_point(points, value: float):
    points = list(points or [])
    if not points:
        return None
    return min(points, key=lambda p: abs(float(getattr(p, "value", np.inf)) - value))


def method_counts(points) -> str:
    counts = Counter(str(getattr(p, "method", "")) for p in points if str(getattr(p, "method", "")))
    return "|".join(f"{method}:{count}" for method, count in sorted(counts.items()))


def point_fields(prefix: str, point) -> dict:
    if point is None:
        return {
            f"{prefix}_constraint_violation": None,
            f"{prefix}_scaled_constraint_violation": None,
            f"{prefix}_projected_grad_norm": None,
            f"{prefix}_objective_grad_norm": None,
            f"{prefix}_method": "",
            f"{prefix}_profile_runtime_sec": None,
        }
    return {
        f"{prefix}_constraint_violation": getattr(point, "constraint_violation", None),
        f"{prefix}_scaled_constraint_violation": getattr(point, "scaled_constraint_violation", None),
        f"{prefix}_projected_grad_norm": getattr(point, "projected_grad_norm", None),
        f"{prefix}_objective_grad_norm": getattr(point, "objective_grad_norm", None),
        f"{prefix}_method": getattr(point, "method", ""),
        f"{prefix}_profile_runtime_sec": getattr(point, "runtime_sec", None),
    }


def run_one_target(
    fit: dict,
    target: str,
    stage: str,
    fit_runtime: float,
    *,
    solver_options=None,
    contract_brackets=True,
) -> dict:
    t0 = time.perf_counter()
    target_kind, target_func = target_function(fit, target)
    target_value = float(target_func(fit["param_values"]))
    target_std = target_std_from_cov(fit, target_func, target_value)
    profile_chi2 = build_constrained_profile_chi2(
        fit["chi2"],
        fit["fitted_params_to_params"],
        target_func,
        fit["fitted_values"],
        target_scale=target_std,
        target_name=target,
        solver_options=solver_options,
    )
    base_profile = profile_chi2(target_value)
    if base_profile > fit["chi2_min"] + 1e-5:
        raise RuntimeError(f"profile chi2 at optimum {base_profile} > chi2_min {fit['chi2_min']}")

    error_p, error_n = binary_search_error(
        profile_chi2,
        target_value,
        fit["chi2_min"],
        target_value - 2.0 * target_std,
        target_value + 2.0 * target_std,
        contract_brackets=contract_brackets,
    )
    diag = dict(binary_search_error.last_diagnostics or {})
    upper_endpoint = float(diag.get("upper_endpoint", target_value + error_p))
    lower_endpoint = float(diag.get("lower_endpoint", target_value - error_n))
    upper_point = closest_point(profile_chi2.diagnostics, upper_endpoint)
    lower_point = closest_point(profile_chi2.diagnostics, lower_endpoint)
    points = list(profile_chi2.diagnostics)
    row = {
        "stage": stage,
        "label": fit["label"],
        "algorithm": fit["algorithm"],
        "status": "ok",
        "target": target,
        "target_kind": target_kind,
        "target_value": target_value,
        "target_std": target_std,
        "chi2_min": float(fit["chi2_min"]),
        "error_p": float(error_p),
        "error_n": float(error_n),
        "upper_endpoint": upper_endpoint,
        "lower_endpoint": lower_endpoint,
        "upper_residual": float(diag.get("upper_residual", math.nan)),
        "lower_residual": float(diag.get("lower_residual", math.nan)),
        "upper_chi2": float(diag.get("upper_chi2", math.nan)),
        "lower_chi2": float(diag.get("lower_chi2", math.nan)),
        "profile_calls": len(points),
        "profile_methods": method_counts(points),
        "kkt_polish_calls": sum("+KKT" in str(getattr(p, "method", "")) for p in points),
        "n_params": int(len(fit["parameters"])),
        "n_nodes": int(len(fit["nodes"])),
        "n_nuisance": int(sum(p.startswith("nuisance_") for p in fit["parameters"])),
        "n_meas": int(len(fit["meas_df"])),
        "fit_valid": fit.get("fit_valid"),
        "hesse_accurate": fit.get("hesse_accurate"),
        "fit_runtime_sec": fit_runtime,
        "target_runtime_sec": time.perf_counter() - t0,
        "exception_type": "",
        "exception": "",
    }
    row.update(point_fields("upper", upper_point))
    row.update(point_fields("lower", lower_point))
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["representative", "all"], default="representative")
    parser.add_argument("--jsonl", default="notes/asym_fit_sweep_results.jsonl")
    parser.add_argument("--csv", default="notes/asym_fit_sweep_results.csv")
    parser.add_argument("--stdout-log", default="notes/logs/asym_fit_sweep_stdout.log")
    parser.add_argument("--fit-label", action="append", default=None)
    parser.add_argument("--start-from", default=None)
    parser.add_argument("--limit-fits", type=int, default=None)
    parser.add_argument("--max-nodes", type=int, default=1)
    parser.add_argument("--max-params", type=int, default=1)
    parser.add_argument("--max-nuisance", type=int, default=2)
    parser.add_argument("--target-timeout-sec", type=int, default=900)
    parser.add_argument("--optimizer", choices=["minuit", "scipy"], default="minuit")
    parser.add_argument("--fit-space", choices=["unconstrained", "constrained"], default="unconstrained")
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    args = parser.parse_args()

    jsonl_path = Path(args.jsonl)
    csv_path = Path(args.csv)
    stdout_log = Path(args.stdout_log)
    stdout_log.parent.mkdir(parents=True, exist_ok=True)
    completed = read_completed(jsonl_path) if args.resume else set()

    fits_df = all_fits()
    fits_df = fits_df[fits_df["algorithm"] != "IGNORE"].copy()
    fits_df = fits_df[fits_df["label"] != "tauhflav"].copy()
    if args.fit_label:
        requested = set(args.fit_label)
        fits_df = fits_df[fits_df["label"].isin(requested)]
    if args.start_from:
        labels_all = list(fits_df["label"])
        fits_df = fits_df.iloc[labels_all.index(args.start_from):]
    if args.limit_fits is not None:
        fits_df = fits_df.head(args.limit_fits)

    labels = list(fits_df["label"])
    print(f"fit labels selected={len(labels)}, completed target rows={len(completed)}")

    old_handler = signal.signal(signal.SIGALRM, _alarm_handler)
    try:
        for i, label in enumerate(labels, start=1):
            fit_runtime = None
            with stdout_log.open("a") as log:
                log.write(f"\n===== fit {i}/{len(labels)} {label} =====\n")
                log.flush()
                try:
                    t_fit = time.perf_counter()
                    with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                        fit = run_fit(label, verbose=False, optimizer=args.optimizer, fit_space=args.fit_space)
                    fit_runtime = time.perf_counter() - t_fit
                    if fit is None:
                        row = {
                            "stage": args.stage,
                            "label": label,
                            "status": "skipped",
                            "target": "",
                            "fit_runtime_sec": fit_runtime,
                            "exception": "run_fit returned None",
                        }
                        append_jsonl(jsonl_path, row)
                        append_csv(csv_path, row)
                        print(f"[{i}/{len(labels)}] {label} skipped")
                        continue
                    targets = choose_targets(
                        fit,
                        args.stage,
                        max_nodes=args.max_nodes,
                        max_params=args.max_params,
                        max_nuisance=args.max_nuisance,
                    )
                except Exception as exc:  # noqa: BLE001 - sweep must continue
                    row = {
                        "stage": args.stage,
                        "label": label,
                        "status": "fit_error",
                        "target": "",
                        "fit_runtime_sec": fit_runtime,
                        "exception_type": type(exc).__name__,
                        "exception": str(exc),
                    }
                    append_jsonl(jsonl_path, row)
                    append_csv(csv_path, row)
                    log.write(f"FIT EXCEPTION {type(exc).__name__}: {exc}\n")
                    print(f"[{i}/{len(labels)}] {label} fit_error {type(exc).__name__}: {exc}")
                    continue

            for target in targets:
                key = (args.stage, label, target)
                if key in completed:
                    continue
                row = None
                with stdout_log.open("a") as log:
                    log.write(f"\n--- target {label} / {target} ---\n")
                    log.flush()
                    try:
                        signal.alarm(args.target_timeout_sec)
                        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                            row = run_one_target(fit, target, args.stage, fit_runtime)
                        signal.alarm(0)
                    except Exception as exc:  # noqa: BLE001 - row-level diagnostics
                        signal.alarm(0)
                        row = {
                            "stage": args.stage,
                            "label": label,
                            "algorithm": fit.get("algorithm"),
                            "status": "error",
                            "target": target,
                            "n_params": int(len(fit["parameters"])),
                            "n_nodes": int(len(fit["nodes"])),
                            "n_nuisance": int(sum(p.startswith("nuisance_") for p in fit["parameters"])),
                            "n_meas": int(len(fit["meas_df"])),
                            "fit_valid": fit.get("fit_valid"),
                            "hesse_accurate": fit.get("hesse_accurate"),
                            "fit_runtime_sec": fit_runtime,
                            "exception_type": type(exc).__name__,
                            "exception": str(exc),
                        }
                        log.write(f"TARGET EXCEPTION {type(exc).__name__}: {exc}\n")
                append_jsonl(jsonl_path, row)
                append_csv(csv_path, row)
                print(
                    f"[{i}/{len(labels)}] {label} target={target} {row['status']} "
                    f"runtime={row.get('target_runtime_sec')} "
                    f"resid=({row.get('upper_residual')},{row.get('lower_residual')})"
                )
                completed.add(key)
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
