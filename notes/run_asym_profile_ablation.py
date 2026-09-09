#!/usr/bin/env python
"""Focused asymmetric-profile ablation harness.

This intentionally lives under notes/: it is validation tooling, not package
API.  It reuses the existing sweep helpers but restricts work to hard targets
that are useful for quick solver-method iteration.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import gc
import json
import math
import signal
import sys
import time
from collections import defaultdict
from dataclasses import asdict, is_dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np

from pdgfits.asym_errors import ProfileSolverOptions
from pdgfits.avg import run_avg
from pdgfits.fit import run_fit
from pdgfits.query import avg_queries

from notes.run_asym_avg_sweep import build_row_for_result
from notes.run_asym_fit_sweep import FIELDNAMES as FIT_FIELDNAMES
from notes.run_asym_fit_sweep import run_one_target


FIT_ABLATION_FIELDNAMES = ["variant", *FIT_FIELDNAMES]
AVG_FIELDNAMES = [
    "node",
    "status",
    "raw_n_meas",
    "post_n_meas",
    "n_params",
    "n_nuisance",
    "has_adjust",
    "runtime_sec",
    "upper_residual",
    "lower_residual",
    "fresh_upper_residual",
    "fresh_lower_residual",
    "profile_calls",
    "profile_methods",
    "exception_type",
    "exception",
]


VARIANTS = {
    "slsqp_kkt": (
        ProfileSolverOptions(
            use_slsqp=True,
            polish_slsqp=True,
            use_exact_hessian=False,
            use_kkt_polish=True,
            use_descent_check=False,
        ),
        True,
        "SLSQP plus KKT polish only; no exact-Hessian fallback or descent check.",
    ),
    "exact_kkt": (
        ProfileSolverOptions(
            use_slsqp=False,
            use_exact_hessian=True,
            use_kkt_polish=True,
            use_descent_check=False,
        ),
        True,
        "Exact-Hessian trust-constr plus KKT only.",
    ),
    "no_descent": (
        ProfileSolverOptions(
            use_slsqp=True,
            polish_slsqp=False,
            use_exact_hessian=True,
            use_kkt_polish=True,
            use_descent_check=False,
        ),
        True,
        "SLSQP, exact-Hessian trust-constr, and KKT; no descent certificate.",
    ),
    "minimal": (
        ProfileSolverOptions(
            use_slsqp=True,
            polish_slsqp=False,
            use_exact_hessian=True,
            use_kkt_polish=True,
            use_descent_check=True,
        ),
        True,
        "SLSQP, exact-Hessian trust-constr, KKT, descent certificate, bracket contraction.",
    ),
    "minimal_no_contract": (
        ProfileSolverOptions(
            use_slsqp=True,
            polish_slsqp=False,
            use_exact_hessian=True,
            use_kkt_polish=True,
            use_descent_check=True,
        ),
        False,
        "Minimal variant but with bracket contraction disabled.",
    ),
}


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


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(clean(row), sort_keys=True) + "\n")


def append_csv(path: Path, row: dict, fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow({key: clean(row.get(key)) for key in fieldnames})


def read_csv_rows(path: str | Path) -> list[dict]:
    with Path(path).open(newline="") as f:
        return list(csv.DictReader(f))


def add_target(targets: dict[str, set[str]], label: str, target: str) -> None:
    if label and target:
        targets[label].add(target)


def select_fit_targets(*, include_lam_all: bool, include_chi: bool) -> dict[str, list[str]]:
    targets: dict[str, set[str]] = defaultdict(set)

    for row in read_csv_rows("notes/asym_fit_failure_focus_final.csv"):
        add_target(targets, row.get("label", ""), row.get("target", ""))

    sweep_rows = read_csv_rows("notes/asym_fit_sweep_results.csv")
    for row in sweep_rows:
        label = row.get("label", "")
        target = row.get("target", "")
        if label == "Upsilon(2S)":
            add_target(targets, label, target)
        if label == "Lam-b-0" and (
            include_lam_all
            or target
            in {
                "S040.10",
                "S040.29",
                "S040.9",
                "S040R29",
                "S040R9",
                "nuisance_S042.30",
            }
        ):
            add_target(targets, label, target)
        if label == "eta_c J/psi psi(2S)" and target == "M026.45":
            add_target(targets, label, target)
        if include_chi and label == "chi_c012 psi(2S)":
            add_target(targets, label, target)

    return {label: sorted(values) for label, values in sorted(targets.items())}


def abs_residual(row: dict) -> float:
    values = []
    for key in ("fresh_upper_residual", "fresh_lower_residual", "upper_residual", "lower_residual"):
        try:
            values.append(abs(float(row.get(key) or 0.0)))
        except (TypeError, ValueError):
            pass
    return max(values) if values else 0.0


def select_avg_nodes(limit_slowest: int, limit_residual: int, limit_nuisance: int) -> list[str]:
    rows = read_csv_rows("notes/asym_avg_sweep_results.csv")
    selected = {"M026R08"}

    for row in sorted(rows, key=lambda r: float(r.get("runtime_sec") or 0.0), reverse=True)[:limit_slowest]:
        selected.add(row["node"])
    for row in sorted(rows, key=abs_residual, reverse=True)[:limit_residual]:
        selected.add(row["node"])
    for row in sorted(
        rows,
        key=lambda r: (int(float(r.get("n_nuisance") or 0)), float(r.get("runtime_sec") or 0.0)),
        reverse=True,
    )[:limit_nuisance]:
        selected.add(row["node"])
    return sorted(selected)


def run_average_focus(nodes: list[str], csv_path: Path, jsonl_path: Path, stdout_log: Path) -> None:
    avg_df, corr_df_dict = avg_queries(verbose=False)
    for idx, node in enumerate(nodes, start=1):
        meas_df_node = avg_df[avg_df["node"] == node]
        corr_df_node = corr_df_dict[node]
        raw_n = int(len(meas_df_node))
        t0 = time.perf_counter()
        with stdout_log.open("a") as log:
            log.write(f"\n===== avg {idx}/{len(nodes)} {node} =====\n")
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
                        "exception": "run_avg returned None",
                    }
                else:
                    row = build_row_for_result(
                        node,
                        result,
                        raw_n,
                        corr_df_node,
                        runtime,
                        max_nuisance=20,
                    )
            except Exception as exc:  # noqa: BLE001 - diagnostic harness
                runtime = time.perf_counter() - t0
                row = {
                    "node": node,
                    "status": "error",
                    "raw_n_meas": raw_n,
                    "runtime_sec": runtime,
                    "exception_type": type(exc).__name__,
                    "exception": str(exc),
                }
                log.write(f"AVG EXCEPTION {type(exc).__name__}: {exc}\n")

        append_jsonl(jsonl_path, row)
        append_csv(csv_path, row, AVG_FIELDNAMES)
        print(
            f"[avg {idx}/{len(nodes)}] {node} {row['status']} "
            f"runtime={row.get('runtime_sec', 0):.2f}s "
            f"fresh_resid=({row.get('fresh_upper_residual')},{row.get('fresh_lower_residual')})"
        )


def run_fit_focus(
    variants: list[str],
    targets_by_label: dict[str, list[str]],
    csv_path: Path,
    jsonl_path: Path,
    stdout_log: Path,
    target_timeout_sec: int,
) -> None:
    old_handler = signal.signal(signal.SIGALRM, _alarm_handler)
    try:
        for variant in variants:
            solver_options, contract_brackets, _ = VARIANTS[variant]
            for label_idx, (label, targets) in enumerate(targets_by_label.items(), start=1):
                with stdout_log.open("a") as log:
                    log.write(f"\n===== variant {variant} fit {label_idx}/{len(targets_by_label)} {label} =====\n")
                    log.flush()
                    try:
                        t_fit = time.perf_counter()
                        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                            fit = run_fit(label, verbose=False)
                        fit_runtime = time.perf_counter() - t_fit
                    except Exception as exc:  # noqa: BLE001 - diagnostic harness
                        row = {
                            "variant": variant,
                            "stage": "focused-ablation",
                            "label": label,
                            "status": "fit_error",
                            "target": "",
                            "exception_type": type(exc).__name__,
                            "exception": str(exc),
                        }
                        append_jsonl(jsonl_path, row)
                        append_csv(csv_path, row, FIT_ABLATION_FIELDNAMES)
                        log.write(f"FIT EXCEPTION {type(exc).__name__}: {exc}\n")
                        print(f"[{variant}] {label} fit_error {type(exc).__name__}: {exc}")
                        continue

                for target_idx, target in enumerate(targets, start=1):
                    with stdout_log.open("a") as log:
                        log.write(f"\n--- variant {variant} target {label} / {target} ---\n")
                        log.flush()
                        try:
                            signal.alarm(target_timeout_sec)
                            with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                                row = run_one_target(
                                    fit,
                                    target,
                                    "focused-ablation",
                                    fit_runtime,
                                    solver_options=solver_options,
                                    contract_brackets=contract_brackets,
                                )
                            signal.alarm(0)
                        except Exception as exc:  # noqa: BLE001 - row-level diagnostics
                            signal.alarm(0)
                            row = {
                                "stage": "focused-ablation",
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
                    row["variant"] = variant
                    append_jsonl(jsonl_path, row)
                    append_csv(csv_path, row, FIT_ABLATION_FIELDNAMES)
                    print(
                        f"[{variant} {label_idx}/{len(targets_by_label)} {target_idx}/{len(targets)}] "
                        f"{label} / {target} {row['status']} runtime={row.get('target_runtime_sec')} "
                        f"resid=({row.get('upper_residual')},{row.get('lower_residual')})"
                    )
                    gc.collect()
                try:
                    jax.clear_caches()
                except Exception:
                    pass
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", action="append", choices=sorted(VARIANTS), default=None)
    parser.add_argument("--fit-csv", default="notes/asym_profile_ablation_fit.csv")
    parser.add_argument("--fit-jsonl", default="notes/asym_profile_ablation_fit.jsonl")
    parser.add_argument("--avg-csv", default="notes/asym_profile_ablation_avg.csv")
    parser.add_argument("--avg-jsonl", default="notes/asym_profile_ablation_avg.jsonl")
    parser.add_argument("--stdout-log", default="notes/logs/asym_profile_ablation_stdout.log")
    parser.add_argument("--target-timeout-sec", type=int, default=240)
    parser.add_argument("--skip-avgs", action="store_true")
    parser.add_argument("--skip-fits", action="store_true")
    parser.add_argument("--include-lam-all", action="store_true")
    parser.add_argument("--include-chi", action="store_true")
    parser.add_argument(
        "--fit-target",
        action="append",
        default=None,
        help="Restrict fit run to one target, formatted as LABEL::TARGET. May be repeated.",
    )
    parser.add_argument("--avg-slowest", type=int, default=5)
    parser.add_argument("--avg-residual", type=int, default=5)
    parser.add_argument("--avg-nuisance", type=int, default=5)
    args = parser.parse_args()

    variants = args.variant or ["slsqp_kkt", "exact_kkt", "no_descent", "minimal"]
    stdout_log = Path(args.stdout_log)
    stdout_log.parent.mkdir(parents=True, exist_ok=True)

    if not args.skip_avgs:
        avg_nodes = select_avg_nodes(args.avg_slowest, args.avg_residual, args.avg_nuisance)
        print(f"average focused nodes={len(avg_nodes)}: {', '.join(avg_nodes)}")
        run_average_focus(avg_nodes, Path(args.avg_csv), Path(args.avg_jsonl), stdout_log)

    if not args.skip_fits:
        if args.fit_target:
            selected: dict[str, set[str]] = defaultdict(set)
            for item in args.fit_target:
                if "::" not in item:
                    raise ValueError(f"--fit-target must be LABEL::TARGET, got {item!r}")
                label, target = item.split("::", 1)
                add_target(selected, label, target)
            targets_by_label = {label: sorted(values) for label, values in sorted(selected.items())}
        else:
            targets_by_label = select_fit_targets(
                include_lam_all=args.include_lam_all,
                include_chi=args.include_chi,
            )
        n_targets = sum(len(v) for v in targets_by_label.values())
        print(f"fit variants={variants}")
        print(f"fit labels={len(targets_by_label)}, targets={n_targets}")
        for name in variants:
            print(f"variant {name}: {VARIANTS[name][2]}")
        run_fit_focus(
            variants,
            targets_by_label,
            Path(args.fit_csv),
            Path(args.fit_jsonl),
            stdout_log,
            args.target_timeout_sec,
        )


if __name__ == "__main__":
    main()
