#!/usr/bin/env python
"""Focused benchmark for asymmetric-error endpoint search variants.

This script is intentionally notes-only validation tooling.  It reuses the
existing fit/average runners and records the metrics needed to compare small
endpoint-search changes: residuals, profile-call counts, method distribution,
runtime, and any per-search evaluation counters exposed by binary_search_error.
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

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np

from pdgfits.asym_errors import binary_search_error
from pdgfits.avg import run_avg
from pdgfits.fit import run_fit
from pdgfits.query import avg_queries

from notes.run_asym_avg_sweep import FIELDNAMES as AVG_SWEEP_FIELDNAMES
from notes.run_asym_avg_sweep import build_row_for_result
from notes.run_asym_fit_sweep import FIELDNAMES as FIT_SWEEP_FIELDNAMES
from notes.run_asym_fit_sweep import run_one_target


FIT_HARD_TARGETS = {
    "B0": [
        "S042.181",
        "S042.390",
        "S042.69",
        "S042B09",
        "S042B31",
        "S042B95",
        "nuisance_M070.1",
        "nuisance_M070.50",
        "nuisance_M070.54",
    ],
    "B0S-BR": [
        "S086.37",
        "S086.5",
        "S086.7",
        "S086R04",
        "S086R29",
        "S086R3",
        "nuisance_M004.1",
        "nuisance_M092.16",
        "nuisance_S041.3",
    ],
    "K_3^*(1780)": ["M060.6"],
    "Lam-b-0": ["S040.10", "S040.29", "S040R10", "S040R29"],
    "chi_c012 psi(2S)": ["M055B1", "M055B10", "M055B11", "M056.11", "M056.6"],
    "eta_c J/psi psi(2S)": [
        "M026.31",
        "M026.45",
        "M026G01",
        "M026G03",
        "M026G07",
        "M026W",
        "nuisance_M071.13",
        "nuisance_M071.254",
    ],
    "eta_c(2S)": ["M059.4"],
}

EXTRA_ENDPOINT_FIELDS = [
    "endpoint_search_method",
    "endpoint_function_evals",
    "endpoint_cache_hits",
    "upper_function_evals",
    "lower_function_evals",
    "upper_cache_hits",
    "lower_cache_hits",
    "upper_bracket_evals",
    "lower_bracket_evals",
    "upper_bisect_evals",
    "lower_bisect_evals",
]

FIT_FIELDNAMES = ["variant", *FIT_SWEEP_FIELDNAMES, *EXTRA_ENDPOINT_FIELDS]
AVG_FIELDNAMES = ["variant", *AVG_SWEEP_FIELDNAMES, *EXTRA_ENDPOINT_FIELDS]


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


def _float(row: dict, key: str) -> float:
    try:
        return float(row.get(key) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _abs_endpoint_residual(row: dict) -> float:
    return max(
        abs(_float(row, key))
        for key in ("fresh_upper_residual", "fresh_lower_residual", "upper_residual", "lower_residual")
    )


def select_avg_nodes(limit_slowest: int, limit_residual: int, limit_nuisance: int) -> list[str]:
    rows = read_csv_rows("notes/asym_avg_sweep_results.csv")
    selected = {"M026R08"}
    for row in sorted(rows, key=lambda r: _float(r, "runtime_sec"), reverse=True)[:limit_slowest]:
        selected.add(row["node"])
    for row in sorted(rows, key=_abs_endpoint_residual, reverse=True)[:limit_residual]:
        selected.add(row["node"])
    for row in sorted(
        rows,
        key=lambda r: (int(_float(r, "n_nuisance")), _float(r, "runtime_sec")),
        reverse=True,
    )[:limit_nuisance]:
        selected.add(row["node"])
    return sorted(selected)


def parse_fit_targets(items: list[str] | None) -> dict[str, list[str]]:
    if not items:
        return {label: list(targets) for label, targets in sorted(FIT_HARD_TARGETS.items())}
    selected: dict[str, set[str]] = defaultdict(set)
    for item in items:
        if "::" not in item:
            raise ValueError(f"--fit-target must be LABEL::TARGET, got {item!r}")
        label, target = item.split("::", 1)
        selected[label].add(target)
    return {label: sorted(targets) for label, targets in sorted(selected.items())}


def add_endpoint_fields(row: dict, diag: dict | None) -> dict:
    diag = dict(diag or {})
    side_counts = diag.get("side_counts") if isinstance(diag.get("side_counts"), dict) else {}
    row["endpoint_search_method"] = diag.get("search_method")
    row["endpoint_function_evals"] = diag.get("function_evals")
    row["endpoint_cache_hits"] = diag.get("cache_hits")
    for side in ("upper", "lower"):
        counts = side_counts.get(side, {}) if isinstance(side_counts.get(side), dict) else {}
        row[f"{side}_function_evals"] = counts.get("function_evals")
        row[f"{side}_cache_hits"] = counts.get("cache_hits")
        row[f"{side}_bracket_evals"] = counts.get("bracket_evals")
        row[f"{side}_bisect_evals"] = counts.get("bisect_evals")
    return row


def run_fit_benchmark(
    *,
    variant: str,
    targets_by_label: dict[str, list[str]],
    csv_path: Path,
    jsonl_path: Path,
    stdout_log: Path,
    target_timeout_sec: int,
) -> None:
    old_handler = signal.signal(signal.SIGALRM, _alarm_handler)
    try:
        for label_idx, (label, targets) in enumerate(targets_by_label.items(), start=1):
            with stdout_log.open("a") as log:
                log.write(f"\n===== fit {label_idx}/{len(targets_by_label)} {label} =====\n")
                log.flush()
                try:
                    t_fit = time.perf_counter()
                    with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                        fit = run_fit(label, verbose=False)
                    fit_runtime = time.perf_counter() - t_fit
                except Exception as exc:  # noqa: BLE001 - benchmark continues
                    row = {
                        "variant": variant,
                        "stage": "endpoint-search-hard",
                        "label": label,
                        "status": "fit_error",
                        "target": "",
                        "exception_type": type(exc).__name__,
                        "exception": str(exc),
                    }
                    append_jsonl(jsonl_path, row)
                    append_csv(csv_path, row, FIT_FIELDNAMES)
                    log.write(f"FIT EXCEPTION {type(exc).__name__}: {exc}\n")
                    print(f"[fit {label_idx}/{len(targets_by_label)}] {label} fit_error {type(exc).__name__}: {exc}")
                    continue

            for target_idx, target in enumerate(targets, start=1):
                with stdout_log.open("a") as log:
                    log.write(f"\n--- target {label} / {target} ---\n")
                    log.flush()
                    try:
                        signal.alarm(target_timeout_sec)
                        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                            row = run_one_target(fit, target, "endpoint-search-hard", fit_runtime)
                        signal.alarm(0)
                        row = add_endpoint_fields(row, binary_search_error.last_diagnostics)
                    except Exception as exc:  # noqa: BLE001 - row-level diagnostics
                        signal.alarm(0)
                        row = {
                            "stage": "endpoint-search-hard",
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
                append_csv(csv_path, row, FIT_FIELDNAMES)
                print(
                    f"[fit {label_idx}/{len(targets_by_label)} {target_idx}/{len(targets)}] "
                    f"{label} / {target} {row['status']} runtime={row.get('target_runtime_sec')} "
                    f"calls={row.get('profile_calls')} resid=({row.get('upper_residual')},{row.get('lower_residual')})"
                )
                gc.collect()
            try:
                jax.clear_caches()
            except Exception:
                pass
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def run_avg_benchmark(
    *,
    variant: str,
    nodes: list[str],
    csv_path: Path,
    jsonl_path: Path,
    stdout_log: Path,
) -> None:
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
                        "variant": variant,
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
                    row = add_endpoint_fields(row, result.get("asym_error_diagnostics"))
                    row["variant"] = variant
            except Exception as exc:  # noqa: BLE001 - benchmark continues
                runtime = time.perf_counter() - t0
                row = {
                    "variant": variant,
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
            f"[avg {idx}/{len(nodes)}] {node} {row['status']} runtime={row.get('runtime_sec', 0):.2f}s "
            f"calls={row.get('profile_calls')} fresh_resid=({row.get('fresh_upper_residual')},{row.get('fresh_lower_residual')})"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", default="baseline")
    parser.add_argument("--fit-csv", default="notes/asym_endpoint_search_fit_benchmark.csv")
    parser.add_argument("--fit-jsonl", default="notes/asym_endpoint_search_fit_benchmark.jsonl")
    parser.add_argument("--avg-csv", default="notes/asym_endpoint_search_avg_benchmark.csv")
    parser.add_argument("--avg-jsonl", default="notes/asym_endpoint_search_avg_benchmark.jsonl")
    parser.add_argument("--stdout-log", default="notes/logs/asym_endpoint_search_benchmark_stdout.log")
    parser.add_argument("--skip-fits", action="store_true")
    parser.add_argument("--skip-avgs", action="store_true")
    parser.add_argument("--fit-target", action="append", default=None, help="LABEL::TARGET; may be repeated")
    parser.add_argument("--avg-node", action="append", default=None)
    parser.add_argument("--avg-slowest", type=int, default=5)
    parser.add_argument("--avg-residual", type=int, default=5)
    parser.add_argument("--avg-nuisance", type=int, default=5)
    parser.add_argument("--target-timeout-sec", type=int, default=900)
    args = parser.parse_args()

    stdout_log = Path(args.stdout_log)
    stdout_log.parent.mkdir(parents=True, exist_ok=True)

    if not args.skip_fits:
        targets_by_label = parse_fit_targets(args.fit_target)
        print(
            f"fit hard labels={len(targets_by_label)}, "
            f"targets={sum(len(v) for v in targets_by_label.values())}, variant={args.variant}"
        )
        run_fit_benchmark(
            variant=args.variant,
            targets_by_label=targets_by_label,
            csv_path=Path(args.fit_csv),
            jsonl_path=Path(args.fit_jsonl),
            stdout_log=stdout_log,
            target_timeout_sec=args.target_timeout_sec,
        )

    if not args.skip_avgs:
        nodes = sorted(args.avg_node) if args.avg_node else select_avg_nodes(
            args.avg_slowest,
            args.avg_residual,
            args.avg_nuisance,
        )
        print(f"average hard nodes={len(nodes)}, variant={args.variant}: {', '.join(nodes)}")
        run_avg_benchmark(
            variant=args.variant,
            nodes=nodes,
            csv_path=Path(args.avg_csv),
            jsonl_path=Path(args.avg_jsonl),
            stdout_log=stdout_log,
        )


if __name__ == "__main__":
    main()
