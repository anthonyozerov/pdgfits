#!/usr/bin/env python
"""Round 20 B0 unconstrained-refit/globality stability audit.

This notes-only harness reuses the round-18 toy campaign and round-19 B0 audit
helpers, but deliberately avoids endpoint searches.  It reruns only the
unconstrained toy refit from the production/reference start plus deterministic
restarts/refinements from the returned MLE and, for suspicious B0 rows, the
fixed-truth profile solution.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import jax
from jax import numpy as jnp
import numpy as np

import run_round15_wilks_calibration_smoke as round15
import run_round16_calibration_pilot as round16
import run_round19_b0_consistency_audit as round19

from pdgfits.fit import run_fit


jax.config.update("jax_enable_x64", True)


CASE_SPECS = {
    case.case_id: case
    for case in round16.DEFAULT_CASES
}

CASE_INDEX = {
    case.case_id: index
    for index, case in enumerate(round16.DEFAULT_CASES)
}

SELECTION_FIELDNAMES = [
    "selection_reason",
    "dgp",
    "case_id",
    "stratum",
    "label",
    "target",
    "toy_index",
    "seed",
    "original_q",
    "reported_chi2",
    "profile_true_chi2",
    "refit_valid_csv",
    "refit_accurate_csv",
    "refit_edm_csv",
    "min_decay_param_mle_csv",
    "max_abs_decay_fitted_coord_mle_csv",
    "chart_saturation_warning_mle_csv",
]

ATTEMPT_FIELDNAMES = [
    "dgp",
    "case_id",
    "stratum",
    "label",
    "target",
    "toy_index",
    "seed",
    "start_name",
    "optimizer",
    "start_chi2",
    "result_chi2",
    "improvement_vs_reported",
    "improvement_vs_reference_rerun",
    "success",
    "valid",
    "accurate",
    "edm",
    "nfcn",
    "ngrad",
    "nfev",
    "njev",
    "gradient_norm",
    "target_value",
    "target_minus_truth",
    "min_decay_param",
    "max_decay_param",
    "max_abs_decay_fitted_coord",
    "message",
]

ROW_FIELDNAMES = [
    "selection_reason",
    "dgp",
    "case_id",
    "stratum",
    "label",
    "target",
    "toy_index",
    "seed",
    "status",
    "runtime_sec",
    "original_q",
    "reported_chi2",
    "reference_rerun_chi2",
    "reference_rerun_minus_reported",
    "reference_rerun_valid",
    "reference_rerun_accurate",
    "reference_rerun_edm",
    "profile_true_chi2_csv",
    "profile_true_chi2_recomputed",
    "profile_true_chi2_used",
    "profile_true_recomputed_minus_csv",
    "profile_start_used",
    "best_restart_chi2",
    "best_restart_start",
    "best_restart_optimizer",
    "best_restart_success",
    "best_restart_valid",
    "best_restart_accurate",
    "best_restart_edm",
    "best_restart_nfev",
    "best_restart_nfcn",
    "improvement_vs_reported",
    "improvement_vs_reference_rerun",
    "q_after_best_refit",
    "profile_consistency_failure_original",
    "profile_consistency_failure_after_best",
    "profile_consistency_failure_disappears",
    "lowered_gt_1e_4",
    "lowered_gt_1e_3",
    "lowered_gt_1e_2",
    "min_decay_param_mle",
    "max_decay_param_mle",
    "max_abs_decay_fitted_coord_mle",
    "min_decay_param_best",
    "max_decay_param_best",
    "max_abs_decay_fitted_coord_best",
    "chart_saturation_warning_mle_csv",
    "n_attempts",
    "n_finite_attempts",
    "n_failed_attempts",
    "exception_type",
    "exception",
]

FAILURE_FIELDNAMES = [
    "selection_reason",
    "dgp",
    "case_id",
    "stratum",
    "label",
    "target",
    "toy_index",
    "seed",
    "phase",
    "exception_type",
    "exception",
    "runtime_sec",
]


def clean_value(value: Any) -> Any:
    return round15.clean_value(value)


def json_default(obj: Any) -> Any:
    return round15.json_default(obj)


def finite_or_none(value: Any) -> float | None:
    return round15.finite_or_none(value)


def bool_from_csv(value: Any) -> bool:
    return str(value).strip().lower() == "true"


def float_from_csv(row: dict[str, Any], key: str) -> float:
    value = row.get(key)
    if value in (None, ""):
        return math.nan
    return float(value)


def int_from_csv(row: dict[str, Any], key: str) -> int:
    return int(str(row[key]))


def append_row(path: Path, row: dict[str, Any], fieldnames: list[str], *, jsonl: bool = False) -> None:
    round15.append_row(path, row, fieldnames, jsonl=jsonl)


def ensure_artifact(path: Path, fieldnames: list[str], *, jsonl: bool = False) -> None:
    round15.ensure_artifact(path, fieldnames, jsonl=jsonl)


def reset_outputs(paths: list[Path]) -> None:
    round15.reset_outputs(paths)


def read_campaign_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [row for row in csv.DictReader(handle) if row.get("status") == "ok"]


def selection_key(row: dict[str, Any]) -> tuple[str, str, int]:
    return (str(row["dgp"]), str(row["case_id"]), int_from_csv(row, "toy_index"))


def add_selection(
    selected: dict[tuple[str, str, int], dict[str, Any]],
    row: dict[str, Any],
    reason: str,
) -> None:
    key = selection_key(row)
    if key in selected:
        selected[key]["selection_reason"] += f";{reason}"
        return
    selected[key] = {**row, "selection_reason": reason}


def closest_by_q(rows: list[dict[str, Any]], q_target: float, n: int) -> list[dict[str, Any]]:
    finite = [row for row in rows if math.isfinite(float_from_csv(row, "profile_true_delta_chi2"))]
    return sorted(finite, key=lambda row: abs(float_from_csv(row, "profile_true_delta_chi2") - q_target))[:n]


def build_selection(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = read_campaign_rows(Path(args.campaign_csv))
    selected: dict[tuple[str, str, int], dict[str, Any]] = {}

    b0_rows = [row for row in rows if row.get("case_id") == "fit_b0_s042b95"]
    for row in b0_rows:
        q = float_from_csv(row, "profile_true_delta_chi2")
        if row.get("dgp") == "split_normal_pdg_resid" and q < args.b0_split_q_lt:
            add_selection(selected, row, f"b0_split_q_lt_{args.b0_split_q_lt:g}")
        elif row.get("dgp") == "local_gaussian_sym" and q < args.b0_local_q_lt:
            add_selection(selected, row, f"b0_local_q_lt_{args.b0_local_q_lt:g}")

    # Keep immediate neighbors of any negative-q B0 row, since round 19 found
    # that the nearby rows are useful controls for local recurrence claims.
    b0_by_dgp_toy = {
        (row["dgp"], int_from_csv(row, "toy_index")): row
        for row in b0_rows
    }
    for row in b0_rows:
        q = float_from_csv(row, "profile_true_delta_chi2")
        if q < 0.0:
            toy = int_from_csv(row, "toy_index")
            for offset in range(-args.neighbor_radius, args.neighbor_radius + 1):
                neighbor = b0_by_dgp_toy.get((row["dgp"], toy + offset))
                if neighbor is not None:
                    add_selection(selected, neighbor, f"negative_q_neighbor_radius_{args.neighbor_radius}")

    for dgp in round16.DGP_CHOICES:
        group = [row for row in b0_rows if row.get("dgp") == dgp]
        saturated = sorted(
            group,
            key=lambda row: float_from_csv(row, "max_abs_decay_fitted_coord_mle"),
            reverse=True,
        )[: args.b0_saturation_reps_per_dgp]
        for row in saturated:
            add_selection(selected, row, f"b0_top_saturation_{args.b0_saturation_reps_per_dgp}")
        for row in closest_by_q(group, 1.0, args.b0_mid_q_reps_per_dgp):
            add_selection(selected, row, f"b0_q_near_1_{args.b0_mid_q_reps_per_dgp}")

    comparison_case_ids = [
        "fit_b0s_s08637",
        "fit_eta_m026g01",
        "fit_g2000_k002m",
        "fit_g2000_k003m",
    ]
    for case_id in comparison_case_ids:
        for dgp in round16.DGP_CHOICES:
            group = [
                row
                for row in rows
                if row.get("case_id") == case_id and row.get("dgp") == dgp
            ]
            small = sorted(group, key=lambda row: float_from_csv(row, "profile_true_delta_chi2"))[
                : args.comparison_small_q_per_case_dgp
            ]
            for row in small:
                add_selection(selected, row, f"comparison_small_q_{args.comparison_small_q_per_case_dgp}")

            if case_id.startswith("fit_g2000"):
                for row in closest_by_q(group, 1.0, args.clean_q_near_1_per_case_dgp):
                    add_selection(selected, row, f"clean_q_near_1_{args.clean_q_near_1_per_case_dgp}")

    selected_rows = list(selected.values())
    selected_rows.sort(
        key=lambda row: (
            CASE_INDEX.get(str(row.get("case_id")), 999),
            str(row.get("dgp")),
            int_from_csv(row, "toy_index"),
        )
    )
    return selected_rows


def write_selection(rows: list[dict[str, Any]], csv_path: Path, jsonl_path: Path) -> None:
    ensure_artifact(csv_path, SELECTION_FIELDNAMES)
    ensure_artifact(jsonl_path, SELECTION_FIELDNAMES, jsonl=True)
    for row in rows:
        out = {
            "selection_reason": row.get("selection_reason"),
            "dgp": row.get("dgp"),
            "case_id": row.get("case_id"),
            "stratum": row.get("stratum"),
            "label": row.get("label"),
            "target": row.get("target"),
            "toy_index": int_from_csv(row, "toy_index"),
            "seed": int_from_csv(row, "seed"),
            "original_q": float_from_csv(row, "profile_true_delta_chi2"),
            "reported_chi2": float_from_csv(row, "toy_chi2_min"),
            "profile_true_chi2": float_from_csv(row, "profile_true_chi2"),
            "refit_valid_csv": bool_from_csv(row.get("refit_valid")),
            "refit_accurate_csv": bool_from_csv(row.get("refit_accurate")),
            "refit_edm_csv": finite_or_none(row.get("refit_edm")),
            "min_decay_param_mle_csv": finite_or_none(row.get("min_decay_param_mle")),
            "max_abs_decay_fitted_coord_mle_csv": finite_or_none(row.get("max_abs_decay_fitted_coord_mle")),
            "chart_saturation_warning_mle_csv": bool_from_csv(row.get("chart_saturation_warning_mle")),
        }
        append_row(csv_path, out, SELECTION_FIELDNAMES)
        append_row(jsonl_path, out, SELECTION_FIELDNAMES, jsonl=True)


def completed_keys(path: Path) -> set[tuple[str, str, int]]:
    if not path.exists() or path.stat().st_size == 0:
        return set()
    out: set[tuple[str, str, int]] = set()
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("status") != "ok":
                continue
            try:
                out.add((str(row["dgp"]), str(row["case_id"]), int(row["toy_index"])))
            except (KeyError, TypeError, ValueError):
                continue
    return out


def norm_or_none(values: np.ndarray) -> float | None:
    if not np.all(np.isfinite(values)):
        return None
    return float(np.linalg.norm(values))


def prefixed_attempt(
    base: dict[str, Any],
    row: dict[str, Any],
    reported_chi2: float,
    reference_chi2: float,
) -> dict[str, Any]:
    result_chi2 = finite_or_none(row.get("result_chi2"))
    out = {
        **base,
        **{key: row.get(key) for key in ATTEMPT_FIELDNAMES if key not in base},
    }
    if result_chi2 is not None:
        out["improvement_vs_reported"] = reported_chi2 - result_chi2
        out["improvement_vs_reference_rerun"] = reference_chi2 - result_chi2
    else:
        out["improvement_vs_reported"] = None
        out["improvement_vs_reference_rerun"] = None
    return out


def add_start(starts: list[tuple[str, np.ndarray]], name: str, x: np.ndarray) -> None:
    round19.add_start(starts, name, x)


def b0_floor_starts(audit: round19.ToyAudit, x_hat: np.ndarray) -> list[tuple[str, np.ndarray]]:
    starts: list[tuple[str, np.ndarray]] = []
    if len(audit.decay_idxs) == 0:
        return starts
    mle_params = audit.params(x_hat)
    for floor in (1e-6, 3e-6, 1e-5, 1e-4):
        params = mle_params.copy()
        params[audit.decay_idxs] = np.clip(params[audit.decay_idxs], floor, 1.0 - floor)
        add_start(starts, f"clip_decay_floor_{floor:g}", audit.physical_to_fitted(params))
    for factor in (0.5, 2.0, 10.0):
        params = mle_params.copy()
        tiny = audit.decay_idxs[params[audit.decay_idxs] <= 1e-6]
        if len(tiny) == 0:
            continue
        params[tiny] = np.clip(params[tiny] * factor, 1e-12, 1.0 - 1e-12)
        add_start(starts, f"tiny_decay_factor_{factor:g}", audit.physical_to_fitted(params))
    return starts


def run_reference_minuit(
    audit: round19.ToyAudit,
    start: np.ndarray,
    truth_target: float,
    reported_chi2: float,
) -> tuple[dict[str, Any], np.ndarray]:
    rows = round19.run_unconstrained_from_start(
        audit,
        "production_reference_fit_start",
        start,
        truth_target,
        reported_chi2,
        reported_chi2,
        ("minuit",),
    )
    return rows[0]


def should_compute_profile_start(row: dict[str, Any], args: argparse.Namespace) -> bool:
    if row.get("case_id") != "fit_b0_s042b95":
        return False
    q = float_from_csv(row, "profile_true_delta_chi2")
    if q < args.profile_start_q_lt:
        return True
    if q < 0:
        return True
    return False


def run_one_row(
    campaign_row: dict[str, Any],
    fit: dict[str, Any],
    context: dict[str, Any],
    target_func: Any,
    target_std: float,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    t0 = time.perf_counter()
    case_id = str(campaign_row["case_id"])
    dgp = str(campaign_row["dgp"])
    toy_index = int_from_csv(campaign_row, "toy_index")
    seed = int_from_csv(campaign_row, "seed")
    reported_chi2 = float_from_csv(campaign_row, "toy_chi2_min")
    original_q = float_from_csv(campaign_row, "profile_true_delta_chi2")
    profile_chi2_csv = float_from_csv(campaign_row, "profile_true_chi2")

    y_toy, _sample_info = round19.make_toy(seed, fit, context, dgp)
    audit = round19.ToyAudit(fit, target_func, target_std, y_toy)
    truth_params = np.asarray(fit["param_values"], dtype=np.float64)
    truth_target = float(target_func(jnp.asarray(truth_params, dtype=jnp.float64)))
    reference_start = np.asarray(fit["fitted_values"], dtype=np.float64)

    base = {
        "dgp": dgp,
        "case_id": case_id,
        "stratum": campaign_row.get("stratum"),
        "label": campaign_row.get("label"),
        "target": campaign_row.get("target"),
        "toy_index": toy_index,
        "seed": seed,
    }

    attempts: list[dict[str, Any]] = []
    reference_attempt, x_hat = run_reference_minuit(audit, reference_start, truth_target, reported_chi2)
    reference_chi2 = float(reference_attempt["result_chi2"])
    attempts.append(prefixed_attempt(base, reference_attempt, reported_chi2, reference_chi2))

    candidate_starts: list[tuple[str, np.ndarray, tuple[str, ...]]] = [
        ("returned_mle_restart", x_hat, ("minuit", "scipy")),
    ]
    profile_chi2_recomputed: float | None = None
    profile_start_used = False
    if should_compute_profile_start(campaign_row, args):
        profile_row, profile_x = audit.run_profile_from_start("reported_refit_mle", x_hat, truth_target)
        if profile_row.get("result_chi2") is not None:
            profile_chi2_recomputed = float(profile_row["result_chi2"])
            profile_start_used = True
            candidate_starts.append(("fixed_truth_profile_solution", profile_x, ("minuit", "scipy")))

    if case_id == "fit_b0_s042b95":
        for start_name, start in b0_floor_starts(audit, x_hat):
            candidate_starts.append((start_name, start, ("minuit",)))

    if not args.skip_truth_physical_start and case_id == "fit_b0_s042b95":
        add = []
        add_start(add, "truth_physical_params", audit.physical_to_fitted(truth_params))
        for start_name, start in add:
            candidate_starts.append((start_name, start, ("minuit",)))

    best_chi2 = reference_chi2
    for start_name, start, optimizers in candidate_starts:
        for attempt_row, _x_result in round19.run_unconstrained_from_start(
            audit,
            start_name,
            start,
            truth_target,
            reported_chi2,
            best_chi2,
            optimizers,
        ):
            result_chi2 = finite_or_none(attempt_row.get("result_chi2"))
            if result_chi2 is not None:
                best_chi2 = min(best_chi2, result_chi2)
            attempts.append(prefixed_attempt(base, attempt_row, reported_chi2, reference_chi2))

    finite_attempts = [attempt for attempt in attempts if attempt.get("result_chi2") is not None]
    best = min(finite_attempts, key=lambda row: float(row["result_chi2"]))
    best_chi2 = float(best["result_chi2"])
    q_after = profile_chi2_csv - best_chi2
    chart_mle = round15.chart_summary(fit, x_hat)
    best_chart = {
        "min_decay_param_best": best.get("min_decay_param"),
        "max_decay_param_best": best.get("max_decay_param"),
        "max_abs_decay_fitted_coord_best": best.get("max_abs_decay_fitted_coord"),
    }
    improvement_vs_reported = reported_chi2 - best_chi2
    improvement_vs_reference = reference_chi2 - best_chi2
    row = {
        **base,
        "selection_reason": campaign_row.get("selection_reason"),
        "status": "ok",
        "runtime_sec": time.perf_counter() - t0,
        "original_q": original_q,
        "reported_chi2": reported_chi2,
        "reference_rerun_chi2": reference_chi2,
        "reference_rerun_minus_reported": reference_chi2 - reported_chi2,
        "reference_rerun_valid": reference_attempt.get("valid"),
        "reference_rerun_accurate": reference_attempt.get("accurate"),
        "reference_rerun_edm": reference_attempt.get("edm"),
        "profile_true_chi2_csv": profile_chi2_csv,
        "profile_true_chi2_recomputed": profile_chi2_recomputed,
        "profile_true_chi2_used": profile_chi2_csv,
        "profile_true_recomputed_minus_csv": (
            profile_chi2_recomputed - profile_chi2_csv
            if profile_chi2_recomputed is not None
            else None
        ),
        "profile_start_used": profile_start_used,
        "best_restart_chi2": best_chi2,
        "best_restart_start": best.get("start_name"),
        "best_restart_optimizer": best.get("optimizer"),
        "best_restart_success": best.get("success"),
        "best_restart_valid": best.get("valid"),
        "best_restart_accurate": best.get("accurate"),
        "best_restart_edm": best.get("edm"),
        "best_restart_nfev": best.get("nfev"),
        "best_restart_nfcn": best.get("nfcn"),
        "improvement_vs_reported": improvement_vs_reported,
        "improvement_vs_reference_rerun": improvement_vs_reference,
        "q_after_best_refit": q_after,
        "profile_consistency_failure_original": original_q < -1e-4,
        "profile_consistency_failure_after_best": q_after < -1e-4,
        "profile_consistency_failure_disappears": (original_q < -1e-4 and q_after >= -1e-4),
        "lowered_gt_1e_4": improvement_vs_reported > 1e-4,
        "lowered_gt_1e_3": improvement_vs_reported > 1e-3,
        "lowered_gt_1e_2": improvement_vs_reported > 1e-2,
        "min_decay_param_mle": chart_mle.get("min_decay_param_mle"),
        "max_decay_param_mle": chart_mle.get("max_decay_param_mle"),
        "max_abs_decay_fitted_coord_mle": chart_mle.get("max_abs_decay_fitted_coord_mle"),
        **best_chart,
        "chart_saturation_warning_mle_csv": bool_from_csv(campaign_row.get("chart_saturation_warning_mle")),
        "n_attempts": len(attempts),
        "n_finite_attempts": len(finite_attempts),
        "n_failed_attempts": len(attempts) - len(finite_attempts),
        "exception_type": None,
        "exception": None,
    }
    return row, attempts


def summarize(row_csv: Path, attempt_csv: Path, failure_csv: Path, selection_csv: Path, args: argparse.Namespace) -> dict[str, Any]:
    rows = []
    if row_csv.exists() and row_csv.stat().st_size > 0:
        with row_csv.open("r", encoding="utf-8", newline="") as handle:
            rows = [row for row in csv.DictReader(handle) if row.get("status") == "ok"]
    attempts = []
    if attempt_csv.exists() and attempt_csv.stat().st_size > 0:
        with attempt_csv.open("r", encoding="utf-8", newline="") as handle:
            attempts = list(csv.DictReader(handle))
    failures = []
    if failure_csv.exists() and failure_csv.stat().st_size > 0:
        with failure_csv.open("r", encoding="utf-8", newline="") as handle:
            failures = list(csv.DictReader(handle))
    selected = []
    if selection_csv.exists() and selection_csv.stat().st_size > 0:
        with selection_csv.open("r", encoding="utf-8", newline="") as handle:
            selected = list(csv.DictReader(handle))

    def as_float(row: dict[str, Any], key: str) -> float:
        try:
            return float(row.get(key) or "nan")
        except ValueError:
            return math.nan

    thresholds = (1e-4, 1e-3, 1e-2)

    def threshold_counts(group: list[dict[str, Any]]) -> dict[str, int]:
        return {
            f"lowered_gt_{threshold:g}": int(sum(as_float(row, "improvement_vs_reported") > threshold for row in group))
            for threshold in thresholds
        }

    group_summaries = []
    group_keys = sorted({(row["case_id"], row["dgp"]) for row in rows})
    for case_id, dgp in group_keys:
        group = [row for row in rows if row.get("case_id") == case_id and row.get("dgp") == dgp]
        improvements = np.asarray([as_float(row, "improvement_vs_reported") for row in group], dtype=float)
        improvements = improvements[np.isfinite(improvements)]
        q_values = np.asarray([as_float(row, "original_q") for row in group], dtype=float)
        q_values = q_values[np.isfinite(q_values)]
        group_summaries.append(
            {
                "case_id": case_id,
                "dgp": dgp,
                "n_rows": len(group),
                **threshold_counts(group),
                "max_improvement": finite_or_none(np.max(improvements)) if len(improvements) else None,
                "median_improvement": finite_or_none(np.median(improvements)) if len(improvements) else None,
                "min_original_q": finite_or_none(np.min(q_values)) if len(q_values) else None,
                "profile_consistency_failures_original": int(
                    sum(bool_from_csv(row.get("profile_consistency_failure_original")) for row in group)
                ),
                "profile_consistency_failures_after_best": int(
                    sum(bool_from_csv(row.get("profile_consistency_failure_after_best")) for row in group)
                ),
                "best_rows": [
                    {
                        "toy_index": int(row["toy_index"]),
                        "seed": int(row["seed"]),
                        "original_q": as_float(row, "original_q"),
                        "improvement": as_float(row, "improvement_vs_reported"),
                        "q_after_best_refit": as_float(row, "q_after_best_refit"),
                        "best_start": row.get("best_restart_start"),
                        "best_optimizer": row.get("best_restart_optimizer"),
                        "min_decay_param_mle": as_float(row, "min_decay_param_mle"),
                        "max_abs_decay_fitted_coord_mle": as_float(row, "max_abs_decay_fitted_coord_mle"),
                    }
                    for row in sorted(group, key=lambda r: as_float(r, "improvement_vs_reported"), reverse=True)[:5]
                ],
            }
        )

    selection_counts = Counter(row.get("selection_reason", "") for row in selected)
    attempt_counts = Counter(
        (row.get("start_name", ""), row.get("optimizer", ""))
        for row in attempts
    )
    all_improvements = np.asarray([as_float(row, "improvement_vs_reported") for row in rows], dtype=float)
    all_improvements = all_improvements[np.isfinite(all_improvements)]
    b0_rows = [row for row in rows if row.get("case_id") == "fit_b0_s042b95"]
    non_b0_rows = [row for row in rows if row.get("case_id") != "fit_b0_s042b95"]
    clean_rows = [row for row in rows if row.get("stratum") == "clean_control"]

    return {
        "metadata": {
            "campaign_csv": args.campaign_csv,
            "b0_split_q_lt": args.b0_split_q_lt,
            "b0_local_q_lt": args.b0_local_q_lt,
            "profile_start_q_lt": args.profile_start_q_lt,
            "thresholds": list(thresholds),
            "notes": "No endpoint searches are run by this harness.",
        },
        "row_counts": {
            "selected_rows": len(selected),
            "completed_rows": len(rows),
            "attempt_rows": len(attempts),
            "failure_rows": len(failures),
            "b0_rows": len(b0_rows),
            "non_b0_rows": len(non_b0_rows),
            "clean_control_rows": len(clean_rows),
        },
        "overall": {
            **threshold_counts(rows),
            "max_improvement": finite_or_none(np.max(all_improvements)) if len(all_improvements) else None,
            "median_improvement": finite_or_none(np.median(all_improvements)) if len(all_improvements) else None,
            "profile_consistency_failures_original": int(
                sum(bool_from_csv(row.get("profile_consistency_failure_original")) for row in rows)
            ),
            "profile_consistency_failures_after_best": int(
                sum(bool_from_csv(row.get("profile_consistency_failure_after_best")) for row in rows)
            ),
            "profile_consistency_failures_disappear": int(
                sum(bool_from_csv(row.get("profile_consistency_failure_disappears")) for row in rows)
            ),
        },
        "b0": threshold_counts(b0_rows),
        "non_b0": threshold_counts(non_b0_rows),
        "clean_controls": threshold_counts(clean_rows),
        "group_summaries": group_summaries,
        "selection_reason_counts": dict(selection_counts),
        "attempt_counts": {f"{k[0]}::{k[1]}": v for k, v in attempt_counts.items()},
        "failure_rows": failures[:20],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-csv", default="notes/codex-refactor/round18_calibration_campaign_v1_toys.csv")
    parser.add_argument("--output-prefix", default="notes/codex-refactor/round20_b0_refit_stability_audit")
    parser.add_argument("--b0-split-q-lt", type=float, default=0.02)
    parser.add_argument("--b0-local-q-lt", type=float, default=0.02)
    parser.add_argument("--b0-saturation-reps-per-dgp", type=int, default=10)
    parser.add_argument("--b0-mid-q-reps-per-dgp", type=int, default=4)
    parser.add_argument("--neighbor-radius", type=int, default=1)
    parser.add_argument("--comparison-small-q-per-case-dgp", type=int, default=4)
    parser.add_argument("--clean-q-near-1-per-case-dgp", type=int, default=2)
    parser.add_argument("--profile-start-q-lt", type=float, default=0.001)
    parser.add_argument("--skip-truth-physical-start", action="store_true")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    prefix = Path(args.output_prefix)
    selection_csv = prefix.with_name(prefix.name + "_selection.csv")
    selection_jsonl = prefix.with_name(prefix.name + "_selection.jsonl")
    row_csv = prefix.with_name(prefix.name + "_rows.csv")
    row_jsonl = prefix.with_name(prefix.name + "_rows.jsonl")
    attempt_csv = prefix.with_name(prefix.name + "_attempts.csv")
    attempt_jsonl = prefix.with_name(prefix.name + "_attempts.jsonl")
    failure_csv = prefix.with_name(prefix.name + "_failures.csv")
    failure_jsonl = prefix.with_name(prefix.name + "_failures.jsonl")
    summary_json = prefix.with_name(prefix.name + "_summary.json")

    output_paths = [
        selection_csv,
        selection_jsonl,
        row_csv,
        row_jsonl,
        attempt_csv,
        attempt_jsonl,
        failure_csv,
        failure_jsonl,
        summary_json,
    ]
    if args.overwrite:
        reset_outputs(output_paths)

    ensure_artifact(row_csv, ROW_FIELDNAMES)
    ensure_artifact(row_jsonl, ROW_FIELDNAMES, jsonl=True)
    ensure_artifact(attempt_csv, ATTEMPT_FIELDNAMES)
    ensure_artifact(attempt_jsonl, ATTEMPT_FIELDNAMES, jsonl=True)
    ensure_artifact(failure_csv, FAILURE_FIELDNAMES)
    ensure_artifact(failure_jsonl, FAILURE_FIELDNAMES, jsonl=True)

    selected_rows = build_selection(args)
    if args.max_rows is not None:
        selected_rows = selected_rows[: args.max_rows]
    if args.overwrite or not selection_csv.exists() or selection_csv.stat().st_size == 0:
        write_selection(selected_rows, selection_csv, selection_jsonl)

    done = completed_keys(row_csv)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in selected_rows:
        grouped[str(row["case_id"])].append(row)

    for case_id, case_rows in grouped.items():
        case = CASE_SPECS[case_id]
        print(f"loading {case_id}: {case.label}::{case.target} ({len(case_rows)} selected)", flush=True)
        with contextlib.redirect_stdout(io.StringIO()):
            fit = run_fit(case.label, verbose=False)
            if fit is None:
                raise RuntimeError(f"run_fit returned None for {case.label}")
            context = round15.build_fit_context(case.label, fit)
        target_func = round15.target_func_for_fit(fit, case.target)
        truth_target = float(target_func(jnp.asarray(fit["param_values"], dtype=jnp.float64)))
        target_std = round15.target_std_for_fit(fit, target_func, truth_target)
        print(
            f"case {case_id}: npar={len(fit['parameters'])} nmeas={len(context['y_observed'])} "
            f"truth={truth_target:.8g} target_std={target_std:.4g}",
            flush=True,
        )

        for campaign_row in case_rows:
            key = selection_key(campaign_row)
            if key in done:
                continue
            t0 = time.perf_counter()
            try:
                row, attempts = run_one_row(campaign_row, fit, context, target_func, target_std, args)
                append_row(row_csv, row, ROW_FIELDNAMES)
                append_row(row_jsonl, row, ROW_FIELDNAMES, jsonl=True)
                for attempt in attempts:
                    append_row(attempt_csv, attempt, ATTEMPT_FIELDNAMES)
                    append_row(attempt_jsonl, attempt, ATTEMPT_FIELDNAMES, jsonl=True)
                print(
                    f"  {row['dgp']} toy {row['toy_index']}: q={row['original_q']:.6g} "
                    f"improvement={row['improvement_vs_reported']:.6g} "
                    f"best={row['best_restart_start']}/{row['best_restart_optimizer']}",
                    flush=True,
                )
            except Exception as exc:  # noqa: BLE001 - artifact records exact failure
                failure = {
                    "selection_reason": campaign_row.get("selection_reason"),
                    "dgp": campaign_row.get("dgp"),
                    "case_id": case_id,
                    "stratum": campaign_row.get("stratum"),
                    "label": campaign_row.get("label"),
                    "target": campaign_row.get("target"),
                    "toy_index": int_from_csv(campaign_row, "toy_index"),
                    "seed": int_from_csv(campaign_row, "seed"),
                    "phase": "refit_audit",
                    "exception_type": type(exc).__name__,
                    "exception": round15.clean_message(exc, limit=1000),
                    "runtime_sec": time.perf_counter() - t0,
                }
                append_row(failure_csv, failure, FAILURE_FIELDNAMES)
                append_row(failure_jsonl, failure, FAILURE_FIELDNAMES, jsonl=True)
                print(
                    f"  {campaign_row['dgp']} toy {campaign_row['toy_index']}: "
                    f"FAILED {type(exc).__name__}: {exc}",
                    flush=True,
                )

    summary = summarize(row_csv, attempt_csv, failure_csv, selection_csv, args)
    summary_json.write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=json_default) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True, default=json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
