#!/usr/bin/env python
"""Round 21 guarded B0/BR-BRU unconstrained-refit policy comparator.

This is a notes-only harness.  It expands the round-20 refit/globality audit to
all round-18 B0::S042B95 toys and evaluates policy variants using only starts
that a production refit could have available: the returned MLE and deterministic
decay-floor/tiny-parameter perturbations.  Fixed-truth profile starts are
deliberately disabled here because production cannot use toy truth.
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
from typing import Any, Callable

import jax
from jax import numpy as jnp
import numpy as np

import run_round15_wilks_calibration_smoke as round15
import run_round16_calibration_pilot as round16
import run_round19_b0_consistency_audit as round19
import run_round20_b0_refit_stability_audit as round20

from pdgfits.fit import run_fit


jax.config.update("jax_enable_x64", True)


# Round 20 used fixed-truth profile starts diagnostically for small-q B0 rows.
# Round 21 is a production-policy comparator, so disable that unavailable start.
round20.should_compute_profile_start = lambda _row, _args: False


POLICY_FIELDNAMES = [
    "selection_reason",
    "dgp",
    "case_id",
    "stratum",
    "label",
    "target",
    "toy_index",
    "seed",
    "policy",
    "trigger_name",
    "trigger_fired",
    "production_eligible",
    "n_extra_attempts",
    "extra_attempt_names",
    "best_policy_chi2",
    "best_policy_start",
    "best_policy_optimizer",
    "improvement_vs_reported",
    "improvement_vs_reference_rerun",
    "original_q",
    "q_after_policy",
    "q_shift",
    "lowered_gt_1e_4",
    "lowered_gt_1e_3",
    "lowered_gt_1e_2",
    "min_decay_param_mle",
    "max_abs_decay_fitted_coord_mle",
    "returned_mle_best_improvement",
    "loose_saturation_trigger",
    "extreme_saturation_trigger",
    "cheap_hint_gt_1e_5_trigger",
    "cheap_hint_gt_1e_4_trigger",
    "oracle_small_q_trigger",
    "runtime_sec",
]

ACTUAL_FIELDNAMES = [
    "case",
    "policy",
    "trigger_name",
    "trigger_fired",
    "production_eligible",
    "n_extra_attempts",
    "extra_attempt_names",
    "reported_chi2_min",
    "best_policy_chi2",
    "best_policy_start",
    "best_policy_optimizer",
    "improvement_vs_reported",
    "min_decay_param_mle",
    "max_abs_decay_fitted_coord_mle",
    "returned_mle_best_improvement",
    "loose_saturation_trigger",
    "extreme_saturation_trigger",
    "cheap_hint_gt_1e_5_trigger",
    "cheap_hint_gt_1e_4_trigger",
]


def clean_value(value: Any) -> Any:
    return round15.clean_value(value)


def json_default(obj: Any) -> Any:
    return round15.json_default(obj)


def finite_or_none(value: Any) -> float | None:
    return round15.finite_or_none(value)


def as_float(row: dict[str, Any], key: str, default: float = math.nan) -> float:
    try:
        value = row.get(key)
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def as_int(row: dict[str, Any], key: str, default: int = 0) -> int:
    try:
        return int(str(row[key]))
    except (KeyError, TypeError, ValueError):
        return default


def as_bool(value: Any) -> bool:
    return str(value).strip().lower() == "true"


def row_key(row: dict[str, Any]) -> tuple[str, str, int]:
    return (str(row["dgp"]), str(row["case_id"]), as_int(row, "toy_index"))


def read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=json_default) + "\n",
        encoding="utf-8",
    )


def closest_by_q(rows: list[dict[str, Any]], q_target: float, n: int) -> list[dict[str, Any]]:
    finite = [row for row in rows if math.isfinite(as_float(row, "profile_true_delta_chi2"))]
    return sorted(finite, key=lambda row: abs(as_float(row, "profile_true_delta_chi2") - q_target))[:n]


def add_selection(
    selected: dict[tuple[str, str, int], dict[str, Any]],
    row: dict[str, Any],
    reason: str,
) -> None:
    key = row_key(row)
    if key in selected:
        selected[key]["selection_reason"] += f";{reason}"
        return
    selected[key] = {**row, "selection_reason": reason}


def build_selection(args: argparse.Namespace) -> list[dict[str, Any]]:
    campaign_rows = [
        row
        for row in round20.read_campaign_rows(Path(args.campaign_csv))
        if row.get("status") == "ok"
    ]
    selected: dict[tuple[str, str, int], dict[str, Any]] = {}

    for row in campaign_rows:
        if row.get("case_id") == "fit_b0_s042b95":
            add_selection(selected, row, "all_b0_s042b95_round18_campaign")

    for case_id in ("fit_b0s_s08637", "fit_eta_m026g01"):
        for dgp in round16.DGP_CHOICES:
            group = [
                row
                for row in campaign_rows
                if row.get("case_id") == case_id and row.get("dgp") == dgp
            ]
            for row in sorted(group, key=lambda item: as_float(item, "profile_true_delta_chi2"))[
                : args.profile_shape_small_q_per_case_dgp
            ]:
                add_selection(selected, row, f"profile_shape_small_q_{args.profile_shape_small_q_per_case_dgp}")
            for row in closest_by_q(group, 1.0, args.profile_shape_q_near_1_per_case_dgp):
                add_selection(selected, row, f"profile_shape_q_near_1_{args.profile_shape_q_near_1_per_case_dgp}")

    for case_id in ("fit_g2000_k002m", "fit_g2000_k003m"):
        for dgp in round16.DGP_CHOICES:
            group = [
                row
                for row in campaign_rows
                if row.get("case_id") == case_id and row.get("dgp") == dgp
            ]
            for row in sorted(group, key=lambda item: as_float(item, "profile_true_delta_chi2"))[
                : args.clean_small_q_per_case_dgp
            ]:
                add_selection(selected, row, f"clean_small_q_{args.clean_small_q_per_case_dgp}")
            for row in closest_by_q(group, 1.0, args.clean_q_near_1_per_case_dgp):
                add_selection(selected, row, f"clean_q_near_1_{args.clean_q_near_1_per_case_dgp}")

    selected_rows = list(selected.values())
    selected_rows.sort(
        key=lambda row: (
            round20.CASE_INDEX.get(str(row.get("case_id")), 999),
            str(row.get("dgp")),
            as_int(row, "toy_index"),
        )
    )
    if args.max_rows is not None:
        selected_rows = selected_rows[: args.max_rows]
    return selected_rows


def write_selection(rows: list[dict[str, Any]], csv_path: Path, jsonl_path: Path) -> None:
    round20.write_selection(rows, csv_path, jsonl_path)


def attempt_key(row: dict[str, Any]) -> tuple[str, str, int]:
    return (str(row["dgp"]), str(row["case_id"]), as_int(row, "toy_index"))


def is_returned_attempt(attempt: dict[str, Any]) -> bool:
    return attempt.get("start_name") == "returned_mle_restart"


def is_floor_attempt(attempt: dict[str, Any]) -> bool:
    name = str(attempt.get("start_name") or "")
    return name.startswith("clip_decay_floor_") or name.startswith("tiny_decay_factor_")


def finite_attempts(attempts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [attempt for attempt in attempts if math.isfinite(as_float(attempt, "result_chi2"))]


def best_attempt(
    baseline_chi2: float,
    attempts: list[dict[str, Any]],
) -> tuple[float, str, str]:
    finite = finite_attempts(attempts)
    if not finite:
        return baseline_chi2, "baseline_reported", ""
    best = min(finite, key=lambda row: as_float(row, "result_chi2"))
    best_chi2 = as_float(best, "result_chi2")
    if best_chi2 < baseline_chi2:
        return best_chi2, str(best.get("start_name") or ""), str(best.get("optimizer") or "")
    return baseline_chi2, "baseline_reported", ""


def trigger_values(row: dict[str, Any], attempts: list[dict[str, Any]]) -> dict[str, Any]:
    min_decay = as_float(row, "min_decay_param_mle")
    max_coord = as_float(row, "max_abs_decay_fitted_coord_mle")
    has_decay = math.isfinite(min_decay) and math.isfinite(max_coord)
    returned_best_chi2, _start, _optimizer = best_attempt(
        as_float(row, "reported_chi2"),
        [attempt for attempt in attempts if is_returned_attempt(attempt)],
    )
    returned_improvement = as_float(row, "reported_chi2") - returned_best_chi2
    loose = bool(has_decay and min_decay <= 1e-6 and max_coord >= 50.0)
    extreme = bool(has_decay and min_decay <= 5e-7 and max_coord >= 80.0)
    cheap_1e5 = bool(loose and returned_improvement > 1e-5)
    cheap_1e4 = bool(loose and returned_improvement > 1e-4)
    oracle_small_q = bool(loose and as_float(row, "original_q") < 0.02)
    return {
        "returned_mle_best_improvement": returned_improvement,
        "loose_saturation_trigger": loose,
        "extreme_saturation_trigger": extreme,
        "cheap_hint_gt_1e_5_trigger": cheap_1e5,
        "cheap_hint_gt_1e_4_trigger": cheap_1e4,
        "oracle_small_q_trigger": oracle_small_q,
    }


def policies_for_row(row: dict[str, Any], attempts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    triggers = trigger_values(row, attempts)
    returned = [attempt for attempt in attempts if is_returned_attempt(attempt)]
    floor = [attempt for attempt in attempts if is_floor_attempt(attempt)]
    baseline_chi2 = as_float(row, "reported_chi2")
    reference_chi2 = as_float(row, "reference_rerun_chi2")
    profile_chi2 = as_float(row, "profile_true_chi2_used")

    definitions: list[tuple[str, str, bool, bool, list[dict[str, Any]]]] = [
        ("baseline_reported", "none", False, True, []),
        ("returned_mle_restart", "always", True, True, returned),
        (
            "loose_saturation_floor_only",
            "loose_saturation",
            triggers["loose_saturation_trigger"],
            True,
            floor if triggers["loose_saturation_trigger"] else [],
        ),
        (
            "loose_saturation_returned_plus_floor",
            "loose_saturation",
            triggers["loose_saturation_trigger"],
            True,
            returned + (floor if triggers["loose_saturation_trigger"] else []),
        ),
        (
            "extreme_saturation_returned_plus_floor",
            "extreme_saturation",
            triggers["extreme_saturation_trigger"],
            True,
            returned + (floor if triggers["extreme_saturation_trigger"] else []),
        ),
        (
            "cheap_hint_1e_5_returned_then_floor",
            "loose_saturation_and_returned_improvement_gt_1e-5",
            triggers["cheap_hint_gt_1e_5_trigger"],
            True,
            returned + (floor if triggers["cheap_hint_gt_1e_5_trigger"] else []),
        ),
        (
            "cheap_hint_1e_4_returned_then_floor",
            "loose_saturation_and_returned_improvement_gt_1e-4",
            triggers["cheap_hint_gt_1e_4_trigger"],
            True,
            returned + (floor if triggers["cheap_hint_gt_1e_4_trigger"] else []),
        ),
        (
            "oracle_small_q_returned_plus_floor",
            "loose_saturation_and_profile_q_lt_0.02",
            triggers["oracle_small_q_trigger"],
            False,
            returned + (floor if triggers["oracle_small_q_trigger"] else []),
        ),
    ]

    out = []
    for policy, trigger_name, trigger_fired, production_eligible, selected_attempts in definitions:
        best_chi2, best_start, best_optimizer = best_attempt(baseline_chi2, selected_attempts)
        q_after = profile_chi2 - best_chi2 if math.isfinite(profile_chi2) else math.nan
        original_q = as_float(row, "original_q")
        improvement = baseline_chi2 - best_chi2
        extra_names = sorted(
            {
                f"{attempt.get('start_name')}::{attempt.get('optimizer')}"
                for attempt in selected_attempts
            }
        )
        out.append(
            {
                "selection_reason": row.get("selection_reason"),
                "dgp": row.get("dgp"),
                "case_id": row.get("case_id"),
                "stratum": row.get("stratum"),
                "label": row.get("label"),
                "target": row.get("target"),
                "toy_index": as_int(row, "toy_index"),
                "seed": as_int(row, "seed"),
                "policy": policy,
                "trigger_name": trigger_name,
                "trigger_fired": bool(trigger_fired),
                "production_eligible": bool(production_eligible),
                "n_extra_attempts": len(selected_attempts),
                "extra_attempt_names": ";".join(extra_names),
                "best_policy_chi2": best_chi2,
                "best_policy_start": best_start,
                "best_policy_optimizer": best_optimizer,
                "improvement_vs_reported": improvement,
                "improvement_vs_reference_rerun": reference_chi2 - best_chi2,
                "original_q": original_q,
                "q_after_policy": q_after,
                "q_shift": q_after - original_q if math.isfinite(q_after) and math.isfinite(original_q) else None,
                "lowered_gt_1e_4": improvement > 1e-4,
                "lowered_gt_1e_3": improvement > 1e-3,
                "lowered_gt_1e_2": improvement > 1e-2,
                "min_decay_param_mle": as_float(row, "min_decay_param_mle"),
                "max_abs_decay_fitted_coord_mle": as_float(row, "max_abs_decay_fitted_coord_mle"),
                **triggers,
                "runtime_sec": as_float(row, "runtime_sec"),
            }
        )
    return out


def write_policy_rows(row_csv: Path, attempt_csv: Path, policy_csv: Path, policy_jsonl: Path) -> list[dict[str, Any]]:
    rows = [row for row in read_csv(row_csv) if row.get("status") == "ok"]
    attempts_by_key: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for attempt in read_csv(attempt_csv):
        attempts_by_key[attempt_key(attempt)].append(attempt)

    if policy_csv.exists():
        policy_csv.unlink()
    if policy_jsonl.exists():
        policy_jsonl.unlink()
    round15.ensure_artifact(policy_csv, POLICY_FIELDNAMES)
    round15.ensure_artifact(policy_jsonl, POLICY_FIELDNAMES, jsonl=True)

    policy_rows: list[dict[str, Any]] = []
    for row in rows:
        for policy_row in policies_for_row(row, attempts_by_key.get(row_key(row), [])):
            round15.append_row(policy_csv, policy_row, POLICY_FIELDNAMES)
            round15.append_row(policy_jsonl, policy_row, POLICY_FIELDNAMES, jsonl=True)
            policy_rows.append(policy_row)
    return policy_rows


def threshold_counts(q_values: np.ndarray) -> dict[str, Any]:
    q_values = q_values[np.isfinite(q_values)]
    out: dict[str, Any] = {"n": int(len(q_values))}
    for name, (direction, threshold) in round16.Q_THRESHOLDS.items():
        if direction == "le":
            count = int(np.sum(q_values <= threshold))
        else:
            count = int(np.sum(q_values > threshold))
        out[f"{name}_count"] = count
        out[f"{name}_fraction"] = finite_or_none(count / len(q_values)) if len(q_values) else None
        out[f"{name}_threshold"] = threshold
    return out


def policy_group_name(row: dict[str, Any]) -> str:
    case_id = row.get("case_id")
    if case_id == "fit_b0_s042b95":
        return "b0_all_campaign"
    if row.get("stratum") == "clean_control":
        return "clean_controls"
    return "b0s_eta_profile_shape"


def summarize(
    row_csv: Path,
    attempt_csv: Path,
    failure_csv: Path,
    selection_csv: Path,
    policy_rows: list[dict[str, Any]],
    actual_rows: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    rows = [row for row in read_csv(row_csv) if row.get("status") == "ok"]
    attempts = read_csv(attempt_csv)
    failures = read_csv(failure_csv)
    selected = read_csv(selection_csv)
    thresholds = (1e-4, 1e-3, 1e-2)

    def gt_counts(group: list[dict[str, Any]]) -> dict[str, int]:
        return {
            f"lowered_gt_{threshold:g}": int(
                sum(as_float(row, "improvement_vs_reported") > threshold for row in group)
            )
            for threshold in thresholds
        }

    policy_summaries = []
    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in policy_rows:
        groups[(policy_group_name(row), str(row.get("case_id")), str(row.get("dgp")), str(row.get("policy")))].append(row)

    for (group_name, case_id, dgp, policy), group in sorted(groups.items()):
        improvements = np.asarray([as_float(row, "improvement_vs_reported") for row in group], dtype=float)
        improvements = improvements[np.isfinite(improvements)]
        q_original = np.asarray([as_float(row, "original_q") for row in group], dtype=float)
        q_after = np.asarray([as_float(row, "q_after_policy") for row in group], dtype=float)
        trigger_count = int(sum(as_bool(row.get("trigger_fired")) for row in group))
        extra_attempt_count = int(sum(as_int(row, "n_extra_attempts") for row in group))
        policy_summaries.append(
            {
                "group": group_name,
                "case_id": case_id,
                "dgp": dgp,
                "policy": policy,
                "n_rows": len(group),
                "trigger_count": trigger_count,
                "production_eligible": as_bool(group[0].get("production_eligible")) if group else None,
                "extra_attempt_count": extra_attempt_count,
                "extra_attempts_per_row": finite_or_none(extra_attempt_count / len(group)) if group else None,
                **gt_counts(group),
                "max_improvement": finite_or_none(np.max(improvements)) if len(improvements) else None,
                "median_improvement": finite_or_none(np.median(improvements)) if len(improvements) else None,
                "original_q_thresholds": threshold_counts(q_original),
                "corrected_q_thresholds": threshold_counts(q_after),
                "threshold_count_deltas": {
                    key: threshold_counts(q_after).get(key) - threshold_counts(q_original).get(key)
                    for key in threshold_counts(q_original)
                    if key.endswith("_count")
                },
            }
        )

    material_policy = "loose_saturation_returned_plus_floor"
    material_rows = [
        row
        for row in policy_rows
        if row.get("policy") == material_policy and as_float(row, "improvement_vs_reported") > 1e-4
    ]
    material_keys = {(row.get("dgp"), row.get("case_id"), row.get("toy_index")) for row in material_rows}
    catches_by_row = []
    by_material_key: dict[tuple[Any, Any, Any], list[dict[str, Any]]] = defaultdict(list)
    for row in policy_rows:
        key = (row.get("dgp"), row.get("case_id"), row.get("toy_index"))
        if key in material_keys:
            by_material_key[key].append(row)
    for key, group in sorted(by_material_key.items(), key=lambda item: (str(item[0][1]), str(item[0][0]), int(item[0][2]))):
        best_full = [row for row in group if row.get("policy") == material_policy][0]
        catches_by_row.append(
            {
                "dgp": key[0],
                "case_id": key[1],
                "toy_index": key[2],
                "original_q": as_float(best_full, "original_q"),
                "full_policy_improvement": as_float(best_full, "improvement_vs_reported"),
                "q_after_full_policy": as_float(best_full, "q_after_policy"),
                "min_decay_param_mle": as_float(best_full, "min_decay_param_mle"),
                "max_abs_decay_fitted_coord_mle": as_float(best_full, "max_abs_decay_fitted_coord_mle"),
                "policy_improvements": {
                    str(row.get("policy")): as_float(row, "improvement_vs_reported")
                    for row in sorted(group, key=lambda item: str(item.get("policy")))
                },
                "policy_best_starts": {
                    str(row.get("policy")): f"{row.get('best_policy_start')}::{row.get('best_policy_optimizer')}"
                    for row in sorted(group, key=lambda item: str(item.get("policy")))
                },
                "production_triggers": {
                    "loose_saturation": as_bool(best_full.get("loose_saturation_trigger")),
                    "extreme_saturation": as_bool(best_full.get("extreme_saturation_trigger")),
                    "cheap_hint_gt_1e-5": as_bool(best_full.get("cheap_hint_gt_1e_5_trigger")),
                    "cheap_hint_gt_1e-4": as_bool(best_full.get("cheap_hint_gt_1e_4_trigger")),
                },
                "oracle_small_q": as_bool(best_full.get("oracle_small_q_trigger")),
            }
        )

    rows_by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_group[policy_group_name(row)].append(row)

    runtime_by_group = {
        group: {
            "n_rows": len(group_rows),
            "total_runtime_sec": finite_or_none(sum(as_float(row, "runtime_sec", 0.0) for row in group_rows)),
            "mean_runtime_sec": finite_or_none(
                sum(as_float(row, "runtime_sec", 0.0) for row in group_rows) / len(group_rows)
            )
            if group_rows
            else None,
            "median_runtime_sec": finite_or_none(
                np.median([as_float(row, "runtime_sec", 0.0) for row in group_rows])
            )
            if group_rows
            else None,
        }
        for group, group_rows in rows_by_group.items()
    }

    attempt_counts = Counter((row.get("start_name", ""), row.get("optimizer", "")) for row in attempts)
    selection_counts = Counter(row.get("selection_reason", "") for row in selected)
    clean_policy_rows = [
        row
        for row in policy_rows
        if row.get("policy") == material_policy and row.get("stratum") == "clean_control"
    ]
    actual_summary = {
        row["policy"]: {
            "improvement_vs_reported": as_float(row, "improvement_vs_reported"),
            "trigger_fired": as_bool(row.get("trigger_fired")),
            "n_extra_attempts": as_int(row, "n_extra_attempts"),
            "best_start": row.get("best_policy_start"),
            "best_optimizer": row.get("best_policy_optimizer"),
        }
        for row in actual_rows
    }

    return {
        "metadata": {
            "campaign_csv": args.campaign_csv,
            "output_prefix": args.output_prefix,
            "policy_material_reference": material_policy,
            "profile_truth_starts_disabled": True,
            "notes": "No endpoint searches and no fixed-truth profile starts are run by this harness.",
            "production_trigger_definitions": {
                "loose_saturation": "n_decay_params>0, min_decay_param_mle<=1e-6, max_abs_decay_fitted_coord_mle>=50",
                "extreme_saturation": "n_decay_params>0, min_decay_param_mle<=5e-7, max_abs_decay_fitted_coord_mle>=80",
                "cheap_hint": "loose_saturation plus returned-MLE restart/scipy improvement above threshold",
                "oracle_small_q": "loose_saturation plus profile q<0.02; retrospective only, not production eligible",
            },
        },
        "row_counts": {
            "selected_rows": len(selected),
            "completed_rows": len(rows),
            "attempt_rows": len(attempts),
            "policy_rows": len(policy_rows),
            "failure_rows": len(failures),
            "actual_policy_rows": len(actual_rows),
        },
        "selection_reason_counts": dict(selection_counts),
        "attempt_counts": {f"{key[0]}::{key[1]}": value for key, value in attempt_counts.items()},
        "runtime_by_group": runtime_by_group,
        "policy_summaries": policy_summaries,
        "material_rows_caught_by_full_policy": catches_by_row,
        "clean_control_full_policy": {
            **gt_counts(clean_policy_rows),
            "max_improvement": finite_or_none(
                np.max([as_float(row, "improvement_vs_reported") for row in clean_policy_rows])
            )
            if clean_policy_rows
            else None,
        },
        "actual_snapshot_summary": actual_summary,
        "failures": failures[:20],
    }


def actual_snapshot_attempts(
    fit: dict[str, Any],
    context: dict[str, Any],
    target_func: Callable[[Any], Any],
    target_std: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    y_observed = np.asarray(context["y_observed"], dtype=np.float64)
    audit = round19.ToyAudit(fit, target_func, target_std, y_observed)
    truth_params = np.asarray(fit["param_values"], dtype=np.float64)
    truth_target = float(target_func(jnp.asarray(truth_params, dtype=jnp.float64)))
    x_hat = np.asarray(fit["fitted_values"], dtype=np.float64)
    reported_chi2 = audit.chi2_np(x_hat)
    chart = audit.chart_summary(x_hat)
    base = {"dgp": "actual_snapshot", "case_id": "fit_b0_s042b95", "toy_index": -1}

    attempts: list[dict[str, Any]] = []
    best_seen = reported_chi2
    for start_name, start, optimizers in [
        ("returned_mle_restart", x_hat, ("minuit", "scipy")),
        *[(name, start, ("minuit",)) for name, start in round20.b0_floor_starts(audit, x_hat)],
    ]:
        for attempt, _x in round19.run_unconstrained_from_start(
            audit,
            start_name,
            start,
            truth_target,
            reported_chi2,
            best_seen,
            optimizers,
        ):
            result_chi2 = finite_or_none(attempt.get("result_chi2"))
            if result_chi2 is not None:
                best_seen = min(best_seen, result_chi2)
            attempts.append({**base, **attempt})

    row = {
        "selection_reason": "actual_snapshot",
        "dgp": "actual_snapshot",
        "case_id": "fit_b0_s042b95",
        "stratum": "actual_snapshot",
        "label": "B0",
        "target": "S042B95",
        "toy_index": -1,
        "seed": -1,
        "reported_chi2": reported_chi2,
        "reference_rerun_chi2": reported_chi2,
        "profile_true_chi2_used": math.nan,
        "original_q": math.nan,
        "runtime_sec": math.nan,
        "min_decay_param_mle": chart.get("min_decay_param"),
        "max_abs_decay_fitted_coord_mle": chart.get("max_abs_decay_fitted_coord"),
    }
    return attempts, row


def write_actual_policy_artifact(actual_json: Path, actual_csv: Path, actual_jsonl: Path) -> list[dict[str, Any]]:
    payload = json.loads(actual_json.read_text(encoding="utf-8"))
    rows = payload["policy_rows"]
    if actual_csv.exists():
        actual_csv.unlink()
    if actual_jsonl.exists():
        actual_jsonl.unlink()
    round15.ensure_artifact(actual_csv, ACTUAL_FIELDNAMES)
    round15.ensure_artifact(actual_jsonl, ACTUAL_FIELDNAMES, jsonl=True)
    for row in rows:
        round15.append_row(actual_csv, row, ACTUAL_FIELDNAMES)
        round15.append_row(actual_jsonl, row, ACTUAL_FIELDNAMES, jsonl=True)
    return rows


def run_actual_snapshot_check(prefix: Path) -> list[dict[str, Any]]:
    actual_json = prefix.with_name(prefix.name + "_actual_snapshot.json")
    actual_csv = prefix.with_name(prefix.name + "_actual_snapshot_policy_rows.csv")
    actual_jsonl = prefix.with_name(prefix.name + "_actual_snapshot_policy_rows.jsonl")
    print("loading actual snapshot B0::S042B95", flush=True)
    case = round20.CASE_SPECS["fit_b0_s042b95"]
    with contextlib.redirect_stdout(io.StringIO()):
        fit = run_fit(case.label, verbose=False)
        if fit is None:
            raise RuntimeError(f"run_fit returned None for {case.label}")
        context = round15.build_fit_context(case.label, fit)
    target_func = round15.target_func_for_fit(fit, case.target)
    truth_target = float(target_func(jnp.asarray(fit["param_values"], dtype=jnp.float64)))
    target_std = round15.target_std_for_fit(fit, target_func, truth_target)
    attempts, actual_row = actual_snapshot_attempts(fit, context, target_func, target_std)
    policy_rows = []
    for row in policies_for_row(actual_row, attempts):
        policy_rows.append(
            {
                "case": "B0::S042B95",
                "policy": row["policy"],
                "trigger_name": row["trigger_name"],
                "trigger_fired": row["trigger_fired"],
                "production_eligible": row["production_eligible"],
                "n_extra_attempts": row["n_extra_attempts"],
                "extra_attempt_names": row["extra_attempt_names"],
                "reported_chi2_min": actual_row["reported_chi2"],
                "best_policy_chi2": row["best_policy_chi2"],
                "best_policy_start": row["best_policy_start"],
                "best_policy_optimizer": row["best_policy_optimizer"],
                "improvement_vs_reported": row["improvement_vs_reported"],
                "min_decay_param_mle": actual_row["min_decay_param_mle"],
                "max_abs_decay_fitted_coord_mle": actual_row["max_abs_decay_fitted_coord_mle"],
                "returned_mle_best_improvement": row["returned_mle_best_improvement"],
                "loose_saturation_trigger": row["loose_saturation_trigger"],
                "extreme_saturation_trigger": row["extreme_saturation_trigger"],
                "cheap_hint_gt_1e_5_trigger": row["cheap_hint_gt_1e_5_trigger"],
                "cheap_hint_gt_1e_4_trigger": row["cheap_hint_gt_1e_4_trigger"],
            }
        )
    payload = {
        "case": "B0::S042B95",
        "reported_chi2_min": actual_row["reported_chi2"],
        "min_decay_param_mle": actual_row["min_decay_param_mle"],
        "max_abs_decay_fitted_coord_mle": actual_row["max_abs_decay_fitted_coord_mle"],
        "attempts": attempts,
        "policy_rows": policy_rows,
    }
    write_json(actual_json, payload)
    return write_actual_policy_artifact(actual_json, actual_csv, actual_jsonl)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-csv", default="notes/codex-refactor/round18_calibration_campaign_v1_toys.csv")
    parser.add_argument("--output-prefix", default="notes/codex-refactor/round21_guarded_refit_policy_comparator")
    parser.add_argument("--profile-shape-small-q-per-case-dgp", type=int, default=16)
    parser.add_argument("--profile-shape-q-near-1-per-case-dgp", type=int, default=4)
    parser.add_argument("--clean-small-q-per-case-dgp", type=int, default=4)
    parser.add_argument("--clean-q-near-1-per-case-dgp", type=int, default=2)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--max-runtime-sec", type=float, default=None)
    parser.add_argument("--skip-actual-snapshot", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    # Attributes consumed by the reused round-20 row runner.
    args.profile_start_q_lt = -math.inf
    args.skip_truth_physical_start = True

    prefix = Path(args.output_prefix)
    selection_csv = prefix.with_name(prefix.name + "_selection.csv")
    selection_jsonl = prefix.with_name(prefix.name + "_selection.jsonl")
    row_csv = prefix.with_name(prefix.name + "_rows.csv")
    row_jsonl = prefix.with_name(prefix.name + "_rows.jsonl")
    attempt_csv = prefix.with_name(prefix.name + "_attempts.csv")
    attempt_jsonl = prefix.with_name(prefix.name + "_attempts.jsonl")
    failure_csv = prefix.with_name(prefix.name + "_failures.csv")
    failure_jsonl = prefix.with_name(prefix.name + "_failures.jsonl")
    policy_csv = prefix.with_name(prefix.name + "_policy_rows.csv")
    policy_jsonl = prefix.with_name(prefix.name + "_policy_rows.jsonl")
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
        policy_csv,
        policy_jsonl,
        summary_json,
        prefix.with_name(prefix.name + "_actual_snapshot.json"),
        prefix.with_name(prefix.name + "_actual_snapshot_policy_rows.csv"),
        prefix.with_name(prefix.name + "_actual_snapshot_policy_rows.jsonl"),
    ]
    if args.overwrite:
        round20.reset_outputs(output_paths)

    round20.ensure_artifact(row_csv, round20.ROW_FIELDNAMES)
    round20.ensure_artifact(row_jsonl, round20.ROW_FIELDNAMES, jsonl=True)
    round20.ensure_artifact(attempt_csv, round20.ATTEMPT_FIELDNAMES)
    round20.ensure_artifact(attempt_jsonl, round20.ATTEMPT_FIELDNAMES, jsonl=True)
    round20.ensure_artifact(failure_csv, round20.FAILURE_FIELDNAMES)
    round20.ensure_artifact(failure_jsonl, round20.FAILURE_FIELDNAMES, jsonl=True)

    selected_rows = build_selection(args)
    if args.overwrite or not selection_csv.exists() or selection_csv.stat().st_size == 0:
        write_selection(selected_rows, selection_csv, selection_jsonl)

    done = round20.completed_keys(row_csv)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in selected_rows:
        grouped[str(row["case_id"])].append(row)

    t_start = time.perf_counter()
    stop_for_runtime = False
    for case_id, case_rows in grouped.items():
        if stop_for_runtime:
            break
        case = round20.CASE_SPECS[case_id]
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
            if args.max_runtime_sec is not None and time.perf_counter() - t_start > args.max_runtime_sec:
                print(f"stopping for --max-runtime-sec={args.max_runtime_sec}", flush=True)
                stop_for_runtime = True
                break
            key = round20.selection_key(campaign_row)
            if key in done:
                continue
            row_t0 = time.perf_counter()
            try:
                row, attempts = round20.run_one_row(campaign_row, fit, context, target_func, target_std, args)
                round20.append_row(row_csv, row, round20.ROW_FIELDNAMES)
                round20.append_row(row_jsonl, row, round20.ROW_FIELDNAMES, jsonl=True)
                for attempt in attempts:
                    round20.append_row(attempt_csv, attempt, round20.ATTEMPT_FIELDNAMES)
                    round20.append_row(attempt_jsonl, attempt, round20.ATTEMPT_FIELDNAMES, jsonl=True)
                done.add(key)
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
                    "toy_index": round20.int_from_csv(campaign_row, "toy_index"),
                    "seed": round20.int_from_csv(campaign_row, "seed"),
                    "phase": "round21_refit_policy_comparator",
                    "exception_type": type(exc).__name__,
                    "exception": round15.clean_message(exc, limit=1000),
                    "runtime_sec": time.perf_counter() - row_t0,
                }
                round20.append_row(failure_csv, failure, round20.FAILURE_FIELDNAMES)
                round20.append_row(failure_jsonl, failure, round20.FAILURE_FIELDNAMES, jsonl=True)
                print(
                    f"  {campaign_row['dgp']} toy {campaign_row['toy_index']}: "
                    f"FAILED {type(exc).__name__}: {exc}",
                    flush=True,
                )

    policy_rows = write_policy_rows(row_csv, attempt_csv, policy_csv, policy_jsonl)
    actual_rows: list[dict[str, Any]] = []
    if not args.skip_actual_snapshot:
        actual_rows = run_actual_snapshot_check(prefix)
    summary = summarize(row_csv, attempt_csv, failure_csv, selection_csv, policy_rows, actual_rows, args)
    write_json(summary_json, summary)
    print(json.dumps(summary, indent=2, sort_keys=True, default=json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
