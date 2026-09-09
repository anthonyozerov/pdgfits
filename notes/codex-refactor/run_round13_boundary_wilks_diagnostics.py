#!/usr/bin/env python
"""Round 13 notes-only boundary/Wilks regularity diagnostics.

This script consumes the round 9-12 numerical artifacts and ranks verified
profile endpoints by statistical regularity risk. It intentionally does not
change production code or rerun the expensive endpoint solver by default.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ENDPOINT_TOL = 5e-3
MATERIAL_LOWER_PROFILE_TOL = 1e-4


def finite_or_none(value: Any) -> Any:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, float) and math.isnan(value):
        return False
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y"}


def f(row: pd.Series | dict[str, Any], name: str, default: float = math.nan) -> float:
    value = row.get(name, default)
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def clean_value(value: Any) -> Any:
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        if math.isnan(float(value)) or math.isinf(float(value)):
            return None
        return float(value)
    if value is pd.NA:
        return None
    if isinstance(value, str) and value == "nan":
        return None
    return value


def json_default(obj: Any) -> Any:
    cleaned = clean_value(obj)
    if cleaned is not obj:
        return cleaned
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def write_rows(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    jsonl: bool = False,
    fieldnames_if_empty: list[str] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if jsonl:
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                cleaned = {key: clean_value(value) for key, value in row.items()}
                handle.write(json.dumps(cleaned, default=json_default, sort_keys=True) + "\n")
        return

    fieldnames: list[str] = list(fieldnames_if_empty or [])
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: clean_value(row.get(key)) for key in fieldnames})


def load_csv(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"required artifact not found: {path}")
    return pd.read_csv(p)


def case_side_key(row: pd.Series | dict[str, Any]) -> tuple[str, str]:
    return str(row["case_id"]), str(row["side"])


def compute_asymmetry_ratios(round09: pd.DataFrame) -> dict[str, dict[str, float]]:
    by_case: dict[str, dict[str, float]] = {}
    for case_id, group in round09.groupby("case_id"):
        lower = group[group["side"] == "lower"]
        upper = group[group["side"] == "upper"]
        if lower.empty or upper.empty:
            continue
        lower_error = abs(f(lower.iloc[0], "profile_error"))
        upper_error = abs(f(upper.iloc[0], "profile_error"))
        denom = min(lower_error, upper_error)
        ratio = max(lower_error, upper_error) / denom if denom > 0 else math.nan
        signed = upper_error / lower_error if lower_error > 0 else math.nan
        by_case[str(case_id)] = {
            "profile_lower_error": lower_error,
            "profile_upper_error": upper_error,
            "profile_asymmetry_ratio": ratio,
            "profile_upper_over_lower_error": signed,
        }
    return by_case


def curve_metrics(round09: pd.DataFrame, evaluations: pd.DataFrame) -> dict[tuple[str, str], dict[str, Any]]:
    detail = evaluations[evaluations.get("record_type").isin(["diagnostic_root_detail", "parabolic_profile_detail"])].copy()
    if detail.empty:
        return {}

    out: dict[tuple[str, str], dict[str, Any]] = {}
    for _, row in round09.iterrows():
        case_id, side = case_side_key(row)
        target_value = f(row, "target_value")
        chi2_min = f(row, "chi2_min")
        profile_error = abs(f(row, "profile_error"))
        if not all(math.isfinite(x) for x in (target_value, chi2_min, profile_error)) or profile_error <= 0:
            continue

        subset = detail[(detail["case_id"] == case_id) & (detail["side"] == side)].copy()
        if subset.empty:
            continue
        rows = []
        for _, detail_row in subset.iterrows():
            value = f(detail_row, "profile_value", f(detail_row, "value"))
            chi2 = f(detail_row, "selected_chi2")
            if not math.isfinite(value) or not math.isfinite(chi2):
                continue
            signed_disp = value - target_value
            if side == "lower" and signed_disp > 1e-14:
                continue
            if side == "upper" and signed_disp < -1e-14:
                continue
            r = abs(signed_disp) / profile_error
            if r < 0.05 or r > 1.25:
                continue
            delta = chi2 - chi2_min
            if not math.isfinite(delta):
                continue
            ideal = r * r
            ratio = delta / ideal if ideal > 0 else math.nan
            rows.append((r, delta, ideal, ratio))
        if not rows:
            continue
        ratios = np.asarray([item[3] for item in rows if math.isfinite(item[3])], dtype=float)
        deviations = np.asarray([item[1] - item[2] for item in rows], dtype=float)
        out[(case_id, side)] = {
            "profile_curve_sample_points": len(rows),
            "profile_curve_curvature_ratio_min": finite_or_none(np.min(ratios)) if len(ratios) else None,
            "profile_curve_curvature_ratio_max": finite_or_none(np.max(ratios)) if len(ratios) else None,
            "profile_curve_curvature_ratio_range": finite_or_none(np.max(ratios) - np.min(ratios)) if len(ratios) else None,
            "profile_curve_quad_deviation_rms": finite_or_none(math.sqrt(float(np.mean(deviations * deviations)))),
            "profile_curve_quad_deviation_max_abs": finite_or_none(float(np.max(np.abs(deviations)))),
        }
    return out


def measurement_asymmetry(error_p: np.ndarray, error_n: np.ndarray) -> dict[str, Any]:
    error_p = np.asarray(error_p, dtype=float)
    error_n = np.asarray(error_n, dtype=float)
    denom = np.maximum((np.abs(error_p) + np.abs(error_n)) / 2.0, 1e-300)
    frac = np.abs(error_p - error_n) / denom
    finite = frac[np.isfinite(frac)]
    if len(finite) == 0:
        return {
            "n_rows": 0,
            "n_asymmetric_rows": 0,
            "max_error_asymmetry_frac": None,
            "median_error_asymmetry_frac": None,
        }
    return {
        "n_rows": int(len(finite)),
        "n_asymmetric_rows": int(np.sum(finite > 1e-10)),
        "max_error_asymmetry_frac": float(np.max(finite)),
        "median_error_asymmetry_frac": float(np.median(finite)),
    }


def scan_input_asymmetry(round09: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Parse snapshot inputs and summarize asymmetric reported errors.

    This is a cheap data-model scan, not a refit. It gives context for whether
    nonlinear profile shape is plausibly coming from asymmetric measurement
    interpolation rather than optimizer behavior.
    """

    from pdgfits.avg import run_avg
    from pdgfits.fit import run_fit
    from pdgfits.query import avg_queries

    by_case: dict[str, dict[str, Any]] = {}
    avg_cache: dict[str, Any] = {}
    fit_cache: dict[str, Any] = {}

    cases = (
        round09[["case_id", "kind", "label_or_node", "target"]]
        .drop_duplicates()
        .sort_values("case_id")
        .to_dict("records")
    )
    for case in cases:
        case_id = str(case["case_id"])
        kind = str(case["kind"])
        label_or_node = str(case["label_or_node"])
        target = str(case["target"])
        try:
            if kind == "avg":
                if not avg_cache:
                    avg_df, corr_df_dict = avg_queries(verbose=False)
                    avg_cache["avg_df"] = avg_df
                    avg_cache["corr_df_dict"] = corr_df_dict
                avg_df = avg_cache["avg_df"]
                corr_df_dict = avg_cache["corr_df_dict"]
                result = run_avg(label_or_node, avg_df[avg_df["node"] == label_or_node], corr_df_dict[label_or_node])
            elif kind == "fit":
                if label_or_node not in fit_cache:
                    fit_cache[label_or_node] = run_fit(label_or_node, verbose=False)
                result = fit_cache[label_or_node]
            else:
                raise ValueError(f"unknown kind {kind}")

            if result is None:
                raise RuntimeError("case returned no fit/average result")
            meas_df = result["meas_df"]
            all_summary = measurement_asymmetry(meas_df["error_p"].to_numpy(), meas_df["error_n"].to_numpy())
            target_df = meas_df[meas_df["node"] == target]
            target_summary = measurement_asymmetry(target_df["error_p"].to_numpy(), target_df["error_n"].to_numpy())
            by_case[case_id] = {
                "input_scan_status": "ok",
                "input_all_n_rows": all_summary["n_rows"],
                "input_all_n_asymmetric_rows": all_summary["n_asymmetric_rows"],
                "input_all_max_error_asymmetry_frac": all_summary["max_error_asymmetry_frac"],
                "input_all_median_error_asymmetry_frac": all_summary["median_error_asymmetry_frac"],
                "input_target_n_rows": target_summary["n_rows"],
                "input_target_n_asymmetric_rows": target_summary["n_asymmetric_rows"],
                "input_target_max_error_asymmetry_frac": target_summary["max_error_asymmetry_frac"],
                "input_target_median_error_asymmetry_frac": target_summary["median_error_asymmetry_frac"],
            }
        except Exception as exc:  # noqa: BLE001 - failures are reported, not fatal
            by_case[case_id] = {
                "input_scan_status": "failed",
                "input_scan_error": f"{type(exc).__name__}: {exc}",
            }
    return by_case


def max_material_improvement(case_id: str, side: str, round10_globality: pd.DataFrame) -> dict[str, Any]:
    if round10_globality.empty:
        return {}
    subset = round10_globality[(round10_globality["case_id"] == case_id) & (round10_globality["side"] == side)]
    if subset.empty:
        return {
            "round10_globality_checked": False,
            "round10_max_improvement_vs_previous_best": None,
            "round10_material_lower_profile_found": False,
        }
    improvement = subset["improvement_vs_previous_best"].map(lambda x: f({"x": x}, "x", 0.0)).max()
    return {
        "round10_globality_checked": True,
        "round10_max_improvement_vs_previous_best": finite_or_none(improvement),
        "round10_material_lower_profile_found": bool(improvement > MATERIAL_LOWER_PROFILE_TOL),
    }


def chart_crosscheck(case_id: str, side: str, round11_fixed: pd.DataFrame) -> dict[str, Any]:
    if round11_fixed.empty:
        return {}
    subset = round11_fixed[(round11_fixed["case_id"] == case_id) & (round11_fixed["side"] == side)]
    if subset.empty:
        return {
            "round11_chart_checked": False,
            "round11_max_chart_improvement_vs_production": None,
            "round11_material_lower_chart_profile_found": False,
        }
    improvement = subset["improvement_vs_production"].map(lambda x: f({"x": x}, "x", 0.0)).max()
    return {
        "round11_chart_checked": True,
        "round11_max_chart_improvement_vs_production": finite_or_none(improvement),
        "round11_material_lower_chart_profile_found": bool(improvement > MATERIAL_LOWER_PROFILE_TOL),
        "round11_charts_tested": "|".join(sorted(str(x) for x in subset["chart"].dropna().unique())),
    }


def vm_crosscheck(case_id: str, side: str, round12_vm: pd.DataFrame) -> dict[str, Any]:
    if round12_vm.empty:
        return {}
    subset = round12_vm[
        (round12_vm["case_id"] == case_id)
        & (round12_vm["side"] == side)
        & (round12_vm["solver"] == "robust_trust")
        & (round12_vm["start_mode"] == "hesse_one_sigma")
    ]
    if subset.empty:
        return {
            "round12_vm_checked": False,
            "round12_vm_best_abs_endpoint_delta": None,
            "round12_vm_best_abs_verification_residual": None,
        }
    return {
        "round12_vm_checked": True,
        "round12_vm_best_abs_endpoint_delta": finite_or_none(subset["abs_endpoint_delta_vs_production"].min()),
        "round12_vm_best_abs_verification_residual": finite_or_none(subset["verification_residual_vs_target"].abs().min()),
        "round12_vm_iterations": finite_or_none(subset["iterations"].min()),
    }


def classify(row: dict[str, Any]) -> dict[str, Any]:
    endpoint_residual = abs(float(row.get("endpoint_residual") or math.inf))
    numerical_ok = (
        endpoint_residual <= ENDPOINT_TOL
        and not row.get("endpoint_correctness_failure", False)
        and not row.get("round10_material_lower_profile_found", False)
        and not row.get("round11_material_lower_chart_profile_found", False)
    )

    min_decay = row.get("min_decay_param")
    max_coord = row.get("max_abs_decay_fitted_coord")
    target_ratio = row.get("target_value_to_min_side_error_ratio")
    hesse_rel = abs(float(row.get("hesse_profile_rel_diff") or 0.0))
    hesse_resid = abs(float(row.get("hesse_endpoint_residual") or 0.0))
    curve_rms = float(row.get("profile_curve_quad_deviation_rms") or 0.0)
    curve_ratio_range = float(row.get("profile_curve_curvature_ratio_range") or 0.0)
    asym_ratio = float(row.get("profile_asymmetry_ratio") or 1.0)
    input_target_asym = row.get("input_target_max_error_asymmetry_frac")
    input_all_asym = row.get("input_all_max_error_asymmetry_frac")
    input_asym = max(
        float(input_target_asym or 0.0),
        float(input_all_asym or 0.0),
    )
    status = str(row.get("mncross_status") or "")
    method = str(row.get("production_method") or "")

    severe_boundary = (
        bool(row.get("parameter_or_chart_boundary", False))
        or (min_decay is not None and min_decay < 1e-6)
        or (max_coord is not None and max_coord > 20.0)
    )
    tiny_or_near_boundary = (
        severe_boundary
        or (min_decay is not None and min_decay < 1e-4)
        or (target_ratio is not None and target_ratio < 5.0)
    )
    one_sided_target = bool(target_ratio is not None and target_ratio < 3.0)
    profile_nonlinear = hesse_resid > 0.02 or hesse_rel > 0.015 or curve_rms > 0.02 or curve_ratio_range > 0.20
    hesse_bad = hesse_resid > 0.02 or hesse_rel > 0.015
    profile_asymmetric = asym_ratio > 1.05
    residual_edge = endpoint_residual > 0.0045
    certificate_risky = "descent-check" in method or "high-projected-gradient" in status or "call-heavy" in status

    score = 0
    reasons: list[str] = []
    if not numerical_ok:
        score += 100
        reasons.append("numerical endpoint/profile check concern")
    if severe_boundary:
        score += 45
        reasons.append("severe physical/chart boundary proximity")
    elif tiny_or_near_boundary:
        score += 25
        reasons.append("tiny BR/BRU or near-boundary coordinate")
    if one_sided_target:
        score += 35
        reasons.append("target estimate close enough to zero for one-sided behavior")
    if profile_nonlinear:
        score += 25
        reasons.append("nonquadratic profile or HESSE/profile disagreement")
    if profile_asymmetric:
        score += 10
        reasons.append("lower/upper profile errors visibly asymmetric")
    if input_asym > 0.25 and profile_nonlinear:
        score += 15
        reasons.append("asymmetric reported-error interpolation likely contributes")
    elif input_asym > 0.25:
        score += 5
        reasons.append("asymmetric reported errors present")
    if residual_edge:
        score += 8
        reasons.append("endpoint residual near current bisection tolerance")
    if certificate_risky:
        score += 6
        reasons.append("local numerical certificate is weaker/call-heavy")

    if not numerical_ok:
        wilks_risk = "numerical-first"
        dominant = "endpoint-correctness"
    elif severe_boundary:
        wilks_risk = "high"
        dominant = "boundary/tiny-parameter"
    elif tiny_or_near_boundary and profile_nonlinear:
        wilks_risk = "medium-high"
        dominant = "boundary-plus-nonquadratic-profile"
    elif profile_nonlinear:
        wilks_risk = "medium"
        dominant = "nonquadratic/asymmetric-profile"
    elif tiny_or_near_boundary:
        wilks_risk = "medium"
        dominant = "tiny-parameter-boundary"
    elif residual_edge:
        wilks_risk = "low-medium"
        dominant = "endpoint-tolerance-not-regularity"
    else:
        wilks_risk = "low"
        dominant = "locally-quadratic-interior"

    if one_sided_target:
        dominant = "one-sided-target-boundary"
    if hesse_bad and input_asym > 0.25 and not severe_boundary:
        dominant = "asymmetric-chi2/profile-shape"

    return {
        "numerical_endpoint_ok": numerical_ok,
        "wilks_regularity_risk": wilks_risk,
        "dominant_risk_driver": dominant,
        "regularity_risk_score": score,
        "regularity_risk_reasons": "|".join(reasons) if reasons else "none",
        "severe_boundary_or_chart_saturation": severe_boundary,
        "tiny_or_near_boundary_coordinate": tiny_or_near_boundary,
        "one_sided_target_boundary": one_sided_target,
        "profile_nonlinear": profile_nonlinear,
        "hesse_profile_disagreement": hesse_bad,
        "profile_asymmetric": profile_asymmetric,
        "asymmetric_input_errors_present": input_asym > 0.25,
        "residual_tolerance_edge": residual_edge,
        "certificate_risky": certificate_risky,
    }


def build_rows(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    round09 = load_csv(args.round09_results_csv)
    evaluations = load_csv(args.round09_evaluations_csv)
    round10_status = load_csv(args.round10_status_csv)
    round10_globality = load_csv(args.round10_globality_csv)
    round11_fixed = load_csv(args.round11_fixed_csv)
    round12_vm = load_csv(args.round12_vm_csv)

    asymmetry_by_case = compute_asymmetry_ratios(round09)
    curve_by_key = curve_metrics(round09, evaluations)
    input_by_case = {} if args.skip_input_scan else scan_input_asymmetry(round09)
    status_by_key = {
        (str(row["case_id"]), str(row["side"])): row
        for _, row in round10_status.iterrows()
    }

    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for _, r09 in round09.iterrows():
        case_id, side = case_side_key(r09)
        status_row = status_by_key.get((case_id, side))
        endpoint_residual = f(r09, "production_residual")
        profile_error = abs(f(r09, "profile_error"))
        target_value = f(r09, "target_value")
        lower_upper = asymmetry_by_case.get(case_id, {})
        min_side_error = min(
            lower_upper.get("profile_lower_error", math.inf),
            lower_upper.get("profile_upper_error", math.inf),
        )
        if not math.isfinite(min_side_error) or min_side_error <= 0:
            target_ratio = None
        else:
            target_ratio = abs(target_value) / min_side_error if math.isfinite(target_value) else None

        status_fields = {}
        if status_row is not None:
            status_fields = {
                "mncross_status": status_row.get("mncross_status"),
                "endpoint_correctness_failure": parse_bool(status_row.get("endpoint_correctness_failure")),
                "lower_fixed_target_chi2_material": parse_bool(status_row.get("lower_fixed_target_chi2_material")),
                "parameter_or_chart_boundary": parse_bool(status_row.get("parameter_or_chart_boundary")),
                "call_heavy_or_call_limit_like": parse_bool(status_row.get("call_heavy_or_call_limit_like")),
                "descent_check": parse_bool(status_row.get("descent_check")),
                "high_projected_or_kkt_gradient": parse_bool(status_row.get("high_projected_or_kkt_gradient")),
                "hesse_profile_nonlinear_round10": parse_bool(status_row.get("hesse_profile_nonlinear")),
                "disconnected_or_globality_suspect_round10": parse_bool(status_row.get("disconnected_or_globality_suspect")),
            }

        min_decay_candidates = [
            f(r09, "diagnostic_root_selected_min_decay_param", math.inf),
            f(r09, "parabolic_profile_selected_min_decay_param", math.inf),
            f(r09, "mle_min_decay_param", math.inf),
        ]
        max_coord_candidates = [
            f(r09, "diagnostic_root_selected_max_abs_decay_fitted_coord", 0.0),
            f(r09, "parabolic_profile_selected_max_abs_decay_fitted_coord", 0.0),
            f(r09, "mle_max_abs_decay_fitted_coord", 0.0),
        ]
        min_decay = min(min_decay_candidates)
        max_coord = max(max_coord_candidates)

        row: dict[str, Any] = {
            "case_id": case_id,
            "kind": r09.get("kind"),
            "label_or_node": r09.get("label_or_node"),
            "target": r09.get("target"),
            "side": side,
            "n_parameters": finite_or_none(r09.get("n_parameters")),
            "n_nodes": finite_or_none(r09.get("n_nodes")),
            "target_value": finite_or_none(target_value),
            "profile_endpoint": finite_or_none(r09.get("production_endpoint")),
            "profile_error": finite_or_none(profile_error),
            "endpoint_residual": finite_or_none(endpoint_residual),
            "production_method": r09.get("production_profile_method"),
            "production_profile_nfev": finite_or_none(r09.get("production_profile_nfev")),
            "production_side_function_evals": finite_or_none(r09.get("production_side_function_evals")),
            "production_projected_grad_norm": finite_or_none(r09.get("production_projected_grad_norm")),
            "production_scaled_constraint_violation": finite_or_none(r09.get("production_scaled_constraint_violation")),
            "parabolic_endpoint": finite_or_none(r09.get("parabolic_endpoint")),
            "parabolic_error": finite_or_none(r09.get("parabolic_error")),
            "hesse_profile_rel_diff": finite_or_none(r09.get("parabolic_profile_error_rel_disagreement")),
            "hesse_endpoint_residual": finite_or_none(r09.get("parabolic_profile_residual")),
            "profile_endpoint_abs_disagreement_vs_hesse": finite_or_none(r09.get("parabolic_profile_endpoint_abs_disagreement")),
            "min_decay_param": finite_or_none(min_decay),
            "max_abs_decay_fitted_coord": finite_or_none(max_coord),
            "target_value_to_min_side_error_ratio": finite_or_none(target_ratio),
            "round09_risk_score": finite_or_none(r09.get("risk_score")),
            "round09_status_tags": r09.get("status_tags"),
            **lower_upper,
            **curve_by_key.get((case_id, side), {}),
            **input_by_case.get(case_id, {}),
            **status_fields,
            **max_material_improvement(case_id, side, round10_globality),
            **chart_crosscheck(case_id, side, round11_fixed),
            **vm_crosscheck(case_id, side, round12_vm),
        }
        row.update(classify(row))
        rows.append(row)

    rows.sort(
        key=lambda row: (
            -int(row.get("regularity_risk_score") or 0),
            str(row.get("case_id")),
            str(row.get("side")),
        )
    )
    for idx, row in enumerate(rows, start=1):
        row["regularity_risk_rank"] = idx

    summary = {
        "result_rows": len(rows),
        "failure_rows": len(failures),
        "numerical_endpoint_issue_rows": sum(1 for row in rows if not row["numerical_endpoint_ok"]),
        "wilks_risk_counts": pd.Series([row["wilks_regularity_risk"] for row in rows]).value_counts().to_dict(),
        "dominant_risk_counts": pd.Series([row["dominant_risk_driver"] for row in rows]).value_counts().to_dict(),
        "highest_risk_case_sides": rows[:10],
    }
    return rows, failures, summary


def main_impl(args: argparse.Namespace) -> int:
    t0 = time.perf_counter()
    rows, failures, summary = build_rows(args)
    write_rows(Path(args.results_csv), rows)
    write_rows(Path(args.results_jsonl), rows, jsonl=True)
    write_rows(
        Path(args.failures_csv),
        failures,
        fieldnames_if_empty=["case_id", "kind", "label_or_node", "target", "side", "exception"],
    )
    write_rows(Path(args.failures_jsonl), failures, jsonl=True)
    Path(args.summary_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary_json).write_text(json.dumps(summary, default=json_default, indent=2, sort_keys=True) + "\n")
    print(f"wrote {len(rows)} result rows")
    print(f"wrote {len(failures)} failure rows")
    print("wilks risk counts:", summary["wilks_risk_counts"])
    print("dominant risk counts:", summary["dominant_risk_counts"])
    print(f"runtime_sec={time.perf_counter() - t0:.3f}")
    return 0 if not failures else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--round09-results-csv", default="notes/codex-refactor/round09_hesse_profile_diagnostics_results.csv")
    parser.add_argument("--round09-evaluations-csv", default="notes/codex-refactor/round09_hesse_profile_diagnostics_evaluations.csv")
    parser.add_argument("--round10-status-csv", default="notes/codex-refactor/round10_mncross_status_summary.csv")
    parser.add_argument("--round10-globality-csv", default="notes/codex-refactor/round10_stratified_globality_results.csv")
    parser.add_argument("--round11-fixed-csv", default="notes/codex-refactor/round11_brbru_chart_conditioning_fixed_checks.csv")
    parser.add_argument("--round12-vm-csv", default="notes/codex-refactor/round12_robust_vm_endpoint_results.csv")
    parser.add_argument("--results-csv", default="notes/codex-refactor/round13_boundary_wilks_diagnostics_results.csv")
    parser.add_argument("--results-jsonl", default="notes/codex-refactor/round13_boundary_wilks_diagnostics_results.jsonl")
    parser.add_argument("--failures-csv", default="notes/codex-refactor/round13_boundary_wilks_diagnostics_failures.csv")
    parser.add_argument("--failures-jsonl", default="notes/codex-refactor/round13_boundary_wilks_diagnostics_failures.jsonl")
    parser.add_argument("--summary-json", default="notes/codex-refactor/round13_boundary_wilks_diagnostics_summary.json")
    parser.add_argument("--stdout-log", default=None)
    parser.add_argument("--skip-input-scan", action="store_true", help="Skip cheap parsed-input asymmetry scan.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.stdout_log:
        path = Path(args.stdout_log)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            with contextlib.redirect_stdout(handle), contextlib.redirect_stderr(handle):
                return main_impl(args)
    return main_impl(args)


if __name__ == "__main__":
    sys.exit(main())
