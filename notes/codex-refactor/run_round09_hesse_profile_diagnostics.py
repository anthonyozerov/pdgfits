#!/usr/bin/env python
"""Round 09 notes-only HESSE/parabolic-vs-profile diagnostics.

This harness asks whether cheap local signals can rank profile-likelihood
endpoints by numerical risk.  It does not alter production code, does not touch
build_chi2.py, and never treats HESSE/parabolic endpoints as reported errors.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import jax
from jax import numpy as jnp
import numpy as np

from pdgfits.asym_errors import CallableProfileProblem, calc_asym_errors, find_profile_root
from pdgfits.avg import run_avg
from pdgfits.fit import run_fit
from pdgfits.query import avg_queries


jax.config.update("jax_enable_x64", True)


def _load_round08_module():
    path = Path(__file__).with_name("run_round08_minos_covariance_starts.py")
    spec = importlib.util.spec_from_file_location("round08_minos_covariance_starts", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import round08 helper from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


round08 = _load_round08_module()


@dataclass(frozen=True)
class CaseSpec:
    case_id: str
    kind: str
    label_or_node: str
    target: str
    fairness: str
    rationale: str


DEFAULT_CASES = [
    CaseSpec(
        "avg_m002w",
        "avg",
        "M002W",
        "M002W",
        "fair_direct_average_coordinate",
        "one-parameter direct average from round 07",
    ),
    CaseSpec(
        "avg_m026r08",
        "avg",
        "M026R08",
        "M026R08",
        "fair_direct_average_coordinate_with_nuisance",
        "direct average coordinate with nuisance adjustments from rounds 07/08",
    ),
    CaseSpec(
        "fit_g2000_k002m",
        "fit",
        "G(2000),G(1800)",
        "K002M",
        "fair_direct_fit_coordinate",
        "nearly quadratic two-parameter MASS fit",
    ),
    CaseSpec(
        "fit_g2000_k003m",
        "fit",
        "G(2000),G(1800)",
        "K003M",
        "fair_direct_fit_coordinate",
        "second direct coordinate in the same two-parameter MASS fit",
    ),
    CaseSpec(
        "fit_eta_m026w",
        "fit",
        "eta_c J/psi psi(2S)",
        "M026W",
        "fair_direct_fit_coordinate_in_bru_fit",
        "round 7/8 hard direct coordinate with descent-check endpoints",
    ),
    CaseSpec(
        "fit_b0_s042b95",
        "fit",
        "B0",
        "S042B95",
        "caveated_arbitrary_scalar_node",
        "hard BRU node/ratio target with severe chart saturation",
    ),
    CaseSpec(
        "fit_b0s_s08637",
        "fit",
        "B0S-BR",
        "S086.37",
        "caveated_mapped_physical_parameter",
        "well-conditioned mapped BRU physical parameter from round 8",
    ),
    CaseSpec(
        "fit_b0s_s086r04",
        "fit",
        "B0S-BR",
        "S086R04",
        "caveated_arbitrary_scalar_node",
        "broader hard candidate with many trust-constr/KKT calls in the fit sweep",
    ),
    CaseSpec(
        "fit_eta_m026g01",
        "fit",
        "eta_c J/psi psi(2S)",
        "M026G01",
        "caveated_arbitrary_scalar_node",
        "tiny BRU node accepted with asymmetric endpoint certificates in the fit sweep",
    ),
    CaseSpec(
        "fit_eta_nuisance_m071254",
        "fit",
        "eta_c J/psi psi(2S)",
        "nuisance_M071.254",
        "fair_direct_fit_coordinate_but_statistical_nuisance",
        "high-residual/high-gradient nuisance endpoint from the fit sweep",
    ),
]


def finite_or_none(value: Any) -> Any:
    return round08.finite_or_none(value)


def write_rows(path: Path, rows: list[dict[str, Any]], *, jsonl: bool = False) -> None:
    round08.write_rows(path, rows, jsonl=jsonl)


def clean_message(message: Any) -> str:
    return round08.clean_message(message)


def as_float(value: Any) -> float:
    return float(np.asarray(value, dtype=np.float64))


def covariance_summary(covariance: np.ndarray | None) -> dict[str, Any]:
    if covariance is None:
        return {
            "covariance_available": False,
            "covariance_condition": None,
            "covariance_min_singular": None,
            "covariance_max_singular": None,
        }
    cov = np.asarray(covariance, dtype=np.float64)
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1] or not np.all(np.isfinite(cov)):
        return {
            "covariance_available": False,
            "covariance_condition": None,
            "covariance_min_singular": None,
            "covariance_max_singular": None,
        }
    try:
        singular = np.linalg.svd(cov, compute_uv=False)
    except np.linalg.LinAlgError:
        return {
            "covariance_available": False,
            "covariance_condition": None,
            "covariance_min_singular": None,
            "covariance_max_singular": None,
        }
    finite = singular[np.isfinite(singular)]
    positive = finite[finite > max(np.finfo(float).eps, 1e-300)]
    if len(positive) == 0:
        condition = math.inf
        min_s = None
        max_s = None
    else:
        min_s = float(np.min(positive))
        max_s = float(np.max(positive))
        condition = max_s / min_s if min_s > 0 else math.inf
    return {
        "covariance_available": True,
        "covariance_condition": finite_or_none(condition),
        "covariance_min_singular": min_s,
        "covariance_max_singular": max_s,
    }


def fit_target_func(fit: dict[str, Any], target: str) -> Callable[[Any], Any]:
    return round08.target_func_for_fit(fit, target)


def avg_target_func(avg_result: dict[str, Any], target: str) -> Callable[[Any], Any]:
    if target in avg_result["parameters"]:
        idx = avg_result["parameters"].index(target)
        return lambda params: params[idx]
    if target in avg_result["nodes"]:
        return avg_result["node_funcs"][avg_result["nodes"].index(target)]
    raise KeyError(f"{target} not found in average result")


def hesse_info(
    result: dict[str, Any],
    target_func: Callable[[Any], Any],
    covariance: np.ndarray | None,
) -> dict[str, Any]:
    fitted_values = np.asarray(result["fitted_values"], dtype=np.float64)
    fitted_to_params = result["fitted_params_to_params"]

    def target_fitted(fp):
        return target_func(fitted_to_params(fp))

    info: dict[str, Any] = covariance_summary(covariance)
    try:
        grad = np.asarray(
            jax.grad(target_fitted)(jnp.asarray(fitted_values, dtype=jnp.float64)),
            dtype=np.float64,
        )
    except Exception as exc:  # noqa: BLE001 - diagnostics record unavailability
        info.update(
            {
                "hesse_available": False,
                "hesse_failure": f"{type(exc).__name__}: {exc}",
                "target_grad_norm": None,
                "target_variance": None,
                "parabolic_error": None,
                "cov_grad_norm": None,
                "cov_dx_per_sigma_norm": None,
            }
        )
        return info

    info["target_grad_norm"] = finite_or_none(float(np.linalg.norm(grad)))
    if covariance is None:
        info.update(
            {
                "hesse_available": False,
                "hesse_failure": "missing_covariance",
                "target_variance": None,
                "parabolic_error": None,
                "cov_grad_norm": None,
                "cov_dx_per_sigma_norm": None,
            }
        )
        return info

    cov = np.asarray(covariance, dtype=np.float64)
    try:
        cov_grad = cov @ grad
        variance = float(grad @ cov_grad)
    except Exception as exc:  # noqa: BLE001
        info.update(
            {
                "hesse_available": False,
                "hesse_failure": f"{type(exc).__name__}: {exc}",
                "target_variance": None,
                "parabolic_error": None,
                "cov_grad_norm": None,
                "cov_dx_per_sigma_norm": None,
            }
        )
        return info

    parabolic_error = math.sqrt(variance) if math.isfinite(variance) and variance > 0 else math.nan
    dx_norm = (
        float(np.linalg.norm(cov_grad / parabolic_error))
        if math.isfinite(parabolic_error) and parabolic_error > 0
        else math.nan
    )
    info.update(
        {
            "hesse_available": bool(math.isfinite(parabolic_error) and parabolic_error > 0),
            "hesse_failure": None if math.isfinite(parabolic_error) and parabolic_error > 0 else "nonpositive_target_variance",
            "target_variance": finite_or_none(variance),
            "parabolic_error": finite_or_none(parabolic_error),
            "cov_grad_norm": finite_or_none(float(np.linalg.norm(cov_grad))),
            "cov_dx_per_sigma_norm": finite_or_none(dx_norm),
        }
    )
    return info


def endpoint_from_root(root: Any, side: str) -> Any:
    return root.lower if side == "lower" else root.upper


def side_counts_from_asym(asym: dict[str, Any], side: str) -> dict[str, Any]:
    counts = (asym.get("side_counts") or {}).get(side) or {}
    return {
        "production_side_function_evals": counts.get("function_evals"),
        "production_side_cache_hits": counts.get("cache_hits"),
        "production_side_bracket_evals": counts.get("bracket_evals"),
        "production_side_bisect_evals": counts.get("bisect_evals"),
    }


def production_fields(asym: dict[str, Any], side: str) -> dict[str, Any]:
    fields = round08.current_row_fields(asym, side)
    fields.update(side_counts_from_asym(asym, side))
    return fields


def best_record(records: list[dict[str, Any]], start_kind: str) -> dict[str, Any]:
    candidates = [
        record for record in records
        if record.get("start_kind") == start_kind and record.get("candidate_chi2") is not None
    ]
    if not candidates:
        return {}
    return min(candidates, key=lambda record: float(record["candidate_chi2"]))


def candidate_fields(point: Any) -> dict[str, Any]:
    detail = getattr(point, "round08_detail", {}) or {}
    records = list(detail.get("candidate_records") or [])
    current = best_record(records, "current")
    covariance = best_record(records, "covariance")
    out: dict[str, Any] = {
        "diagnostic_current_best_chi2": detail.get("current_best_chi2"),
        "diagnostic_covariance_best_chi2": detail.get("covariance_best_chi2"),
        "diagnostic_covariance_minus_current_chi2": detail.get("covariance_minus_current_chi2"),
        "diagnostic_covariance_found_lower_than_current": detail.get("covariance_found_lower_than_current"),
        "diagnostic_n_candidate_starts": detail.get("n_candidate_starts"),
        "diagnostic_candidate_nfev_total": detail.get("candidate_nfev_total"),
        "diagnostic_selected_start_kind": detail.get("selected_start_kind"),
        "diagnostic_selected_start_name": detail.get("selected_start_name"),
    }
    for prefix, record in (("current_candidate", current), ("covariance_candidate", covariance)):
        for key in (
            "candidate_chi2",
            "candidate_ok",
            "candidate_method",
            "candidate_nfev",
            "candidate_projected_grad_norm",
            "predictor_available",
            "predictor_denom",
            "predictor_dx_norm",
            "predictor_grad_norm",
            "predictor_cov_grad_norm",
            "cov_raw_min_decay_param",
            "cov_raw_max_decay_param",
            "cov_raw_max_abs_decay_fitted_coord",
            "cov_raw_chart_saturation_warning",
            "cov_projected_min_decay_param",
            "cov_projected_max_decay_param",
            "cov_projected_max_abs_decay_fitted_coord",
            "cov_projected_chart_saturation_warning",
            "covariance_projection_norm",
        ):
            if key in record:
                out[f"{prefix}_{key}"] = record.get(key)
    return out


def point_fields(prefix: str, point: Any, chi2_min: float) -> dict[str, Any]:
    out = {
        f"{prefix}_chi2": getattr(point, "chi2", None),
        f"{prefix}_residual": (
            getattr(point, "chi2", math.nan) - (float(chi2_min) + 1.0)
            if getattr(point, "chi2", None) is not None else None
        ),
        f"{prefix}_success": getattr(point, "success", None),
        f"{prefix}_method": getattr(point, "method", None),
        f"{prefix}_message": clean_message(getattr(point, "message", "")),
        f"{prefix}_nfev": getattr(point, "nfev", None),
        f"{prefix}_nit": getattr(point, "nit", None),
        f"{prefix}_runtime_sec": getattr(point, "runtime_sec", None),
        f"{prefix}_constraint_violation": getattr(point, "constraint_violation", None),
        f"{prefix}_scaled_constraint_violation": getattr(point, "scaled_constraint_violation", None),
        f"{prefix}_projected_grad_norm": getattr(point, "projected_grad_norm", None),
        f"{prefix}_objective_grad_norm": getattr(point, "objective_grad_norm", None),
        f"{prefix}_descent_improvement": getattr(point, "descent_improvement", None),
    }
    detail = getattr(point, "round08_detail", {}) or {}
    for key in (
        "selected_min_decay_param",
        "selected_max_decay_param",
        "selected_max_abs_decay_fitted_coord",
        "selected_chart_saturation_warning",
        "selected_decay_active_lower_1e8",
        "selected_decay_active_upper_1e8",
    ):
        if key in detail:
            out[f"{prefix}_{key}"] = detail.get(key)
    return out


def evaluate_parabolic_endpoint(
    case: CaseSpec,
    result: dict[str, Any],
    target_func: Callable[[Any], Any],
    covariance: np.ndarray,
    target_value: float,
    target_std: float,
    side: str,
    endpoint: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if case.kind == "fit":
        profile = round08.build_constrained_profile_for_policy(
            result,
            target_func,
            target_value,
            target_std,
            policy="current_plus_covariance",
            target_name=case.target,
        )
    else:
        profile = round08.build_avg_profile_for_policy(
            result,
            covariance,
            policy="current_plus_covariance",
            target_name=case.target,
        )

    try:
        profile(endpoint)
        point = profile.last_point
        fields = point_fields("parabolic_profile", point, float(result["chi2_min"]))
        fields.update({f"parabolic_profile_{k}": v for k, v in candidate_fields(point).items()})
        failed = False
        exception = None
    except Exception as exc:  # noqa: BLE001 - expected diagnostic failure path
        fields = {
            "parabolic_profile_chi2": None,
            "parabolic_profile_residual": None,
            "parabolic_profile_success": False,
            "parabolic_profile_method": None,
            "parabolic_profile_message": None,
        }
        failed = True
        exception = f"{type(exc).__name__}: {exc}"

    eval_rows: list[dict[str, Any]] = []
    for detail in getattr(profile, "round08_details", []):
        base = {
            key: value
            for key, value in detail.items()
            if key != "candidate_records"
        }
        base.update(
            {
                "record_type": "parabolic_profile_detail",
                "case_id": case.case_id,
                "kind": case.kind,
                "label_or_node": case.label_or_node,
                "target": case.target,
                "side": side,
                "profile_value": endpoint,
            }
        )
        eval_rows.append(base)
        for candidate in detail.get("candidate_records", []):
            eval_rows.append(
                {
                    **candidate,
                    "record_type": "parabolic_profile_candidate",
                    "case_id": case.case_id,
                    "kind": case.kind,
                    "label_or_node": case.label_or_node,
                    "target": case.target,
                    "side": side,
                    "profile_value": endpoint,
                }
            )
    fields["parabolic_profile_failed"] = failed
    fields["parabolic_profile_exception"] = exception
    return fields, eval_rows


def run_diagnostic_root(
    case: CaseSpec,
    result: dict[str, Any],
    target_func: Callable[[Any], Any],
    covariance: np.ndarray,
    target_value: float,
    target_std: float,
    lb: float,
    ub: float,
) -> tuple[Any, list[dict[str, Any]]]:
    if case.kind == "fit":
        profile = round08.build_constrained_profile_for_policy(
            result,
            target_func,
            target_value,
            target_std,
            policy="current_plus_covariance",
            target_name=case.target,
        )
    else:
        profile = round08.build_avg_profile_for_policy(
            result,
            covariance,
            policy="current_plus_covariance",
            target_name=case.target,
        )
    problem = CallableProfileProblem(
        profile_chi2=profile,
        target_name=case.target,
        target_value=target_value,
        chi2_min=float(result["chi2_min"]),
        lower_initial=lb,
        upper_initial=ub,
    )
    root = find_profile_root(problem, target_value, float(result["chi2_min"]), lb, ub)
    eval_rows: list[dict[str, Any]] = []
    for detail in getattr(profile, "round08_details", []):
        flat = {
            key: value
            for key, value in detail.items()
            if key != "candidate_records"
        }
        flat.update(
            {
                "record_type": "diagnostic_root_detail",
                "case_id": case.case_id,
                "kind": case.kind,
                "label_or_node": case.label_or_node,
                "target": case.target,
                "side": "lower" if detail.get("value", target_value) < target_value else "upper",
                "profile_value": detail.get("value"),
            }
        )
        eval_rows.append(flat)
        for candidate in detail.get("candidate_records", []):
            eval_rows.append(
                {
                    **candidate,
                    "record_type": "diagnostic_root_candidate",
                    "case_id": case.case_id,
                    "kind": case.kind,
                    "label_or_node": case.label_or_node,
                    "target": case.target,
                    "side": "lower" if detail.get("value", target_value) < target_value else "upper",
                    "profile_value": detail.get("value"),
                }
            )
    return root, eval_rows


def risk_fields(row: dict[str, Any]) -> dict[str, Any]:
    def f(name: str, default: float = math.nan) -> float:
        value = row.get(name)
        try:
            out = float(value)
        except (TypeError, ValueError):
            return default
        return out if math.isfinite(out) else default

    valid_residual = abs(f("production_residual", math.inf)) <= 5e-3
    near_residual_limit = abs(f("production_residual", 0.0)) > 2.5e-3
    root_calls = f("production_side_function_evals", f("production_root_function_evals", 0.0))
    call_heavy = root_calls >= 10
    method = str(row.get("production_profile_method") or "")
    descent_check = "descent-check" in method
    projected_grad = f("production_projected_grad_norm", 0.0)
    high_projected_grad = projected_grad > 1.0
    hesse_rel = f("parabolic_profile_error_rel_disagreement", 0.0)
    hesse_resid = abs(f("parabolic_profile_residual", 0.0))
    parabolic_disagreement = hesse_rel > 0.05 or hesse_resid > 0.25
    chart_warning = any(
        bool(row.get(name))
        for name in (
            "mle_chart_saturation_warning",
            "diagnostic_root_selected_chart_saturation_warning",
            "parabolic_profile_selected_chart_saturation_warning",
            "diagnostic_root_covariance_candidate_cov_projected_chart_saturation_warning",
            "parabolic_profile_diagnostic_covariance_candidate_cov_projected_chart_saturation_warning",
        )
    )
    min_decay = min(
        [
            f(name, math.inf)
            for name in (
                "mle_min_decay_param",
                "diagnostic_root_selected_min_decay_param",
                "parabolic_profile_selected_min_decay_param",
            )
        ]
    )
    max_coord = max(
        [
            f(name, 0.0)
            for name in (
                "mle_max_abs_decay_fitted_coord",
                "diagnostic_root_selected_max_abs_decay_fitted_coord",
                "parabolic_profile_selected_max_abs_decay_fitted_coord",
            )
        ]
    )
    near_boundary = chart_warning or min_decay < 1e-6 or max_coord > 10.0
    covariance_condition = f("covariance_condition", 0.0)
    cov_bad_condition = covariance_condition > 1e14
    cov_pred_available = row.get("diagnostic_root_covariance_candidate_predictor_available")
    if cov_pred_available is None:
        cov_pred_available = row.get("parabolic_profile_diagnostic_covariance_candidate_predictor_available")
    cov_missing = cov_pred_available is False or cov_pred_available is None
    cov_minus_current = f("diagnostic_root_diagnostic_covariance_minus_current_chi2", 0.0)
    para_cov_minus_current = f("parabolic_profile_diagnostic_covariance_minus_current_chi2", 0.0)
    new_lower_chi2 = cov_minus_current < -1e-6 or para_cov_minus_current < -1e-6
    invalid_failed = (
        not bool(row.get("production_profile_success", True))
        or not valid_residual
        or bool(row.get("parabolic_profile_failed"))
    )

    covariance_suspect = near_boundary or cov_bad_condition or cov_missing or parabolic_disagreement
    if new_lower_chi2:
        covariance_expectation = "stop_new_lower_fixed_target_chi2"
    elif covariance_suspect:
        covariance_expectation = "suspect"
    elif hesse_rel <= 0.02 and hesse_resid <= 0.1 and not descent_check:
        covariance_expectation = "locally_trustworthy"
    else:
        covariance_expectation = "mixed_requires_profile_verification"

    score = 0
    if new_lower_chi2:
        score += 100
    if invalid_failed:
        score += 40
    elif near_residual_limit:
        score += 12
    if near_boundary:
        score += 30
    if parabolic_disagreement:
        score += 25
    elif hesse_rel > 0.02 or hesse_resid > 0.1:
        score += 10
    if descent_check:
        score += 15
    if call_heavy:
        score += 10
    if high_projected_grad:
        score += 10
    if cov_bad_condition or cov_missing:
        score += 8

    tags = ["valid-residual" if valid_residual else "invalid-residual"]
    if new_lower_chi2:
        tags.append("new-lower-chi2")
    if near_boundary:
        tags.append("boundary/chart-saturation")
    if call_heavy:
        tags.append("call-heavy")
    if descent_check:
        tags.append("descent-check")
    if high_projected_grad:
        tags.append("high-projected-gradient")
    if parabolic_disagreement:
        tags.append("parabolic-profile-disagreement")
    if covariance_expectation == "locally_trustworthy":
        tags.append("covariance-start-trustworthy")
    else:
        tags.append("covariance-start-suspect")
    if invalid_failed:
        tags.append("invalid/failed")

    return {
        "risk_score": score,
        "valid_residual": valid_residual,
        "near_residual_limit": near_residual_limit,
        "call_heavy": call_heavy,
        "descent_check": descent_check,
        "high_projected_grad": high_projected_grad,
        "parabolic_disagreement": parabolic_disagreement,
        "near_boundary_or_chart_saturation": near_boundary,
        "covariance_bad_condition": cov_bad_condition,
        "covariance_predictor_missing": cov_missing,
        "new_lower_fixed_target_chi2_found": new_lower_chi2,
        "invalid_or_failed": invalid_failed,
        "covariance_start_expectation": covariance_expectation,
        "status_tags": "|".join(tags),
    }


def add_ranks(rows: list[dict[str, Any]]) -> None:
    sorted_rows = sorted(
        rows,
        key=lambda row: (
            -int(row.get("risk_score") or 0),
            str(row.get("case_id")),
            str(row.get("side")),
        ),
    )
    for rank, row in enumerate(sorted_rows, start=1):
        row["risk_rank"] = rank


def run_case(
    case: CaseSpec,
    *,
    avg_cache: dict[str, Any],
    fit_cache: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if case.kind == "fit":
        if case.label_or_node not in fit_cache:
            fit_cache[case.label_or_node] = run_fit(case.label_or_node, verbose=False)
        result = fit_cache[case.label_or_node]
        if result is None:
            raise RuntimeError(f"run_fit returned None for {case.label_or_node}")
        target_func = fit_target_func(result, case.target)
        asym = calc_asym_errors(result, targets=[case.target])[case.target]
        covariance = np.asarray(result["covariance"], dtype=np.float64)
        target_value = float(asym["value"])
        target_std = round08.target_std_for_fit(result, target_func, target_value)
        lb = target_value - 2.0 * target_std
        ub = target_value + 2.0 * target_std
    elif case.kind == "avg":
        if not avg_cache:
            avg_df, corr_df_dict = avg_queries(verbose=False)
            avg_cache["avg_df"] = avg_df
            avg_cache["corr_df_dict"] = corr_df_dict
        avg_df = avg_cache["avg_df"]
        corr_df_dict = avg_cache["corr_df_dict"]
        result = run_avg(
            case.label_or_node,
            avg_df[avg_df["node"] == case.label_or_node],
            corr_df_dict[case.label_or_node],
        )
        if result is None:
            raise RuntimeError(f"run_avg returned None for {case.label_or_node}")
        target_func = avg_target_func(result, case.target)
        asym = {
            "value": float(result["param_values"][result["parameters"].index(case.target)]),
            "error_p": float(result["error_p"]),
            "error_n": float(result["error_n"]),
            **result["asym_error_diagnostics"],
        }
        covariance = round08.covariance_for_avg(result)
        target_value = float(asym["value"])
        target_std = math.sqrt(float(covariance[result["parameters"].index(case.target), result["parameters"].index(case.target)]))
        lb = target_value - 2.5 * max(float(asym["lower_error"]), abs(float(asym["error_n"])), 1e-12)
        ub = target_value + 2.5 * max(float(asym["upper_error"]), abs(float(asym["error_p"])), 1e-12)
    else:
        raise ValueError(f"unknown case kind {case.kind}")

    hesse = hesse_info(result, target_func, covariance)
    mle_chart = round08.chart_summary(
        list(result["parameters"]),
        result["fitted_params_to_params"],
        np.asarray(result["fitted_values"], dtype=np.float64),
    )

    root, eval_rows = run_diagnostic_root(
        case,
        result,
        target_func,
        covariance,
        target_value,
        target_std,
        lb,
        ub,
    )

    rows: list[dict[str, Any]] = []
    parabolic_error = hesse.get("parabolic_error")
    for side in ("lower", "upper"):
        sign = -1.0 if side == "lower" else 1.0
        production_endpoint = float(asym[f"{side}_endpoint"])
        production_error = float(asym[f"{side}_error"])
        if parabolic_error is None:
            parabolic_endpoint = None
        else:
            parabolic_endpoint = target_value + sign * float(parabolic_error)

        if parabolic_endpoint is not None:
            parabolic_fields, parabolic_eval_rows = evaluate_parabolic_endpoint(
                case,
                result,
                target_func,
                covariance,
                target_value,
                target_std,
                side,
                float(parabolic_endpoint),
            )
            eval_rows.extend(parabolic_eval_rows)
        else:
            parabolic_fields = {
                "parabolic_profile_failed": True,
                "parabolic_profile_exception": "missing_parabolic_endpoint",
            }

        root_endpoint = endpoint_from_root(root, side)
        root_point = root_endpoint.point
        root_detail = getattr(root_point, "round08_detail", {}) or {}
        base_row = {
            "case_id": case.case_id,
            "kind": case.kind,
            "label_or_node": case.label_or_node,
            "target": case.target,
            "fairness": case.fairness,
            "rationale": case.rationale,
            "side": side,
            "n_parameters": len(result["parameters"]),
            "n_nodes": len(result["nodes"]),
            "chi2_min": float(result["chi2_min"]),
            "target_chi2": float(result["chi2_min"]) + 1.0,
            "target_value": target_value,
            "target_std_used_for_bracket": target_std,
            "initial_lb": lb,
            "initial_ub": ub,
            "parabolic_endpoint": parabolic_endpoint,
            "parabolic_error": parabolic_error,
            "profile_error": production_error,
            "parabolic_profile_endpoint_abs_disagreement": (
                abs(float(parabolic_endpoint) - production_endpoint)
                if parabolic_endpoint is not None else None
            ),
            "parabolic_profile_error_abs_disagreement": (
                abs(float(parabolic_error) - production_error)
                if parabolic_error is not None else None
            ),
            "parabolic_profile_error_rel_disagreement": (
                abs(float(parabolic_error) - production_error) / max(abs(production_error), 1e-300)
                if parabolic_error is not None else None
            ),
            **production_fields(asym, side),
            "diagnostic_root_endpoint": root_endpoint.endpoint,
            "diagnostic_root_error": root_endpoint.error,
            "diagnostic_root_chi2": root_endpoint.chi2,
            "diagnostic_root_residual": root_endpoint.residual,
            "diagnostic_root_function_evals": root.function_evals,
            "diagnostic_root_cache_hits": root.cache_hits,
            "diagnostic_root_side_function_evals": root_endpoint.counts.get("function_evals"),
            "diagnostic_root_side_bracket_evals": root_endpoint.counts.get("bracket_evals"),
            "diagnostic_root_side_bisect_evals": root_endpoint.counts.get("bisect_evals"),
            **point_fields("diagnostic_root", root_point, float(result["chi2_min"])),
            **{f"diagnostic_root_{k}": v for k, v in candidate_fields(root_point).items()},
            **{f"mle_{key}": value for key, value in mle_chart.items()},
            **hesse,
            **parabolic_fields,
        }
        for key in (
            "selected_min_decay_param",
            "selected_max_decay_param",
            "selected_max_abs_decay_fitted_coord",
            "selected_chart_saturation_warning",
            "selected_decay_active_lower_1e8",
            "selected_decay_active_upper_1e8",
        ):
            if key in root_detail:
                base_row[f"diagnostic_root_{key}"] = root_detail.get(key)
        base_row["endpoint_delta_diagnostic_vs_production"] = (
            float(base_row["diagnostic_root_endpoint"]) - production_endpoint
        )
        base_row["profile_chi2_delta_diagnostic_vs_production"] = (
            float(base_row["diagnostic_root_chi2"]) - float(base_row["production_profile_chi2"])
        )
        base_row.update(risk_fields(base_row))
        rows.append(base_row)

    return rows, eval_rows


def parse_cases(selected: list[str] | None) -> list[CaseSpec]:
    if not selected:
        return DEFAULT_CASES
    by_id = {case.case_id: case for case in DEFAULT_CASES}
    cases = []
    for case_id in selected:
        if case_id not in by_id:
            raise SystemExit(f"unknown case id {case_id}; available: {', '.join(by_id)}")
        cases.append(by_id[case_id])
    return cases


def main_impl(args: argparse.Namespace) -> int:
    cases = parse_cases(args.case)
    rows: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    avg_cache: dict[str, Any] = {}
    fit_cache: dict[str, Any] = {}

    for case in cases:
        print(f"\n=== {case.case_id}: {case.kind} {case.label_or_node} / {case.target} ===")
        t0 = time.perf_counter()
        try:
            case_rows, case_eval_rows = run_case(case, avg_cache=avg_cache, fit_cache=fit_cache)
            rows.extend(case_rows)
            eval_rows.extend(case_eval_rows)
            print(f"completed {case.case_id} in {time.perf_counter() - t0:.3f}s")
            for row in case_rows:
                print(
                    row["side"],
                    "risk",
                    row["risk_score"],
                    "prod_resid",
                    row["production_residual"],
                    "hesse_rel",
                    row["parabolic_profile_error_rel_disagreement"],
                    "hesse_eval_resid",
                    row.get("parabolic_profile_residual"),
                    "tags",
                    row["status_tags"],
                )
                if row["new_lower_fixed_target_chi2_found"]:
                    print("LOWER FIXED-TARGET CHI2 CANDIDATE FLAGGED.")
                    if not args.keep_going:
                        print("Stopping because --keep-going was disabled.")
                        add_ranks(rows)
                        write_all(args, rows, eval_rows, failures)
                        return 2
        except Exception as exc:  # noqa: BLE001 - notes-only diagnostic records failure
            failure = {
                "case_id": case.case_id,
                "kind": case.kind,
                "label_or_node": case.label_or_node,
                "target": case.target,
                "exception": f"{type(exc).__name__}: {exc}",
            }
            failures.append(failure)
            print(f"FAILED {case.case_id}: {failure['exception']}")
            if not args.keep_going:
                break

    add_ranks(rows)
    write_all(args, rows, eval_rows, failures)
    print(f"\nwrote {len(rows)} result rows")
    print(f"wrote {len(eval_rows)} evaluation rows")
    print(f"wrote {len(failures)} failure rows")
    return 0 if not failures else 1


def write_all(
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
    failures: list[dict[str, Any]],
) -> None:
    rows_sorted = sorted(rows, key=lambda row: int(row.get("risk_rank") or 10**9))
    write_rows(Path(args.results_csv), rows_sorted)
    write_rows(Path(args.results_jsonl), rows_sorted, jsonl=True)
    write_rows(Path(args.evaluations_csv), eval_rows)
    write_rows(Path(args.evaluations_jsonl), eval_rows, jsonl=True)
    write_rows(Path(args.failures_csv), failures)
    write_rows(Path(args.failures_jsonl), failures, jsonl=True)
    summary = {
        "result_rows": len(rows),
        "evaluation_rows": len(eval_rows),
        "failure_rows": len(failures),
        "new_lower_fixed_target_chi2_rows": sum(
            1 for row in rows if row.get("new_lower_fixed_target_chi2_found")
        ),
        "highest_risk": rows_sorted[:10],
    }
    Path(args.summary_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary_json).write_text(json.dumps(summary, default=round08.json_default, indent=2, sort_keys=True) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append", help="Case id to run; repeatable. Defaults to all cases.")
    parser.add_argument("--results-csv", default="notes/codex-refactor/round09_hesse_profile_diagnostics_results.csv")
    parser.add_argument("--results-jsonl", default="notes/codex-refactor/round09_hesse_profile_diagnostics_results.jsonl")
    parser.add_argument("--evaluations-csv", default="notes/codex-refactor/round09_hesse_profile_diagnostics_evaluations.csv")
    parser.add_argument("--evaluations-jsonl", default="notes/codex-refactor/round09_hesse_profile_diagnostics_evaluations.jsonl")
    parser.add_argument("--failures-csv", default="notes/codex-refactor/round09_hesse_profile_diagnostics_failures.csv")
    parser.add_argument("--failures-jsonl", default="notes/codex-refactor/round09_hesse_profile_diagnostics_failures.jsonl")
    parser.add_argument("--summary-json", default="notes/codex-refactor/round09_hesse_profile_diagnostics_summary.json")
    parser.add_argument("--stdout-log", default=None)
    parser.add_argument("--keep-going", action="store_true", default=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.stdout_log:
        path = Path(args.stdout_log)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f, contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
            return main_impl(args)
    return main_impl(args)


if __name__ == "__main__":
    sys.exit(main())
