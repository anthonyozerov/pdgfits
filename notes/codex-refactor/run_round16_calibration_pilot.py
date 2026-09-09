#!/usr/bin/env python
"""Round 16 bounded Wilks-calibration pilot.

This notes-only harness extends the round-15 smoke test by treating the toy
data-generating process as an explicit experimental factor.  It deliberately
keeps the same production chi2/refit/profile machinery and changes only the
toy draws and artifact shape.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Callable

import jax
from jax import numpy as jnp
import numpy as np

import run_round15_wilks_calibration_smoke as round15

from pdgfits.asym_errors import CallableProfileProblem, build_constrained_profile_chi2, find_profile_root
from pdgfits.fit import run_fit


jax.config.update("jax_enable_x64", True)


DGP_CHOICES = ("local_gaussian_sym", "split_normal_pdg_resid")

TOY_FIELDNAMES = [
    "dgp",
    "case_id",
    "stratum",
    "label",
    "target",
    "toy_index",
    "seed",
    "status",
    "runtime_sec",
    "n_parameters",
    "n_measurements",
    "truth_target",
    "toy_target_mle",
    "toy_target_minus_truth",
    "target_std_baseline",
    "toy_chi2_min",
    "toy_chi2_at_truth_params",
    "toy_delta_chi2_truth_params",
    "profile_true_chi2",
    "profile_true_delta_chi2",
    "profile_true_delta_minus_chi2_1",
    "profile_true_success",
    "profile_true_method",
    "profile_true_nfev",
    "profile_true_projected_grad_norm",
    "profile_true_scaled_constraint_violation",
    "profile_true_message",
    "wilks_q_le_1",
    "wilks_q_le_chisq1_median",
    "wilks_q_gt_chisq1_90",
    "wilks_q_gt_chisq1_95",
    "refit_method",
    "refit_success",
    "refit_valid",
    "refit_accurate",
    "refit_nfcn",
    "refit_ngrad",
    "refit_message",
    "refit_edm",
    "refit_profile_consistency_failure",
    "min_decay_param_mle",
    "max_decay_param_mle",
    "max_abs_decay_fitted_coord_mle",
    "decay_active_lower_1e8_mle",
    "decay_active_lower_1e6_mle",
    "decay_active_upper_1e8_mle",
    "chart_saturation_warning_mle",
    "n_decay_params",
    "toy_noise_norm",
    "toy_noise_mahalanobis_clipped",
    "split_positive_resid_count",
    "split_negative_resid_count",
    "split_positive_scale_mean",
    "split_negative_scale_mean",
    "corr_min_eig",
    "corr_n_negative_eig",
    "cov_min_eig",
    "cov_n_clipped_eig",
    "endpoint_requested",
    "endpoint_status",
    "endpoint_interval_contains_truth",
    "endpoint_lower",
    "endpoint_upper",
    "endpoint_lower_residual",
    "endpoint_upper_residual",
    "endpoint_root_function_evals",
    "endpoint_root_runtime_sec",
    "exception_type",
    "exception",
]

ENDPOINT_FIELDNAMES = ["dgp", *round15.ENDPOINT_FIELDNAMES]

FAILURE_FIELDNAMES = [
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

DEFAULT_CASES = [
    round15.CaseSpec(
        "fit_b0_s042b95",
        "boundary_tiny_parameter",
        "B0",
        "S042B95",
        "round-13 high-risk boundary/tiny BRU coordinate case",
    ),
    round15.CaseSpec(
        "fit_b0s_s08637",
        "asymmetric_profile_shape",
        "B0S-BR",
        "S086.37",
        "round-15 q-tail smoke signal and round-13 medium-high profile-shape risk",
    ),
    round15.CaseSpec(
        "fit_eta_m026g01",
        "asymmetric_profile_shape",
        "eta_c J/psi psi(2S)",
        "M026G01",
        "round-13 asymmetric/profile-shape case deferred from round 16",
    ),
    round15.CaseSpec(
        "fit_g2000_k002m",
        "clean_control",
        "G(2000),G(1800)",
        "K002M",
        "round-13 locally quadratic interior direct-coordinate control",
    ),
    round15.CaseSpec(
        "fit_g2000_k003m",
        "clean_control",
        "G(2000),G(1800)",
        "K003M",
        "second locally quadratic interior coordinate in the same clean fit",
    ),
]


Q_THRESHOLDS = {
    "q_le_chisq1_median": ("le", 0.454936423119572),
    "q_le_1": ("le", 1.0),
    "q_gt_chisq1_90": ("gt", 2.705543454095404),
    "q_gt_chisq1_95": ("gt", 3.841458820694124),
}


def binomial_summary(count: int, n: int) -> dict[str, Any]:
    if n <= 0:
        return {
            "count": int(count),
            "n": int(n),
            "fraction": None,
            "standard_error": None,
            "wilson_95_low": None,
            "wilson_95_high": None,
        }
    p = count / n
    se = math.sqrt(p * (1.0 - p) / n)
    z = 1.959963984540054
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = z * math.sqrt((p * (1.0 - p) / n) + z * z / (4.0 * n * n)) / denom
    return {
        "count": int(count),
        "n": int(n),
        "fraction": p,
        "standard_error": se,
        "wilson_95_low": max(0.0, center - half),
        "wilson_95_high": min(1.0, center + half),
    }


def q_threshold_summaries(q: np.ndarray) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    q = q[np.isfinite(q)]
    for name, (direction, threshold) in Q_THRESHOLDS.items():
        if direction == "le":
            count = int(np.sum(q <= threshold))
        elif direction == "gt":
            count = int(np.sum(q > threshold))
        else:
            raise ValueError(f"unknown threshold direction: {direction}")
        entry = binomial_summary(count, int(len(q)))
        entry["threshold"] = threshold
        entry["direction"] = direction
        out[name] = entry
    return out


def endpoint_requested_for_toy(toy_index: int, args: argparse.Namespace) -> bool:
    if toy_index < args.endpoint_toys_per_case_dgp:
        return True
    if args.endpoint_stride <= 0:
        return False
    if toy_index % args.endpoint_stride != 0:
        return False
    if args.endpoint_stride_max_toys is None:
        return True
    return toy_index // args.endpoint_stride < args.endpoint_stride_max_toys


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


def correlation_factor(corr_mat: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    corr = np.asarray(corr_mat, dtype=np.float64)
    corr = 0.5 * (corr + corr.T)
    corr_eig = np.linalg.eigvalsh(corr)
    eigvals, eigvecs = np.linalg.eigh(corr)
    clipped = np.clip(eigvals, 0.0, None)
    cov = eigvecs @ np.diag(clipped) @ eigvecs.T
    diag = np.sqrt(np.clip(np.diag(cov), np.finfo(float).tiny, None))
    normalizer = np.diag(1.0 / diag)
    factor = normalizer @ eigvecs @ np.diag(np.sqrt(clipped))
    scale = max(float(np.max(np.abs(eigvals))), 1.0)
    info = {
        "corr_min_eig": round15.finite_or_none(np.min(corr_eig)),
        "corr_n_negative_eig": int(np.sum(corr_eig < -1e-10)),
        "cov_min_eig": round15.finite_or_none(np.min(eigvals)),
        "cov_n_clipped_eig": int(np.sum(eigvals < -1e-12 * scale)),
    }
    return factor, info


def draw_toy(
    rng: np.random.Generator,
    dgp: str,
    mean_y: np.ndarray,
    context: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    error_n = np.asarray(context["error_n"], dtype=np.float64)
    error_p = np.asarray(context["error_p"], dtype=np.float64)

    if dgp == "local_gaussian_sym":
        sigma = round15.sym_error_at_zero(error_n, error_p)
        factor, sample_info = round15.sampling_factor(context["corr_mat"], sigma)
        z = rng.standard_normal(len(mean_y))
        noise = factor @ z
        sample_info.update(
            {
                "toy_noise_mahalanobis_clipped": float(np.dot(z, z)),
                "split_positive_resid_count": None,
                "split_negative_resid_count": None,
                "split_positive_scale_mean": None,
                "split_negative_scale_mean": None,
            }
        )
        return mean_y + noise, sample_info

    if dgp == "split_normal_pdg_resid":
        factor, sample_info = correlation_factor(context["corr_mat"])
        z = factor @ rng.standard_normal(len(mean_y))
        positive = z >= 0.0
        # The production chi2 residual is y - mu.  A positive residual means
        # the truth lies below the measurement, so the PDG lower error is the
        # matching one-sigma scale; negative residuals use the upper error.
        scale = np.where(positive, error_n, error_p)
        noise = z * scale
        sample_info.update(
            {
                "toy_noise_mahalanobis_clipped": float(np.dot(z, z)),
                "split_positive_resid_count": int(np.sum(positive)),
                "split_negative_resid_count": int(np.sum(~positive)),
                "split_positive_scale_mean": round15.finite_or_none(np.mean(error_n[positive]))
                if np.any(positive)
                else None,
                "split_negative_scale_mean": round15.finite_or_none(np.mean(error_p[~positive]))
                if np.any(~positive)
                else None,
            }
        )
        return mean_y + noise, sample_info

    raise ValueError(f"unknown DGP: {dgp}")


def endpoint_row(
    dgp: str,
    case: round15.CaseSpec,
    toy_index: int,
    seed: int,
    side: str,
    root: Any,
    truth_target: float,
    toy_target: float,
) -> dict[str, Any]:
    row = round15.endpoint_row(case, toy_index, seed, side, root, truth_target, toy_target)
    return {"dgp": dgp, **row}


def run_case_toy(
    dgp: str,
    case: round15.CaseSpec,
    fit: dict[str, Any],
    context: dict[str, Any],
    target_func: Callable[[Any], Any],
    target_std: float,
    toy_index: int,
    seed: int,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)

    truth_fp = np.asarray(fit["fitted_values"], dtype=np.float64)
    truth_params = np.asarray(fit["param_values"], dtype=np.float64)
    truth_target = float(target_func(jnp.asarray(truth_params, dtype=jnp.float64)))
    mean_y = np.asarray(
        context["mu_adjust"](jnp.asarray(truth_params, dtype=jnp.float64)),
        dtype=np.float64,
    )
    y_toy, sample_info = draw_toy(rng, dgp, mean_y, context)

    refit, toy_chi2 = round15.refit_toy(fit, y_toy)
    x_hat = np.asarray(refit["x_hat"], dtype=np.float64)
    toy_chi2_min = float(refit["chi2_min"])
    toy_params = np.asarray(
        fit["fitted_params_to_params"](jnp.asarray(x_hat, dtype=jnp.float64)),
        dtype=np.float64,
    )
    toy_target = float(target_func(jnp.asarray(toy_params, dtype=jnp.float64)))
    toy_chi2_at_truth = float(toy_chi2(jnp.asarray(truth_fp, dtype=jnp.float64)))

    profile_chi2 = build_constrained_profile_chi2(
        toy_chi2,
        fit["fitted_params_to_params"],
        target_func,
        x_hat,
        target_scale=target_std,
        target_name=case.target,
    )
    true_problem = CallableProfileProblem(
        profile_chi2,
        target_name=case.target,
        target_value=toy_target,
        chi2_min=toy_chi2_min,
    )
    true_point = true_problem.evaluate(truth_target)
    q_true = float(true_point.chi2 - toy_chi2_min)

    row = {
        "dgp": dgp,
        "case_id": case.case_id,
        "stratum": case.stratum,
        "label": case.label,
        "target": case.target,
        "toy_index": toy_index,
        "seed": seed,
        "status": "ok",
        "runtime_sec": None,
        "n_parameters": len(fit["parameters"]),
        "n_measurements": len(context["y_observed"]),
        "truth_target": truth_target,
        "toy_target_mle": toy_target,
        "toy_target_minus_truth": toy_target - truth_target,
        "target_std_baseline": target_std,
        "toy_chi2_min": toy_chi2_min,
        "toy_chi2_at_truth_params": toy_chi2_at_truth,
        "toy_delta_chi2_truth_params": toy_chi2_at_truth - toy_chi2_min,
        "profile_true_chi2": float(true_point.chi2),
        "profile_true_delta_chi2": q_true,
        "profile_true_delta_minus_chi2_1": q_true - 1.0,
        "profile_true_success": bool(true_point.success),
        "profile_true_method": true_point.method,
        "profile_true_nfev": true_point.nfev,
        "profile_true_projected_grad_norm": true_point.projected_grad_norm,
        "profile_true_scaled_constraint_violation": true_point.scaled_constraint_violation,
        "profile_true_message": round15.clean_message(true_point.message),
        "wilks_q_le_1": bool(q_true <= 1.0),
        "wilks_q_le_chisq1_median": bool(q_true <= 0.454936423119572),
        "wilks_q_gt_chisq1_90": bool(q_true > 2.705543454095404),
        "wilks_q_gt_chisq1_95": bool(q_true > 3.841458820694124),
        "refit_profile_consistency_failure": bool(q_true < -1e-4),
        "toy_noise_norm": float(np.linalg.norm(y_toy - mean_y)),
        "endpoint_requested": endpoint_requested_for_toy(toy_index, args),
        "endpoint_status": "not_requested",
        **refit,
        **round15.chart_summary(fit, x_hat),
        **sample_info,
    }
    row.pop("x_hat", None)

    endpoint_rows: list[dict[str, Any]] = []
    if endpoint_requested_for_toy(toy_index, args):
        row["endpoint_status"] = "attempted"
        lb = toy_target - args.bracket_sigmas * target_std
        ub = toy_target + args.bracket_sigmas * target_std
        if not lb < toy_target < ub:
            raise RuntimeError(f"invalid endpoint bracket lb={lb}, val={toy_target}, ub={ub}")
        root = find_profile_root(
            CallableProfileProblem(
                profile_chi2,
                target_name=case.target,
                target_value=toy_target,
                chi2_min=toy_chi2_min,
                lower_initial=lb,
                upper_initial=ub,
            ),
            residual_tol=args.endpoint_residual_tol,
        )
        lower = root.lower.endpoint
        upper = root.upper.endpoint
        row.update(
            {
                "endpoint_status": "ok",
                "endpoint_interval_contains_truth": bool(lower <= truth_target <= upper),
                "endpoint_lower": lower,
                "endpoint_upper": upper,
                "endpoint_lower_residual": root.lower.residual,
                "endpoint_upper_residual": root.upper.residual,
                "endpoint_root_function_evals": root.function_evals,
                "endpoint_root_runtime_sec": root.runtime_sec,
            }
        )
        endpoint_rows.append(endpoint_row(dgp, case, toy_index, seed, "lower", root, truth_target, toy_target))
        endpoint_rows.append(endpoint_row(dgp, case, toy_index, seed, "upper", root, truth_target, toy_target))

    row["runtime_sec"] = time.perf_counter() - t0
    return row, endpoint_rows


def summarize(toy_csv: Path, endpoint_csv: Path, failure_csv: Path, output: Path, metadata: dict[str, Any]) -> None:
    rows: list[dict[str, Any]] = []
    if toy_csv.exists() and toy_csv.stat().st_size > 0:
        with toy_csv.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    endpoint_rows: list[dict[str, Any]] = []
    if endpoint_csv.exists() and endpoint_csv.stat().st_size > 0:
        with endpoint_csv.open("r", encoding="utf-8", newline="") as handle:
            endpoint_rows = list(csv.DictReader(handle))
    failure_rows: list[dict[str, Any]] = []
    if failure_csv.exists() and failure_csv.stat().st_size > 0:
        with failure_csv.open("r", encoding="utf-8", newline="") as handle:
            failure_rows = list(csv.DictReader(handle))

    case_summaries = []
    keys = sorted({(row["dgp"], row["case_id"]) for row in rows})
    for dgp, case_id in keys:
        case_rows = [
            row
            for row in rows
            if row.get("dgp") == dgp and row.get("case_id") == case_id and row.get("status") == "ok"
        ]
        q = np.asarray([float(row["profile_true_delta_chi2"]) for row in case_rows], dtype=float)
        q = q[np.isfinite(q)]
        min_decay_values = np.asarray(
            [float(row.get("min_decay_param_mle") or "nan") for row in case_rows],
            dtype=float,
        )
        finite_min_decay = min_decay_values[np.isfinite(min_decay_values)]
        max_coord_values = np.asarray(
            [float(row.get("max_abs_decay_fitted_coord_mle") or "nan") for row in case_rows],
            dtype=float,
        )
        finite_max_coord = max_coord_values[np.isfinite(max_coord_values)]
        endpoint_case_rows = [
            row
            for row in case_rows
            if row.get("endpoint_status") == "ok"
        ]
        contains = [
            str(row.get("endpoint_interval_contains_truth")).lower() == "true"
            for row in endpoint_case_rows
        ]
        endpoint_residuals = []
        for row in endpoint_case_rows:
            for key in ("endpoint_lower_residual", "endpoint_upper_residual"):
                try:
                    endpoint_residuals.append(abs(float(row.get(key) or "nan")))
                except ValueError:
                    pass
        q_sorted = np.sort(q)
        thresholds = q_threshold_summaries(q)
        case_summaries.append(
            {
                "dgp": dgp,
                "case_id": case_id,
                "stratum": case_rows[0].get("stratum") if case_rows else None,
                "n_ok_toys": int(len(q)),
                "n_endpoint_toys": int(len(endpoint_case_rows)),
                "q_mean": round15.finite_or_none(np.mean(q)) if len(q) else None,
                "q_median": round15.finite_or_none(np.median(q)) if len(q) else None,
                "q_min": round15.finite_or_none(np.min(q)) if len(q) else None,
                "q_max": round15.finite_or_none(np.max(q)) if len(q) else None,
                "q_p25": round15.finite_or_none(np.quantile(q_sorted, 0.25)) if len(q_sorted) else None,
                "q_p75": round15.finite_or_none(np.quantile(q_sorted, 0.75)) if len(q_sorted) else None,
                "q_le_chisq1_median_count": thresholds["q_le_chisq1_median"]["count"],
                "q_le_chisq1_median_fraction": thresholds["q_le_chisq1_median"]["fraction"],
                "q_le_1_count": thresholds["q_le_1"]["count"],
                "q_le_1_fraction": thresholds["q_le_1"]["fraction"],
                "q_gt_chisq1_90_count": thresholds["q_gt_chisq1_90"]["count"],
                "q_gt_chisq1_90_fraction": thresholds["q_gt_chisq1_90"]["fraction"],
                "q_gt_chisq1_95_count": thresholds["q_gt_chisq1_95"]["count"],
                "q_gt_chisq1_95_fraction": thresholds["q_gt_chisq1_95"]["fraction"],
                "q_threshold_summaries": thresholds,
                "endpoint_contains_truth_count": int(sum(contains)),
                "endpoint_contains_truth_fraction": round15.finite_or_none(np.mean(contains)) if contains else None,
                "max_abs_endpoint_residual": round15.finite_or_none(np.nanmax(endpoint_residuals))
                if endpoint_residuals
                else None,
                "min_decay_param_min": round15.finite_or_none(np.min(finite_min_decay))
                if len(finite_min_decay)
                else None,
                "max_abs_decay_fitted_coord_max": round15.finite_or_none(np.max(finite_max_coord))
                if len(finite_max_coord)
                else None,
                "chart_saturation_toys": int(
                    sum(str(row.get("chart_saturation_warning_mle")).lower() == "true" for row in case_rows)
                ),
                "profile_consistency_failures": int(
                    sum(str(row.get("refit_profile_consistency_failure")).lower() == "true" for row in case_rows)
                ),
                "q_values": [round(float(value), 10) for value in q],
            }
        )

    stratum_summaries = []
    stratum_keys = sorted({(row["dgp"], row["stratum"]) for row in rows if row.get("status") == "ok"})
    for dgp, stratum in stratum_keys:
        group_rows = [
            row
            for row in rows
            if row.get("dgp") == dgp and row.get("stratum") == stratum and row.get("status") == "ok"
        ]
        q = np.asarray([float(row["profile_true_delta_chi2"]) for row in group_rows], dtype=float)
        q = q[np.isfinite(q)]
        thresholds = q_threshold_summaries(q)
        stratum_summaries.append(
            {
                "dgp": dgp,
                "stratum": stratum,
                "n_ok_toys": int(len(q)),
                "case_ids": sorted({str(row.get("case_id")) for row in group_rows}),
                "q_mean": round15.finite_or_none(np.mean(q)) if len(q) else None,
                "q_median": round15.finite_or_none(np.median(q)) if len(q) else None,
                "q_min": round15.finite_or_none(np.min(q)) if len(q) else None,
                "q_max": round15.finite_or_none(np.max(q)) if len(q) else None,
                "q_threshold_summaries": thresholds,
                "profile_consistency_failures": int(
                    sum(str(row.get("refit_profile_consistency_failure")).lower() == "true" for row in group_rows)
                ),
                "chart_saturation_toys": int(
                    sum(str(row.get("chart_saturation_warning_mle")).lower() == "true" for row in group_rows)
                ),
            }
        )

    endpoint_residuals = []
    endpoint_method_counts: dict[str, int] = {}
    endpoint_success_false = 0
    endpoint_negative_residual_failures = 0
    for row in endpoint_rows:
        method = row.get("profile_method") or ""
        endpoint_method_counts[method] = endpoint_method_counts.get(method, 0) + 1
        if str(row.get("profile_success")).lower() != "true":
            endpoint_success_false += 1
        try:
            residual = float(row.get("residual") or "nan")
        except ValueError:
            continue
        if math.isfinite(residual):
            endpoint_residuals.append(abs(residual))
            if residual < -5e-3 or residual > 5e-3:
                endpoint_negative_residual_failures += 1

    toy_q_values = []
    for row in rows:
        if row.get("status") != "ok":
            continue
        try:
            toy_q_values.append(float(row["profile_true_delta_chi2"]))
        except (KeyError, TypeError, ValueError):
            pass
    finite_toy_q_values = [q for q in toy_q_values if math.isfinite(q)]
    endpoint_summary = {
        "n_endpoint_side_rows": len(endpoint_rows),
        "n_endpoint_toy_rows": len({(row.get("dgp"), row.get("case_id"), row.get("toy_index")) for row in endpoint_rows}),
        "max_abs_endpoint_residual": round15.finite_or_none(np.max(endpoint_residuals)) if endpoint_residuals else None,
        "endpoint_residual_gt_tolerance_count": int(endpoint_negative_residual_failures),
        "profile_method_counts": endpoint_method_counts,
        "profile_success_false_count": int(endpoint_success_false),
    }

    summary = {
        "metadata": metadata,
        "n_toy_rows": len(rows),
        "n_endpoint_rows": len(endpoint_rows),
        "n_failure_rows": len(failure_rows),
        "case_summaries": case_summaries,
        "stratum_summaries": stratum_summaries,
        "endpoint_summary": endpoint_summary,
        "profile_consistency_failures": int(
            sum(str(row.get("refit_profile_consistency_failure")).lower() == "true" for row in rows)
        ),
        "negative_q_count": int(sum(q < -1e-4 for q in finite_toy_q_values)),
        "min_q": round15.finite_or_none(min(finite_toy_q_values)) if finite_toy_q_values else None,
        "expected_chisq1_q_le_1_fraction": 0.6826894921370859,
        "expected_chisq1_median": 0.454936423119572,
        "expected_chisq1_q90": 2.705543454095404,
        "expected_chisq1_q95": 3.841458820694124,
        "q_threshold_reference": {
            name: {"direction": direction, "threshold": threshold}
            for name, (direction, threshold) in Q_THRESHOLDS.items()
        },
        "limitations": [
            "The split-normal DGP is a Gaussian-copula two-piece-normal toy, not a normalized likelihood derived from the production asymmetric chi2.",
            "The split-normal DGP is mode-centered at the fitted model mean; asymmetric scales imply nonzero marginal means.",
            "Correlation matrices are eigenvalue-clipped for toy generation when necessary.",
            "Toy counts are pilot-scale and support design decisions, not publication-grade coverage claims.",
            "Boundary activity is recorded at the toy MLE; constrained profile endpoint internals expose less boundary detail.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True, default=round15.json_default), encoding="utf-8")


def parse_cases(selected: list[str] | None) -> list[round15.CaseSpec]:
    if not selected:
        return DEFAULT_CASES
    by_id = {case.case_id: case for case in DEFAULT_CASES}
    out = []
    for item in selected:
        if item not in by_id:
            raise ValueError(f"unknown case {item}; choices: {', '.join(sorted(by_id))}")
        out.append(by_id[item])
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", action="append", dest="cases", help="case_id to run; may repeat")
    parser.add_argument("--dgp", action="append", dest="dgps", choices=DGP_CHOICES, help="DGP to run; may repeat")
    parser.add_argument("--toys-per-case-dgp", type=int, default=16)
    parser.add_argument("--endpoint-toys-per-case-dgp", type=int, default=1)
    parser.add_argument(
        "--endpoint-stride",
        type=int,
        default=0,
        help="also compute endpoints when toy_index is a multiple of this positive stride",
    )
    parser.add_argument(
        "--endpoint-stride-max-toys",
        type=int,
        default=None,
        help="optional cap on stride-selected endpoint toys per case/DGP",
    )
    parser.add_argument("--base-seed", type=int, default=2026062516)
    parser.add_argument("--bracket-sigmas", type=float, default=3.0)
    parser.add_argument("--endpoint-residual-tol", type=float, default=5e-3)
    parser.add_argument("--max-runtime-sec", type=float, default=900.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--toy-csv", type=Path, default=Path("notes/codex-refactor/round16_calibration_pilot_toys.csv"))
    parser.add_argument("--toy-jsonl", type=Path, default=Path("notes/codex-refactor/round16_calibration_pilot_toys.jsonl"))
    parser.add_argument(
        "--endpoint-csv",
        type=Path,
        default=Path("notes/codex-refactor/round16_calibration_pilot_endpoints.csv"),
    )
    parser.add_argument(
        "--endpoint-jsonl",
        type=Path,
        default=Path("notes/codex-refactor/round16_calibration_pilot_endpoints.jsonl"),
    )
    parser.add_argument(
        "--failure-csv",
        type=Path,
        default=Path("notes/codex-refactor/round16_calibration_pilot_failures.csv"),
    )
    parser.add_argument(
        "--failure-jsonl",
        type=Path,
        default=Path("notes/codex-refactor/round16_calibration_pilot_failures.jsonl"),
    )
    parser.add_argument("--summary-json", type=Path, default=Path("notes/codex-refactor/round16_calibration_pilot_summary.json"))
    args = parser.parse_args()

    if args.endpoint_toys_per_case_dgp > args.toys_per_case_dgp:
        raise ValueError("--endpoint-toys-per-case-dgp cannot exceed --toys-per-case-dgp")
    if args.endpoint_stride < 0:
        raise ValueError("--endpoint-stride must be nonnegative")
    if args.endpoint_stride_max_toys is not None and args.endpoint_stride_max_toys < 0:
        raise ValueError("--endpoint-stride-max-toys must be nonnegative")
    if args.overwrite and args.resume:
        raise ValueError("--overwrite and --resume are mutually exclusive")

    output_paths = [
        args.toy_csv,
        args.toy_jsonl,
        args.endpoint_csv,
        args.endpoint_jsonl,
        args.failure_csv,
        args.failure_jsonl,
        args.summary_json,
    ]
    if args.overwrite:
        round15.reset_outputs(output_paths)
    round15.ensure_artifact(args.toy_csv, TOY_FIELDNAMES)
    round15.ensure_artifact(args.toy_jsonl, TOY_FIELDNAMES, jsonl=True)
    round15.ensure_artifact(args.endpoint_csv, ENDPOINT_FIELDNAMES)
    round15.ensure_artifact(args.endpoint_jsonl, ENDPOINT_FIELDNAMES, jsonl=True)
    round15.ensure_artifact(args.failure_csv, FAILURE_FIELDNAMES)
    round15.ensure_artifact(args.failure_jsonl, FAILURE_FIELDNAMES, jsonl=True)

    cases = parse_cases(args.cases)
    dgps = args.dgps or list(DGP_CHOICES)
    done = completed_keys(args.toy_csv) if args.resume else set()
    start_time = time.perf_counter()
    metadata = {
        "toys_per_case_dgp": args.toys_per_case_dgp,
        "endpoint_toys_per_case_dgp": args.endpoint_toys_per_case_dgp,
        "endpoint_stride": args.endpoint_stride,
        "endpoint_stride_max_toys": args.endpoint_stride_max_toys,
        "base_seed": args.base_seed,
        "case_ids": [case.case_id for case in cases],
        "dgps": dgps,
        "max_runtime_sec": args.max_runtime_sec,
        "endpoint_residual_tol": args.endpoint_residual_tol,
        "bracket_sigmas": args.bracket_sigmas,
        "command_cwd": os.getcwd(),
    }

    for case_index, case in enumerate(cases):
        if time.perf_counter() - start_time > args.max_runtime_sec:
            break
        print(f"loading {case.case_id}: {case.label}::{case.target}", flush=True)
        with contextlib.redirect_stdout(io.StringIO()):
            fit = run_fit(case.label, verbose=False)
            context = round15.build_fit_context(case.label, fit)
        if fit is None:
            raise RuntimeError(f"run_fit returned None for {case.label}")
        target_func = round15.target_func_for_fit(fit, case.target)
        truth_target = float(target_func(jnp.asarray(fit["param_values"], dtype=jnp.float64)))
        target_std = round15.target_std_for_fit(fit, target_func, truth_target)
        print(
            f"case {case.case_id}: npar={len(fit['parameters'])} nmeas={len(context['y_observed'])} "
            f"truth={truth_target:.8g} target_std={target_std:.4g}",
            flush=True,
        )

        for dgp_index, dgp in enumerate(dgps):
            for toy_index in range(args.toys_per_case_dgp):
                if time.perf_counter() - start_time > args.max_runtime_sec:
                    break
                if (dgp, case.case_id, toy_index) in done:
                    continue
                seed = args.base_seed + 1_000_000 * case_index + 100_000 * dgp_index + toy_index
                t_toy = time.perf_counter()
                try:
                    row, endpoint_rows = run_case_toy(
                        dgp,
                        case,
                        fit,
                        context,
                        target_func,
                        target_std,
                        toy_index,
                        seed,
                        args,
                    )
                    round15.append_row(args.toy_csv, row, TOY_FIELDNAMES)
                    round15.append_row(args.toy_jsonl, row, TOY_FIELDNAMES, jsonl=True)
                    for erow in endpoint_rows:
                        round15.append_row(args.endpoint_csv, erow, ENDPOINT_FIELDNAMES)
                        round15.append_row(args.endpoint_jsonl, erow, ENDPOINT_FIELDNAMES, jsonl=True)
                    print(
                        f"  {dgp} toy {toy_index}: q={row['profile_true_delta_chi2']:.4g} "
                        f"q<=1={row['wilks_q_le_1']} endpoint={row['endpoint_status']} "
                        f"runtime={row['runtime_sec']:.1f}s",
                        flush=True,
                    )
                except Exception as exc:  # noqa: BLE001 - artifact records exact failure
                    failure = {
                        "dgp": dgp,
                        "case_id": case.case_id,
                        "stratum": case.stratum,
                        "label": case.label,
                        "target": case.target,
                        "toy_index": toy_index,
                        "seed": seed,
                        "phase": "toy",
                        "exception_type": type(exc).__name__,
                        "exception": round15.clean_message(exc, limit=1000),
                        "runtime_sec": time.perf_counter() - t_toy,
                    }
                    round15.append_row(args.failure_csv, failure, FAILURE_FIELDNAMES)
                    round15.append_row(args.failure_jsonl, failure, FAILURE_FIELDNAMES, jsonl=True)
                    print(f"  {dgp} toy {toy_index}: FAILED {type(exc).__name__}: {exc}", flush=True)

    metadata["elapsed_sec"] = time.perf_counter() - start_time
    summarize(args.toy_csv, args.endpoint_csv, args.failure_csv, args.summary_json, metadata)
    print(f"wrote summary: {args.summary_json}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
