#!/usr/bin/env python
"""Round 10 notes-only MnCross-style status and stratified globality checks.

This harness consumes the round-09 profile/root diagnostics, writes a structured
crossing/status table, then reruns deliberately broader fixed-target profile
checks on the highest-risk endpoint values.  It does not alter production code
or the asymmetric chi2 model.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import importlib.util
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
from jax import numpy as jnp
import numpy as np

from pdgfits.avg import run_avg
from pdgfits.fit import run_fit
from pdgfits.param_maps import get_decay_info
from pdgfits.query import avg_queries


jax.config.update("jax_enable_x64", True)


def _load_helper(filename: str, module_name: str):
    path = Path(__file__).with_name(filename)
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import helper {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


round03 = _load_helper("run_asym_descent_multistart.py", "round03_descent_multistart")
round08 = _load_helper("run_round08_minos_covariance_starts.py", "round08_minos_covariance_starts")


RESIDUAL_TOL = 5e-3
NUMERICAL_TIE_TOL = 1e-5
DESCENT_FUN_TOL = 1e-4
MATERIAL_ENDPOINT_TOL = 1e-3


@dataclass(frozen=True)
class TargetSpec:
    case_id: str
    kind: str
    label_or_node: str
    target: str
    side: str
    value_source: str
    strata: str


def finite_or_none(value: Any) -> Any:
    return round08.finite_or_none(value)


def clean_message(message: Any) -> str:
    return round08.clean_message(message)


def parse_float(value: Any, default: float = math.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [finite_or_none(x) for x in value.tolist()]
    if isinstance(value, np.generic):
        return finite_or_none(value.item())
    return finite_or_none(value)


def write_rows(path: Path, rows: list[dict[str, Any]], *, jsonl: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if jsonl:
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, default=json_default, sort_keys=True) + "\n")
        return
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: finite_or_none(row.get(key)) for key in fieldnames})


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def best_round09_delta(row: dict[str, Any], prefix: str) -> float:
    current = parse_float(row.get(f"{prefix}_diagnostic_current_best_chi2"))
    covariance = parse_float(row.get(f"{prefix}_diagnostic_covariance_best_chi2"))
    if math.isfinite(current) and math.isfinite(covariance):
        return covariance - current
    explicit = parse_float(row.get(f"{prefix}_diagnostic_covariance_minus_current_chi2"))
    return explicit


def status_tags_for_row(row: dict[str, Any]) -> list[str]:
    tags = str(row.get("status_tags") or "").split("|")
    return [tag for tag in tags if tag]


def build_status_summary(round09_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in round09_rows:
        prod_resid = parse_float(row.get("production_residual"))
        prod_method = str(row.get("production_profile_method") or "")
        prod_success = boolish(row.get("production_profile_success"))
        side_evals = parse_float(row.get("production_side_function_evals"), 0.0)
        root_evals = parse_float(row.get("production_root_function_evals"), 0.0)
        projected_grad = parse_float(row.get("production_projected_grad_norm"), 0.0)
        scaled_cviol = parse_float(row.get("production_scaled_constraint_violation"), 0.0)
        hesse_rel = parse_float(row.get("parabolic_profile_error_rel_disagreement"), 0.0)
        hesse_resid = parse_float(row.get("parabolic_profile_residual"), 0.0)
        best_cov_delta = min(
            best_round09_delta(row, "diagnostic_root"),
            best_round09_delta(row, "parabolic_profile"),
        )
        min_decay = min(
            parse_float(row.get("mle_min_decay_param"), math.inf),
            parse_float(row.get("diagnostic_root_selected_min_decay_param"), math.inf),
            parse_float(row.get("parabolic_profile_selected_min_decay_param"), math.inf),
        )
        max_chart = max(
            parse_float(row.get("mle_max_abs_decay_fitted_coord"), 0.0),
            parse_float(row.get("diagnostic_root_selected_max_abs_decay_fitted_coord"), 0.0),
            parse_float(row.get("parabolic_profile_selected_max_abs_decay_fitted_coord"), 0.0),
        )
        chart_boundary = (
            boolish(row.get("near_boundary_or_chart_saturation"))
            or boolish(row.get("mle_chart_saturation_warning"))
            or min_decay < 1e-6
            or max_chart > 10.0
        )
        valid_residual = prod_success and math.isfinite(prod_resid) and abs(prod_resid) <= RESIDUAL_TOL
        residual_edge = math.isfinite(prod_resid) and abs(prod_resid) >= 0.9 * RESIDUAL_TOL
        residual_near = math.isfinite(prod_resid) and abs(prod_resid) >= 0.5 * RESIDUAL_TOL
        lower_fixed_target = math.isfinite(best_cov_delta) and best_cov_delta < -1e-6
        lower_tie = math.isfinite(best_cov_delta) and best_cov_delta < 0 and abs(best_cov_delta) <= NUMERICAL_TIE_TOL
        lower_material = math.isfinite(best_cov_delta) and abs(best_cov_delta) > DESCENT_FUN_TOL
        invalid_profile = (not prod_success) or (not math.isfinite(prod_resid))
        call_heavy = max(side_evals, root_evals) >= 10
        descent_check = "descent-check" in prod_method
        high_gradient = projected_grad > 1.0
        hesse_nonlinear = abs(hesse_resid) >= 0.025 or hesse_rel >= 0.02
        cov_suspect = "covariance-start-suspect" in status_tags_for_row(row)
        globality_suspect = (
            lower_fixed_target
            or (chart_boundary and (descent_check or high_gradient))
            or (descent_check and high_gradient and (call_heavy or hesse_nonlinear))
        )

        mncross_statuses: list[str] = []
        mncross_statuses.append("valid-residual" if valid_residual else "invalid-residual")
        if lower_fixed_target:
            mncross_statuses.append("lower-fixed-target-chi2")
        if lower_material:
            mncross_statuses.append("material-lower-fixed-target-chi2")
        if chart_boundary:
            mncross_statuses.append("parameter/chart-boundary")
        if call_heavy:
            mncross_statuses.append("call-heavy/call-limit-like")
        if invalid_profile:
            mncross_statuses.append("invalid-profile-solve")
        if residual_edge:
            mncross_statuses.append("residual-tolerance-edge")
        elif residual_near:
            mncross_statuses.append("residual-near-tolerance")
        if descent_check:
            mncross_statuses.append("descent-check")
        if high_gradient:
            mncross_statuses.append("high-projected-gradient")
        if hesse_nonlinear:
            mncross_statuses.append("HESSE-profile-nonlinear")
        if cov_suspect:
            mncross_statuses.append("covariance-start-suspect")
        if globality_suspect:
            mncross_statuses.append("disconnected/globality-suspect")

        out.append(
            {
                "case_id": row.get("case_id"),
                "kind": row.get("kind"),
                "label_or_node": row.get("label_or_node"),
                "target": row.get("target"),
                "side": row.get("side"),
                "risk_rank_round09": row.get("risk_rank"),
                "risk_score_round09": row.get("risk_score"),
                "mncross_status": "|".join(mncross_statuses),
                "valid_residual": valid_residual,
                "endpoint_correctness_failure": bool(math.isfinite(prod_resid) and abs(prod_resid) > RESIDUAL_TOL),
                "new_lower_minimum": False,
                "lower_fixed_target_chi2": lower_fixed_target,
                "lower_fixed_target_chi2_tie": lower_tie,
                "lower_fixed_target_chi2_material": lower_material,
                "parameter_or_chart_boundary": chart_boundary,
                "call_heavy_or_call_limit_like": call_heavy,
                "invalid_profile_solve": invalid_profile,
                "residual_tolerance_edge": residual_edge,
                "descent_check": descent_check,
                "high_projected_or_kkt_gradient": high_gradient,
                "hesse_profile_nonlinear": hesse_nonlinear,
                "covariance_start_suspect": cov_suspect,
                "disconnected_or_globality_suspect": globality_suspect,
                "production_residual": prod_resid,
                "production_method": prod_method,
                "production_side_function_evals": side_evals,
                "production_root_function_evals": root_evals,
                "production_projected_grad_norm": projected_grad,
                "production_scaled_constraint_violation": scaled_cviol,
                "hesse_profile_rel_diff": hesse_rel,
                "hesse_endpoint_residual": hesse_resid,
                "min_decay_param": finite_or_none(min_decay),
                "max_abs_decay_fitted_coord": finite_or_none(max_chart),
                "best_covariance_minus_current_chi2": finite_or_none(best_cov_delta),
                "round09_status_tags": row.get("status_tags"),
            }
        )
    out.sort(key=lambda r: (-(parse_float(r.get("risk_score_round09"), 0.0)), str(r.get("case_id")), str(r.get("side"))))
    return out


def target_specs_from_status(status_rows: list[dict[str, Any]], round09_by_key: dict[tuple[str, str, str], dict[str, Any]]) -> list[TargetSpec]:
    specs_by_key: dict[tuple[str, str, str], TargetSpec] = {}

    def add(case_id: str, side: str, value_source: str, strata: str) -> None:
        candidates = [row for row in status_rows if row["case_id"] == case_id and row["side"] == side]
        if not candidates:
            raise KeyError(f"missing round09/status row for {case_id} {side}")
        row = candidates[0]
        key = (case_id, side, value_source)
        existing = specs_by_key.get(key)
        if existing is not None:
            strata_parts = list(dict.fromkeys(str(existing.strata).split("|") + [strata]))
            specs_by_key[key] = TargetSpec(
                case_id=existing.case_id,
                kind=existing.kind,
                label_or_node=existing.label_or_node,
                target=existing.target,
                side=existing.side,
                value_source=existing.value_source,
                strata="|".join(strata_parts),
            )
            return
        spec = TargetSpec(
            case_id=case_id,
            kind=str(row["kind"]),
            label_or_node=str(row["label_or_node"]),
            target=str(row["target"]),
            side=side,
            value_source=value_source,
            strata=strata,
        )
        specs_by_key[key] = spec

    for row in status_rows:
        if bool(row["descent_check"]) and bool(row["high_projected_or_kkt_gradient"]):
            add(str(row["case_id"]), str(row["side"]), "production_endpoint", "round09_descent_check_high_gradient")

    add("fit_eta_m026w", "lower", "production_endpoint", "round09_sub_1e-5_lower_chi2_candidate")
    add("fit_eta_nuisance_m071254", "lower", "production_endpoint", "round09_sub_1e-5_lower_chi2_candidate")
    add("fit_eta_nuisance_m071254", "upper", "parabolic_endpoint", "round09_sub_1e-5_lower_chi2_candidate_parabolic")
    add("fit_b0_s042b95", "lower", "production_endpoint", "chart_saturated_bru")
    add("fit_b0_s042b95", "upper", "production_endpoint", "chart_saturated_bru")
    add("fit_b0s_s08637", "lower", "production_endpoint", "mild_hesse_profile_nonlinear")
    add("fit_b0s_s08637", "upper", "production_endpoint", "mild_hesse_profile_nonlinear")
    add("fit_eta_m026g01", "upper", "production_endpoint", "mild_hesse_profile_nonlinear")
    add("avg_m002w", "lower", "production_endpoint", "residual_tolerance_edge_average")

    return list(specs_by_key.values())


def row_for_spec(spec: TargetSpec, round09_by_key: dict[tuple[str, str, str], dict[str, Any]]) -> dict[str, Any]:
    try:
        return round09_by_key[(spec.case_id, spec.side, "")]
    except KeyError:
        pass
    for (case_id, side, _value_source), row in round09_by_key.items():
        if case_id == spec.case_id and side == spec.side:
            return row
    raise KeyError(f"missing round09 row for {spec}")


def fixed_value_and_baselines(spec: TargetSpec, row: dict[str, Any]) -> dict[str, Any]:
    if spec.value_source == "production_endpoint":
        fixed_value = parse_float(row.get("production_endpoint"))
        previous_best_chi2 = parse_float(row.get("production_profile_chi2"))
        current_baseline_chi2 = parse_float(row.get("diagnostic_root_diagnostic_current_best_chi2"), previous_best_chi2)
        covariance_baseline_chi2 = parse_float(row.get("diagnostic_root_diagnostic_covariance_best_chi2"))
        round09_cov_minus_current = parse_float(row.get("diagnostic_root_diagnostic_covariance_minus_current_chi2"))
    elif spec.value_source == "parabolic_endpoint":
        fixed_value = parse_float(row.get("parabolic_endpoint"))
        previous_best_chi2 = parse_float(row.get("parabolic_profile_chi2"))
        current_baseline_chi2 = parse_float(row.get("parabolic_profile_diagnostic_current_best_chi2"), previous_best_chi2)
        covariance_baseline_chi2 = parse_float(row.get("parabolic_profile_diagnostic_covariance_best_chi2"))
        round09_cov_minus_current = parse_float(row.get("parabolic_profile_diagnostic_covariance_minus_current_chi2"))
    else:
        raise ValueError(f"unknown value_source {spec.value_source}")
    return {
        "fixed_value": fixed_value,
        "previous_best_chi2": previous_best_chi2,
        "current_baseline_chi2": current_baseline_chi2,
        "covariance_baseline_chi2": covariance_baseline_chi2,
        "round09_covariance_minus_current_chi2": round09_cov_minus_current,
    }


def build_extra_fit_starts(fit: dict[str, Any], target_func: Any, fixed_value: float, target_value: float) -> list[tuple[str, str, np.ndarray]]:
    fp_best = np.asarray(fit["fitted_values"], dtype=np.float64)
    starts: list[tuple[str, str, np.ndarray]] = []
    try:
        @jax.jit
        def target_fitted(fp):
            return target_func(fit["fitted_params_to_params"](fp))

        target_grad = np.asarray(jax.grad(target_fitted)(jnp.asarray(fp_best, dtype=jnp.float64)), dtype=np.float64)
        pred, _info = round08.covariance_prediction(
            fp_best,
            fit.get("covariance"),
            target_grad,
            float(fixed_value) - float(target_value),
        )
        if pred is not None and np.all(np.isfinite(pred)):
            starts.append(("covariance_predicted_raw", "covariance_predicted", np.asarray(pred, dtype=np.float64)))
    except Exception:
        pass

    try:
        _, decay_idxs = get_decay_info(list(fit["parameters"]))
        decay_idxs = np.asarray(decay_idxs, dtype=int)
        if len(decay_idxs):
            params = np.asarray(fit["fitted_params_to_params"](jnp.asarray(fp_best, dtype=jnp.float64)), dtype=np.float64)
            decay_params = params[decay_idxs]
            finite = np.isfinite(decay_params)
            if np.any(finite):
                local_idxs = decay_idxs[finite]
                min_idx = int(local_idxs[np.argmin(decay_params[finite])])
                x = fp_best.copy()
                x[min_idx] = math.copysign(max(abs(float(x[min_idx])) * 1.5, 50.0), float(x[min_idx]) if x[min_idx] != 0 else -1.0)
                starts.append(("boundary_min_decay_more_saturated", "chart_boundary", x))
                x = fp_best.copy()
                x[min_idx] = 0.5 * x[min_idx]
                starts.append(("boundary_min_decay_less_saturated", "chart_boundary", x))
                x = fp_best.copy()
                x[decay_idxs] = 0.5 * x[decay_idxs]
                starts.append(("all_decay_coords_half", "chart_boundary", x))
                x = fp_best.copy()
                x[decay_idxs] = 1.25 * x[decay_idxs]
                starts.append(("all_decay_coords_expand", "chart_boundary", x))
    except Exception:
        pass
    return starts


def stable_seed_offset(*parts: str) -> int:
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
    return int(digest[:12], 16) % 100000


def chart_fields(fit: dict[str, Any], x: np.ndarray | None, prefix: str) -> dict[str, Any]:
    if x is None:
        return {}
    fields = round08.chart_summary(list(fit["parameters"]), fit["fitted_params_to_params"], x)
    return {f"{prefix}_{key}": value for key, value in fields.items()}


def candidate_to_row(candidate: Any, spec: TargetSpec, base: dict[str, Any], fit: dict[str, Any] | None = None) -> dict[str, Any]:
    row = {
        **base,
        "record_type": "candidate",
        "start_name": candidate.start_name,
        "start_kind": candidate.start_kind,
        "candidate_ok": candidate.ok,
        "candidate_chi2": candidate.chi2,
        "candidate_residual": candidate.chi2 - base["target_chi2"],
        "candidate_improvement_vs_previous_best": base["previous_best_chi2"] - candidate.chi2,
        "candidate_improvement_vs_current_baseline": base["current_baseline_chi2"] - candidate.chi2,
        "candidate_method": candidate.method,
        "candidate_constraint_violation": candidate.constraint_violation,
        "candidate_scaled_constraint_violation": candidate.scaled_constraint_violation,
        "candidate_projected_grad_norm": candidate.projected_grad_norm,
        "candidate_objective_grad_norm": candidate.objective_grad_norm,
        "candidate_descent_improvement": candidate.descent_improvement,
        "candidate_projected_start_chi2": candidate.projected_start_chi2,
        "candidate_projected_start_scaled_constraint_violation": candidate.projected_start_scaled_constraint_violation,
        "candidate_nfev": candidate.nfev,
        "candidate_nit": candidate.nit,
        "candidate_runtime_sec": candidate.runtime_sec,
        "candidate_message": clean_message(candidate.message),
    }
    if fit is not None:
        row.update(chart_fields(fit, candidate.x, "candidate"))
    return row


def run_fit_fixed_target(
    spec: TargetSpec,
    round09_row: dict[str, Any],
    fit_cache: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if spec.label_or_node not in fit_cache:
        t_fit = time.perf_counter()
        fit_cache[spec.label_or_node] = run_fit(spec.label_or_node, verbose=False)
        if fit_cache[spec.label_or_node] is None:
            raise RuntimeError(f"run_fit returned None for {spec.label_or_node}")
        fit_cache[f"{spec.label_or_node}__runtime_sec"] = time.perf_counter() - t_fit
    fit = fit_cache[spec.label_or_node]
    target_func = round08.target_func_for_fit(fit, spec.target)
    target_value = parse_float(round09_row.get("target_value"))
    if not math.isfinite(target_value):
        target_value = float(target_func(fit["param_values"]))
    target_std = round08.target_std_for_fit(fit, target_func, target_value)
    baselines = fixed_value_and_baselines(spec, round09_row)
    fixed_value = baselines["fixed_value"]
    if not math.isfinite(fixed_value):
        raise RuntimeError(f"fixed value is not finite for {spec}")

    solver = round03.build_fixed_target_solver(fit, target_func, fixed_value, target_std)
    rng = np.random.default_rng(args.seed + stable_seed_offset(spec.case_id, spec.side, spec.value_source))
    fp_best = np.asarray(fit["fitted_values"], dtype=np.float64)
    starts, sqrt_cov = round03.build_start_points(
        fp_best,
        fit.get("covariance"),
        rng=rng,
        cov_dirs=args.cov_dirs,
        cov_scale=args.cov_scale,
        random_starts=args.random_starts,
        random_scale=args.random_scale,
    )
    starts.extend(build_extra_fit_starts(fit, target_func, fixed_value, target_value))
    starts = round03._dedupe_starts(starts)

    base = {
        "case_id": spec.case_id,
        "kind": spec.kind,
        "label_or_node": spec.label_or_node,
        "target": spec.target,
        "side": spec.side,
        "value_source": spec.value_source,
        "strata": spec.strata,
        "fixed_value": fixed_value,
        "target_value": target_value,
        "target_std": target_std,
        "chi2_min": float(fit["chi2_min"]),
        "target_chi2": float(fit["chi2_min"]) + 1.0,
        "previous_best_chi2": baselines["previous_best_chi2"],
        "current_baseline_chi2": baselines["current_baseline_chi2"],
        "covariance_baseline_chi2": baselines["covariance_baseline_chi2"],
        "round09_covariance_minus_current_chi2": baselines["round09_covariance_minus_current_chi2"],
        "production_residual": parse_float(round09_row.get("production_residual")),
        "production_method": round09_row.get("production_profile_method"),
        "fit_runtime_sec": fit_cache.get(f"{spec.label_or_node}__runtime_sec"),
    }

    candidates = []
    candidate_rows: list[dict[str, Any]] = []
    for start_name, start_kind, start in starts[:1]:
        candidate = solver(start_name, start_kind, start)
        candidates.append(candidate)
        candidate_rows.append(candidate_to_row(candidate, spec, base, fit))

    endpoint_x = candidates[0].x if candidates else None
    starts = round03.add_endpoint_perturbation_starts(
        starts,
        endpoint_x,
        sqrt_cov,
        rng=rng,
        n_endpoint_random=args.endpoint_random_starts,
        endpoint_random_scale=args.endpoint_random_scale,
    )
    starts = round03._dedupe_starts(starts)
    for start_name, start_kind, start in starts[1:]:
        candidate = solver(start_name, start_kind, start)
        candidates.append(candidate)
        candidate_rows.append(candidate_to_row(candidate, spec, base, fit))

    ok = [candidate for candidate in candidates if candidate.ok and math.isfinite(candidate.chi2)]
    finite = [candidate for candidate in candidates if math.isfinite(candidate.chi2)]
    best = min(ok or finite, key=lambda candidate: candidate.chi2) if (ok or finite) else None
    if best is None:
        raise RuntimeError(f"no finite candidates for {spec}")
    best_residual = best.chi2 - base["target_chi2"]
    improvement_previous = base["previous_best_chi2"] - best.chi2
    improvement_current = base["current_baseline_chi2"] - best.chi2
    row = {
        **base,
        "status": "ok",
        "n_starts": len(candidates),
        "n_ok": sum(c.ok for c in candidates),
        "n_failed": sum(not c.ok for c in candidates),
        "best_chi2": best.chi2,
        "best_residual": best_residual,
        "best_start_name": best.start_name,
        "best_start_kind": best.start_kind,
        "best_method": best.method,
        "best_projected_grad_norm": best.projected_grad_norm,
        "best_scaled_constraint_violation": best.scaled_constraint_violation,
        "best_descent_improvement": best.descent_improvement,
        "best_nfev": best.nfev,
        "improvement_vs_previous_best": improvement_previous,
        "improvement_vs_current_baseline": improvement_current,
        "improvement_vs_covariance_baseline": (
            base["covariance_baseline_chi2"] - best.chi2
            if math.isfinite(base["covariance_baseline_chi2"]) else math.nan
        ),
        "material_lower_than_previous_best": bool(improvement_previous > DESCENT_FUN_TOL),
        "material_lower_than_current_baseline": bool(improvement_current > DESCENT_FUN_TOL),
        "endpoint_residual_failure_at_fixed_value": bool(abs(best_residual) > RESIDUAL_TOL),
        "endpoint_residual_hard_failure_at_fixed_value": bool(abs(best_residual) > max(2 * RESIDUAL_TOL, 1e-2)),
        "would_move_endpoint_materially": bool(
            abs(best_residual) > RESIDUAL_TOL and improvement_previous > MATERIAL_ENDPOINT_TOL
        ),
        "exception_type": "",
        "exception": "",
    }
    row.update(chart_fields(fit, best.x, "best"))
    return row, candidate_rows


def run_avg_fixed_target(
    spec: TargetSpec,
    round09_row: dict[str, Any],
    avg_cache: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not avg_cache:
        avg_df, corr_df_dict = avg_queries(verbose=False)
        avg_cache["avg_df"] = avg_df
        avg_cache["corr_df_dict"] = corr_df_dict
    avg_df = avg_cache["avg_df"]
    corr_df_dict = avg_cache["corr_df_dict"]
    result = run_avg(
        spec.label_or_node,
        avg_df[avg_df["node"] == spec.label_or_node],
        corr_df_dict[spec.label_or_node],
    )
    if result is None:
        raise RuntimeError(f"run_avg returned None for {spec.label_or_node}")
    baselines = fixed_value_and_baselines(spec, round09_row)
    fixed_value = baselines["fixed_value"]
    chi2 = result["chi2"]
    params = np.asarray(result["param_values"], dtype=np.float64).copy()
    primary_idx = list(result["parameters"]).index(spec.target)
    params[primary_idx] = fixed_value
    profile_chi2 = float(chi2(jnp.asarray(params, dtype=jnp.float64)))
    target_chi2 = float(result["chi2_min"]) + 1.0
    base = {
        "case_id": spec.case_id,
        "kind": spec.kind,
        "label_or_node": spec.label_or_node,
        "target": spec.target,
        "side": spec.side,
        "value_source": spec.value_source,
        "strata": spec.strata,
        "fixed_value": fixed_value,
        "target_value": parse_float(round09_row.get("target_value")),
        "target_std": None,
        "chi2_min": float(result["chi2_min"]),
        "target_chi2": target_chi2,
        "previous_best_chi2": baselines["previous_best_chi2"],
        "current_baseline_chi2": baselines["current_baseline_chi2"],
        "covariance_baseline_chi2": baselines["covariance_baseline_chi2"],
        "round09_covariance_minus_current_chi2": baselines["round09_covariance_minus_current_chi2"],
        "production_residual": parse_float(round09_row.get("production_residual")),
        "production_method": round09_row.get("production_profile_method"),
    }
    residual = profile_chi2 - target_chi2
    row = {
        **base,
        "status": "ok",
        "n_starts": 1,
        "n_ok": 1,
        "n_failed": 0,
        "best_chi2": profile_chi2,
        "best_residual": residual,
        "best_start_name": "direct_average_coordinate",
        "best_start_kind": "direct",
        "best_method": "direct",
        "best_projected_grad_norm": None,
        "best_scaled_constraint_violation": 0.0,
        "best_descent_improvement": 0.0,
        "best_nfev": 0,
        "improvement_vs_previous_best": base["previous_best_chi2"] - profile_chi2,
        "improvement_vs_current_baseline": base["current_baseline_chi2"] - profile_chi2,
        "improvement_vs_covariance_baseline": math.nan,
        "material_lower_than_previous_best": bool(base["previous_best_chi2"] - profile_chi2 > DESCENT_FUN_TOL),
        "material_lower_than_current_baseline": bool(base["current_baseline_chi2"] - profile_chi2 > DESCENT_FUN_TOL),
        "endpoint_residual_failure_at_fixed_value": bool(abs(residual) > RESIDUAL_TOL),
        "endpoint_residual_hard_failure_at_fixed_value": bool(abs(residual) > max(2 * RESIDUAL_TOL, 1e-2)),
        "would_move_endpoint_materially": bool(abs(residual) > RESIDUAL_TOL),
        "exception_type": "",
        "exception": "",
    }
    candidate = {
        **base,
        "record_type": "candidate",
        "start_name": "direct_average_coordinate",
        "start_kind": "direct",
        "candidate_ok": True,
        "candidate_chi2": profile_chi2,
        "candidate_residual": residual,
        "candidate_improvement_vs_previous_best": row["improvement_vs_previous_best"],
        "candidate_improvement_vs_current_baseline": row["improvement_vs_current_baseline"],
        "candidate_method": "direct",
        "candidate_constraint_violation": 0.0,
        "candidate_scaled_constraint_violation": 0.0,
        "candidate_projected_grad_norm": None,
        "candidate_objective_grad_norm": None,
        "candidate_descent_improvement": 0.0,
        "candidate_nfev": 0,
        "candidate_nit": 0,
        "candidate_runtime_sec": 0.0,
        "candidate_message": "one-parameter average direct fixed-coordinate evaluation",
    }
    return row, [candidate]


def run_stratified_search(
    specs: list[TargetSpec],
    round09_by_case_side: dict[tuple[str, str, str], dict[str, Any]],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    results: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    fit_cache: dict[str, Any] = {}
    avg_cache: dict[str, Any] = {}

    for spec in specs:
        row09 = row_for_spec(spec, round09_by_case_side)
        print(f"=== {spec.case_id} {spec.target} {spec.side} {spec.value_source} ===")
        t0 = time.perf_counter()
        try:
            if spec.kind == "fit":
                row, candidate_rows = run_fit_fixed_target(spec, row09, fit_cache, args)
            elif spec.kind == "avg":
                row, candidate_rows = run_avg_fixed_target(spec, row09, avg_cache)
            else:
                raise ValueError(f"unknown kind {spec.kind}")
            row["runtime_sec"] = time.perf_counter() - t0
            results.append(row)
            candidates.extend(candidate_rows)
            print(
                f"{spec.case_id}/{spec.side}/{spec.value_source}: "
                f"best_resid={row['best_residual']:.6g} "
                f"improve_prev={row['improvement_vs_previous_best']:.6g} "
                f"best={row['best_chi2']:.12g}"
            )
        except Exception as exc:  # noqa: BLE001 - diagnostics should continue
            failure = {
                "case_id": spec.case_id,
                "kind": spec.kind,
                "label_or_node": spec.label_or_node,
                "target": spec.target,
                "side": spec.side,
                "value_source": spec.value_source,
                "strata": spec.strata,
                "status": "error",
                "exception_type": type(exc).__name__,
                "exception": str(exc),
                "runtime_sec": time.perf_counter() - t0,
            }
            failures.append(failure)
            results.append(failure)
            print(f"FAILED {spec}: {type(exc).__name__}: {exc}")
    return results, candidates, failures


def main_impl(args: argparse.Namespace) -> int:
    round09_rows = read_csv_rows(Path(args.round09_results_csv))
    round09_by_case_side = {(row["case_id"], row["side"], ""): row for row in round09_rows}

    status_rows = build_status_summary(round09_rows)
    write_rows(Path(args.status_csv), status_rows)
    write_rows(Path(args.status_jsonl), status_rows, jsonl=True)

    specs = target_specs_from_status(status_rows, round09_by_case_side)
    if args.case:
        allowed = set(args.case)
        specs = [spec for spec in specs if spec.case_id in allowed or f"{spec.case_id}:{spec.side}" in allowed]

    print(f"Wrote {len(status_rows)} status rows; running {len(specs)} fixed-target checks.")
    results, candidate_rows, failures = run_stratified_search(specs, round09_by_case_side, args)
    write_rows(Path(args.results_csv), results)
    write_rows(Path(args.results_jsonl), results, jsonl=True)
    write_rows(Path(args.candidates_csv), candidate_rows)
    write_rows(Path(args.candidates_jsonl), candidate_rows, jsonl=True)
    write_rows(Path(args.failures_csv), failures)
    write_rows(Path(args.failures_jsonl), failures, jsonl=True)
    return 0 if not failures else 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--round09-results-csv", default="notes/codex-refactor/round09_hesse_profile_diagnostics_results.csv")
    parser.add_argument("--status-csv", default="notes/codex-refactor/round10_mncross_status_summary.csv")
    parser.add_argument("--status-jsonl", default="notes/codex-refactor/round10_mncross_status_summary.jsonl")
    parser.add_argument("--results-csv", default="notes/codex-refactor/round10_stratified_globality_results.csv")
    parser.add_argument("--results-jsonl", default="notes/codex-refactor/round10_stratified_globality_results.jsonl")
    parser.add_argument("--candidates-csv", default="notes/codex-refactor/round10_stratified_globality_candidates.csv")
    parser.add_argument("--candidates-jsonl", default="notes/codex-refactor/round10_stratified_globality_candidates.jsonl")
    parser.add_argument("--failures-csv", default="notes/codex-refactor/round10_stratified_globality_failures.csv")
    parser.add_argument("--failures-jsonl", default="notes/codex-refactor/round10_stratified_globality_failures.jsonl")
    parser.add_argument("--stdout-log", default="notes/logs/round10_mncross_stratified_globality_stdout.log")
    parser.add_argument("--case", action="append", help="optional case_id or case_id:side filter")
    parser.add_argument("--seed", type=int, default=20260625)
    parser.add_argument("--cov-dirs", type=int, default=2)
    parser.add_argument("--cov-scale", type=float, default=0.75)
    parser.add_argument("--random-starts", type=int, default=4)
    parser.add_argument("--random-scale", type=float, default=0.75)
    parser.add_argument("--endpoint-random-starts", type=int, default=4)
    parser.add_argument("--endpoint-random-scale", type=float, default=0.35)
    args = parser.parse_args()

    stdout_log = Path(args.stdout_log)
    stdout_log.parent.mkdir(parents=True, exist_ok=True)
    with stdout_log.open("w", encoding="utf-8") as log:
        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            code = main_impl(args)
    raise SystemExit(code)


if __name__ == "__main__":
    main()
