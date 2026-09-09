#!/usr/bin/env python
"""Round 15 tiny Wilks-calibration smoke test.

This is a notes-only exploratory harness.  It runs a small parametric toy
smoke test under the fitted snapshot chi2 model, then records whether the
profile-likelihood test statistic for the fitted "truth" visibly deviates from
the chi-square-1 heuristic on risk strata identified in round 13.

It is deliberately not a coverage study.
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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import jax
from jax import numpy as jnp
import numpy as np

from pdgfits.asym_errors import (
    CallableProfileProblem,
    build_constrained_profile_chi2,
    find_profile_root,
)
from pdgfits.build_funcs import (
    get_adjust,
    get_mu_adjust,
    get_mu_vectorized,
    get_node_funcs,
    get_parameter_funcs,
    get_translate_dep,
)
from pdgfits.corr_mat import get_corr_mat
from pdgfits.fit import (
    _condition_seed_units,
    _large_coordinate_scale_info,
    _run_minuit_candidate,
    _run_unconstrained_scipy,
    run_fit,
)
from pdgfits.param_maps import (
    build_param_map_scaled,
    build_param_map_sigmoid,
    build_param_map_softmax,
    get_decay_info,
)
from pdgfits.preprocess import preprocess
from pdgfits.query import fit_queries


jax.config.update("jax_enable_x64", True)


TOY_FIELDNAMES = [
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

ENDPOINT_FIELDNAMES = [
    "case_id",
    "stratum",
    "label",
    "target",
    "toy_index",
    "seed",
    "side",
    "status",
    "truth_target",
    "toy_target_mle",
    "endpoint",
    "error",
    "endpoint_contains_truth_side",
    "profile_chi2",
    "target_chi2",
    "residual",
    "profile_success",
    "profile_method",
    "profile_nfev",
    "profile_nit",
    "profile_runtime_sec",
    "constraint_violation",
    "scaled_constraint_violation",
    "projected_grad_norm",
    "objective_grad_norm",
    "descent_improvement",
    "side_function_evals",
    "side_bracket_evals",
    "side_bisect_evals",
    "profile_message",
]

FAILURE_FIELDNAMES = [
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


@dataclass(frozen=True)
class CaseSpec:
    case_id: str
    stratum: str
    label: str
    target: str
    rationale: str


DEFAULT_CASES = [
    CaseSpec(
        "fit_b0_s042b95",
        "boundary_tiny_parameter",
        "B0",
        "S042B95",
        "round-13 high-risk boundary/tiny BRU coordinate case",
    ),
    CaseSpec(
        "fit_b0s_s08637",
        "asymmetric_profile_shape",
        "B0S-BR",
        "S086.37",
        "round-13 medium-high asymmetric/profile-shape case; cheaper than eta M026G01",
    ),
    CaseSpec(
        "fit_g2000_k002m",
        "clean_control",
        "G(2000),G(1800)",
        "K002M",
        "round-13 locally-quadratic interior direct-coordinate control",
    ),
]


def clean_value(value: Any) -> Any:
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return value if math.isfinite(value) else None
    if value is None:
        return None
    return value


def json_default(obj: Any) -> Any:
    cleaned = clean_value(obj)
    if cleaned is not obj:
        return cleaned
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def finite_or_none(value: Any) -> Any:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def clean_message(message: Any, limit: int = 500) -> str:
    text = "" if message is None else str(message).replace("\n", " ")
    return text[:limit]


def append_row(path: Path, row: dict[str, Any], fieldnames: list[str], *, jsonl: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if jsonl:
        with path.open("a", encoding="utf-8") as handle:
            cleaned = {key: clean_value(row.get(key)) for key in fieldnames}
            handle.write(json.dumps(cleaned, default=json_default, sort_keys=True) + "\n")
        return

    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        if write_header:
            writer.writeheader()
        writer.writerow({key: clean_value(row.get(key)) for key in fieldnames})


def ensure_artifact(path: Path, fieldnames: list[str], *, jsonl: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        return
    if jsonl:
        path.touch()
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()


def reset_outputs(paths: list[Path]) -> None:
    for path in paths:
        if path.exists():
            path.unlink()


def completed_keys(path: Path) -> set[tuple[str, int]]:
    if not path.exists() or path.stat().st_size == 0:
        return set()
    out: set[tuple[str, int]] = set()
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("status") != "ok":
                continue
            try:
                out.add((str(row["case_id"]), int(row["toy_index"])))
            except (KeyError, TypeError, ValueError):
                continue
    return out


def target_func_for_fit(fit: dict[str, Any], target: str) -> Callable[[Any], Any]:
    if target in fit["nodes"]:
        return fit["node_funcs"][fit["nodes"].index(target)]
    if target in fit["parameters"]:
        return fit["parameter_funcs"][fit["parameters"].index(target)]
    raise KeyError(f"{target} not found in fit nodes or parameters")


def as_float(value: Any) -> float:
    return float(np.asarray(value, dtype=np.float64))


def target_std_for_fit(fit: dict[str, Any], target_func: Callable[[Any], Any], target_value: float) -> float:
    covariance = fit.get("covariance")
    if covariance is None:
        return max(abs(float(target_value)), 1.0) * 1e-3
    fitted_values = fit["fitted_values"]
    param_values = fit["param_values"]
    fitted_to_params = fit["fitted_params_to_params"]
    try:
        jac_params = jax.jacobian(fitted_to_params)(fitted_values)
        param_cov = jac_params @ covariance @ jac_params.T
        jac_target = jax.jacobian(target_func)(param_values)
        target_var = as_float(jac_target @ param_cov @ jac_target.T)
        target_std = math.sqrt(target_var) if target_var >= 0 else math.nan
    except Exception:  # noqa: BLE001 - diagnostic fallback
        target_std = math.nan
    if not math.isfinite(target_std) or target_std <= 0:
        target_std = max(abs(float(target_value)), 1.0) * 1e-3
    return float(target_std)


def build_fit_context(label: str, reference_fit: dict[str, Any]) -> dict[str, Any]:
    """Rebuild enough preprocessed context to get the fitted model mean."""
    algorithm, measurement_type, fit_df, rel_df, meas_df, corr_df, fit_seed_df, tree_df = fit_queries(
        label,
        verbose=False,
    )
    fit_df, rel_df, meas_df, corr_df, fit_seed_df, dep_meas_data, adjust_data = preprocess(
        fit_df,
        rel_df,
        meas_df,
        corr_df,
        fit_seed_df,
        tree_df,
        algorithm,
        measurement_type,
    )
    parameters = list(fit_seed_df["parameter_key"].unique())
    nodes = list(np.unique(rel_df["node"]))
    particles = list(np.unique([n[:4] for n in nodes + parameters if not n.startswith("nuisance_")]))

    node_funcs = get_node_funcs(nodes, parameters, fit_df, rel_df)
    parameter_funcs = get_parameter_funcs(parameters)
    mu = get_mu_vectorized(parameters, nodes, meas_df, fit_df, rel_df)
    translate_dep = get_translate_dep(dep_meas_data, parameters, nodes, parameter_funcs, node_funcs)
    adjust = get_adjust(adjust_data, parameters, nodes, parameter_funcs, node_funcs)
    mu_adjust = get_mu_adjust(mu, adjust, translate_dep)
    corr_mat = np.asarray(get_corr_mat(meas_df, corr_df), dtype=np.float64)

    is_bru = algorithm == "BRU" or (
        algorithm in ["BR", "BR (NO MATRIX)", "BR PRINT"] and len(particles) > 1
    )
    is_br = algorithm in ["BR", "BR (NO MATRIX)", "BR PRINT"] and len(particles) == 1
    if is_bru:
        _, _, decay_param_idxs = build_param_map_sigmoid(parameters)
    elif is_br:
        _, _, decay_param_idxs, _ = build_param_map_softmax(parameters)
    else:
        decay_param_idxs = None

    param_init = jnp.array(
        [fit_seed_df[fit_seed_df["parameter_key"] == p]["seed"].iloc[0] for p in parameters],
        dtype=jnp.float64,
    )
    if decay_param_idxs is not None:
        for particle in particles:
            bool_select = np.array([p.startswith(particle + ".") for p in parameters])
            if not bool_select.any():
                continue
            decay_seeds = param_init[bool_select]
            if is_br:
                decay_seed_sum = decay_seeds.sum()
                if decay_seed_sum > 0:
                    param_init = param_init.at[bool_select].set(decay_seeds / decay_seed_sum)
            elif decay_seeds.max() > 1:
                decay_seeds = decay_seeds / 100
                param_init = param_init.at[bool_select].set(decay_seeds)
            param_init = param_init.at[bool_select].set(
                jnp.clip(param_init[bool_select], 1e-6, 1 - 1e-6)
            )
    param_init = _condition_seed_units(
        parameters,
        fit_df,
        rel_df,
        meas_df,
        param_init,
        skip_idxs=decay_param_idxs,
    )
    scale_idxs, scales = _large_coordinate_scale_info(
        parameters,
        param_init,
        skip_idxs=decay_param_idxs,
    )
    if len(scale_idxs) > 0:
        # The reference fit already owns the actual composed map.  This block is
        # only a consistency check for cases with large-coordinate scaling.
        build_param_map_scaled(lambda x: x, lambda x: x, scale_idxs, scales)

    if parameters != list(reference_fit["parameters"]):
        raise RuntimeError(f"parameter order mismatch for {label}")
    if list(nodes) != list(reference_fit["nodes"]):
        raise RuntimeError(f"node order mismatch for {label}")

    return {
        "algorithm": algorithm,
        "parameters": parameters,
        "nodes": nodes,
        "meas_df": meas_df,
        "corr_mat": corr_mat,
        "mu_adjust": mu_adjust,
        "error_n": np.asarray(meas_df["error_n"], dtype=np.float64),
        "error_p": np.asarray(meas_df["error_p"], dtype=np.float64),
        "y_observed": np.asarray(meas_df["value"], dtype=np.float64),
    }


def sym_error_at_zero(error_n: np.ndarray, error_p: np.ndarray) -> np.ndarray:
    return 2.0 * error_n * error_p / (error_n + error_p)


def sampling_factor(corr_mat: np.ndarray, sigma: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    corr = np.asarray(corr_mat, dtype=np.float64)
    corr = 0.5 * (corr + corr.T)
    corr_eig = np.linalg.eigvalsh(corr)
    cov = (sigma[:, None] * corr) * sigma[None, :]
    cov = 0.5 * (cov + cov.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    clipped = np.clip(eigvals, 0.0, None)
    factor = eigvecs @ np.diag(np.sqrt(clipped))
    scale = max(float(np.max(np.abs(eigvals))), 1.0)
    info = {
        "corr_min_eig": finite_or_none(np.min(corr_eig)),
        "corr_n_negative_eig": int(np.sum(corr_eig < -1e-10)),
        "cov_min_eig": finite_or_none(np.min(eigvals)),
        "cov_n_clipped_eig": int(np.sum(eigvals < -1e-12 * scale)),
    }
    return factor, info


def chart_summary(fit: dict[str, Any], fitted_values: np.ndarray) -> dict[str, Any]:
    params = np.asarray(
        fit["fitted_params_to_params"](jnp.asarray(fitted_values, dtype=jnp.float64)),
        dtype=np.float64,
    )
    _, decay_idxs = get_decay_info(list(fit["parameters"]))
    decay_idxs = np.asarray(decay_idxs, dtype=int)
    out = {
        "n_decay_params": int(len(decay_idxs)),
        "min_decay_param_mle": None,
        "max_decay_param_mle": None,
        "max_abs_decay_fitted_coord_mle": None,
        "decay_active_lower_1e8_mle": None,
        "decay_active_lower_1e6_mle": None,
        "decay_active_upper_1e8_mle": None,
        "chart_saturation_warning_mle": False,
    }
    if len(decay_idxs) == 0:
        return out
    decay_params = params[decay_idxs]
    decay_fitted = np.asarray(fitted_values, dtype=np.float64)[decay_idxs]
    min_decay = float(np.nanmin(decay_params))
    max_decay = float(np.nanmax(decay_params))
    max_abs_coord = float(np.nanmax(np.abs(decay_fitted)))
    out.update(
        {
            "min_decay_param_mle": min_decay,
            "max_decay_param_mle": max_decay,
            "max_abs_decay_fitted_coord_mle": max_abs_coord,
            "decay_active_lower_1e8_mle": int(np.sum(decay_params <= 1e-8)),
            "decay_active_lower_1e6_mle": int(np.sum(decay_params <= 1e-6)),
            "decay_active_upper_1e8_mle": int(np.sum(decay_params >= 1 - 1e-8)),
            "chart_saturation_warning_mle": bool(
                max_abs_coord > 10 or min_decay < 1e-6 or max_decay > 1 - 1e-6
            ),
        }
    )
    return out


def refit_toy(fit: dict[str, Any], y_toy: np.ndarray) -> tuple[dict[str, Any], Callable[[Any], Any]]:
    y_jax = jnp.asarray(y_toy, dtype=jnp.float64)
    chi2_open = fit["chi2_open"]

    def chi2(fp):
        return chi2_open(fp, y_jax)

    val_and_grad = jax.jit(jax.value_and_grad(chi2))
    grad_jax = jax.jit(jax.grad(chi2))
    hessp_jax = jax.jit(lambda x, p: jax.jvp(jax.grad(chi2), (x,), (p,))[1])

    def chi2_val(x):
        return float(chi2(jnp.asarray(x, dtype=jnp.float64)))

    def chi2_grad(x):
        return np.asarray(grad_jax(jnp.asarray(x, dtype=jnp.float64)), dtype=np.float64)

    def scipy_obj(x):
        val, grad = val_and_grad(jnp.asarray(x, dtype=jnp.float64))
        return float(val), np.asarray(grad, dtype=np.float64)

    def scipy_hessp(x, p):
        return np.asarray(
            hessp_jax(jnp.asarray(x, dtype=jnp.float64), jnp.asarray(p, dtype=jnp.float64)),
            dtype=np.float64,
        )

    x0 = np.asarray(fit["fitted_values"], dtype=np.float64)
    refit_info: dict[str, Any] = {"refit_method": "minuit"}
    with contextlib.redirect_stdout(io.StringIO()):
        m = _run_minuit_candidate(chi2_val, chi2_grad, x0, False, None, None)
    x_hat = np.asarray(m.values, dtype=np.float64)
    f_hat = float(m.fval)
    finite_minuit = np.isfinite(f_hat) and np.all(np.isfinite(x_hat))
    if finite_minuit:
        refit_info.update(
            {
                "refit_success": bool(m.valid),
                "refit_valid": bool(m.valid),
                "refit_accurate": bool(m.accurate),
                "refit_nfcn": int(m.nfcn),
                "refit_ngrad": int(m.ngrad),
                "refit_message": "",
                "refit_edm": finite_or_none(m.fmin.edm),
            }
        )
        if bool(m.valid):
            return {
                "x_hat": x_hat,
                "chi2_min": f_hat,
                **refit_info,
            }, chi2

    scipy = _run_unconstrained_scipy(scipy_obj, chi2_val, scipy_hessp, x0)
    if np.isfinite(getattr(scipy, "fun", np.inf)) and np.all(np.isfinite(getattr(scipy, "x", []))):
        if (not finite_minuit) or float(scipy.fun) < f_hat + max(1e-8, 1e-10 * abs(f_hat)):
            return {
                "x_hat": np.asarray(scipy.x, dtype=np.float64),
                "chi2_min": float(scipy.fun),
                "refit_method": "scipy_fallback",
                "refit_success": bool(scipy.success),
                "refit_valid": None,
                "refit_accurate": None,
                "refit_nfcn": getattr(scipy, "nfev", None),
                "refit_ngrad": getattr(scipy, "njev", None),
                "refit_message": clean_message(getattr(scipy, "message", "")),
                "refit_edm": None,
            }, chi2

    if not finite_minuit:
        raise RuntimeError("toy refit failed: Minuit and scipy fallback returned non-finite results")
    return {
        "x_hat": x_hat,
        "chi2_min": f_hat,
        **refit_info,
    }, chi2


def endpoint_row(
    case: CaseSpec,
    toy_index: int,
    seed: int,
    side: str,
    root: Any,
    truth_target: float,
    toy_target: float,
) -> dict[str, Any]:
    endpoint = root.lower if side == "lower" else root.upper
    contains = (
        endpoint.endpoint <= truth_target <= toy_target
        if side == "lower"
        else toy_target <= truth_target <= endpoint.endpoint
    )
    point = endpoint.point
    return {
        "case_id": case.case_id,
        "stratum": case.stratum,
        "label": case.label,
        "target": case.target,
        "toy_index": toy_index,
        "seed": seed,
        "side": side,
        "status": "ok",
        "truth_target": truth_target,
        "toy_target_mle": toy_target,
        "endpoint": endpoint.endpoint,
        "error": endpoint.error,
        "endpoint_contains_truth_side": bool(contains),
        "profile_chi2": endpoint.chi2,
        "target_chi2": root.target_chi2,
        "residual": endpoint.residual,
        "profile_success": point.success,
        "profile_method": point.method,
        "profile_nfev": point.nfev,
        "profile_nit": point.nit,
        "profile_runtime_sec": point.runtime_sec,
        "constraint_violation": point.constraint_violation,
        "scaled_constraint_violation": point.scaled_constraint_violation,
        "projected_grad_norm": point.projected_grad_norm,
        "objective_grad_norm": point.objective_grad_norm,
        "descent_improvement": point.descent_improvement,
        "side_function_evals": endpoint.counts.get("function_evals"),
        "side_bracket_evals": endpoint.counts.get("bracket_evals"),
        "side_bisect_evals": endpoint.counts.get("bisect_evals"),
        "profile_message": clean_message(point.message),
    }


def run_case_toy(
    case: CaseSpec,
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
    sigma = sym_error_at_zero(context["error_n"], context["error_p"])
    factor, sample_info = sampling_factor(context["corr_mat"], sigma)
    z = rng.standard_normal(len(mean_y))
    noise = factor @ z
    y_toy = mean_y + noise
    toy_noise_mahalanobis = float(np.dot(z, z))

    refit, toy_chi2 = refit_toy(fit, y_toy)
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
        "profile_true_message": clean_message(true_point.message),
        "wilks_q_le_1": bool(q_true <= 1.0),
        "wilks_q_le_chisq1_median": bool(q_true <= 0.454936423119572),
        "refit_profile_consistency_failure": bool(q_true < -1e-4),
        "toy_noise_norm": float(np.linalg.norm(noise)),
        "toy_noise_mahalanobis_clipped": toy_noise_mahalanobis,
        "endpoint_requested": bool(toy_index < args.endpoint_toys_per_case),
        "endpoint_status": "not_requested",
        **refit,
        **chart_summary(fit, x_hat),
        **sample_info,
    }
    row.pop("x_hat", None)

    endpoint_rows: list[dict[str, Any]] = []
    if toy_index < args.endpoint_toys_per_case:
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
        endpoint_rows.append(endpoint_row(case, toy_index, seed, "lower", root, truth_target, toy_target))
        endpoint_rows.append(endpoint_row(case, toy_index, seed, "upper", root, truth_target, toy_target))

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
    for case_id in sorted({row["case_id"] for row in rows}):
        case_rows = [row for row in rows if row["case_id"] == case_id and row.get("status") == "ok"]
        q = np.asarray([float(row["profile_true_delta_chi2"]) for row in case_rows], dtype=float)
        q = q[np.isfinite(q)]
        min_decay_values = np.asarray(
            [float(row.get("min_decay_param_mle") or "nan") for row in case_rows],
            dtype=float,
        )
        finite_min_decay = min_decay_values[np.isfinite(min_decay_values)]
        endpoint_case_rows = [
            row for row in case_rows if row.get("endpoint_status") == "ok"
        ]
        contains = [
            str(row.get("endpoint_interval_contains_truth")).lower() == "true"
            for row in endpoint_case_rows
        ]
        q_le_1 = [value <= 1.0 for value in q]
        case_summaries.append(
            {
                "case_id": case_id,
                "stratum": case_rows[0].get("stratum") if case_rows else None,
                "n_ok_toys": int(len(q)),
                "n_endpoint_toys": int(len(endpoint_case_rows)),
                "q_mean": finite_or_none(np.mean(q)) if len(q) else None,
                "q_median": finite_or_none(np.median(q)) if len(q) else None,
                "q_min": finite_or_none(np.min(q)) if len(q) else None,
                "q_max": finite_or_none(np.max(q)) if len(q) else None,
                "q_le_1_count": int(sum(q_le_1)),
                "q_le_1_fraction": finite_or_none(np.mean(q_le_1)) if len(q_le_1) else None,
                "endpoint_contains_truth_count": int(sum(contains)),
                "endpoint_contains_truth_fraction": finite_or_none(np.mean(contains)) if contains else None,
                "max_abs_endpoint_residual": finite_or_none(
                    np.nanmax(
                        np.abs(
                            [
                                float(row.get("endpoint_lower_residual") or "nan")
                                for row in endpoint_case_rows
                            ]
                            + [
                                float(row.get("endpoint_upper_residual") or "nan")
                                for row in endpoint_case_rows
                            ]
                        )
                    )
                )
                if endpoint_case_rows
                else None,
                "min_decay_param_min": finite_or_none(np.min(finite_min_decay))
                if len(finite_min_decay)
                else None,
                "chart_saturation_toys": int(
                    sum(str(row.get("chart_saturation_warning_mle")).lower() == "true" for row in case_rows)
                ),
            }
        )

    summary = {
        "metadata": metadata,
        "n_toy_rows": len(rows),
        "n_endpoint_rows": len(endpoint_rows),
        "n_failure_rows": len(failure_rows),
        "case_summaries": case_summaries,
        "expected_chisq1_q_le_1_fraction": 0.6826894921370859,
        "expected_chisq1_median": 0.454936423119572,
        "limitations": [
            "local Gaussian toys use symmetrized errors at zero residual, not the full asymmetric interpolation as a generative distribution",
            "correlation matrices are eigenvalue-clipped only for toy generation",
            "toy counts are intentionally too small for coverage claims",
            "boundary activity is recorded at the toy MLE; constrained profile points do not expose optimized coordinates",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True, default=json_default), encoding="utf-8")


def parse_cases(selected: list[str] | None) -> list[CaseSpec]:
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
    parser.add_argument("--toys-per-case", type=int, default=6)
    parser.add_argument("--endpoint-toys-per-case", type=int, default=2)
    parser.add_argument("--base-seed", type=int, default=2026062515)
    parser.add_argument("--bracket-sigmas", type=float, default=3.0)
    parser.add_argument("--endpoint-residual-tol", type=float, default=5e-3)
    parser.add_argument("--max-runtime-sec", type=float, default=900.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--toy-csv", type=Path, default=Path("notes/codex-refactor/round15_wilks_smoke_toys.csv"))
    parser.add_argument("--toy-jsonl", type=Path, default=Path("notes/codex-refactor/round15_wilks_smoke_toys.jsonl"))
    parser.add_argument(
        "--endpoint-csv",
        type=Path,
        default=Path("notes/codex-refactor/round15_wilks_smoke_endpoints.csv"),
    )
    parser.add_argument(
        "--endpoint-jsonl",
        type=Path,
        default=Path("notes/codex-refactor/round15_wilks_smoke_endpoints.jsonl"),
    )
    parser.add_argument(
        "--failure-csv",
        type=Path,
        default=Path("notes/codex-refactor/round15_wilks_smoke_failures.csv"),
    )
    parser.add_argument(
        "--failure-jsonl",
        type=Path,
        default=Path("notes/codex-refactor/round15_wilks_smoke_failures.jsonl"),
    )
    parser.add_argument("--summary-json", type=Path, default=Path("notes/codex-refactor/round15_wilks_smoke_summary.json"))
    args = parser.parse_args()

    if args.endpoint_toys_per_case > args.toys_per_case:
        raise ValueError("--endpoint-toys-per-case cannot exceed --toys-per-case")
    output_paths = [
        args.toy_csv,
        args.toy_jsonl,
        args.endpoint_csv,
        args.endpoint_jsonl,
        args.failure_csv,
        args.failure_jsonl,
        args.summary_json,
    ]
    if args.overwrite and args.resume:
        raise ValueError("--overwrite and --resume are mutually exclusive")
    if args.overwrite:
        reset_outputs(output_paths)
    ensure_artifact(args.toy_csv, TOY_FIELDNAMES)
    ensure_artifact(args.toy_jsonl, TOY_FIELDNAMES, jsonl=True)
    ensure_artifact(args.endpoint_csv, ENDPOINT_FIELDNAMES)
    ensure_artifact(args.endpoint_jsonl, ENDPOINT_FIELDNAMES, jsonl=True)
    ensure_artifact(args.failure_csv, FAILURE_FIELDNAMES)
    ensure_artifact(args.failure_jsonl, FAILURE_FIELDNAMES, jsonl=True)

    cases = parse_cases(args.cases)
    done = completed_keys(args.toy_csv) if args.resume else set()
    start_time = time.perf_counter()
    metadata = {
        "toys_per_case": args.toys_per_case,
        "endpoint_toys_per_case": args.endpoint_toys_per_case,
        "base_seed": args.base_seed,
        "case_ids": [case.case_id for case in cases],
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
            context = build_fit_context(case.label, fit)
        if fit is None:
            raise RuntimeError(f"run_fit returned None for {case.label}")
        target_func = target_func_for_fit(fit, case.target)
        truth_target = float(target_func(jnp.asarray(fit["param_values"], dtype=jnp.float64)))
        target_std = target_std_for_fit(fit, target_func, truth_target)
        print(
            f"case {case.case_id}: npar={len(fit['parameters'])} nmeas={len(context['y_observed'])} "
            f"truth={truth_target:.8g} target_std={target_std:.4g}",
            flush=True,
        )

        for toy_index in range(args.toys_per_case):
            if time.perf_counter() - start_time > args.max_runtime_sec:
                break
            if (case.case_id, toy_index) in done:
                continue
            seed = args.base_seed + 100_000 * case_index + toy_index
            t_toy = time.perf_counter()
            try:
                row, endpoint_rows = run_case_toy(
                    case,
                    fit,
                    context,
                    target_func,
                    target_std,
                    toy_index,
                    seed,
                    args,
                )
                append_row(args.toy_csv, row, TOY_FIELDNAMES)
                append_row(args.toy_jsonl, row, TOY_FIELDNAMES, jsonl=True)
                for erow in endpoint_rows:
                    append_row(args.endpoint_csv, erow, ENDPOINT_FIELDNAMES)
                    append_row(args.endpoint_jsonl, erow, ENDPOINT_FIELDNAMES, jsonl=True)
                print(
                    f"  toy {toy_index}: q={row['profile_true_delta_chi2']:.4g} "
                    f"covered={row['wilks_q_le_1']} endpoint={row['endpoint_status']} "
                    f"runtime={row['runtime_sec']:.1f}s",
                    flush=True,
                )
            except Exception as exc:  # noqa: BLE001 - artifact records exact failure
                failure = {
                    "case_id": case.case_id,
                    "stratum": case.stratum,
                    "label": case.label,
                    "target": case.target,
                    "toy_index": toy_index,
                    "seed": seed,
                    "phase": "toy",
                    "exception_type": type(exc).__name__,
                    "exception": clean_message(exc, limit=1000),
                    "runtime_sec": time.perf_counter() - t_toy,
                }
                append_row(args.failure_csv, failure, FAILURE_FIELDNAMES)
                append_row(args.failure_jsonl, failure, FAILURE_FIELDNAMES, jsonl=True)
                print(f"  toy {toy_index}: FAILED {type(exc).__name__}: {exc}", flush=True)

    metadata["elapsed_sec"] = time.perf_counter() - start_time
    summarize(args.toy_csv, args.endpoint_csv, args.failure_csv, args.summary_json, metadata)
    print(f"wrote summary: {args.summary_json}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
