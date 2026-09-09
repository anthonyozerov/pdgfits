#!/usr/bin/env python
"""Round 07 notes-only iminuit/MINOS comparator for direct-coordinate targets."""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import numpy as np
from iminuit import Minuit
import iminuit
from jax import numpy as jnp

from pdgfits.asym_errors import calc_asym_errors
from pdgfits.avg import run_avg
from pdgfits.fit import run_fit
from pdgfits.query import avg_queries


jax.config.update("jax_enable_x64", True)


@dataclass(frozen=True)
class CaseSpec:
    case_id: str
    kind: str
    label_or_node: str
    target: str
    rationale: str
    expected_fairness: str


DEFAULT_CASES = [
    CaseSpec(
        case_id="avg_m002w",
        kind="avg",
        label_or_node="M002W",
        target="M002W",
        rationale="one-parameter average; profiled value is the only coordinate",
        expected_fairness="fair_direct_coordinate",
    ),
    CaseSpec(
        case_id="avg_m026r08",
        kind="avg",
        label_or_node="M026R08",
        target="M026R08",
        rationale="average primary coordinate with two nuisance adjustments",
        expected_fairness="fair_direct_coordinate_with_nuisance",
    ),
    CaseSpec(
        case_id="fit_g2000_k002m",
        kind="fit",
        label_or_node="G(2000),G(1800)",
        target="K002M",
        rationale="two-parameter MASS fit; target is an actual fitted coordinate",
        expected_fairness="fair_direct_fit_coordinate",
    ),
    CaseSpec(
        case_id="fit_g2000_k003m",
        kind="fit",
        label_or_node="G(2000),G(1800)",
        target="K003M",
        rationale="same two-parameter MASS fit; second direct fitted coordinate",
        expected_fairness="fair_direct_fit_coordinate",
    ),
    CaseSpec(
        case_id="fit_eta_m026w",
        kind="fit",
        label_or_node="eta_c J/psi psi(2S)",
        target="M026W",
        rationale=(
            "hard round-05/06 target; M026W is a non-decay fit parameter, so MINOS "
            "can profile the same coordinate without pretending a node function is a parameter"
        ),
        expected_fairness="fair_hard_direct_fit_coordinate",
    ),
]


REJECTED_CASES = [
    {
        "case_id": "reject_b0_s042b95",
        "label_or_node": "B0",
        "target": "S042B95",
        "reason": (
            "S042B95 is a relationship/node ratio target, not a Minuit coordinate. "
            "Running MINOS on any one underlying B0 decay coordinate would answer a different question."
        ),
    },
    {
        "case_id": "reject_b0s_s08637_default_chart",
        "label_or_node": "B0S-BR",
        "target": "S086.37",
        "reason": (
            "S086.37 is a physical BRU parameter but production profiles it through the sigmoid "
            "mapped fitted chart. MINOS on the default fitted coordinate would give an internal "
            "chart-coordinate interval, while MINOS on a bounded physical chart changes the numerical "
            "profile formulation. This is a useful future caveated experiment, not a clean MINOS oracle."
        ),
    },
]


def finite_or_none(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (bool, str, int)):
        return value
    try:
        out = float(value)
    except (TypeError, ValueError):
        return value
    return out if math.isfinite(out) else None


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


def clean_message(message: Any) -> str:
    text = str(message)
    return " ".join(text.split())


def make_minuit(chi2, grad, start, names, *, limits: dict[str, tuple[float | None, float | None]] | None = None) -> Minuit:
    names = list(names)
    start = np.asarray(start, dtype=np.float64)

    def chi2_np(x):
        value = chi2(jnp.asarray(x, dtype=jnp.float64))
        return float(np.asarray(value, dtype=np.float64))

    def grad_np(x):
        value = grad(jnp.asarray(x, dtype=jnp.float64))
        return np.asarray(value, dtype=np.float64)

    m = Minuit(chi2_np, start, grad=grad_np, name=names)
    m.errordef = 1.0
    if limits:
        for name, lim in limits.items():
            m.limits[name] = lim
    return m


def run_minos(chi2, grad, start, names, target, *, limits=None) -> dict[str, Any]:
    t0 = time.perf_counter()
    m = make_minuit(chi2, grad, start, names, limits=limits)
    m.migrad()
    m.hesse()
    migrad_runtime_sec = time.perf_counter() - t0

    t1 = time.perf_counter()
    try:
        m.minos(target)
        minos_exception = None
    except Exception as exc:  # noqa: BLE001 - notes-only diagnostic records failure
        minos_exception = f"{type(exc).__name__}: {exc}"
    minos_runtime_sec = time.perf_counter() - t1

    target_value = float(m.values[target])
    try:
        hesse_error = float(m.errors[target])
    except Exception:  # noqa: BLE001 - ErrorView membership is not dict-like
        hesse_error = math.nan
    merror = m.merrors.get(target)

    result = {
        "minuit_version": iminuit.__version__,
        "minuit_valid": bool(m.valid),
        "minuit_accurate": bool(m.accurate),
        "minuit_fval": float(m.fval),
        "minuit_edm": finite_or_none(getattr(m.fmin, "edm", None)),
        "minuit_nfcn": int(m.nfcn),
        "minuit_ngrad": int(m.ngrad),
        "minuit_migrad_runtime_sec": migrad_runtime_sec,
        "minuit_minos_runtime_sec": minos_runtime_sec,
        "minuit_value": target_value,
        "hesse_error": hesse_error,
        "minos_exception": minos_exception,
        "minos_lower_error": None,
        "minos_upper_error": None,
        "minos_lower_endpoint": None,
        "minos_upper_endpoint": None,
        "minos_lower_valid": False,
        "minos_upper_valid": False,
        "minos_lower_new_min": None,
        "minos_upper_new_min": None,
        "minos_lower_at_limit": None,
        "minos_upper_at_limit": None,
        "minos_nfcn": None,
        "minuit_values": np.asarray(m.values, dtype=np.float64),
    }
    if merror is not None:
        result.update(
            {
                "minos_lower_error": float(merror.lower),
                "minos_upper_error": float(merror.upper),
                "minos_lower_endpoint": target_value + float(merror.lower),
                "minos_upper_endpoint": target_value + float(merror.upper),
                "minos_lower_valid": bool(merror.lower_valid),
                "minos_upper_valid": bool(merror.upper_valid),
                "minos_lower_new_min": bool(merror.lower_new_min),
                "minos_upper_new_min": bool(merror.upper_new_min),
                "minos_lower_at_limit": bool(merror.at_lower_limit),
                "minos_upper_at_limit": bool(merror.at_upper_limit),
                "minos_nfcn": int(merror.nfcn),
            }
        )
    return result


def fixed_coordinate_profile(
    chi2,
    grad,
    start,
    names,
    target,
    fixed_value,
    *,
    limits=None,
) -> dict[str, Any]:
    names = list(names)
    idx = names.index(target)
    fixed_value = float(fixed_value)
    start = np.asarray(start, dtype=np.float64).copy()
    start[idx] = fixed_value
    if len(start) == 1:
        m = make_minuit(chi2, grad, start, names, limits=limits)
        value = float(m.fcn(start))
        return {
            "profile_chi2": value,
            "profile_valid": True,
            "profile_accurate": None,
            "profile_nfcn": 1,
            "profile_ngrad": 0,
            "profile_message": "direct_one_coordinate",
            "profile_runtime_sec": 0.0,
        }

    t0 = time.perf_counter()
    m = make_minuit(chi2, grad, start, names, limits=limits)
    m.values[target] = fixed_value
    m.fixed[target] = True
    m.migrad()
    message_parts = ["migrad"]
    if not m.valid:
        m.simplex()
        m.migrad()
        message_parts.append("simplex+migrad")
    runtime = time.perf_counter() - t0
    return {
        "profile_chi2": float(m.fval),
        "profile_valid": bool(m.valid),
        "profile_accurate": bool(m.accurate),
        "profile_nfcn": int(m.nfcn),
        "profile_ngrad": int(m.ngrad),
        "profile_message": "+".join(message_parts),
        "profile_runtime_sec": runtime,
    }


def pdgfit_result_for_case(case: CaseSpec, avg_data_cache: dict[str, Any]) -> dict[str, Any]:
    if case.kind == "avg":
        if not avg_data_cache:
            avg_df, corr_df_dict = avg_queries(verbose=False)
            avg_data_cache["avg_df"] = avg_df
            avg_data_cache["corr_df_dict"] = corr_df_dict
        avg_df = avg_data_cache["avg_df"]
        corr_df_dict = avg_data_cache["corr_df_dict"]
        node = case.label_or_node
        result = run_avg(node, avg_df[avg_df["node"] == node], corr_df_dict[node])
        if result is None:
            raise RuntimeError(f"run_avg returned None for {node}")
        value = float(result["param_values"][result["parameters"].index(case.target)])
        asym = {
            "value": value,
            "error_p": float(result["error_p"]),
            "error_n": float(result["error_n"]),
            **result["asym_error_diagnostics"],
        }
        return result | {"round07_asym": asym}

    if case.kind == "fit":
        fit = run_fit(case.label_or_node, verbose=False)
        if fit is None:
            raise RuntimeError(f"run_fit returned None for {case.label_or_node}")
        if case.target not in fit["parameters"]:
            raise RuntimeError(f"{case.target} is not a direct parameter in {case.label_or_node}")
        asym_by_target = calc_asym_errors(fit, targets=[case.target])
        return fit | {"round07_asym": asym_by_target[case.target]}

    raise ValueError(f"unknown case kind {case.kind}")


def rows_for_case(case: CaseSpec, pdg_result: dict[str, Any]) -> list[dict[str, Any]]:
    names = list(pdg_result["parameters"])
    target = case.target
    start = np.asarray(pdg_result["fitted_values"], dtype=np.float64)
    chi2 = pdg_result["chi2"]
    grad = pdg_result["chi2_grad"]
    chi2_min = float(pdg_result["chi2_min"])
    asym = pdg_result["round07_asym"]
    target_value = float(asym["value"])
    minos = run_minos(chi2, grad, start, names, target)
    target_chi2 = chi2_min + 1.0
    minuit_target_chi2 = float(minos["minuit_fval"]) + 1.0

    rows = []
    for side in ("lower", "upper"):
        sign = -1.0 if side == "lower" else 1.0
        pdg_endpoint = float(asym[f"{side}_endpoint"])
        pdg_error = float(asym[f"{side}_error"])
        minos_endpoint = minos[f"minos_{side}_endpoint"]
        minos_error = minos[f"minos_{side}_error"]
        minos_valid = bool(minos[f"minos_{side}_valid"])
        hesse_error = float(minos["hesse_error"])
        hesse_endpoint = float(minos["minuit_value"] + sign * hesse_error)

        fresh_pdg = fixed_coordinate_profile(chi2, grad, start, names, target, pdg_endpoint)
        if minos_endpoint is not None:
            fresh_minos = fixed_coordinate_profile(chi2, grad, start, names, target, minos_endpoint)
        else:
            fresh_minos = {
                "profile_chi2": None,
                "profile_valid": False,
                "profile_accurate": None,
                "profile_nfcn": None,
                "profile_ngrad": None,
                "profile_message": "minos_endpoint_missing",
                "profile_runtime_sec": None,
            }
        fresh_hesse = fixed_coordinate_profile(chi2, grad, start, names, target, hesse_endpoint)

        minos_abs_error = abs(float(minos_error)) if minos_error is not None else None
        hesse_minos_error_delta = (
            minos_abs_error - hesse_error if minos_abs_error is not None and math.isfinite(hesse_error) else None
        )
        row = {
            "case_id": case.case_id,
            "kind": case.kind,
            "label_or_node": case.label_or_node,
            "target": target,
            "side": side,
            "rationale": case.rationale,
            "fairness": case.expected_fairness,
            "n_parameters": len(names),
            "chi2_min_pdg": chi2_min,
            "target_chi2_pdg": target_chi2,
            "target_value_pdg": target_value,
            "pdg_endpoint": pdg_endpoint,
            "pdg_error": pdg_error,
            "pdg_endpoint_chi2_from_repo": float(asym[f"{side}_chi2"]),
            "pdg_residual_from_repo": float(asym[f"{side}_residual"]),
            "pdg_profile_method": asym.get(f"{side}_profile_point", {}).get("method"),
            "pdg_profile_message": asym.get(f"{side}_profile_point", {}).get("message"),
            "pdg_root_function_evals": asym.get("function_evals"),
            "pdg_root_cache_hits": asym.get("cache_hits"),
            "minuit_version": minos["minuit_version"],
            "minuit_valid": minos["minuit_valid"],
            "minuit_accurate": minos["minuit_accurate"],
            "minuit_fval": minos["minuit_fval"],
            "minuit_delta_chi2_min": float(minos["minuit_fval"]) - chi2_min,
            "minuit_edm": minos["minuit_edm"],
            "minuit_nfcn_total": minos["minuit_nfcn"],
            "minuit_ngrad_total": minos["minuit_ngrad"],
            "minuit_value": minos["minuit_value"],
            "minuit_value_delta_pdg": float(minos["minuit_value"]) - target_value,
            "hesse_error": hesse_error,
            "hesse_endpoint": hesse_endpoint,
            "minos_endpoint": minos_endpoint,
            "minos_error": minos_error,
            "minos_abs_error": minos_abs_error,
            "minos_valid": minos_valid,
            "minos_new_min": minos[f"minos_{side}_new_min"],
            "minos_at_limit": minos[f"minos_{side}_at_limit"],
            "minos_nfcn": minos["minos_nfcn"],
            "minos_exception": minos["minos_exception"],
            "endpoint_delta_minos_minus_pdg": (
                float(minos_endpoint) - pdg_endpoint if minos_endpoint is not None else None
            ),
            "error_delta_minos_minus_pdg": (
                minos_abs_error - pdg_error if minos_abs_error is not None else None
            ),
            "hesse_minos_error_delta": hesse_minos_error_delta,
            "hesse_pdg_error_delta": hesse_error - pdg_error,
            "fresh_pdg_profile_chi2": fresh_pdg["profile_chi2"],
            "fresh_pdg_residual_vs_pdg_target": (
                float(fresh_pdg["profile_chi2"]) - target_chi2
                if fresh_pdg["profile_chi2"] is not None else None
            ),
            "fresh_pdg_profile_valid": fresh_pdg["profile_valid"],
            "fresh_pdg_profile_nfcn": fresh_pdg["profile_nfcn"],
            "fresh_pdg_profile_message": clean_message(fresh_pdg["profile_message"]),
            "fresh_minos_profile_chi2": fresh_minos["profile_chi2"],
            "fresh_minos_residual_vs_pdg_target": (
                float(fresh_minos["profile_chi2"]) - target_chi2
                if fresh_minos["profile_chi2"] is not None else None
            ),
            "fresh_minos_residual_vs_minuit_target": (
                float(fresh_minos["profile_chi2"]) - minuit_target_chi2
                if fresh_minos["profile_chi2"] is not None else None
            ),
            "fresh_minos_profile_valid": fresh_minos["profile_valid"],
            "fresh_minos_profile_nfcn": fresh_minos["profile_nfcn"],
            "fresh_minos_profile_message": clean_message(fresh_minos["profile_message"]),
            "fresh_hesse_profile_chi2": fresh_hesse["profile_chi2"],
            "fresh_hesse_residual_vs_pdg_target": (
                float(fresh_hesse["profile_chi2"]) - target_chi2
                if fresh_hesse["profile_chi2"] is not None else None
            ),
            "fresh_hesse_profile_valid": fresh_hesse["profile_valid"],
            "fresh_hesse_profile_nfcn": fresh_hesse["profile_nfcn"],
        }
        rows.append(row)
    return rows


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
    failures: list[dict[str, Any]] = []
    avg_data_cache: dict[str, Any] = {}
    print(f"iminuit_version={iminuit.__version__}")
    for case in cases:
        print(f"\n=== {case.case_id}: {case.kind} {case.label_or_node} / {case.target} ===")
        t0 = time.perf_counter()
        try:
            pdg_result = pdgfit_result_for_case(case, avg_data_cache)
            case_rows = rows_for_case(case, pdg_result)
            rows.extend(case_rows)
            print(f"completed {case.case_id} in {time.perf_counter() - t0:.3f}s")
            for row in case_rows:
                print(
                    row["side"],
                    "pdg",
                    row["pdg_endpoint"],
                    "minos",
                    row["minos_endpoint"],
                    "fresh_minos_resid",
                    row["fresh_minos_residual_vs_pdg_target"],
                    "hesse_delta",
                    row["hesse_minos_error_delta"],
                )
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

    write_rows(Path(args.results_csv), rows)
    write_rows(Path(args.results_jsonl), rows, jsonl=True)
    write_rows(Path(args.failures_csv), failures)
    write_rows(Path(args.failures_jsonl), failures, jsonl=True)
    write_rows(Path(args.rejections_csv), REJECTED_CASES)
    write_rows(Path(args.rejections_jsonl), REJECTED_CASES, jsonl=True)
    print(f"\nwrote {len(rows)} result rows")
    print(f"wrote {len(failures)} failure rows")
    return 0 if not failures else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append", help="Case id to run; repeatable. Defaults to all cases.")
    parser.add_argument("--results-csv", default="notes/codex-refactor/round07_minos_comparator_results.csv")
    parser.add_argument("--results-jsonl", default="notes/codex-refactor/round07_minos_comparator_results.jsonl")
    parser.add_argument("--failures-csv", default="notes/codex-refactor/round07_minos_comparator_failures.csv")
    parser.add_argument("--failures-jsonl", default="notes/codex-refactor/round07_minos_comparator_failures.jsonl")
    parser.add_argument("--rejections-csv", default="notes/codex-refactor/round07_minos_comparator_rejections.csv")
    parser.add_argument("--rejections-jsonl", default="notes/codex-refactor/round07_minos_comparator_rejections.jsonl")
    parser.add_argument("--stdout-log", default=None)
    parser.add_argument("--keep-going", action="store_true", default=True)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.stdout_log:
        path = Path(args.stdout_log)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f, contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
            return main_impl(args)
    return main_impl(args)


if __name__ == "__main__":
    sys.exit(main())
