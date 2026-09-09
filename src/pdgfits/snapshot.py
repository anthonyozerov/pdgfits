"""Capture and replay the small DB surface used by pdgfits.

Usage:
    python -m pdgfits.snapshot capture --out data/pdg-snapshot

Then run offline with:
    PDGFITS_DATA_BACKEND=snapshot \
    PDGFITS_SNAPSHOT_DIR=data/pdg-snapshot \
    python -m pdgfits.run_fits --fit_label eta_958
"""
import argparse
import copy
import io
import json
import os
from contextlib import redirect_stdout
from datetime import datetime, timezone

import pandas as pd

import pdgfits.preprocess as pp
import pdgfits.query as query


SKIP_LABELS = {"tauhflav"}


def _capture_pdg_most_precise_value(node, seen):
    if node in seen:
        return query._read_snapshot_pickle(
            "pdg_most_precise_value", f"{query._snapshot_key(node)}.pkl"
        )
    out = query.pdg_most_precise_value(node)
    query._write_snapshot_pickle(
        out, "pdg_most_precise_value", f"{query._snapshot_key(node)}.pkl"
    )
    seen.add(node)
    return out


def _capture_nuisance_corr(nuisance_params, seen):
    key = query._nuisance_corr_key(nuisance_params)
    if key in seen:
        return query._read_snapshot_pickle("nuisance_corr", f"{key}.pkl")
    out = query.nuisance_corr(nuisance_params, verbose=False)
    query._write_snapshot_pickle(out, "nuisance_corr", f"{key}.pkl")
    seen.add(key)
    return out


def _patch_preprocess_db_calls(seen_pmpv, seen_ncorr):
    real_pmpv = pp.pdg_most_precise_value
    real_ncorr = pp.nuisance_corr

    def rec_pmpv(node):
        return _capture_pdg_most_precise_value(node, seen_pmpv)

    def rec_ncorr(nuisance_params, verbose=True):
        return _capture_nuisance_corr(nuisance_params, seen_ncorr)

    pp.pdg_most_precise_value = rec_pmpv
    pp.nuisance_corr = rec_ncorr
    return real_pmpv, real_ncorr


def _restore_preprocess_db_calls(real_pmpv, real_ncorr):
    pp.pdg_most_precise_value = real_pmpv
    pp.nuisance_corr = real_ncorr


def _capture_fit(label, seen_pmpv, seen_ncorr, quiet=True):
    inputs = query.fit_queries(label, verbose=False)
    query._write_snapshot_pickle(
        copy.deepcopy(inputs), "fits", f"{query._snapshot_key(label)}.pkl"
    )

    algorithm, measurement_type, *dfs = copy.deepcopy(inputs)
    fit_df, rel_df, meas_df, corr_df, fit_seed_df, tree_df = dfs
    real_pmpv, real_ncorr = _patch_preprocess_db_calls(seen_pmpv, seen_ncorr)
    try:
        if quiet:
            with redirect_stdout(io.StringIO()):
                outputs = pp.preprocess(
                    fit_df, rel_df, meas_df, corr_df, fit_seed_df, tree_df,
                    algorithm, measurement_type,
                )
        else:
            outputs = pp.preprocess(
                fit_df, rel_df, meas_df, corr_df, fit_seed_df, tree_df,
                algorithm, measurement_type,
            )
    finally:
        _restore_preprocess_db_calls(real_pmpv, real_ncorr)

    # Optional diagnostics use PDG summary values for fitted parameters/nodes.
    out_fit_df, out_rel_df, _, _, out_fit_seed_df, _, _ = outputs
    for parameter in out_fit_seed_df.get("parameter_key", pd.Series(dtype=str)).dropna().unique():
        node = str(parameter).removeprefix("nuisance_")
        _capture_pdg_most_precise_value(node, seen_pmpv)
    for node in out_rel_df.get("node", pd.Series(dtype=str)).dropna().unique():
        _capture_pdg_most_precise_value(str(node), seen_pmpv)
    for node in out_fit_df.get("node", pd.Series(dtype=str)).dropna().unique():
        _capture_pdg_most_precise_value(str(node), seen_pmpv)


def _avg_preprocess_inputs(node, meas_df_node, corr_df_node):
    meas_df_node = meas_df_node.copy()
    for col in ("systematic_error_clump", "systematic_error_clump2"):
        if col in meas_df_node.columns:
            meas_df_node[col] = None

    fit_df = pd.DataFrame({"node": [node], "type": ["+"], "data_type": [None]})
    rel_df = pd.DataFrame({
        "node": [node], "par_code": [None], "parameter": [node],
        "coefficient": [1.0], "summation": [1],
        "coeff_par_code": [None], "coeff_parameter": [None],
    })
    fit_seed_df = pd.DataFrame({
        "par_code": [None], "parameter": [node], "seed": [0.0],
    })
    tree_df = pd.DataFrame({
        "node": pd.Series([], dtype=str),
        "data_type": pd.Series([], dtype=str),
    })
    return fit_df, rel_df, meas_df_node, corr_df_node.copy(), fit_seed_df, tree_df


def _capture_average_preprocess(avg_df, corr_df_dict, nodes, seen_pmpv, seen_ncorr, quiet=True):
    real_pmpv, real_ncorr = _patch_preprocess_db_calls(seen_pmpv, seen_ncorr)
    try:
        for node in nodes:
            inputs = _avg_preprocess_inputs(
                node, avg_df[avg_df["node"] == node], corr_df_dict[node]
            )
            if quiet:
                with redirect_stdout(io.StringIO()):
                    pp.preprocess(*inputs, algorithm="AVG", measurement_type="")
            else:
                pp.preprocess(*inputs, algorithm="AVG", measurement_type="")
            # Manual use of diagnostics.compare_to_pdg()/meas_diagnostics() on
            # an average result needs the PDG summary value for the averaged node.
            try:
                _capture_pdg_most_precise_value(str(node), seen_pmpv)
            except (AssertionError, ValueError):
                query._write_snapshot_pickle(
                    (None, None, None),
                    "pdg_most_precise_value",
                    f"{query._snapshot_key(node)}.pkl",
                )
                seen_pmpv.add(str(node))
    finally:
        _restore_preprocess_db_calls(real_pmpv, real_ncorr)


def _select_fit_labels(fits_df, requested, include_tauhflav):
    if requested:
        return requested
    labels = [
        row["label"]
        for _, row in fits_df.iterrows()
        if row["algorithm"] != "IGNORE"
    ]
    if not include_tauhflav:
        labels = [label for label in labels if label not in SKIP_LABELS]
    return labels


def capture(args):
    previous_backend = os.environ.get(query.DATA_BACKEND_ENV)
    previous_snapshot_dir = os.environ.get(query.SNAPSHOT_DIR_ENV)
    os.environ[query.DATA_BACKEND_ENV] = "db"
    os.environ[query.SNAPSHOT_DIR_ENV] = args.out

    seen_pmpv = set()
    seen_ncorr = set()
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "format": "pdgfits-simple-snapshot-v1",
        "fits": [],
        "fit_errors": {},
        "averages": None,
        "average_errors": {},
    }

    try:
        os.makedirs(args.out, exist_ok=True)
        fits_df = query.all_fits()
        query._write_snapshot_pickle(fits_df, "all_fits.pkl")

        if not args.no_fits:
            labels = _select_fit_labels(fits_df, args.fit_label, args.include_tauhflav)
            if args.limit is not None:
                labels = labels[:args.limit]
            for i, label in enumerate(labels, 1):
                print(f"[fit {i}/{len(labels)}] {label}")
                try:
                    _capture_fit(label, seen_pmpv, seen_ncorr, quiet=not args.verbose)
                except Exception as exc:  # noqa: BLE001 - continue capturing others
                    manifest["fit_errors"][label] = f"{type(exc).__name__}: {exc}"
                    print(f"  skipped: {manifest['fit_errors'][label]}")
                    continue
                manifest["fits"].append(label)

        if not args.no_averages:
            print("[averages] querying")
            avg_df, corr_df_dict = query.avg_queries(verbose=False)
            query._write_snapshot_pickle((avg_df, corr_df_dict), "avg_queries.pkl")
            nodes = list(avg_df["node"].unique())
            if args.avg_node:
                requested = set(args.avg_node)
                nodes = [node for node in nodes if node in requested]
            if args.limit is not None:
                nodes = nodes[:args.limit]
            print(f"[averages] preparing {len(nodes)} nodes")
            try:
                _capture_average_preprocess(
                    avg_df, corr_df_dict, nodes, seen_pmpv, seen_ncorr,
                    quiet=not args.verbose,
                )
            except Exception as exc:  # noqa: BLE001 - snapshot is still useful
                manifest["average_errors"]["preprocess"] = f"{type(exc).__name__}: {exc}"
                print(f"  average preprocess skipped: {manifest['average_errors']['preprocess']}")
            manifest["averages"] = nodes

        manifest["pdg_most_precise_value_count"] = len(seen_pmpv)
        manifest["nuisance_corr_count"] = len(seen_ncorr)
        with open(os.path.join(args.out, "manifest.json"), "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, sort_keys=True)
    finally:
        if previous_backend is None:
            os.environ.pop(query.DATA_BACKEND_ENV, None)
        else:
            os.environ[query.DATA_BACKEND_ENV] = previous_backend
        if previous_snapshot_dir is None:
            os.environ.pop(query.SNAPSHOT_DIR_ENV, None)
        else:
            os.environ[query.SNAPSHOT_DIR_ENV] = previous_snapshot_dir

    print(f"Snapshot written to {args.out}")
    print(
        f"Captured {len(manifest['fits'])} fits, "
        f"{0 if manifest['averages'] is None else len(manifest['averages'])} average nodes, "
        f"{manifest['pdg_most_precise_value_count']} PDG value lookups, "
        f"{manifest['nuisance_corr_count']} nuisance-correlation lookups."
    )


def main():
    parser = argparse.ArgumentParser(description="Capture a small offline pdgfits snapshot.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    capture_parser = subparsers.add_parser("capture")
    capture_parser.add_argument("--out", required=True, help="Output snapshot directory")
    capture_parser.add_argument("--fit-label", action="append", help="Capture one fit label; repeatable")
    capture_parser.add_argument("--avg-node", action="append", help="Prepare one average node; repeatable")
    capture_parser.add_argument("--no-fits", action="store_true", help="Do not capture fit_queries() outputs")
    capture_parser.add_argument("--no-averages", action="store_true", help="Do not capture avg_queries() outputs")
    capture_parser.add_argument("--include-tauhflav", action="store_true", help="Include tauhflav in all-fit capture")
    capture_parser.add_argument("--limit", type=int, help="Capture only the first N fits/average nodes")
    capture_parser.add_argument("--verbose", action="store_true", help="Show preprocessing output during capture")
    capture_parser.set_defaults(func=capture)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
