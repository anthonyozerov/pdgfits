"""Capture golden fixtures for preprocess() characterization tests.

Run ONCE against a live DB tunnel to snapshot, for every non-IGNORE fit:
  - the inputs to preprocess() (copied BEFORE the call, since preprocess mutates them)
  - the results of the two DB calls preprocess makes internally
    (pdg_most_precise_value, nuisance_corr), so the golden test can replay
    them with no DB
  - the outputs of preprocess()

Usage (with tunnel up and `pdg` env active):
    python tests/fixtures/capture_preprocess_fixtures.py

Writes one pickle per fit to tests/fixtures/preprocess/<label>.pkl and prints a
coverage summary of which special-case paths each fixture exercises.
"""
import io
import os
import re
import sys
import copy
import pickle
import warnings
from contextlib import redirect_stdout

warnings.filterwarnings("ignore")

import jax
jax.config.update("jax_enable_x64", True)

import pdgfits.preprocess as pp
from pdgfits.query import all_fits, fit_queries

FIX_DIR = os.path.join(os.path.dirname(__file__), "preprocess")

# Labels to skip: IGNORE algorithm is filtered separately; tauhflav is excluded
# by the integration suite too (pathological / very slow).
SKIP_LABELS = {"tauhflav"}


def slugify(label):
    """Filesystem-safe stem for a fit label (the true label lives in the pickle)."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", label).strip("_")


def _snapshot_inputs(inputs):
    """Deep-copy the (algorithm, measurement_type, *dfs) input tuple."""
    return copy.deepcopy(inputs)


def detect_paths(inputs):
    """Cheaply tag which preprocess special-case branches a fit will hit.

    Used only for the coverage summary, computed from raw inputs.
    """
    algorithm, measurement_type, fit_df, rel_df, meas_df, corr_df, fit_seed_df, tree_df = inputs
    paths = set()
    meas = meas_df["measurement"].astype(str)
    if measurement_type == "UNIV" or any(
        n in list(meas_df["node"]) for n in ["S024DM", "S024DTT", "S022DM"]
    ):
        paths.add("univ_equality")
    if meas.str.contains("dep_meas").any():
        paths.add("dep_meas")
    if meas.str.contains("br_adjust").any():
        paths.add("br_adjust")
    if meas_df["ignore_minus"].notna().any():
        paths.add("ignore_minus")
    if meas_df["text"].isin(["keV", "eV"]).any():
        paths.add("unit_scale")
    if ((fit_df["data_type"] == "T") & (algorithm != "SPECIALT")).any():
        paths.add("lifetime")
    if fit_df["type"].isin(["G+", "G*", "R+"]).any():
        paths.add("partial_width_sum")
    if meas_df["systematic_error_clump2"].notna().any():
        paths.add("clump2")
    if meas_df["systematic_error_clump"].notna().any():
        paths.add("clump")
    return paths


def capture_one(label):
    """Run fit_queries + preprocess for one label, recording DB calls.

    Returns a fixture dict, or raises on failure.
    """
    inputs = fit_queries(label, verbose=False)
    inputs_snapshot = _snapshot_inputs(inputs)

    # Record the two DB calls preprocess makes internally so the golden test
    # can replay them offline.
    recorded_pmpv = {}   # node -> (value, error_p, error_n)
    recorded_ncorr = {}  # frozenset(nuisance_params) -> DataFrame

    real_pmpv = pp.pdg_most_precise_value
    real_ncorr = pp.nuisance_corr

    def rec_pmpv(node):
        out = real_pmpv(node)
        recorded_pmpv[node] = out
        return out

    def rec_ncorr(nuisance_params, verbose=True):
        out = real_ncorr(nuisance_params, verbose=False)
        recorded_ncorr[frozenset(nuisance_params)] = out
        return out

    algorithm, measurement_type, fit_df, rel_df, meas_df, corr_df, fit_seed_df, tree_df = inputs
    pp.pdg_most_precise_value = rec_pmpv
    pp.nuisance_corr = rec_ncorr
    try:
        buf = io.StringIO()
        with redirect_stdout(buf):
            outputs = pp.preprocess(
                fit_df, rel_df, meas_df, corr_df, fit_seed_df, tree_df,
                algorithm, measurement_type,
            )
    finally:
        pp.pdg_most_precise_value = real_pmpv
        pp.nuisance_corr = real_ncorr

    return {
        "label": label,
        "inputs": inputs_snapshot,
        "outputs": outputs,
        "recorded_pmpv": recorded_pmpv,
        "recorded_ncorr": recorded_ncorr,
        "paths": detect_paths(inputs_snapshot),
    }


def main():
    os.makedirs(FIX_DIR, exist_ok=True)
    fits_df = all_fits()
    labels = [
        row["label"]
        for _, row in fits_df.iterrows()
        if row["algorithm"] != "IGNORE" and row["label"] not in SKIP_LABELS
    ]
    print(f"Capturing {len(labels)} fits into {FIX_DIR}")

    coverage = {}
    captured, skipped = [], []
    for i, label in enumerate(labels, 1):
        try:
            fix = capture_one(label)
        except Exception as e:  # noqa: BLE001 - log and continue
            print(f"  [{i}/{len(labels)}] SKIP {label}: {type(e).__name__}: {e}")
            skipped.append((label, repr(e)))
            continue
        with open(os.path.join(FIX_DIR, f"{i:03d}_{slugify(label)}.pkl"), "wb") as f:
            pickle.dump(fix, f)
        for p in fix["paths"]:
            coverage.setdefault(p, []).append(label)
        captured.append(label)
        print(f"  [{i}/{len(labels)}] OK   {label}  paths={sorted(fix['paths'])}")

    print(f"\nCaptured {len(captured)}, skipped {len(skipped)}")
    print("\nPath coverage (#fits exercising each branch):")
    for p in sorted(coverage):
        print(f"  {p:20s} {len(coverage[p]):3d}  e.g. {coverage[p][0]}")
    uncovered = {
        "univ_equality", "dep_meas", "br_adjust", "ignore_minus", "unit_scale",
        "lifetime", "partial_width_sum", "clump2", "clump",
    } - set(coverage)
    if uncovered:
        print(f"\nWARNING: no fixture exercises: {sorted(uncovered)}")
    if skipped:
        print("\nSkipped labels:")
        for label, err in skipped:
            print(f"  {label}: {err}")


if __name__ == "__main__":
    sys.exit(main())
