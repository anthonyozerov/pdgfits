"""Capture a numerical regression set from the local offline PDG snapshot.

Run in separate processes against the source before and after a refactor.
Includes all fit central values/covariances, hard profile targets, all averages
with nuisance parameters, and a spread of direct averages. No live DB calls.
"""

import argparse
from collections import defaultdict
import contextlib
import csv
import gc
import io
import json
import os
from pathlib import Path
import time

import jax
import numpy as np

from pdgfits.asym_errors import calc_asym_errors
from pdgfits.avg import run_avg
from pdgfits.fit import run_fit
from pdgfits.query import all_fits, avg_queries


def capture(output):
    if os.environ.get("PDGFITS_DATA_BACKEND") != "snapshot":
        raise RuntimeError("Set PDGFITS_DATA_BACKEND=snapshot explicitly")
    root = Path(__file__).resolve().parents[1]
    with (root / "notes/asym_fit_sweep_results.csv").open() as f:
        old = list(csv.DictReader(f))
    targets = defaultdict(list)
    for r in old:
        if r["status"] != "ok":
            targets[r["label"]].append(r["target"])
    targets["G(2000),G(1800)"] = ["K003DM"]
    targets["Upsilon(2S)"] = ["M052R22", "nuisance_M048.8"]
    targets["Lam-b-0"] = ["S040R29"]
    labels = [r.label for r in all_fits().itertuples()
              if r.algorithm != "IGNORE" and r.label != "tauhflav"]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as stream:
        def write(row):
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            stream.flush()

        for index, label in enumerate(labels):
            t0 = time.monotonic()
            with contextlib.redirect_stdout(io.StringIO()):
                fit = run_fit(label, verbose=False)
                if fit is None:
                    raise RuntimeError(f"Fit skipped: {label}")
                profiles = calc_asym_errors(fit, targets[label]) if label in targets else {}
            write({"id": f"fit:{label}", "chi2": float(fit["chi2_min"]),
                   "parameters": fit["parameters"], "values": np.asarray(fit["param_values"]).tolist(),
                   "fitted_values": np.asarray(fit["fitted_values"]).tolist(),
                   "covariance": np.asarray(fit["covariance"]).tolist(),
                   "nodes": fit["nodes"],
                   "node_values": [float(f(fit["param_values"])) for f in fit["node_funcs"]],
                   "profiles": {k: {f: float(v[f]) for f in ["value", "error_p", "error_n", "upper_residual", "lower_residual"]}
                                for k, v in profiles.items()}})
            print(f"fit {index+1}/{len(labels)} {label}: {time.monotonic()-t0:.1f}s", flush=True)
            del fit
            jax.clear_caches()
            gc.collect()

        for label, space in [("B0S-BR", "unconstrained"), ("B0S-BR", "constrained"),
                             ("K_L eta+-,00 phase", "unconstrained"),
                             ("K_L eta+-,00 ph CPT", "unconstrained")]:
            with contextlib.redirect_stdout(io.StringIO()):
                fit = run_fit(label, verbose=False, optimizer="scipy", fit_space=space)
            write({"id": f"scipy:{label}:{space}", "chi2": float(fit["chi2_min"]),
                   "values": np.asarray(fit["param_values"]).tolist(),
                   "covariance": np.asarray(fit["covariance"]).tolist()})
            del fit
            jax.clear_caches()
            gc.collect()

        with (root / "notes/asym_avg_sweep_results.csv").open() as f:
            rows = list(csv.DictReader(f))
        selected = {r["node"] for r in rows if int(r["n_nuisance"]) > 0}
        selected.update(r["node"] for r in rows[::45])
        selected.update(["M002W", "M026R08"])
        data, correlations = avg_queries(verbose=False)
        for index, node in enumerate(sorted(selected)):
            with contextlib.redirect_stdout(io.StringIO()):
                avg = run_avg(node, data[data.node == node], correlations[node])
            if avg is None:
                raise RuntimeError(f"Average skipped: {node}")
            write({"id": f"average:{node}", "parameters": avg["parameters"],
                   "values": np.asarray(avg["param_values"]).tolist(),
                   "chi2": float(avg["chi2_min"]), "error_p": float(avg["error_p"]),
                   "error_n": float(avg["error_n"]),
                   "upper_residual": float(avg["asym_error_diagnostics"]["upper_residual"]),
                   "lower_residual": float(avg["asym_error_diagnostics"]["lower_residual"])})
            if (index+1) % 25 == 0:
                print(f"average {index+1}/{len(selected)}", flush=True)
            del avg
            if index % 10 == 0:
                jax.clear_caches()
                gc.collect()
    print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    capture(parser.parse_args().output)
