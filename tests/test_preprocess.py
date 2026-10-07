"""Characterization (golden) tests for preprocess().

These pin the *current* behavior of preprocess so the decomposition refactor can
be proven byte-for-byte equivalent. They run with NO database: each fixture
carries the inputs, the recorded results of the two DB calls preprocess makes
internally (pdg_most_precise_value, nuisance_corr), and the expected outputs.

Fixtures are generated locally (and gitignored) from a live DB via:

    python tests/fixtures/capture_preprocess_fixtures.py

If no fixtures are present the golden tests skip; the synthetic tests below
always run.
"""
import os
import copy
import glob
import pickle

import pytest
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

import jax
jax.config.update("jax_enable_x64", True)

import pdgfits.preprocess as pp

FIX_DIR = os.path.join(os.path.dirname(__file__), "fixtures", "preprocess")
FIXTURES = sorted(glob.glob(os.path.join(FIX_DIR, "*.pkl")))

# Names of the five DataFrames in the preprocess output tuple, for clear diffs.
OUTPUT_DF_NAMES = ["fit_df", "rel_df", "meas_df", "corr_df", "fit_seed_df"]


def _load(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def _patch_db_calls(monkeypatch, fix):
    """Replay the DB calls preprocess makes internally from recorded results."""
    pmpv = fix["recorded_pmpv"]
    ncorr = fix["recorded_ncorr"]

    def fake_pmpv(node):
        if node not in pmpv:
            raise AssertionError(
                f"preprocess called pdg_most_precise_value({node!r}) but it was "
                f"not recorded in the fixture (recorded: {sorted(pmpv)})"
            )
        return pmpv[node]

    def fake_ncorr(nuisance_params, verbose=True):
        key = frozenset(nuisance_params)
        if key not in ncorr:
            raise AssertionError(
                f"preprocess called nuisance_corr({nuisance_params!r}) but it was "
                f"not recorded in the fixture"
            )
        return ncorr[key]

    monkeypatch.setattr(pp, "pdg_most_precise_value", fake_pmpv)
    monkeypatch.setattr(pp, "nuisance_corr", fake_ncorr)


def _assert_structure_equal(got, want, ctx):
    """Recursively compare the list-of-(None|tuple|list) outputs.

    Handles dep_meas_data and adjust_data, whose elements are None, tuples, or
    lists of strings/floats.
    """
    assert type(got) == type(want), f"{ctx}: type {type(got)} != {type(want)}"
    if isinstance(want, (list, tuple)):
        assert len(got) == len(want), f"{ctx}: len {len(got)} != {len(want)}"
        for i, (g, w) in enumerate(zip(got, want)):
            _assert_structure_equal(g, w, f"{ctx}[{i}]")
    elif isinstance(want, float):
        assert got == pytest.approx(want, nan_ok=True), f"{ctx}: {got} != {want}"
    else:
        assert got == want, f"{ctx}: {got!r} != {want!r}"


def _canonicalize(df):
    """Sort rows into a deterministic order for comparison.

    Row order is NOT a contract of preprocess: nuisance rows are appended while
    iterating a set of strings (preprocess.py), so the current code's row order
    is already non-deterministic across processes. Everything downstream is built
    consistently from the same meas_df, so order is semantically irrelevant. We
    therefore compare sets-of-rows, not sequences.
    """
    if len(df) == 0:
        return df.reset_index(drop=True)
    key = ["|".join(map(str, row)) for row in df.itertuples(index=False, name=None)]
    return df.assign(_k=key).sort_values("_k", kind="stable").drop(
        columns="_k"
    ).reset_index(drop=True)


def _assert_outputs_equal(got, want):
    assert len(got) == len(want) == 7, "preprocess should return a 7-tuple"
    for name, g, w in zip(OUTPUT_DF_NAMES, got[:5], want[:5]):
        assert isinstance(g, pd.DataFrame), f"{name} is not a DataFrame"
        assert set(g.columns) == set(w.columns), (
            f"{name}: columns differ: {set(g.columns) ^ set(w.columns)}"
        )
        assert_frame_equal(
            _canonicalize(g), _canonicalize(w[g.columns]), obj=name,
            check_like=False,
        )
    # dep_meas_data / adjust_data are per-measurement lists; their non-None
    # entries sit in the deterministic (pre-nuisance) prefix, and nuisance
    # entries are all None, so positional comparison is stable.
    _assert_structure_equal(got[5], want[5], "dep_meas_data")
    _assert_structure_equal(got[6], want[6], "adjust_data")


@pytest.mark.skipif(not FIXTURES, reason="no preprocess fixtures; run the capture script")
@pytest.mark.parametrize(
    "fix_path", FIXTURES, ids=[os.path.basename(p)[:-4] for p in FIXTURES]
)
def test_preprocess_golden(fix_path, monkeypatch):
    fix = _load(fix_path)
    _patch_db_calls(monkeypatch, fix)

    # preprocess mutates its inputs in place, so feed it a fresh deep copy.
    algorithm, measurement_type, *dfs = copy.deepcopy(fix["inputs"])
    fit_df, rel_df, meas_df, corr_df, fit_seed_df, tree_df = dfs

    got = pp.preprocess(
        fit_df, rel_df, meas_df, corr_df, fit_seed_df, tree_df,
        algorithm, measurement_type,
    )
    _assert_outputs_equal(got, fix["outputs"])


# --- Synthetic coverage for the one branch no live fit exercises -------------
# No current fit triggers the ignore_minus branch (preprocess takes abs() of the
# measured value for flagged nodes), so pin it directly.

def _minimal_inputs(meas_rows):
    """Build the smallest input set that reaches the parsing/ignore_minus block.

    Only meas_df content matters here; the other frames are empty-but-typed so
    the early special-case blocks are no-ops.
    """
    meas_cols = [
        "node", "reference_id", "occurrence", "measurement", "measurement_adjust",
        "power_of_ten", "text", "source_year", "systematic_error_clump",
        "systematic_error_clump2", "ignore_minus",
    ]
    meas_df = pd.DataFrame(meas_rows)
    for c in meas_cols:
        if c not in meas_df:
            meas_df[c] = np.nan
    meas_df = meas_df[meas_cols]

    fit_df = pd.DataFrame({"node": meas_df["node"].unique()})
    fit_df["type"] = "+"
    fit_df["data_type"] = np.nan

    rel_df = pd.DataFrame({
        "node": meas_df["node"].unique(),
        "par_code": None, "parameter": meas_df["node"].unique(),
        "coefficient": 1.0, "summation": 1.0,
        "coeff_par_code": None, "coeff_parameter": None,
    })
    fit_seed_df = pd.DataFrame({
        "par_code": None, "parameter": meas_df["node"].unique(), "seed": 1.0,
    })
    tree_df = pd.DataFrame({"node": meas_df["node"].unique(), "data_type": np.nan})
    return fit_df, rel_df, meas_df, corr_df_empty(), fit_seed_df, tree_df


def corr_df_empty():
    return pd.DataFrame(columns=[
        "node_one", "reference_id_one", "occurrence_one",
        "node_two", "reference_id_two", "occurrence_two", "correlation",
    ])


def test_ignore_minus_takes_absolute_value():
    rows = [
        {"node": "X1", "reference_id": 1, "occurrence": 1,
         "measurement": "-1.5+0.1-0.1", "text": "MeV", "ignore_minus": "X1"},
        {"node": "X2", "reference_id": 2, "occurrence": 1,
         "measurement": "-2.0+0.2-0.2", "text": "MeV", "ignore_minus": None},
    ]
    fit_df, rel_df, meas_df, corr_df, fit_seed_df, tree_df = _minimal_inputs(rows)

    out = pp.preprocess(fit_df, rel_df, meas_df, corr_df, fit_seed_df, tree_df,
                        algorithm="MASS", measurement_type="MASS")
    out_meas = out[2]
    # Flagged node X1: abs() applied; unflagged X2: sign preserved.
    assert out_meas.loc[out_meas["node"] == "X1", "value"].iloc[0] == pytest.approx(1.5)
    assert out_meas.loc[out_meas["node"] == "X2", "value"].iloc[0] == pytest.approx(-2.0)


@pytest.mark.parametrize('reverse', [False, True])
def test_shared_systematic_rejects_duplicate_correlation(reverse):
    rows = [
        {'node': 'X', 'reference_id': reference, 'occurrence': occurrence,
         'measurement': '1.0 +- 0.1 +- 0.2', 'text': 'MeV',
         'systematic_error_clump': 'shared'}
        for reference, occurrence in [(11, 1), (22, 2)]
    ]
    fit, relations, measurements, _, seeds, tree = _minimal_inputs(rows)
    first, second = (rows[::-1] if reverse else rows)
    correlations = pd.DataFrame([{
        'node_one': first['node'], 'reference_id_one': first['reference_id'],
        'occurrence_one': first['occurrence'], 'node_two': second['node'],
        'reference_id_two': second['reference_id'], 'occurrence_two': second['occurrence'],
        'correlation': .5,
    }])
    with pytest.raises(ValueError, match='already exists in the correlation table'):
        pp.preprocess(fit, relations, measurements, correlations, seeds, tree, 'MASS', 'MASS')
