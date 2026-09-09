import pandas as pd
from pandas.testing import assert_frame_equal
import pytest

import pdgfits.snapshot as snapshot
import pdgfits.query as query


def _enable_snapshot(monkeypatch, tmp_path):
    monkeypatch.setenv(query.DATA_BACKEND_ENV, "snapshot")
    monkeypatch.setenv(query.SNAPSHOT_DIR_ENV, str(tmp_path))


def test_snapshot_replays_query_outputs(monkeypatch, tmp_path):
    _enable_snapshot(monkeypatch, tmp_path)

    fits_df = pd.DataFrame({
        "label": ["fit-a"],
        "algorithm": ["MASS"],
        "chi_square": [1.25],
    })
    fit_payload = (
        "MASS",
        "M",
        pd.DataFrame({"node": ["N1"], "type": ["+"], "data_type": [None]}),
        pd.DataFrame({"node": ["N1"], "parameter": ["N1"], "coefficient": [1.0]}),
        pd.DataFrame({"node": ["N1"], "measurement": ["1.0+0.1-0.1"]}),
        pd.DataFrame({"node_one": [], "node_two": [], "correlation": []}),
        pd.DataFrame({"parameter": ["N1"], "seed": [1.0]}),
        pd.DataFrame({"node": ["N1"], "data_type": [None]}),
    )
    avg_payload = (
        pd.DataFrame({"node": ["N1"], "measurement": ["1.0+0.1-0.1"]}),
        {"N1": pd.DataFrame({"node_one": ["N1"], "node_two": ["N1"], "correlation": [0.5]})},
    )
    pmpv_payload = (1.0, 0.1, 0.1)
    ncorr_payload = pd.DataFrame({
        "par_code_row": [None],
        "parameter_row": ["N1"],
        "par_code_column": [None],
        "parameter_column": ["N2"],
        "coefficient": [50.0],
    })

    query._write_snapshot_pickle(fits_df, "all_fits.pkl")
    query._write_snapshot_pickle(fit_payload, "fits", f"{query._snapshot_key('fit-a')}.pkl")
    query._write_snapshot_pickle(avg_payload, "avg_queries.pkl")
    query._write_snapshot_pickle(
        pmpv_payload,
        "pdg_most_precise_value",
        f"{query._snapshot_key('N1')}.pkl",
    )
    query._write_snapshot_pickle(
        ncorr_payload,
        "nuisance_corr",
        f"{query._nuisance_corr_key(['nuisance_N1', 'nuisance_N2'])}.pkl",
    )

    assert_frame_equal(query.all_fits(), fits_df)
    got_fit = query.fit_queries("fit-a")
    assert got_fit[:2] == fit_payload[:2]
    for got, want in zip(got_fit[2:], fit_payload[2:]):
        assert_frame_equal(got, want)

    got_avg_df, got_corr = query.avg_queries()
    assert_frame_equal(got_avg_df, avg_payload[0])
    assert_frame_equal(got_corr["N1"], avg_payload[1]["N1"])
    assert query.pdg_most_precise_value("N1") == pmpv_payload
    assert_frame_equal(
        query.nuisance_corr(["nuisance_N2", "nuisance_N1"]),
        ncorr_payload,
    )


def test_snapshot_backend_requires_directory(monkeypatch):
    monkeypatch.setenv(query.DATA_BACKEND_ENV, "snapshot")
    monkeypatch.delenv(query.SNAPSHOT_DIR_ENV, raising=False)

    with pytest.raises(RuntimeError, match=query.SNAPSHOT_DIR_ENV):
        query.all_fits()


def test_capture_writes_snapshot_with_mocked_db(monkeypatch, tmp_path):
    fits_df = pd.DataFrame({"label": ["fit-a"], "algorithm": ["MASS"]})
    fit_payload = (
        "MASS",
        "M",
        pd.DataFrame({"node": ["N1"], "type": ["+"], "data_type": [None]}),
        pd.DataFrame({"node": ["N1"], "parameter": ["N1"], "coefficient": [1.0]}),
        pd.DataFrame({"node": ["N1"], "measurement": ["1.0+0.1-0.1"]}),
        pd.DataFrame({"node_one": [], "node_two": [], "correlation": []}),
        pd.DataFrame({"parameter": ["N1"], "seed": [1.0]}),
        pd.DataFrame({"node": ["N1"], "data_type": [None]}),
    )

    def fake_preprocess(fit_df, rel_df, meas_df, corr_df, fit_seed_df, tree_df,
                        algorithm, measurement_type):
        snapshot.pp.pdg_most_precise_value("N1")
        snapshot.pp.nuisance_corr(["nuisance_N1"])
        fit_seed_df = fit_seed_df.copy()
        fit_seed_df["parameter_key"] = fit_seed_df["parameter"]
        return fit_df, rel_df, meas_df, corr_df, fit_seed_df, [], []

    monkeypatch.setattr(snapshot.query, "all_fits", lambda: fits_df)
    monkeypatch.setattr(snapshot.query, "fit_queries", lambda label, verbose=False: fit_payload)
    monkeypatch.setattr(snapshot.query, "pdg_most_precise_value", lambda node: (1.0, 0.1, 0.1))
    monkeypatch.setattr(
        snapshot.query,
        "nuisance_corr",
        lambda nuisance_params, verbose=False: pd.DataFrame({"parameter_row": ["N1"]}),
    )
    monkeypatch.setattr(snapshot.pp, "preprocess", fake_preprocess)

    args = type("Args", (), {
        "out": str(tmp_path),
        "no_fits": False,
        "fit_label": ["fit-a"],
        "include_tauhflav": False,
        "limit": None,
        "verbose": False,
        "no_averages": True,
        "avg_node": None,
    })()
    snapshot.capture(args)

    _enable_snapshot(monkeypatch, tmp_path)
    assert_frame_equal(query.all_fits(), fits_df)
    assert query.fit_queries("fit-a")[:2] == ("MASS", "M")
    assert query.pdg_most_precise_value("N1") == (1.0, 0.1, 0.1)
