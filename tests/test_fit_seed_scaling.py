import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import pandas as pd
import pytest

from pdgfits.fit import (
    _condition_seed_units,
    _large_coordinate_scale_info,
)


def test_rescale_direct_parameter_seed_power_of_ten_mismatch():
    meas_df = pd.DataFrame({
        "node": ["phase", "delta", "delta"],
        "value": [43.0, 5.24e9, 5.34e9],
    })
    fit_df = pd.DataFrame({"node": [], "type": []})
    rel_df = pd.DataFrame({"node": [], "parameter_key": []})
    params = ["phase", "delta"]
    init = jnp.array([45.0, 0.53])

    got = _condition_seed_units(params, fit_df, rel_df, meas_df, init)

    assert float(got[0]) == pytest.approx(45.0)
    assert float(got[1]) == pytest.approx(5.3e9)


def test_rescale_lifetime_width_seed_from_parsed_lifetime_measurements():
    fit_df = pd.DataFrame({"node": ["S012T"], "type": ["lifetime"]})
    rel_df = pd.DataFrame({"node": ["S012T"], "parameter_key": ["S012W"]})
    meas_df = pd.DataFrame({
        "node": ["S012T", "S012T"],
        "value": [8.958e-11, 8.962e-11],
    })
    params = ["S012W"]
    init = jnp.array([1.12])

    got = _condition_seed_units(params, fit_df, rel_df, meas_df, init)

    assert float(got[0]) == pytest.approx(1.0 / 8.96e-11)


def test_large_parameter_scale_info_skips_decay_coordinates():
    params = ["S013D", "S013.G001", "phase"]
    init = jnp.array([5.3e9, 1e-5, 43.0])

    idxs, scales = _large_coordinate_scale_info(params, init, skip_idxs=[1])

    assert idxs.tolist() == [0]
    assert scales.tolist() == pytest.approx([5.3e9])
