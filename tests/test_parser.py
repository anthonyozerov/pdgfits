import numpy as np
import pytest
import jax
jax.config.update("jax_enable_x64", True)

from pdgfits.parser import (
    parse_measurement, get_scale, parameter_key,
    br_adjust_node, get_dep_meas_data, get_adjust_data,
)


def test_basic_asymmetric_errors():
    val, pos, neg, _ = parse_measurement("1.23+0.05-0.03")
    assert val == pytest.approx(1.23)
    assert pos == pytest.approx(0.05)
    assert neg == pytest.approx(0.03)


def test_trailing_exponent():
    val, pos, neg, _ = parse_measurement("1.23+0.05-0.03E-3")
    assert val == pytest.approx(1.23e-3)
    assert pos == pytest.approx(0.05e-3)
    assert neg == pytest.approx(0.03e-3)


def test_combined_errors_in_quadrature():
    _, pos, neg, _ = parse_measurement("1.0+0.3+0.4-0.3-0.4")
    assert pos == pytest.approx(0.5)
    assert neg == pytest.approx(0.5)


def test_at_stripping():
    val1, pos1, neg1, _ = parse_measurement("1.5+0.1-0.1@SOME_FLAG")
    val2, pos2, neg2, _ = parse_measurement("1.5+0.1-0.1")
    assert val1 == pytest.approx(val2)
    assert pos1 == pytest.approx(pos2)
    assert neg1 == pytest.approx(neg2)


def test_symmetric_plusminus():
    _, pos, neg, _ = parse_measurement("2.0+-0.1")
    assert pos == pytest.approx(0.1)
    assert neg == pytest.approx(0.1)


def test_spaced_plusminus():
    val, pos, neg, _ = parse_measurement("1.524 + - 0.041")
    assert val == pytest.approx(1.524)
    assert pos == pytest.approx(0.041)
    assert neg == pytest.approx(0.041)


def test_no_errors():
    result = parse_measurement("3.14")
    assert result[0] == pytest.approx(3.14)
    assert result[1] == pytest.approx(0.0)
    assert result[2] == pytest.approx(0.0)


def test_parenthesized():
    val, pos, neg, _ = parse_measurement("(1.0+0.1-0.1)")
    assert val == pytest.approx(1.0)
    assert pos == pytest.approx(0.1)
    assert neg == pytest.approx(0.1)


def test_last_err():
    _, _, _, last = parse_measurement("1.0+0.3-0.2")
    assert last == pytest.approx((0.3 + 0.2) / 2)


def test_get_scale_kev():
    assert get_scale("keV") == pytest.approx(1e-3)


def test_get_scale_ev():
    assert get_scale("eV") == pytest.approx(1e-6)


def test_get_scale_unknown():
    assert get_scale("MeV") == 1
    assert get_scale("unknown") == 1


def test_parameter_key_non_null():
    assert parameter_key("S012", "G001") == "S012.G001"


def test_parameter_key_nan():
    assert parameter_key(np.nan, "G001") == "G001"


# --- structured measurement-annotation string parsers -----------------------

def test_br_adjust_node():
    s = "br_adjust: 1.2+0.1-0.1; /, 0.5+0.01-0.01, M001 8"
    assert br_adjust_node(s) == "M001.8"


def test_br_adjust_node_none():
    assert br_adjust_node("1.23+0.05-0.03") is None


def test_get_dep_meas_data():
    s = "dep_meas: 1.23, 0.5 0.1 S035R1, -0.3 0.0 S035R2"
    nodes, slopes, constants = get_dep_meas_data(s)
    assert nodes == ["S035R1", "S035R2"]
    assert slopes == [0.5, -0.3]
    assert constants == [0.1, 0.0]


def test_get_dep_meas_data_none():
    assert get_dep_meas_data("1.23+0.05-0.03") is None


def test_get_adjust_data():
    nodes, rels = get_adjust_data(["*, ADJUST, M001 8", "/, ADJUST, S014 12"])
    assert nodes == ["M001.8", "S014.12"]
    assert rels == ["*", "/"]


def test_get_adjust_data_none():
    assert get_adjust_data(None) is None


def test_get_adjust_data_rejects_unknown_rel():
    with pytest.raises(AssertionError):
        get_adjust_data(["+, ADJUST, M001 8"])
