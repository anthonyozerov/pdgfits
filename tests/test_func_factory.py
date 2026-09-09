import math
import pytest
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

from pdgfits.func_factory import func_factory, ALLOWED_EQUATION_TYPES

# params and coeff helpers
P2 = jnp.array([2.0, 3.0])
P3 = jnp.array([2.0, 3.0, 4.0])

# shape (3, n_params), all zeros → simple dot-only mode
CP2 = [[0, 0], [0, 0], [0, 0]]
CP3 = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]

# coefficient rows: each selects one param
C2_ROW0 = [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]
C2_ROW1 = [[0.0, 1.0], [1.0, 0.0], [0.0, 0.0]]
C3_EACH = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]


def test_plus_first_param():
    f = func_factory("+", C2_ROW0, CP2)
    assert float(f(P2)) == pytest.approx(2.0)


def test_plus_second_param():
    f = func_factory("+", C2_ROW1, CP2)
    assert float(f(P2)) == pytest.approx(3.0)


def test_lifetime():
    f = func_factory("lifetime", C2_ROW0, CP2)
    assert float(f(P2)) == pytest.approx(0.5)


def test_division():
    f = func_factory("/", C2_ROW0, CP2)
    assert float(f(P2)) == pytest.approx(2.0 / 3.0)


def test_gplus():
    f = func_factory("G+", C2_ROW0, CP2)
    assert float(f(P2)) == pytest.approx(6.0)


def test_rplus():
    f = func_factory("R+", C2_ROW0, CP2)
    assert float(f(P2)) == pytest.approx(6.0)


def test_p():
    f = func_factory("P", C2_ROW0, CP2)
    assert float(f(P2)) == pytest.approx(6.0)


def test_gstar():
    f = func_factory("G*", C3_EACH, CP3)
    assert float(f(P3)) == pytest.approx(24.0)


def test_sr():
    f = func_factory("SR", C2_ROW0, CP2)
    assert float(f(P2)) == pytest.approx(math.sqrt(6.0))


def test_sq():
    f = func_factory("SQ", C3_EACH, CP3)
    assert float(f(P3)) == pytest.approx(math.sqrt(6.0) * 4.0)


def test_pdiv():
    f = func_factory("P/", C3_EACH, CP3)
    assert float(f(P3)) == pytest.approx(2.0 * 3.0 / 4.0)


def test_unknown_raises():
    with pytest.raises(ValueError):
        f = func_factory("BOGUS", C2_ROW0, CP2)
        f(P2)


def test_jax_differentiable():
    for eq in ALLOWED_EQUATION_TYPES:
        params = P3 if eq in ("G*", "P/", "SQ") else P2
        cp = CP3 if eq in ("G*", "P/", "SQ") else CP2
        c = C3_EACH if eq in ("G*", "P/", "SQ") else C2_ROW0
        f = func_factory(eq, c, cp)
        grad = jax.grad(f)(params)
        assert jnp.all(jnp.isfinite(grad)), f"Non-finite gradient for equation type '{eq}'"
