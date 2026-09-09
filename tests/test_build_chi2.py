import numpy as np
import pytest
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

from pdgfits.build_chi2 import build_chi2


def identity(fp):
    return fp


def make_chi2(y, mu, error, corr_inv=None, **kwargs):
    y = jnp.array(y, dtype=jnp.float64)
    error = jnp.array(error, dtype=jnp.float64)
    if corr_inv is None:
        corr_inv = jnp.eye(len(y), dtype=jnp.float64)
    return build_chi2(y, mu, error, error, corr_inv, identity, **kwargs)


def test_zero_residuals():
    mu = lambda p: p
    chi2, _, chi2_val, _ = make_chi2([1.0, 2.0], mu, [1.0, 1.0])
    params = jnp.array([1.0, 2.0])
    assert chi2_val(params) == pytest.approx(0.0)


def test_single_meas_symmetric():
    mu = lambda p: jnp.array([2.0])
    chi2, _, chi2_val, _ = make_chi2([1.0], mu, [1.0])
    params = jnp.array([0.0])
    assert chi2_val(params) == pytest.approx(1.0)


def test_chi2_nonnegative():
    rng = np.random.default_rng(42)
    mu = lambda p: jnp.array([1.5, 2.5, 3.5])
    chi2, _, chi2_val, _ = make_chi2([1.0, 2.0, 3.0], mu, [0.5, 0.5, 0.5])
    for _ in range(10):
        params = jnp.array(rng.standard_normal(3))
        assert chi2_val(params) >= 0.0


def test_gradient_at_minimum():
    mu = lambda p: p
    chi2, chi2_grad, _, _ = make_chi2([1.0], mu, [1.0])
    params_min = jnp.array([1.0])
    grad = chi2_grad(params_min)
    assert abs(grad[0]) < 1e-6


def test_asymmetric_errors_select_sigma():
    error_n = jnp.array([0.1], dtype=jnp.float64)
    error_p = jnp.array([1.0], dtype=jnp.float64)
    corr_inv = jnp.eye(1, dtype=jnp.float64)

    mu = lambda p: jnp.array([0.0])
    chi2, _, chi2_val, _ = build_chi2(
        jnp.array([0.0]), mu, error_n, error_p, corr_inv, identity
    )

    # resid > 0 (y > mu): effective sigma → error_n = 0.1 → large chi2
    y_pos = jnp.array([1.0])
    chi2_pos, _, chi2_val_pos, _ = build_chi2(y_pos, mu, error_n, error_p, corr_inv, identity)
    val_pos = chi2_val_pos(jnp.array([0.0]))

    # resid < 0 (y < mu): effective sigma → error_p = 1.0 → small chi2
    y_neg = jnp.array([-1.0])
    chi2_neg, _, chi2_val_neg, _ = build_chi2(y_neg, mu, error_n, error_p, corr_inv, identity)
    val_neg = chi2_val_neg(jnp.array([0.0]))

    # same |resid|=1, but asymmetric → chi2 very different
    assert val_pos > val_neg * 10


def test_translate_dep():
    mu = lambda p: jnp.array([0.0])
    translate_dep = lambda p: jnp.array([1.0])
    chi2, _, chi2_val, _ = make_chi2([0.0], mu, [1.0], translate_dep=translate_dep)
    params = jnp.array([0.0])
    # resid = 0.0*1 + 1.0 - 0.0 = 1.0 → chi2 = 1.0
    assert chi2_val(params) == pytest.approx(1.0)


def test_adjust():
    mu = lambda p: jnp.array([2.0])
    adjust = lambda p: jnp.array([2.0])
    chi2, _, chi2_val, _ = make_chi2([1.0], mu, [1.0], adjust=adjust)
    params = jnp.array([0.0])
    # resid = 1.0*2.0 + 0 - 2.0 = 0.0 → chi2 = 0.0
    assert chi2_val(params) == pytest.approx(0.0)


def test_chi2_val_returns_float():
    mu = lambda p: p
    _, _, chi2_val, _ = make_chi2([1.0], mu, [1.0])
    result = chi2_val(jnp.array([1.0]))
    assert isinstance(result, float)


def test_chi2_grad_returns_numpy():
    mu = lambda p: p
    _, chi2_grad, _, _ = make_chi2([1.0], mu, [1.0])
    result = chi2_grad(jnp.array([1.0]))
    assert isinstance(result, np.ndarray)
