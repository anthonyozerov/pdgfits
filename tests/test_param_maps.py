import numpy as np
import pytest
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

from pdgfits.param_maps import (
    build_param_map_sigmoid,
    build_param_map_softmax,
    build_param_map_scaled,
    get_decay_info,
)

PARAMS_SINGLE = ["S012.G001", "S012.G002", "S012.G003", "mass"]
PARAMS_TWO = ["S012.G001", "S013.G001", "mass"]
PARAMS_NO_DECAY = ["mass", "width"]


# ──────────────────────────────────────────────
# build_param_map_sigmoid
# ──────────────────────────────────────────────

def test_sigmoid_roundtrip():
    f2p, p2f, decay_idxs = build_param_map_sigmoid(PARAMS_SINGLE)
    params = jnp.array([0.3, 0.2, 0.4, 1000.0])
    recovered = f2p(p2f(params))
    assert jnp.allclose(recovered, params, atol=1e-10)


def test_sigmoid_large_positive_logit():
    f2p, _, decay_idxs = build_param_map_sigmoid(PARAMS_SINGLE)
    fitted = jnp.array([100.0, 100.0, 100.0, 1000.0])
    params = f2p(fitted)
    for i in decay_idxs:
        assert float(params[i]) > 0.99


def test_sigmoid_large_negative_logit():
    f2p, _, decay_idxs = build_param_map_sigmoid(PARAMS_SINGLE)
    fitted = jnp.array([-100.0, -100.0, -100.0, 1000.0])
    params = f2p(fitted)
    for i in decay_idxs:
        assert float(params[i]) < 0.01


def test_sigmoid_nondecay_passthrough():
    f2p, p2f, decay_idxs = build_param_map_sigmoid(PARAMS_SINGLE)
    params = jnp.array([0.3, 0.2, 0.4, 999.0])
    non_decay_idx = 3
    assert float(f2p(p2f(params))[non_decay_idx]) == pytest.approx(999.0)


def test_sigmoid_tiny_seed_stays_well_scaled():
    _, p2f, decay_idxs = build_param_map_sigmoid(PARAMS_SINGLE)
    params = jnp.array([1e-5, 0.2, 0.4, 1000.0])
    fitted = p2f(params)

    assert abs(float(fitted[decay_idxs[0]])) < 10.0


# ──────────────────────────────────────────────
# build_param_map_softmax
# ──────────────────────────────────────────────

def test_softmax_decay_sums_to_one():
    f2p, _, decay_idxs, _ = build_param_map_softmax(PARAMS_SINGLE)
    fitted = jnp.array([0.5, -1.0, 0.3, 1000.0])
    params = f2p(fitted)
    decay_sum = float(jnp.sum(params[jnp.array(decay_idxs)]))
    assert decay_sum == pytest.approx(1.0, abs=1e-14)


def test_softmax_roundtrip():
    f2p, p2f, decay_idxs, _ = build_param_map_softmax(PARAMS_SINGLE)
    # choose decay params summing to 1
    params = jnp.array([0.2, 0.5, 0.3, 1000.0])
    recovered = f2p(p2f(params))
    assert jnp.allclose(recovered[jnp.array(decay_idxs)], params[jnp.array(decay_idxs)], atol=1e-10)


def test_softmax_nondecay_passthrough():
    f2p, p2f, _, _ = build_param_map_softmax(PARAMS_SINGLE)
    params = jnp.array([0.2, 0.5, 0.3, 777.0])
    recovered = f2p(p2f(params))
    assert float(recovered[3]) == pytest.approx(777.0)


def test_softmax_two_particles_raises():
    with pytest.raises(AssertionError):
        build_param_map_softmax(PARAMS_TWO)


# ──────────────────────────────────────────────
# get_decay_info
# ──────────────────────────────────────────────

def test_get_decay_info_finds_particle():
    params = ["S012.G001", "S012.G002", "mass"]
    particles, idxs = get_decay_info(params)
    assert list(particles) == ["S012"]
    assert list(idxs) == [0, 1]


def test_get_decay_info_no_decays():
    particles, idxs = get_decay_info(PARAMS_NO_DECAY)
    assert len(particles) == 0
    assert len(idxs) == 0


def test_scaled_map_roundtrip_and_coordinate_scale():
    identity = lambda x: x
    f2p, p2f = build_param_map_scaled(identity, identity, [1], [1e6])
    params = jnp.array([2.0, 8.0e7])
    fitted = p2f(params)

    assert float(fitted[1]) == pytest.approx(80.0)
    assert jnp.allclose(f2p(fitted), params, rtol=1e-12, atol=0)
