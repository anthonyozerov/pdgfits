import contextlib
import copy
import io

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from pdgfits.build_funcs import (
    get_node_funcs, get_parameter_funcs, get_meas_funcs, get_mu,
    get_mu_vectorized,
    get_translate_dep, get_adjust, get_mu_adjust,
)


def make_frames(parameters, nodes, node_types, rel_rows, meas_nodes):
    """Build minimal fit_df, rel_df, meas_df for testing."""
    fit_df = pd.DataFrame({'node': nodes, 'type': node_types})
    rel_df = pd.DataFrame(rel_rows)
    if 'coeff_parameter_key' not in rel_df.columns:
        rel_df['coeff_parameter_key'] = None
    meas_df = pd.DataFrame({'node': meas_nodes})
    return fit_df, rel_df, meas_df


def mu_old(parameters, nodes, fit_df, rel_df, meas_df):
    node_funcs = get_node_funcs(nodes, parameters, fit_df, rel_df, jit=False)
    parameter_funcs = get_parameter_funcs(parameters)
    meas_funcs = get_meas_funcs(
        dict(zip(nodes, node_funcs)),
        dict(zip(parameters, parameter_funcs)),
        meas_df,
    )
    return get_mu(meas_funcs)


def mu_new(parameters, nodes, fit_df, rel_df, meas_df):
    return get_mu_vectorized(parameters, nodes, meas_df, fit_df, rel_df, jit=False)


def check_mu_match(parameters, nodes, fit_df, rel_df, meas_df, param_vals):
    old = mu_old(parameters, nodes, fit_df, rel_df, meas_df)
    new = mu_new(parameters, nodes, fit_df, rel_df, meas_df)
    params = jnp.array(param_vals, dtype=jnp.float64)
    assert jnp.allclose(new(params), old(params), rtol=1e-10, atol=0), (
        f"mu mismatch: new={new(params)}, old={old(params)}"
    )


# Test 1: single '+' node, single measurement, one parameter
def test_single_plus_node_single_meas():
    parameters = ['A']
    nodes = ['A']
    rel_rows = [{'node': 'A', 'parameter_key': 'A', 'coefficient': 1.0, 'summation': 1}]
    fit_df, rel_df, meas_df = make_frames(parameters, nodes, ['+'], rel_rows, ['A'])
    check_mu_match(parameters, nodes, fit_df, rel_df, meas_df, [3.0])


# Test 2: single '+' node, multiple measurements sharing that node
def test_single_plus_node_multi_meas():
    parameters = ['A']
    nodes = ['A']
    rel_rows = [{'node': 'A', 'parameter_key': 'A', 'coefficient': 1.0, 'summation': 1}]
    fit_df, rel_df, meas_df = make_frames(parameters, nodes, ['+'], rel_rows, ['A', 'A', 'A'])
    check_mu_match(parameters, nodes, fit_df, rel_df, meas_df, [5.0])
    new = mu_new(parameters, nodes, fit_df, rel_df, meas_df)
    result = new(jnp.array([5.0]))
    assert result.shape == (3,)
    assert jnp.allclose(result, jnp.array([5.0, 5.0, 5.0]))


# Test 3: two '+' nodes, two parameters, four measurements
def test_two_plus_nodes():
    parameters = ['A', 'B']
    nodes = ['P', 'Q']
    rel_rows = [
        {'node': 'P', 'parameter_key': 'A', 'coefficient': 2.0, 'summation': 1},
        {'node': 'P', 'parameter_key': 'B', 'coefficient': 3.0, 'summation': 1},
        {'node': 'Q', 'parameter_key': 'A', 'coefficient': 1.0, 'summation': 1},
        {'node': 'Q', 'parameter_key': 'B', 'coefficient': 4.0, 'summation': 1},
    ]
    fit_df, rel_df, meas_df = make_frames(
        parameters, nodes, ['+', '+'], rel_rows, ['P', 'P', 'Q', 'Q']
    )
    check_mu_match(parameters, nodes, fit_df, rel_df, meas_df, [1.0, 2.0])
    new = mu_new(parameters, nodes, fit_df, rel_df, meas_df)
    result = new(jnp.array([1.0, 2.0]))
    # P = 2*1 + 3*2 = 8, Q = 1*1 + 4*2 = 9
    assert jnp.allclose(result, jnp.array([8.0, 8.0, 9.0, 9.0]))


# Test 4: nuisance parameter measurement (not a node, measured directly)
def test_nuisance_parameter():
    parameters = ['X', 'nuisance_N']
    nodes = ['X']
    rel_rows = [{'node': 'X', 'parameter_key': 'X', 'coefficient': 1.0, 'summation': 1}]
    fit_df, rel_df, meas_df = make_frames(
        parameters, nodes, ['+'], rel_rows, ['X', 'nuisance_N']
    )
    check_mu_match(parameters, nodes, fit_df, rel_df, meas_df, [7.0, 3.0])
    new = mu_new(parameters, nodes, fit_df, rel_df, meas_df)
    result = new(jnp.array([7.0, 3.0]))
    assert jnp.allclose(result, jnp.array([7.0, 3.0]))


# Test 5: mixed '+' node and 'lifetime' node (nonlinear fallback path)
def test_mixed_plus_and_lifetime():
    parameters = ['x', 'width']
    nodes = ['P_node', 'L_node']
    rel_rows = [
        {'node': 'P_node', 'parameter_key': 'x', 'coefficient': 1.0, 'summation': 1},
        {'node': 'L_node', 'parameter_key': 'width', 'coefficient': 1.0, 'summation': 1},
    ]
    fit_df, rel_df, meas_df = make_frames(
        parameters, nodes, ['+', 'lifetime'], rel_rows,
        ['P_node', 'L_node', 'P_node']
    )
    check_mu_match(parameters, nodes, fit_df, rel_df, meas_df, [2.0, 4.0])
    new = mu_new(parameters, nodes, fit_df, rel_df, meas_df)
    result = new(jnp.array([2.0, 4.0]))
    # P_node = x = 2.0, L_node = 1/width = 0.25
    assert jnp.allclose(result, jnp.array([2.0, 0.25, 2.0]))


# Test 6: '+' node with non-zero coeff_parameter_key (must fall through to func_factory)
def test_plus_with_coeff_parameter_key():
    parameters = ['A', 'B']
    nodes = ['C']
    rel_rows = [{
        'node': 'C', 'parameter_key': 'A', 'coefficient': 1.0,
        'summation': 1, 'coeff_parameter_key': 'B',
    }]
    fit_df, rel_df, meas_df = make_frames(parameters, nodes, ['+'], rel_rows, ['C'])
    check_mu_match(parameters, nodes, fit_df, rel_df, meas_df, [2.0, 3.0])
    new = mu_new(parameters, nodes, fit_df, rel_df, meas_df)
    # C = A * B = 2 * 3 = 6 ('+' with coeff_parameter_key: coefficient * coeff_param * param)
    result = new(jnp.array([2.0, 3.0]))
    assert jnp.allclose(result, jnp.array([6.0]))


# Test 7: get_translate_dep fast path when all entries are None
def test_translate_dep_all_none():
    result_fn = get_translate_dep([None, None, None], [], [], [], [])
    params = jnp.array([1.0, 2.0])
    out = result_fn(params)
    assert jnp.allclose(out, jnp.zeros(3, dtype=jnp.float64))
    assert out.shape == (3,)


# Test 8: get_adjust fast path when all entries are None
def test_adjust_all_none():
    result_fn = get_adjust([None, None, None], [], [], [], [])
    params = jnp.array([1.0, 2.0])
    out = result_fn(params)
    assert jnp.allclose(out, jnp.ones(3, dtype=jnp.float64))
    assert out.shape == (3,)


def test_combined_measurement_corrections_and_derivatives():
    # X names both a physical parameter and a relationship 2*X. Offsets use
    # the physical parameter; the measured prediction uses the relationship.
    parameters = ['X', 'nuisance_A', 'nuisance_B']
    parameter_funcs = get_parameter_funcs(parameters)
    node_funcs = [lambda p: 2*p[0]]
    offsets = [(['X', 'A'], [0.5, 3.0], [8.0, 1.0]), None, None]
    adjustments = [(['A', 'B'], ['/', '*']), None, None]
    original = copy.deepcopy((offsets, adjustments))
    translate = get_translate_dep(offsets, parameters, ['X'], parameter_funcs, node_funcs)
    adjust = get_adjust(adjustments, parameters, ['X'], parameter_funcs, node_funcs)
    predictions = get_mu_adjust(lambda p: jnp.array([2*p[0], p[1], p[2]]), adjust, translate)

    values = jnp.array([10., 2., 4.])
    np.testing.assert_allclose(translate(values), [4., 0., 0.])
    np.testing.assert_allclose(adjust(values), [.5, 1., 1.])
    # Corrected prediction: (2*X - .5*(X-8) - 3*(A-1)) * B/A.
    np.testing.assert_allclose(jax.jit(predictions)(values), [32., 2., 4.])
    np.testing.assert_allclose(jax.jacfwd(predictions)(values),
                               [[3., -22., 8.], [0., 1., 0.], [0., 0., 1.]])
    assert (offsets, adjustments) == original


@pytest.mark.db
def test_mu_vectorized_matches_original_on_avg_nodes():
    from pdgfits.query import avg_queries
    from pdgfits.preprocess import preprocess

    avg_df, corr_df_dict = avg_queries(verbose=False)
    unique_nodes = avg_df['node'].unique()[:20]

    for node in unique_nodes:
        meas_df_node = avg_df[avg_df['node'] == node].copy()
        for col in ('systematic_error_clump', 'systematic_error_clump2'):
            if col in meas_df_node.columns:
                meas_df_node[col] = None

        corr_df_node = corr_df_dict.get(node, pd.DataFrame())

        raw_fit_df = pd.DataFrame({'node': [node], 'type': ['+'], 'data_type': [None]})
        raw_rel_df = pd.DataFrame({
            'node': [node], 'par_code': [None], 'parameter': [node],
            'coefficient': [1.0], 'summation': [1],
            'coeff_par_code': [None], 'coeff_parameter': [None],
        })
        raw_seed_df = pd.DataFrame({'par_code': [None], 'parameter': [node], 'seed': [0.0]})
        raw_tree_df = pd.DataFrame({
            'node': pd.Series([], dtype=str),
            'data_type': pd.Series([], dtype=str),
        })

        with contextlib.redirect_stdout(io.StringIO()):
            fit_df, rel_df, meas_df, corr_df, fit_seed_df, _, _ = preprocess(
                raw_fit_df, raw_rel_df, meas_df_node, corr_df_node.copy(),
                raw_seed_df, raw_tree_df, 'AVG', '',
            )

        if len(meas_df[meas_df['node'] == node]) <= 1:
            continue

        parameters = list(fit_seed_df['parameter_key'].unique())
        nodes_list = list(rel_df['node'].unique())

        seed_val = float(meas_df[meas_df['node'] == node]['value'].mean())
        fit_seed_df = fit_seed_df.copy()
        fit_seed_df.loc[fit_seed_df['parameter_key'] == node, 'seed'] = seed_val

        node_funcs_old = get_node_funcs(nodes_list, parameters, fit_df, rel_df, jit=False)
        parameter_funcs_old = get_parameter_funcs(parameters)
        meas_funcs_old = get_meas_funcs(
            dict(zip(nodes_list, node_funcs_old)),
            dict(zip(parameters, parameter_funcs_old)),
            meas_df,
        )
        mu_old_fn = get_mu(meas_funcs_old)
        mu_new_fn = get_mu_vectorized(parameters, nodes_list, meas_df, fit_df, rel_df, jit=False)

        params_seed = jnp.array(
            [fit_seed_df[fit_seed_df['parameter_key'] == p]['seed'].iloc[0] for p in parameters],
            dtype=jnp.float64,
        )
        params_perturbed = params_seed * 1.1 + 0.01

        for label, params in [('seed', params_seed), ('perturbed', params_perturbed)]:
            old_val = mu_old_fn(params)
            new_val = mu_new_fn(params)
            assert jnp.allclose(new_val, old_val, rtol=1e-10, atol=0), (
                f"mu mismatch for node {node} at {label}: new={new_val}, old={old_val}"
            )
