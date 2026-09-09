import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from pdgfits.build_chi2 import build_chi2
from pdgfits.node_scales import node_scale_geometry, fit_node_scales
from pdgfits.refit import prepare_refit
from pdgfits.pdg_scaling import pdg_fit_scales, pdg_linear_fit


def model_fit(values, lower, upper, correlation, mean, initial, nodes):
    values, lower, upper = map(jnp.asarray, (values, lower, upper))
    chi2, grad, _, opened = build_chi2(values, mean, lower, upper,
                                      jnp.linalg.pinv(jnp.asarray(correlation)), lambda p: p)
    return {'fitted_values': jnp.asarray(initial), 'param_values': jnp.asarray(initial),
            'covariance': jnp.eye(len(initial)), 'chi2': chi2, 'chi2_grad': grad,
            'chi2_open': opened, 'chi2_min': float(chi2(jnp.asarray(initial))),
            'mu_adjust': mean, 'fitted_params_to_params': lambda p: p,
            'meas_df': pd.DataFrame({'node': nodes, 'value': values, 'error_n': lower, 'error_p': upper}),
            'corr_mat': correlation}

def test_scaled_refit_minimizes_same_asymmetric_correlated_objective():
    correlation = np.eye(4); correlation[0, 2] = correlation[2, 0] = .3
    mean = lambda p: jnp.array([p[0], p[0], p[0]*p[1], p[1]])
    fit = model_fit([1., 1.2, 2.3, 2.], [.1, .2, .2, .3], [.2, .1, .4, .2],
                    correlation, mean, [1., 2.], ['a', 'a', 'ab', 'b'])
    scales = np.array([2., 2., 1.3, 1.])
    result = prepare_refit(fit)(scales=scales)
    assert result['chi2_min'] == pytest.approx(float(fit['chi2_open'](result['fitted_values'],
                                               fit['meas_df']['value'].to_numpy(), scales)), abs=1e-9)
    assert np.linalg.norm(jax.grad(result['chi2'])(result['fitted_values'])) < 1e-4

def test_measurement_sensitivity_holds_node_scales_fixed():
    from pdgfits.diagnostics import meas_sensitivity
    fit = model_fit([0., 2., 4.], np.ones(3), np.ones(3), np.eye(3),
                    lambda p: jnp.ones(3)*p[0], [2.], ['a']*3)
    scaled = prepare_refit(fit)(scales=np.array([1., 2., 3.]))
    weights = 1/np.array([1., 2., 3.])**2
    np.testing.assert_allclose(meas_sensitivity(scaled), (weights/weights.sum())[None, :], atol=1e-10)

@pytest.mark.parametrize('exclude', [False, True])
@pytest.mark.parametrize('rho', [0., .4])
def test_general_pdg_scale_pass_agrees_with_linear_reference(exclude, rho):
    design = np.repeat([[1., 0.], [0., 1.], [1., 1.]], 3, axis=0)
    sigma = np.tile([1., 1., 10.], 3)
    nodes = np.repeat(['a', 'b', 'a+b'], 3)
    correlation = np.eye(9)
    correlation[0, 3] = correlation[3, 0] = rho
    y = np.array([-1., 1., 10., -2., 1., 3., 4., 8., 22.])
    fit = model_fit(y, sigma, sigma, correlation, lambda p: jnp.asarray(design)@p, [0., 0.], nodes)
    old = pdg_linear_fit(y, design, correlation*np.outer(sigma, sigma), nodes, exclude_weak=exclude)
    new = pdg_fit_scales(fit, exclude_weak=exclude)
    np.testing.assert_array_equal(new['retained'], old['retained'])
    np.testing.assert_allclose(new['refitted_fit']['param_values'], old['refitted_parameters'], atol=1e-7)
    np.testing.assert_allclose(new['refitted_fit']['covariance'], old['covariance'], rtol=1e-7)

def test_general_pdg_dependency_blocks_keep_the_exact_relation():
    block = np.array([[1., 0.], [0., 1.], [-1., 1.]])
    design = np.tile(block, (2, 1))
    covariance = block@block.T
    errors = np.sqrt(np.diag(covariance))*.1
    correlation = np.kron(np.eye(2), covariance/np.sqrt(np.outer(np.diag(covariance), np.diag(covariance))))
    fit = model_fit([1., 2., 1., 3., 5., 2.], np.tile(errors, 2), np.tile(errors, 2),
                    correlation, lambda p: jnp.asarray(design)@p, [2., 3.5], ['a', 'b', 'difference']*2)
    result = pdg_fit_scales(fit, exclude_weak=False)
    assert all(r['dependent'] for r in result['correlated_blocks'])
    np.testing.assert_allclose(result['refitted_fit']['corr_mat'], correlation, atol=1e-14)
    assert min(result['measurement_scales']) > 2


def test_single_pass_recovers_ordinary_birge_exactly():
    y = np.array([-2., -1., 0., 0., 1., 3.])
    fit = model_fit(y, np.ones(6), np.ones(6), np.eye(6),
                    lambda p: jnp.ones(6)*p[0], [y.mean()], ['a']*6)
    result = fit_node_scales(fit)
    group = result['node_scaling']['groups'][0]
    assert group['expected'] == pytest.approx(5., abs=1e-12)
    assert group['scale'] == pytest.approx(np.sqrt(np.sum((y-y.mean())**2)/5), rel=1e-12)
    assert result['covariance'][0, 0] == pytest.approx(group['scale']**2/6)
    assert result['node_scaling']['method'] == 'local geometry, one pass'


def test_marginal_node_expectations_follow_full_correlated_projection():
    x = np.repeat([[1., 0.], [1., 1.]], 3, axis=0)
    y = np.array([-3., 1., 2., 7., 9., 10.])
    corr = np.eye(6)
    corr[0, 3] = corr[3, 0] = .7
    w = np.linalg.inv(corr)
    beta = np.linalg.solve(x.T@w@x, x.T@w@y)
    residual = y-x@beta
    expected_cov = corr-x@np.linalg.solve(x.T@w@x, x.T)
    answer = node_scale_geometry(residual, -x, corr, ['a']*3+['ab']*3)
    np.testing.assert_allclose(answer['residual_covariance'], expected_cov, atol=1e-12)
    for group in answer['groups']:
        g = group['indices']
        assert group['observed'] == pytest.approx(residual[g]@residual[g])
        assert group['expected'] == pytest.approx(np.trace(expected_cov[np.ix_(g, g)]))
    assert answer['residual_df'] == 4
    assert not np.isclose(sum(g['observed'] for g in answer['groups']), answer['q'])


def test_one_correlated_node_recovers_generalized_birge():
    corr = np.array([[1., .6], [.6, 1.]])
    answer = node_scale_geometry([-2., 2.], -np.ones((2, 1)), corr, ['a', 'a'])
    group = answer['groups'][0]
    assert group['expected'] == pytest.approx(1.)
    assert group['observed'] == pytest.approx(20.)
    assert group['scale'] == pytest.approx(np.sqrt(20.))


def test_local_geometry_is_invariant_to_parameter_units_and_row_order():
    x = np.repeat([[1., 0], [0, 1.], [1., 1.]], 2, axis=0)
    corr = .2*np.ones((6, 6))+.8*np.eye(6)
    nodes = np.repeat(['a', 'b', 'sum'], 2)
    y = np.array([0., 2., 1., 4., 4., 9.])
    order = [5, 3, 1, 4, 2, 0]
    first = node_scale_geometry(y, x, corr, nodes)
    second = node_scale_geometry(y[order], (x*[1e-12, 1e12])[order], corr[np.ix_(order, order)], nodes[order])
    by_node = dict(zip(second['nodes'], second['scales']))
    np.testing.assert_allclose(first['scales'], [by_node[n] for n in first['nodes']], atol=1e-12)


def test_saturated_node_is_flagged_without_an_invented_scale():
    answer = node_scale_geometry([0.], [[1.]], [[1.]], ['a'])
    assert answer['residual_df'] == 0
    assert not answer['groups'][0]['estimable']
    assert answer['scales'][0] == 1


def test_dependent_nodes_share_one_scale_and_keep_raw_relation():
    block = np.array([[1., 0.], [0., 1.], [-1., 1.]])
    design = np.tile(block, (2, 1))
    covariance = block@block.T
    errors = np.sqrt(np.diag(covariance))*.1
    corr = np.kron(np.eye(2), covariance/np.sqrt(np.outer(np.diag(covariance), np.diag(covariance))))
    fit = model_fit([1., 2., 1., 3., 5., 2.], np.tile(errors, 2), np.tile(errors, 2),
                    corr, lambda p: jnp.asarray(design)@p, [2., 3.5], ['a', 'b', 'difference']*2)
    answer = fit_node_scales(fit)
    geometry = answer['node_scaling']
    assert len(geometry['groups']) == 1
    assert geometry['residual_df'] == 2
    assert geometry['groups'][0]['expected'] == pytest.approx(2.)
    assert geometry['scales'][0] > 2
    np.testing.assert_allclose(geometry['scales'], geometry['scales'][0])
    v = corr*np.outer(np.tile(errors, 2)*answer['input_scales'], np.tile(errors, 2)*answer['input_scales'])
    np.testing.assert_allclose(v[:3, :3]@np.array([1., -1., 1.]), 0, atol=1e-12)


def test_asymmetric_knot_is_explicit_and_uses_finite_local_geometry():
    fit = model_fit([-.4, 2.], [1., 2.], [1., 1.], np.eye(2),
                    lambda p: jnp.ones(2)*p[0], [0.], ['a']*2)
    answer = fit_node_scales(fit)
    assert answer['node_scaling']['knot_rows'] == [1]
    assert answer['node_scaling']['groups'][0]['expected'] == pytest.approx(1.)
    assert np.isfinite(answer['node_scaling']['scales']).all()


def test_local_scales_retain_nonlinear_asymmetric_correlated_objective():
    mean = lambda p: jnp.r_[jnp.repeat(p[0], 3), jnp.repeat(p[1], 3), jnp.repeat(p[0]*p[1], 3)]
    y = np.array([1.8, 2.1, 2.2, 2.8, 3.2, 2.9, 4., 7., 9.])
    lower = np.r_[np.full(6, .2), np.full(3, .4)]
    corr = np.eye(9); corr[0, 3] = corr[3, 0] = .35
    fit = model_fit(y, lower, 1.5*lower, corr, mean, [2., 3.], ['a']*3+['b']*3+['ab']*3)
    answer = fit_node_scales(fit)
    assert answer['node_scaling']['scales'][2] > 3
    np.testing.assert_array_equal(answer['corr_mat'], corr)
    assert answer['chi2_min'] == pytest.approx(float(fit['chi2_open'](answer['fitted_values'], y, answer['input_scales'])))
    assert np.linalg.norm(jax.grad(answer['chi2'])(answer['fitted_values'])) < 1e-4


def test_boundary_geometry_conditions_on_the_active_face():
    fit = model_fit([-.4, -.2, .1], [.2]*3, [.2]*3, np.eye(3),
                    lambda p: jnp.ones(3)*p[0], [.01], ['M000.1']*3)
    fit.update(parameters=['M000.1'], algorithm='BRU')
    answer = fit_node_scales(fit)
    assert abs(answer['param_values'][0]) < 1e-8
    assert answer['node_scaling']['active_bounds'] == [0]
    assert answer['node_scaling']['residual_df'] == 3
    assert answer['node_scaling']['scales'][0] == pytest.approx(np.sqrt(1.75))


def test_refit_handles_an_asymmetric_knot_on_a_physical_boundary():
    from pdgfits.refit import physical_refit_model
    fit = model_fit([-2., -2.2], [.2, .2], [1., 1.], np.eye(2),
                    lambda p: jnp.ones(2)*p[0], [.01], ['M000.1']*2)
    fit.update(parameters=['M000.1'], algorithm='BRU')
    result = prepare_refit(physical_refit_model(fit))(scales=np.full(2, 2.))
    assert abs(result['param_values'][0]) < 1e-8
    assert result['chi2_min'] == pytest.approx(2.21, abs=1e-8)
    assert not result['hesse_accurate']


def test_near_perfect_correlation_does_not_discard_a_measured_direction():
    corr = np.array([[1., 1.-1e-12], [1.-1e-12, 1.]])
    result = node_scale_geometry([-1., 1.], np.ones((2, 1)), corr, ['a', 'a'])
    assert result['measurement_rank'] == 2
    expected_q = np.array([-1., 1.])@np.linalg.pinv(corr)@np.array([-1., 1.])
    assert result['q'] == pytest.approx(expected_q, rel=1e-4)
    assert result['groups'][0]['expected'] == pytest.approx(1., abs=1e-4)


def test_refit_follows_descent_along_an_asymmetric_corner():
    # At x=y=0, moving either coordinate alone increases Q, but moving along
    # x+y=0 decreases it. A coordinate-only descent check is insufficient.
    fit = model_fit([-1., .6, .4], [.2, 1., 1.], [1., 1., 1.], np.eye(3),
                    lambda p: jnp.array([p[0]+p[1], p[0], p[1]]), [0., 0.], ['sum', 'x', 'y'])
    answer = prepare_refit(fit)(full_output=False)
    assert answer['chi2_min'] == pytest.approx(1.5, abs=1e-9)
    np.testing.assert_allclose(answer['fitted_values'], [.1, -.1], atol=2e-6)
