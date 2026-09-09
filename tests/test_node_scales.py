import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
from scipy.integrate import quad

from pdgfits.build_chi2 import build_chi2
from pdgfits.node_scales import _error_sampler, fit_node_scales, sample_errors
from pdgfits.refit import prepare_refit
from pdgfits.scalar_average import scalar_average
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


def test_asymmetric_sampler_agrees_with_normalized_density():
    lower, upper = .2, 1.
    def kernel(r):
        sigma = np.clip((2*lower*upper-r*(upper-lower))/(lower+upper), lower, upper)
        return np.exp(-.5*(r/sigma)**2)
    norm = quad(kernel, -10, 10, points=[-upper, lower], epsabs=1e-10)[0]
    mean = quad(lambda r: r*kernel(r), -10, 10, points=[-upper, lower], epsabs=1e-10)[0]/norm
    variance = quad(lambda r: (r-mean)**2*kernel(r), -10, 10, points=[-upper, lower], epsabs=1e-10)[0]/norm
    sample = sample_errors([lower], [upper], np.eye(1), 50000, 37)[:, 0]
    assert sample.mean() == pytest.approx(mean, abs=5*np.sqrt(variance/len(sample)))
    assert sample.var() == pytest.approx(variance, rel=.04)


def test_symmetric_sampler_keeps_cross_node_covariance():
    sigma = np.array([1., 2.])
    correlation = np.array([[1., .6], [.6, 1.]])
    sample = sample_errors(sigma, sigma, correlation, 40000, 14)
    np.testing.assert_allclose(np.cov(sample.T), correlation*np.outer(sigma, sigma), rtol=.025)


def test_refitted_score_recovers_birge_with_reported_monte_carlo_error():
    y = np.array([-2., -1., 0., 0., 1., 3.])
    fit = model_fit(y, np.ones(6), np.ones(6), np.eye(6),
                    lambda p: jnp.ones(6)*p[0], [y.mean()], ['a']*6)
    result = fit_node_scales(fit, draws=256, seed=91, tolerance=1e-5)
    diagnostic = result['node_scaling']
    expected = diagnostic['expected'][0]
    assert expected == pytest.approx(5., abs=5*diagnostic['expected_mc_se'][0])
    assert diagnostic['scales'][0] == pytest.approx(np.sqrt(np.sum((y-y.mean())**2)/expected), rel=1e-4)
    assert diagnostic['score_residual'] < 1e-5


def test_expected_node_scores_match_correlated_gaussian_projection():
    design = np.repeat([[1., 0.], [1., 1.]], 3, axis=0)
    y = np.array([-3., 1., 2., 7., 9., 10.])
    correlation = np.eye(6)
    correlation[0, 3] = correlation[3, 0] = .4
    fit = model_fit(y, np.ones(6), np.ones(6), correlation,
                    lambda p: jnp.asarray(design)@p, [0., 8.], ['a']*3+['ab']*3)
    result = fit_node_scales(fit, draws=512, seed=38)
    first = result['node_scaling']['history'][0]
    weight = np.linalg.inv(correlation)
    information = design.T@weight@design
    projection = weight-weight@design@np.linalg.solve(information, design.T@weight)
    residual = y-design@np.linalg.solve(information, design.T@weight@y)
    for i in range(2):
        indicator = np.repeat([i == 0, i == 1], 3)
        derivative = indicator[:, None]*correlation+correlation*indicator[None, :]
        expected = .5*np.trace(projection@derivative)
        assert abs(first['expected'][i]-expected) < 5*first['expected_mc_se'][i]
        assert first['observed'][i] == pytest.approx((indicator*residual)@weight@residual, abs=1e-8)
    assert np.trace(projection@correlation) == pytest.approx(4.)


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


def test_singular_asymmetric_kernel_is_not_silently_treated_as_density():
    with pytest.raises(np.linalg.LinAlgError):
        sample_errors([.2, .3], [1., 1.], np.ones((2, 2)), 100)


def test_profile_scale_score_at_asymmetric_kink_matches_refitted_difference():
    y, lower, upper = np.array([-.4, 2.]), np.array([1., 2.]), np.array([1., 1.])
    fit = model_fit(y, lower, upper, np.eye(2), lambda p: jnp.ones(2)*p[0], [0.], ['a']*2)
    result = fit_node_scales(fit, draws=128, seed=81)
    observed = result['node_scaling']['history'][0]['observed'][0]
    step = 1e-4
    plus = scalar_average(y, lower*np.exp(step), upper*np.exp(step), np.eye(2))['q']
    minus = scalar_average(y, lower*np.exp(-step), upper*np.exp(-step), np.eye(2))['q']
    assert observed == pytest.approx(-(plus-minus)/(4*step), abs=2e-4)


def test_nonlinear_asymmetric_correlated_fit_localizes_scale():
    mean = lambda p: jnp.r_[jnp.repeat(p[0], 3), jnp.repeat(p[1], 3), jnp.repeat(p[0]*p[1], 3)]
    y = np.array([1.8, 2.1, 2.2, 2.8, 3.2, 2.9, 4., 7., 9.])
    lower = np.r_[np.full(6, .2), np.full(3, .4)]
    correlation = np.eye(9); correlation[0, 3] = correlation[3, 0] = .35
    fit = model_fit(y, lower, 1.5*lower, correlation, mean, [2., 3.], ['a']*3+['b']*3+['ab']*3)
    result = fit_node_scales(fit, draws=128, seed=81)
    scaling = result['node_scaling']
    assert scaling['score_residual'] < .001
    assert scaling['scales'][2] > 3
    np.testing.assert_allclose(scaling['scales'][:2], 1.)
    np.testing.assert_array_equal(result['corr_mat'], correlation)


def test_asymmetric_correlated_sampling_has_zero_expected_scale_score():
    lower, upper = np.array([.2, .5]), np.array([1., .8])
    correlation = np.array([[1., .4], [.4, 1.]])
    values = sample_errors(lower, upper, correlation, 20000, 59)
    _, _, _, opened = build_chi2(jnp.zeros(2), lambda p: p, lower, upper,
                                 jnp.linalg.inv(jnp.asarray(correlation)), lambda p: p)
    scores = np.asarray(jax.jit(jax.vmap(lambda y: -.5*jax.grad(
        lambda logs: opened(jnp.zeros(2), y, jnp.exp(logs)))(jnp.zeros(2))))(values))
    assert np.all(np.abs(scores.mean(axis=0)-1) < 5*scores.std(axis=0)/np.sqrt(len(scores)))


@pytest.mark.parametrize('asymmetric', [False, True])
def test_singular_sampler_keeps_exact_raw_relation_with_unequal_node_scales(asymmetric):
    # Third reported quantity is the difference of the first two. Inflation
    # may change the information within this plane, never the exact identity.
    design = np.array([[1., 0.], [0., 1.], [-1., 1.]])
    covariance = design@design.T
    lower = np.sqrt(np.diag(covariance))
    correlation = covariance/np.outer(lower, lower)
    upper = lower*np.array([1.5, .8, 1.]) if asymmetric else lower
    fit = model_fit([2., 3., 1.], lower, upper, correlation,
                    lambda p: jnp.asarray(design)@p, [2., 3.], ['a', 'b', 'difference'])
    sample, blocks = _error_sampler(fit, 20000, 18)
    errors, weights, effective = sample(np.array([2., 1., 3.]))
    np.testing.assert_allclose(errors[:, 2], errors[:, 1]-errors[:, 0], atol=1e-13)
    assert blocks == [[0, 1, 2]]
    assert effective > 1000
    if not asymmetric:
        precision = np.linalg.pinv(correlation)/np.outer(lower*np.array([2., 1., 3.]), lower*np.array([2., 1., 3.]))
        expected = design@np.linalg.inv(design.T@precision@design)@design.T
        np.testing.assert_allclose(np.cov(errors.T, aweights=weights), expected, atol=.15)


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


def test_batched_refits_match_individual_nonlinear_asymmetric_fits():
    mean = lambda p: jnp.array([p[0], p[0], p[1], p[0]*p[1], p[0]*p[1]])
    fit = model_fit([1., 1.1, 2., 2.1, 1.8], [.2]*5, [.3]*5, np.eye(5), mean, [1., 2.], ['a', 'a', 'b', 'ab', 'ab'])
    refit = prepare_refit(fit)
    values = np.asarray(mean(jnp.array([1., 2.])))+sample_errors([.2]*5, [.3]*5, np.eye(5), 32, 45)
    batch = refit.batch(values, np.ones(5), start=np.zeros(2))
    for y, p in zip(values, batch):
        single = refit(y, full_output=False)
        assert float(fit['chi2_open'](p, y)) <= single['chi2_min']+1e-7


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


def test_node_refits_respect_an_active_physical_boundary():
    fit = model_fit([-.4, -.2, .1], [.2]*3, [.2]*3, np.eye(3),
                    lambda p: jnp.ones(3)*p[0], [.01], ['M000.1']*3)
    fit.update(parameters=['M000.1'], algorithm='BRU')
    result = fit_node_scales(fit, draws=128, seed=37)
    assert abs(result['param_values'][0]) < 1e-8
    assert result['node_scaling']['scales'][0] > 1
    assert result['node_scaling']['score_residual'] <= result['node_scaling']['tolerance']


def test_scale_solution_can_lie_at_a_mean_boundary_and_asymmetric_knot():
    fit = model_fit([-2., -2.2], [.2, .2], [1., 1.], np.eye(2),
                    lambda p: jnp.ones(2)*p[0], [.01], ['M000.1']*2)
    fit.update(parameters=['M000.1'], algorithm='BRU')
    result = fit_node_scales(fit, draws=256, seed=91, tolerance=1e-4, mc_tolerance=0)
    scaling = result['node_scaling']
    assert abs(result['param_values'][0]) < 1e-8
    assert scaling['scales'][0] == pytest.approx(2., abs=1e-5)
    assert scaling['raw_score_residual'] < 1e-4


def test_singular_and_ordinary_blocks_use_independent_random_streams():
    correlation = np.array([[1., 1., 0.], [1., 1., 0.], [0., 0., 1.]])
    fit = model_fit([0., 0., 0.], np.ones(3), np.ones(3), correlation,
                    lambda p: jnp.array([p[0], p[0], p[1]]), [0., 0.], ['a', 'a', 'b'])
    sampler, _ = _error_sampler(fit, 20000, 41)
    errors, _, _ = sampler(np.ones(3))
    np.testing.assert_allclose(errors[:, 0], errors[:, 1], atol=1e-14)
    assert abs(np.corrcoef(errors[:, [0, 2]].T)[0, 1]) < .025
