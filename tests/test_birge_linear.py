import numpy as np
import pytest
from numpy.testing import assert_allclose

from pdgfits.birge import linear_birge, fit_linear_scales


def test_ordinary_birge_and_independent_nodes():
    y = np.array([-3., 0, 3, 10, 10, 10])
    design = np.repeat(np.eye(2), 3, axis=0)
    result = fit_linear_scales(y, design, np.eye(6), ['a'] * 3 + ['b'] * 3)
    assert_allclose(result['parameters'], [0, 10], atol=1e-10)
    assert_allclose(result['scales'], [3, 1], rtol=1e-6)
    assert_allclose(result['initial']['response'], np.eye(2) * 2, atol=1e-14)


def test_correlated_block_preserves_input_correlation():
    v = np.array([[1., .5], [.5, 1.]]) * .01**2
    result = fit_linear_scales([.9, 1.1], np.ones((2, 1)), v, ['a', 'a'])
    assert len(result['nodes']) == 1
    assert_allclose(result['scales'], [20], rtol=1e-6)
    assert_allclose(result['measurement_covariance'], v * 400, rtol=1e-6)
    assert_allclose(result['parameters'], [1], rtol=1e-10)
    identity = fit_linear_scales([.9, 1.1], np.ones((2, 1)), np.eye(2)*.01**2, ['a', 'a'])
    assert_allclose(identity['scales'], [np.sqrt(200)], rtol=1e-6)


def test_residual_expectations_when_scales_differ():
    design = np.repeat([[1., 0], [0, 1.], [1., 1.]], 2, axis=0)
    nodes = np.repeat(['a', 'b', 'sum'], 2)
    result = linear_birge(np.zeros(6), design, np.eye(6), nodes)
    response = result['response']
    assert_allclose(response.sum(1), [4/3] * 3)
    assert_allclose(response, (np.ones((3, 3)) + 9*np.eye(3))/9)
    assert_allclose(response @ np.ones(3), [b['expected_q'] for b in result['blocks']])
    residual_map = np.eye(6) - design @ np.linalg.pinv(design)
    true_covariance = np.diag(np.repeat([1, 4, 9], 2))
    expected = np.diag(residual_map @ true_covariance @ residual_map.T).reshape(3, 2).sum(1)
    assert_allclose(response @ [1, 4, 9], expected)


def test_units_and_row_permutation_do_not_change_scales():
    y = np.array([0., 2, 1, 4, 4, 9])
    design = np.repeat([[1., 0], [0, 1.], [1., 1.]], 2, axis=0)
    nodes = np.repeat(['a', 'b', 'sum'], 2)
    result = fit_linear_scales(y, design, np.eye(6), nodes)
    units = np.array([1e-9, 1e-9, 1e8, 1e8, 1., 1.])
    changed = fit_linear_scales(y*units, design*units[:, None], np.diag(units**2), nodes)
    assert_allclose(changed['parameters'], result['parameters'], atol=1e-6)
    assert_allclose(changed['scales'], result['scales'], rtol=1e-6)
    order = [5, 3, 1, 4, 2, 0]
    reordered = fit_linear_scales(y[order], design[order], np.eye(6), nodes[order])
    assert_allclose(reordered['parameters'], result['parameters'], atol=1e-6)
    by_node = dict(zip(reordered['nodes'], reordered['scales']))
    assert_allclose([by_node[n] for n in ['a', 'b', 'sum']], result['scales'], rtol=1e-6)


def test_saturated_node_is_not_a_measured_scale():
    result = linear_birge([3], [[1]], [[1]], ['a'])
    assert not result['blocks'][0]['estimable']
    assert np.isnan(result['blocks'][0]['scale'])
    with pytest.raises(ValueError, match='positive definite'):
        linear_birge([1, 2], [[1], [1]], [[1, 1], [1, 1]], ['a', 'a'])


def test_cross_node_correlations_allow_different_scales():
    from scipy.optimize import minimize
    y = np.array([-3., 0, 3, 10, 10, 10])
    x = np.repeat(np.eye(2), 3, axis=0)
    v = .2*np.ones((6, 6)) + .8*np.eye(6)
    result = fit_linear_scales(y, x, v, ['a']*3 + ['b']*3)
    assert result['scales'][0] > 3
    assert result['scales'][1] == pytest.approx(1)
    changed = result['measurement_covariance']
    assert_allclose(changed/np.sqrt(np.outer(changed.diagonal(), changed.diagonal())), v)
    # Independent derivative-free minimization of the REML determinant criterion.
    def criterion(log_scales):
        s = np.exp(np.repeat(log_scales, 3))
        cov = v*np.outer(s, s)
        precision = np.linalg.inv(cov)
        info = x.T@precision@x
        beta = np.linalg.solve(info, x.T@precision@y)
        r = y-x@beta
        return (np.linalg.slogdet(cov)[1] + np.linalg.slogdet(info)[1] + r@precision@r)/2
    direct = minimize(criterion, [1, .3], method='Nelder-Mead', bounds=[(0, None)]*2,
                      options={'xatol': 1e-10, 'fatol': 1e-10})
    assert direct.success
    assert_allclose(result['scales'], np.exp(direct.x), rtol=1e-6)


def test_correlated_reml_reaches_boundary_solution():
    # A relative-objective stop can occur before the scale score is small here.
    y = np.array([.314179299187676, -.9146405017445695, 1.5224506738854078,
                  .3230076366592777, -1.696775859686525, -.6923469271064584])
    x = np.repeat([[1., 0], [0, 1.], [1., 1.]], 2, axis=0)
    v = np.eye(6)
    v[np.ix_([0, 2, 4], [0, 2, 4])] = .6*np.eye(3)+.4*np.ones((3, 3))
    fit = fit_linear_scales(y, x, v, np.repeat(['a', 'b', 'sum'], 2))
    assert_allclose(fit['scales'][1:], [1, 1], atol=1e-8)
    assert fit['restricted_nll'] < 2.594
    assert fit['score_residual'] < 1e-6
