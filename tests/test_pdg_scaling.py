import numpy as np
import pytest
from numpy.testing import assert_allclose
from pdgfits.pdg_scaling import pdg_average, pdg_linear_fit
from pdgfits.birge import fit_linear_scales


def test_average_exclusion_keeps_original_mean():
    y, e = np.array([0., 2., 100.]), np.array([1., 1., 20.])
    result = pdg_average(y, e)
    mean = np.average(y, weights=1/e**2)
    assert_allclose(result['value'], mean)
    assert_allclose(result['scale'], np.sqrt(mean**2+(2-mean)**2))
    assert_allclose(result['error_n'], result['scale']/np.sqrt(np.sum(1/e**2)))
    np.testing.assert_array_equal(result['scale_inputs'], [True, True, False])


def test_equal_precision_pull_and_birge_agree():
    y = [-2., 0, 2]
    result = pdg_linear_fit(y, np.ones((3, 1)), np.eye(3), ['a']*3, exclude_weak=False)
    assert_allclose(result['node_scales']['a'], pdg_average(y, np.ones(3), exclude_weak=False)['scale'], rtol=1e-5)


def test_unequal_precision_pull_and_birge_are_not_equivalent():
    y, e = np.array([0., 0., 20.]), np.array([1., 1., 10.])
    result = pdg_linear_fit(y, np.ones((3, 1)), np.diag(e*e), ['a']*3, exclude_weak=False)
    mean = np.average(y, weights=1/e**2)
    u2 = 1/np.sum(1/e**2)
    expected = max(1., np.sqrt(np.mean((y-mean)**2/(e**2-(1-1e-5)**2*u2))))
    assert_allclose(result['node_scales']['a'], expected)
    assert abs(expected-pdg_average(y, e, exclude_weak=False)['scale']) > .01


def test_identity_block_reproduces_richie_example():
    y, x, v = np.array([.9, 1.1]), np.ones((2, 1)), np.eye(2)*.01**2
    ordinary = pdg_linear_fit(y, x, v, ['a']*2, exclude_weak=False)
    correlated = pdg_linear_fit(y, x, v, ['a']*2, exclude_weak=False, correlated_blocks=[[0, 1]])
    assert_allclose(np.sqrt(ordinary['covariance'][0,0]), .1, rtol=2e-5)
    assert_allclose(np.sqrt(correlated['covariance'][0,0]), .01/np.sqrt(2), rtol=1e-10)
    updated = correlated['measurement_covariance']
    assert updated[0,1]/np.sqrt(updated[0,0]*updated[1,1]) < -.99
    reml = fit_linear_scales(y, x, v, ['a']*2)
    assert_allclose(reml['measurement_covariance'], .02*np.eye(2), rtol=1e-6)


def test_fit_excludes_once_refits_and_reports_original_center():
    y, e = np.array([0., 2., 100.]), np.array([1., 1., 20.])
    result = pdg_linear_fit(y, np.ones((3, 1)), np.diag(e*e), ['a']*3)
    assert_allclose(result['parameters'], [np.average(y, weights=1/e**2)])
    assert_allclose(result['refitted_parameters'], [1])
    np.testing.assert_array_equal(result['retained'], [True, True, False])
