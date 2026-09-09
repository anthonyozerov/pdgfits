import numpy as np
import pytest
from scipy.optimize import Bounds, LinearConstraint

from pdgfits.fit import _run_constrained_scipy


def test_constrained_scipy_driver_solves_bounded_linear_constraint():
    target = np.array([0.25, 0.75])

    def scipy_obj(x):
        resid = x - target
        return float(resid @ resid), 2.0 * resid

    def scipy_value(x):
        resid = x - target
        return float(resid @ resid)

    def scipy_hessp(x, p):
        return 2.0 * p

    result = _run_constrained_scipy(
        scipy_obj,
        scipy_value,
        scipy_hessp,
        np.array([0.9, 0.1]),
        Bounds([0.0, 0.0], [1.0, 1.0]),
        [LinearConstraint([[1.0, 1.0]], 1.0, 1.0)],
    )

    assert result.success
    assert result.fun == pytest.approx(0.0, abs=1e-14)
    assert result.x == pytest.approx(target)
