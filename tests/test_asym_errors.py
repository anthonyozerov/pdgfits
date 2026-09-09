import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from pdgfits.asym_errors import (
    CallableProfileProblem,
    ProfilePoint,
    build_coordinate_profile_chi2,
    binary_search_error,
    build_constrained_profile_chi2,
    find_profile_root,
)
from pdgfits.avg import run_avg


def test_boundary_endpoint_is_not_forced_to_cross_objective_level():
    root = find_profile_root(lambda x: (x-.1)**2, .1, 0., -2., 2., limits=(0., np.inf))
    assert root.lower.endpoint == 0.
    assert root.lower.is_bound
    assert root.lower.chi2 < 1
    assert not root.upper.is_bound
    assert abs(root.upper.residual) < .005


def test_root_rejects_failed_profile_even_when_objective_is_finite():
    class FailedProfile:
        def evaluate(self, value):
            return ProfilePoint(value, value**2, False, 0.0, "not minimized")

    with pytest.raises(RuntimeError, match="Unsuccessful profile"):
        find_profile_root(FailedProfile(), 0, 0, -2, 2, contract_brackets=False)


def test_root_honors_requested_residual_tolerance():
    # A jump across the requested objective level cannot supply an endpoint.
    def profile(value):
        return 0.996 if abs(value) < 1 else 1.004

    with pytest.raises(RuntimeError, match="endpoint verification failed"):
        find_profile_root(profile, 0, 0, -2, 2, residual_tol=0.001)


def test_coordinate_profile_does_not_certify_an_unoptimized_start(monkeypatch):
    from scipy.optimize import OptimizeResult
    import pdgfits.profiles as profiles

    def failed_minimize(fun, x, **kwargs):
        value = fun(x)
        return OptimizeResult(x=x, fun=value[0] if isinstance(value, tuple) else value,
                              success=False, message="no progress")

    monkeypatch.setattr(profiles, "minimize", failed_minimize)
    profile = build_coordinate_profile_chi2(
        lambda x: x[0]**2 + (x[1]-x[0])**2, np.array([0., 0.]), 0
    )
    with pytest.raises(RuntimeError, match="Profile minimization failed"):
        profile(1.0)


def test_binary_search_reuses_last_endpoint_evaluation():
    calls = []

    def profile_chi2(x):
        x = float(x)
        calls.append(x)
        return x * x

    err_p, err_n = binary_search_error(profile_chi2, 0.0, 0.0, -2.0, 2.0)

    assert err_p == pytest.approx(1.0)
    assert err_n == pytest.approx(1.0)
    assert calls == [2.0, 1.0, -2.0, -1.0]
    diag = binary_search_error.last_diagnostics
    assert diag["function_evals"] == 4
    assert diag["cache_hits"] == 0
    assert diag["upper_profile_point"]["chi2"] == pytest.approx(1.0)
    assert diag["lower_profile_point"]["chi2"] == pytest.approx(1.0)
    assert binary_search_error.last_root.upper.point.success


def test_constrained_profile_checks_success_and_endpoint_residuals():
    def chi2(x):
        # Nontrivial curved nuisance valley: target is x0, nuisance optimum depends on x0.
        return ((x[0] - 1.25) / 0.7) ** 2 + ((x[1] - 0.4 * x[0]) / 0.2) ** 2

    def target_func(params):
        return params[0]

    fitted = np.array([1.25, 0.5])
    prof = build_constrained_profile_chi2(
        chi2, lambda x: x, target_func, fitted, target_scale=0.7, target_name="x0"
    )
    err_p, err_n = binary_search_error(prof, 1.25, 0.0, 1.25 - 2 * 0.7, 1.25 + 2 * 0.7)
    assert err_p == pytest.approx(0.7, abs=5e-3)
    assert err_n == pytest.approx(0.7, abs=5e-3)
    diag = binary_search_error.last_diagnostics
    assert abs(diag["upper_residual"]) < 1e-2
    assert abs(diag["lower_residual"]) < 1e-2
    assert all(p.success for p in prof.diagnostics)


def test_find_profile_root_accepts_explicit_problem_metadata():
    def profile_chi2(x):
        x = float(x)
        return x * x

    problem = CallableProfileProblem(
        profile_chi2,
        target_name="quadratic",
        target_value=0.0,
        chi2_min=0.0,
        lower_initial=-2.0,
        upper_initial=2.0,
    )
    root = find_profile_root(problem)

    assert root.upper.error == pytest.approx(1.0)
    assert root.lower.error == pytest.approx(1.0)
    assert root.upper.point.method == "callable"
    assert root.diagnostics()["search_method"] == "sqrt-secant"


def test_constrained_profile_respects_tiny_target_scale():
    center = 1.0e-12
    sigma = 2.5e-15

    def chi2(x):
        return ((x[0] - center) / sigma) ** 2 + ((x[1] - 0.25 * x[0]) / sigma) ** 2

    def target_func(params):
        return params[0]

    fitted = np.array([center, 0.25 * center])
    prof = build_constrained_profile_chi2(
        chi2, lambda x: x, target_func, fitted, target_scale=sigma, target_name="tiny"
    )
    err_p, err_n = binary_search_error(
        prof, center, 0.0, center - 2 * sigma, center + 2 * sigma
    )
    assert err_p == pytest.approx(sigma, rel=0, abs=5e-18)
    assert err_n == pytest.approx(sigma, rel=0, abs=5e-18)
    diag = binary_search_error.last_diagnostics
    assert abs(diag["upper_residual"]) < 1e-2
    assert abs(diag["lower_residual"]) < 1e-2


def test_average_asymmetric_errors_verify_profile_endpoints():
    # A small but non-quadratic average-side problem: asymmetric measurement
    # errors and a nonzero measurement correlation exercise build_chi2's PDG
    # interpolation plus the run_avg profile endpoint verification path.
    meas_df = pd.DataFrame(
        {
            "node": ["TAVG", "TAVG", "TAVG"],
            "measurement": ["1.00+0.20-0.10", "1.35+0.12-0.25", "0.82+0.18-0.14"],
            "text": ["MeV", "MeV", "MeV"],
            "ignore_minus": [None, None, None],
            "systematic_error_clump": [None, None, None],
            "systematic_error_clump2": [None, None, None],
            "reference_id": ["A", "B", "C"],
            "occurrence": [1, 1, 1],
        }
    )
    corr_df = pd.DataFrame(
        {
            "node_one": ["TAVG"],
            "reference_id_one": ["A"],
            "occurrence_one": [1],
            "node_two": ["TAVG"],
            "reference_id_two": ["B"],
            "occurrence_two": [1],
            "correlation": [0.25],
        }
    )
    result = run_avg("TAVG", meas_df, corr_df)
    diag = result["asym_error_diagnostics"]
    assert abs(diag["upper_residual"]) < 1e-2
    assert abs(diag["lower_residual"]) < 1e-2
    assert result["error_p"] > 0
    assert result["error_n"] > 0
    assert all(p.success for p in result["profile_diagnostics"])
