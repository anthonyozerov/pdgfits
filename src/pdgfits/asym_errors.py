import jax
from jax import numpy as jnp
from decimal import Decimal
from scipy.optimize import NonlinearConstraint, minimize


def binary_search_error(profile_chi2, val, chi2_min, err_scale, lb, ub):
    """
    Find 1-sigma asymmetric errors by binary search on a profile chi-squared.

    profile_chi2: callable(scalar) -> chi2 profiled over all other parameters
    val: optimal value of the target quantity
    chi2_min: minimum chi2 at the optimum
    err_scale: scale for convergence tolerance (search stops when bracket < 1e-3 * err_scale)
    lb: initial lower bound (lb < val); expanded outward if chi2 doesn't reach chi2_min+1
    ub: initial upper bound (ub > val); expanded outward if chi2 doesn't reach chi2_min+1

    Returns (upper_err, lower_err).
    """
    target = chi2_min + 1
    chi2 = jnp.inf

    while profile_chi2(ub) < target:
        ub = val + 2 * (ub - val)
    lo, hi = val, ub
    while jnp.abs(chi2-target)>0.005:
        mid = (lo + hi) / 2
        chi2 = profile_chi2(mid)
        if chi2 < target:
            lo = mid
        else:
            hi = mid
    assert jnp.abs(chi2 - target) < 0.1, f'Expected chi2 ~ {target}, got {chi2} at mid={mid}, lo={lo}, hi={hi}'

    upper_err = float(hi - val)

    chi2 = jnp.inf
    while profile_chi2(lb) < target:
        lb = val - 2 * (val - lb)
    lo, hi = lb, val
    while jnp.abs(chi2-target)>0.005:
        mid = (lo + hi) / 2
        chi2 = profile_chi2(mid)
        if chi2 < target:
            hi = mid
        else:
            lo = mid
    assert jnp.abs(chi2 - target) < 0.1, f'Expected chi2 ~ {target}, got {chi2} at mid={mid}, lo={lo}, hi={hi}'
    lower_err = float(val - lo)

    return upper_err, lower_err


def calc_asym_errors(fit, targets=None):
    """
    Calculate asymmetric errors via binary search for a subset of nodes or parameters.

    fit: dict returned by run_fit
    targets: list of node or parameter names, or None to use all nodes
    """
    nodes = fit['nodes']
    parameters = fit['parameters']
    node_funcs = fit['node_funcs']
    parameter_funcs = fit['parameter_funcs']
    fitted_params_to_params = fit['fitted_params_to_params']
    fitted_values = fit['fitted_values']
    param_values = fit['param_values']
    chi2 = fit['chi2']
    chi2_grad = fit['chi2_grad']
    chi2_min = fit['chi2_min']
    covariance = fit['covariance']

    if targets is None:
        targets = nodes

    J = jax.jacobian(fitted_params_to_params)(fitted_values)
    param_cov = J @ covariance @ J.T

    for target in targets:
        if target in nodes:
            target_func = node_funcs[nodes.index(target)]
        elif target in parameters:
            target_func = parameter_funcs[parameters.index(target)]
        else:
            print(f'Warning: {target} not found in nodes or parameters, skipping.')
            continue

        target_value = float(target_func(param_values))
        J_target = jax.jacobian(target_func)(param_values)
        target_std = float(jnp.sqrt(J_target @ param_cov @ J_target.T))

        def make_constraint(func):
            @jax.jit
            def constraint_jax(fp):
                return func(fitted_params_to_params(fp))
            return lambda fp: constraint_jax(jnp.array(fp))

        constraint = make_constraint(target_func)

        def profile_chi2(v):
            c = NonlinearConstraint(constraint, lb=v, ub=v)
            res = minimize(fun=chi2, x0=fitted_values, jac=chi2_grad, constraints=[c])
            assert res.success
            return float(res.fun)

        lb = target_value - 10 * target_std
        ub = target_value + 10 * target_std

        upper_err, lower_err = binary_search_error(
            profile_chi2, target_value, chi2_min, target_std, lb, ub
        )

        print(f'{target}: {Decimal(target_value):.5E} + {Decimal(upper_err):.2E} - {Decimal(lower_err):.2E}')
