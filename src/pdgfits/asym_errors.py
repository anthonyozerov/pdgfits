import jax
from jax import numpy as jnp
from decimal import Decimal
from scipy.optimize import NonlinearConstraint, minimize


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

        target_value = target_func(param_values)
        J_target = jax.jacobian(target_func)(param_values)
        target_std = float(jnp.sqrt(J_target @ param_cov @ J_target.T))

        def make_constraint(func):
            @jax.jit
            def constraint_jax(fp):
                return func(fitted_params_to_params(fp))
            return lambda fp: constraint_jax(jnp.array(fp))

        constraint = make_constraint(target_func)

        def chi2_at_val(val):
            c = NonlinearConstraint(constraint, lb=val, ub=val)
            return minimize(fun=chi2, x0=fitted_values, jac=chi2_grad, constraints=[c])

        lb = target_value
        ub = target_value + 10 * target_std
        while (ub - lb) / target_std > 1e-3:
            mid = (lb + ub) / 2
            res = chi2_at_val(mid)
            assert res.success
            if res.fun < chi2_min + 1:
                lb = mid
            else:
                ub = mid
        upper_err = float(ub - target_value)

        lb = target_value - 10 * target_std
        ub = target_value
        while (ub - lb) / target_std > 1e-3:
            mid = (lb + ub) / 2
            res = chi2_at_val(mid)
            if res.fun < chi2_min + 1:
                ub = mid
            else:
                lb = mid
        lower_err = float(target_value - lb)

        target_value = float(target_value)
        print(f'{target}: {Decimal(target_value):.5E} + {Decimal(upper_err):.2E} - {Decimal(lower_err):.2E}')
