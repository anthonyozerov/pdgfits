import numpy as np
import jax
from jax import numpy as jnp
import warnings
from iminuit import Minuit
from scipy.optimize import minimize as scipy_minimize, Bounds, LinearConstraint

from pdgfits.fit_query import query_db
from pdgfits.preprocess import preprocess
from pdgfits.build_funcs import get_node_funcs, get_parameter_funcs, get_meas_funcs, get_mu, get_translate_dep, get_adjust
from pdgfits.func_factory import ALLOWED_EQUATION_TYPES
from pdgfits.corr_mat import get_corr_mat
from pdgfits.build_chi2 import build_chi2
from pdgfits.param_maps import build_param_map_sigmoid, build_param_map_softmax, get_decay_info


def run_fit(label, verbose=True, optimizer='minuit', fit_space='unconstrained'):
    """
    Run a single fit by label.

    Returns a dict with fit results, or None if the fit was skipped.

    Keys:
        label, algorithm, parameters, nodes,
        param_values, fitted_values, chi2_min,
        chi2, chi2_grad,
        fitted_params_to_params, params_to_fitted_params,
        node_funcs, parameter_funcs,
        meas_df, rel_df, fit_df, mu,
        covariance
    """
    algorithm, measurement_type, fit_df, rel_df, meas_df, corr_df, fit_seed_df, tree_df = query_db(label, verbose=False)

    if verbose:
        print(f'algorithm: {algorithm}')

    fit_df, rel_df, meas_df, corr_df, fit_seed_df, dep_meas_data, adjust_data = preprocess(
        fit_df, rel_df, meas_df, corr_df, fit_seed_df, tree_df, algorithm, measurement_type
    )

    parameters = list(fit_seed_df['parameter_key'].unique())
    nodes = list(np.unique(rel_df['node']))
    particles = list(np.unique([n[:4] for n in nodes + parameters if not n.startswith('nuisance_')]))

    if verbose:
        print(f'{len(parameters)} parameters, {len(nodes)} nodes, {len(particles)} particles')
        print('Parameters:', ', '.join(parameters))

    assert all(meas_df['node'].isin(nodes + parameters))

    if any(~fit_df['type'].isin(ALLOWED_EQUATION_TYPES)):
        if verbose:
            print(list(fit_df['type'][~fit_df['type'].isin(ALLOWED_EQUATION_TYPES)]))
            warnings.warn('Some nodes have an equation type which is not supported. Fit skipped.')
        return None

    node_funcs = get_node_funcs(nodes, parameters, fit_df, rel_df)
    parameter_funcs = get_parameter_funcs(parameters)
    meas_funcs = get_meas_funcs(dict(zip(nodes, node_funcs)), dict(zip(parameters, parameter_funcs)), meas_df)
    mu = get_mu(meas_funcs)

    if verbose and any(d is not None for d in dep_meas_data):
        print('Dependent measurements found, these will be accounted for.')
    translate_dep = get_translate_dep(dep_meas_data, parameters, nodes, parameter_funcs, node_funcs)
    adjust = get_adjust(adjust_data, parameters, nodes, parameter_funcs, node_funcs)

    y = jnp.array(meas_df['value'], dtype=jnp.float64)
    error_n = jnp.array(meas_df['error_n'], dtype=jnp.float64)
    error_p = jnp.array(meas_df['error_p'], dtype=jnp.float64)

    corr_mat = get_corr_mat(meas_df, corr_df)
    if verbose and not jnp.all(jnp.linalg.eigvals(corr_mat) >= 0):
        print('!!!correlation matrix is not PSD!!!')
        print('!!!this is probably bad!!!')
    corr_mat_inv = jnp.linalg.pinv(corr_mat)

    is_bru = algorithm == 'BRU' or (algorithm in ['BR', 'BR (NO MATRIX)', 'BR PRINT'] and len(particles) > 1)
    is_br = algorithm in ['BR', 'BR (NO MATRIX)', 'BR PRINT'] and len(particles) == 1
    constrained = fit_space == 'constrained'

    if constrained and is_br and optimizer == 'minuit':
        raise ValueError("Constrained sum-to-one fit is not supported with the minuit optimizer")

    if constrained and (is_bru or is_br):
        _, decay_param_idxs = get_decay_info(parameters)
        fitted_params_to_params = lambda x: x
        params_to_fitted_params = lambda x: x
        fixed_idx = None
    elif is_bru:
        fitted_params_to_params, params_to_fitted_params, decay_param_idxs = build_param_map_sigmoid(parameters)
        fixed_idx = None
    elif is_br:
        fitted_params_to_params, params_to_fitted_params, decay_param_idxs, fixed_idx = build_param_map_softmax(parameters)
    else:
        fitted_params_to_params = lambda x: x
        params_to_fitted_params = lambda x: x
        fixed_idx = None
        decay_param_idxs = None

    chi2, chi2_grad, chi2_val, chi2_open = build_chi2(
        y, mu, error_n, error_p, corr_mat_inv, fitted_params_to_params,
        translate_dep=translate_dep, adjust=adjust
    )

    chi2_val_and_grad = jax.jit(jax.value_and_grad(chi2))
    chi2_hessian = jax.hessian(chi2)

    param_init = jnp.array(
        [fit_seed_df[fit_seed_df['parameter_key'] == p]['seed'].iloc[0] for p in parameters],
        dtype=jnp.float64
    )

    if decay_param_idxs is not None:
        for particle in particles:
            bool_select = np.array([p.startswith(particle + '.') for p in parameters])
            decay_seed_sum = param_init[bool_select].sum()
            if decay_seed_sum >= 1:
                print(f'rescaling {particle} decay fit seed sum down to 1')
                param_init = param_init.at[bool_select].set(param_init[bool_select] / decay_seed_sum)
            param_init = param_init.at[bool_select].set(jnp.clip(param_init[bool_select], 1e-6, 1 - 1e-6))

    fitted_param_init = params_to_fitted_params(param_init)

    assert not jnp.any(jnp.isnan(fitted_param_init) | jnp.isinf(fitted_param_init))
    fitted_params_to_params(fitted_param_init)

    if verbose:
        print('performing fit...')

    fit_valid = None
    hesse_accurate = None

    if optimizer == 'scipy':
        def scipy_obj(x):
            val, grad = chi2_val_and_grad(jnp.array(x, dtype=jnp.float64))
            return float(val), np.array(grad, dtype=np.float64)

        x0 = np.array(fitted_param_init, dtype=np.float64)

        scipy_method = 'Newton-CG'
        scipy_bounds = None
        scipy_constraints = []

        if constrained and decay_param_idxs is not None:
            scipy_method = 'SLSQP'
            n = len(x0)
            lb = np.full(n, -np.inf)
            ub = np.full(n, np.inf)
            for i in decay_param_idxs:
                lb[i] = 0.0
                ub[i] = 1.0
            scipy_bounds = Bounds(lb, ub)
            if is_br:
                A = np.zeros((1, n))
                for i in decay_param_idxs:
                    A[0, int(i)] = 1.0
                scipy_constraints.append(LinearConstraint(A, 1.0, 1.0))

        result = scipy_minimize(scipy_obj, x0, method=scipy_method, jac=True,
                                bounds=scipy_bounds, constraints=scipy_constraints)
        if verbose:
            print(f'scipy success: {result.success}, message: {result.message}')
        chi2_min = float(result.fun)
        fitted_values = jnp.array(result.x, dtype=jnp.float64)
        param_values = fitted_params_to_params(fitted_values)
        hess = chi2_hessian(fitted_values)
        covariance = 2 * jnp.linalg.pinv(hess)
    else:
        m = Minuit(chi2_val, fitted_param_init, grad=chi2_grad)
        m.errordef = 1
        m.strategy = 0

        if constrained and decay_param_idxs is not None:
            for i in decay_param_idxs:
                m.limits[int(i)] = (0, 1)

        m.migrad()
        m.simplex()
        m.migrad()
        m.simplex()
        m.migrad()
        m.simplex()
        m.migrad()
        m.hesse()

        if fixed_idx is not None:
            m.fixed[fixed_idx] = True
            m.hesse()

        chi2_min = float(m.fval)
        fitted_values = jnp.array(m.values, dtype=jnp.float64)
        param_values = fitted_params_to_params(fitted_values)
        covariance = jnp.array(m.covariance, dtype=jnp.float64)
        fit_valid = m.valid
        hesse_accurate = m.accurate

        if verbose:
            print(f'fit valid: {m.valid}, errors from hessian accurate: {m.accurate}')

    return {
        'label': label,
        'algorithm': algorithm,
        'parameters': parameters,
        'nodes': nodes,
        'param_values': param_values,
        'fitted_values': fitted_values,
        'chi2_min': chi2_min,
        'chi2': chi2,
        'chi2_open': chi2_open,
        'chi2_grad': chi2_grad,
        'fitted_params_to_params': fitted_params_to_params,
        'params_to_fitted_params': params_to_fitted_params,
        'node_funcs': node_funcs,
        'parameter_funcs': parameter_funcs,
        'meas_df': meas_df,
        'rel_df': rel_df,
        'fit_df': fit_df,
        'mu': mu,
        'covariance': covariance,
        'fit_valid': fit_valid,
        'hesse_accurate': hesse_accurate,
    }
