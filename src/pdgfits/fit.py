"""Prepare and minimize a joint fit in physical or unconstrained coordinates."""

import numpy as np
import jax
from jax import numpy as jnp
import warnings
from iminuit import Minuit
from scipy.optimize import minimize as scipy_minimize, Bounds, LinearConstraint, OptimizeResult

from pdgfits.query import fit_queries
from pdgfits.preprocess import preprocess
from pdgfits.build_funcs import get_node_funcs, get_parameter_funcs, get_mu_vectorized, get_translate_dep, get_adjust, get_mu_adjust
from pdgfits.func_factory import ALLOWED_EQUATION_TYPES
from pdgfits.corr_mat import get_corr_mat
from pdgfits.build_chi2 import build_chi2
from pdgfits.param_maps import (
    build_param_map_sigmoid,
    build_param_map_softmax,
    build_param_map_scaled,
    get_decay_info,
)


def _condition_seed_units(parameters, fit_df, rel_df, meas_df, param_init, skip_idxs=None):
    """Put seeds with obvious display-unit mismatches into parsed-measurement units."""
    skip_idxs = set() if skip_idxs is None else set(int(i) for i in skip_idxs)
    param_init = jnp.array(param_init, dtype=jnp.float64)
    param_idx = {p: i for i, p in enumerate(parameters)}

    for idx, param in enumerate(parameters):
        if idx in skip_idxs:
            continue

        direct = np.asarray(meas_df.loc[meas_df['node'] == param, 'value'], dtype=np.float64)
        if len(direct) == 0:
            continue

        seed = float(param_init[idx])
        target_scale = float(np.nanmedian(np.abs(direct)))
        if not np.isfinite(seed) or not np.isfinite(target_scale) or target_scale == 0:
            continue

        seed_abs = abs(seed)
        if seed_abs == 0:
            if target_scale >= 1e4:
                param_init = param_init.at[idx].set(float(np.nanmedian(direct)))
            continue

        ratio = target_scale / seed_abs
        if ratio >= 1e4 or ratio <= 1e-4:
            param_init = param_init.at[idx].set(
                seed * 10.0 ** np.round(np.log10(ratio))
            )

    for lifetime_node in fit_df.loc[fit_df['type'] == 'lifetime', 'node']:
        width_params = rel_df.loc[
            rel_df['node'] == lifetime_node, 'parameter_key'
        ].dropna().unique()
        if len(width_params) != 1 or width_params[0] not in param_idx:
            continue

        direct = meas_df.loc[meas_df['node'] == lifetime_node, 'value']
        if len(direct) == 0:
            continue

        lifetime_scale = float(np.nanmedian(np.abs(np.asarray(direct, dtype=np.float64))))
        if not np.isfinite(lifetime_scale) or lifetime_scale == 0:
            continue

        idx = param_idx[width_params[0]]
        seed = float(param_init[idx])
        target_seed = 1.0 / lifetime_scale
        if not np.isfinite(seed) or not np.isfinite(target_seed):
            continue

        seed_abs = abs(seed)
        if seed_abs == 0 or target_seed / seed_abs >= 1e4 or target_seed / seed_abs <= 1e-4:
            param_init = param_init.at[idx].set(target_seed)

    return param_init


def _large_coordinate_scale_info(parameters, param_init, skip_idxs=None):
    """Return linear optimizer scales for large non-decay coordinates."""
    skip_idxs = set() if skip_idxs is None else set(int(i) for i in skip_idxs)
    scale_idxs = []
    scales = []

    for idx, seed in enumerate(np.asarray(param_init, dtype=np.float64)):
        if idx in skip_idxs or not np.isfinite(seed):
            continue
        scale = abs(seed)
        if scale > 1e4:
            scale_idxs.append(idx)
            scales.append(scale)

    return np.array(scale_idxs, dtype=int), np.array(scales, dtype=np.float64)


def _finite_scipy_result(result):
    return (result is not None and np.isfinite(getattr(result, 'fun', np.inf))
            and np.all(np.isfinite(result.x)))


def _best_attempt(best, candidate):
    """Prefer a lower objective, or a successful numerically tied attempt."""
    if _finite_scipy_result(candidate):
        tied = candidate.fun <= best.fun + max(1e-8, 1e-10 * abs(best.fun))
        if candidate.fun < best.fun or (tied and candidate.success and not best.success):
            return candidate
    return best


def _scipy_attempt(objective, start, method, **kwargs):
    try:
        return scipy_minimize(objective, start, method=method, **kwargs)
    except (FloatingPointError, ValueError, np.linalg.LinAlgError) as exc:
        return OptimizeResult(x=np.asarray(start), fun=np.inf, success=False,
                              message=f'{method} failed: {exc}')


def _run_unconstrained_scipy(scipy_obj, scipy_value, scipy_hessp, x0):
    """Trust region, then BFGS and a simplex escape only when needed."""
    start_fun = scipy_value(x0)
    best = OptimizeResult(x=np.array(x0), fun=start_fun, success=False,
                          message='initial point (all scipy attempts failed)')

    def trust(start):
        return _scipy_attempt(scipy_obj, start, 'trust-ncg', jac=True,
                              hessp=scipy_hessp, options={'maxiter': 1000})

    first = trust(x0)
    best = _best_attempt(best, first)
    stalled = not _finite_scipy_result(first) or not first.success or first.fun >= start_fun - 1e-8
    if stalled:
        bfgs = scipy_minimize(scipy_obj, best.x if np.isfinite(best.fun) else x0,
                              method='BFGS', jac=True,
                              options={'maxiter': max(1000, 1000 * len(x0))})
        best = _best_attempt(best, bfgs)
        if _finite_scipy_result(bfgs):
            best = _best_attempt(best, trust(bfgs.x))
    if stalled and not best.success:
        simplex = scipy_minimize(scipy_value, best.x if np.isfinite(best.fun) else x0,
                                 method='Nelder-Mead', options={
                                     'adaptive': True, 'maxiter': max(1000, 100 * len(x0)),
                                     'xatol': 1e-5, 'fatol': 1e-5})
        best = _best_attempt(best, simplex)
        if _finite_scipy_result(simplex):
            best = _best_attempt(best, trust(simplex.x))
    return best


def _run_constrained_scipy(scipy_obj, scipy_value, scipy_hessp, x0, bounds, constraints):
    """Compare SLSQP and exact-Hessian trust region from the supplied seed."""
    x0 = np.asarray(x0, dtype=np.float64)
    best = OptimizeResult(x=x0.copy(), fun=np.inf, success=False,
                          message='no finite constrained scipy start')
    if not np.isfinite(x0).all():
        return best
    best.fun = scipy_value(x0)
    best.message = 'initial point (all constrained scipy attempts failed)'

    def trust(start):
        return _scipy_attempt(scipy_obj, start, 'trust-constr', jac=True,
                              hessp=scipy_hessp, bounds=bounds, constraints=constraints,
                              options={'maxiter': 1000})

    slsqp = _scipy_attempt(scipy_obj, x0, 'SLSQP', jac=True,
                           bounds=bounds, constraints=constraints,
                           options={'maxiter': 1000, 'ftol': 1e-10})
    best = _best_attempt(best, slsqp)
    best = _best_attempt(best, trust(x0))
    if _finite_scipy_result(slsqp) and not slsqp.success:
        best = _best_attempt(best, trust(slsqp.x))
    if not best.success:
        best = _best_attempt(best, trust(best.x))
    return best


def run_fit(label, verbose=True, optimizer='minuit', fit_space='unconstrained'):
    """
    Run a single fit by label.

    Returns a dict with fit results, or None if the fit was skipped.

    Keys (same set as run_avg(); keys not applicable to fits are None):
        node, label, algorithm, parameters, nodes,
        param_values, fitted_values, chi2_min,
        n_meas, error_n, error_p,
        chi2, chi2_open, chi2_grad,
        fitted_params_to_params, params_to_fitted_params,
        node_funcs, parameter_funcs, mu,
        meas_df, rel_df, fit_df,
        covariance, fit_valid, hesse_accurate
    """
    if optimizer not in ('minuit', 'scipy'):
        raise ValueError(f'Unknown optimizer: {optimizer}')
    if fit_space not in ('unconstrained', 'constrained'):
        raise ValueError(f'Unknown fit space: {fit_space}')
    algorithm, measurement_type, fit_df, rel_df, meas_df, corr_df, fit_seed_df, tree_df = fit_queries(label, verbose=False)

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
    mu = get_mu_vectorized(parameters, nodes, meas_df, fit_df, rel_df)

    if verbose and any(d is not None for d in dep_meas_data):
        print('Dependent measurements found, these will be accounted for.')
    translate_dep = get_translate_dep(dep_meas_data, parameters, nodes, parameter_funcs, node_funcs)
    adjust = get_adjust(adjust_data, parameters, nodes, parameter_funcs, node_funcs)
    mu_adjust = get_mu_adjust(mu, adjust, translate_dep)

    y = jnp.array(meas_df['value'], dtype=jnp.float64)
    error_n = jnp.array(meas_df['error_n'], dtype=jnp.float64)
    error_p = jnp.array(meas_df['error_p'], dtype=jnp.float64)

    corr_mat, blocks = get_corr_mat(meas_df, corr_df, return_blocks=True)
    if verbose and not jnp.all(jnp.linalg.eigvals(corr_mat) >= 0):
        warnings.warn('Measurement correlation matrix is not positive semidefinite.')
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

    param_init = jnp.array(
        [fit_seed_df[fit_seed_df['parameter_key'] == p]['seed'].iloc[0] for p in parameters],
        dtype=jnp.float64
    )

    if decay_param_idxs is not None:
        # BR fractions sum to one; normalize their seeds as in sbrfit.f.
        # BRU fractions need not sum to one. Convert percent seeds instead,
        # then stay inside the open interval required by the parameter map.
        for particle in particles:
            bool_select = np.array([p.startswith(particle + '.') for p in parameters])
            if not bool_select.any():
                continue
            decay_seeds = param_init[bool_select]
            if is_br:
                decay_seed_sum = decay_seeds.sum()
                if decay_seed_sum > 0:
                    print(f'normalizing {particle} decay fit seeds to sum to 1')
                    param_init = param_init.at[bool_select].set(decay_seeds / decay_seed_sum)
            elif decay_seeds.max() > 1:
                print(f'converting {particle} decay fit seeds from percent to proportion')
                decay_seeds = decay_seeds / 100
                param_init = param_init.at[bool_select].set(decay_seeds)
            param_init = param_init.at[bool_select].set(
                jnp.clip(param_init[bool_select], 1e-6, 1 - 1e-6)
            )

    param_init = _condition_seed_units(
        parameters, fit_df, rel_df, meas_df, param_init, skip_idxs=decay_param_idxs
    )

    scale_idxs, scales = _large_coordinate_scale_info(
        parameters, param_init, skip_idxs=decay_param_idxs
    )

    if len(scale_idxs) > 0:
        fitted_params_to_params, params_to_fitted_params = build_param_map_scaled(
            fitted_params_to_params, params_to_fitted_params, scale_idxs, scales
        )
        if verbose:
            scaled = ', '.join(
                f'{parameters[i]} / {scale:.3g}'
                for i, scale in zip(scale_idxs, scales)
            )
            print(f'scaling fitted coordinates: {scaled}')

    chi2, chi2_grad, chi2_val, chi2_open = build_chi2(
        y, mu_adjust, error_n, error_p, corr_mat_inv, fitted_params_to_params,
    )

    fitted_param_init = params_to_fitted_params(param_init)

    assert not jnp.any(jnp.isnan(fitted_param_init) | jnp.isinf(fitted_param_init))
    fitted_params_to_params(fitted_param_init)

    if verbose:
        print('performing fit...')

    fit_valid = None
    hesse_accurate = None

    if optimizer == 'scipy':
        chi2_val_and_grad = jax.jit(jax.value_and_grad(chi2))
        chi2_hessp = jax.jit(lambda x, p: jax.jvp(jax.grad(chi2), (x,), (p,))[1])

        def scipy_obj(x):
            val, grad = chi2_val_and_grad(jnp.array(x, dtype=jnp.float64))
            return float(val), np.array(grad, dtype=np.float64)

        def scipy_value(x):
            return float(chi2(jnp.array(x, dtype=jnp.float64)))

        def scipy_hessp(x, p):
            hvp = np.array(
                chi2_hessp(
                    jnp.array(x, dtype=jnp.float64),
                    jnp.array(p, dtype=jnp.float64),
                ),
                dtype=np.float64,
            )
            if not np.all(np.isfinite(hvp)):
                raise FloatingPointError('non-finite Hessian-vector product')
            return hvp

        scipy_bounds = None
        scipy_constraints = []

        if constrained and decay_param_idxs is not None:
            n = len(fitted_param_init)
            lb = np.full(n, -np.inf)
            ub = np.full(n, np.inf)
            lb[decay_param_idxs] = 0.0
            ub[decay_param_idxs] = 1.0
            scipy_bounds = Bounds(lb, ub)
            if is_br:
                A = np.zeros((1, n))
                A[0, decay_param_idxs] = 1.0
                scipy_constraints.append(LinearConstraint(A, 1.0, 1.0))

        x0 = np.array(fitted_param_init, dtype=np.float64)
        if scipy_bounds is None:
            result = _run_unconstrained_scipy(scipy_obj, scipy_value, scipy_hessp, x0)
        else:
            result = _run_constrained_scipy(
                scipy_obj,
                scipy_value,
                scipy_hessp,
                x0,
                scipy_bounds,
                scipy_constraints,
            )
        if verbose:
            print(f'scipy success: {result.success}, message: {result.message}')
        chi2_min = float(result.fun)
        fitted_values = jnp.array(result.x, dtype=jnp.float64)
        param_values = fitted_params_to_params(fitted_values)
        hess = jax.hessian(chi2)(fitted_values)
        covariance = 2 * jnp.linalg.pinv(hess)
    else:
        m = Minuit(chi2_val, fitted_param_init, grad=chi2_grad)
        m.errordef = 1
        m.strategy = 0
        if constrained and decay_param_idxs is not None:
            for index in decay_param_idxs:
                m.limits[int(index)] = (0, 1)
        # Preserve the validated escape/recovery sequence before estimating curvature.
        for _ in range(3):
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
        'node': None,
        'label': label,
        'algorithm': algorithm,
        'parameters': parameters,
        'nodes': nodes,
        'param_values': param_values,
        'fitted_values': fitted_values,
        'chi2_min': chi2_min,
        'n_meas': None,
        'error_n': None,
        'error_p': None,
        'chi2': chi2,
        'chi2_open': chi2_open,
        'chi2_grad': chi2_grad,
        'fitted_params_to_params': fitted_params_to_params,
        'params_to_fitted_params': params_to_fitted_params,
        'node_funcs': node_funcs,
        'parameter_funcs': parameter_funcs,
        'mu': mu,
        'mu_adjust': mu_adjust,
        'corr_mat': corr_mat,
        'correlation_blocks': blocks,
        'meas_df': meas_df,
        'rel_df': rel_df,
        'fit_df': fit_df,
        'covariance': covariance,
        'fit_valid': fit_valid,
        'hesse_accurate': hesse_accurate,
    }
