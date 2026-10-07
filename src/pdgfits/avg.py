"""Average one node, profiling auxiliary inputs with the common fit objective."""

import os
import time

import numpy as np
import pandas as pd

from jax import numpy as jnp
from iminuit import Minuit

from pdgfits.preprocess import preprocess
from pdgfits.build_funcs import get_mu_vectorized, get_translate_dep, get_adjust, get_node_funcs, get_parameter_funcs, get_mu_adjust
from pdgfits.corr_mat import get_corr_mat, correlation_blocks
from pdgfits.build_chi2 import build_chi2, build_product_chi2
from pdgfits.asym_errors import build_coordinate_profile_chi2, find_profile_root
from pdgfits.scalar_average import scalar_average
from pdgfits.refit import prepare_refit


def run_avg(node, meas_df_node, corr_df_node, skip_avg=False, contours=False, contours_dir=None, corr_floor=0.0):
    """
    Run a weighted average for a single node.

    Returns a dict with the same keys as run_fit(); keys not applicable to
    averages (label, algorithm, covariance, fit_valid, hesse_accurate) are None.
    """
    print(node + '-'*20)
    meas_df_node = meas_df_node.copy()
    for col in ('systematic_error_clump', 'systematic_error_clump2'):
        if col in meas_df_node.columns:
            meas_df_node[col] = None

    fit_df = pd.DataFrame({'node': [node], 'type': ['+'], 'data_type': [None]})
    rel_df = pd.DataFrame({
        'node': [node], 'par_code': [None], 'parameter': [node],
        'coefficient': [1.0], 'summation': [1],
        'coeff_par_code': [None], 'coeff_parameter': [None],
    })
    fit_seed_df = pd.DataFrame({'par_code': [None], 'parameter': [node], 'seed': [0.0]})
    tree_df = pd.DataFrame({'node': pd.Series([], dtype=str), 'data_type': pd.Series([], dtype=str)})

    fit_df, rel_df, meas_df, corr_df, fit_seed_df, dep_meas_data, adjust_data = preprocess(
        fit_df, rel_df, meas_df_node, corr_df_node.copy(), fit_seed_df, tree_df, 'AVG', ''
    )
    if len(meas_df[meas_df['node']==node]) <= 1:
        print(f'Not enough measurements for node {node}, skipping.')
        return None

    parameters = list(fit_seed_df['parameter_key'].unique())
    nodes_list = list(rel_df['node'].unique())
    direct = (len(parameters) == 1 and all(d is None for d in dep_meas_data)
              and all(d is None for d in adjust_data))

    mu = get_mu_vectorized(parameters, nodes_list, meas_df, fit_df, rel_df, jit=False)
    node_funcs = get_node_funcs(nodes_list, parameters, fit_df, rel_df, jit=False)
    parameter_funcs = get_parameter_funcs(parameters)
    translate_dep = get_translate_dep(dep_meas_data, parameters, nodes_list, parameter_funcs, node_funcs)
    adjust = get_adjust(adjust_data, parameters, nodes_list, parameter_funcs, node_funcs)
    mu_adjust = get_mu_adjust(mu, adjust, translate_dep)

    # Initialize the primary value from corrected measurements; keep the
    # auxiliary seeds supplied by preprocessing. Build the parameter vector once.
    seeds = fit_seed_df.drop_duplicates('parameter_key').set_index('parameter_key')['seed']
    param_init = seeds.loc[parameters].to_numpy(dtype=np.float64, copy=True)
    primary_idx = parameters.index(node)
    primary_mask = meas_df['node'] == node
    param_init[primary_idx] = np.nan
    if direct:
        param_init[primary_idx] = float(meas_df['value'].mean())
    else:
        corrected = meas_df['value'] * adjust(param_init) + translate_dep(param_init)
        param_init[primary_idx] = np.mean(corrected[primary_mask])

    y = jnp.array(meas_df['value'], dtype=jnp.float64)
    error_n = jnp.array(meas_df['error_n'], dtype=jnp.float64)
    error_p = jnp.array(meas_df['error_p'], dtype=jnp.float64)

    corr_mat, blocks = get_corr_mat(meas_df, corr_df, return_blocks=True)
    if corr_floor > 0.0:
        off_diag = 1.0 - jnp.eye(corr_mat.shape[0])
        corr_mat = jnp.maximum(corr_mat, corr_floor * off_diag)
        blocks = correlation_blocks(np.asarray(corr_mat))
    corr_mat_inv = jnp.asarray(np.linalg.pinv(np.asarray(corr_mat)))

    if all(d is None for d in dep_meas_data):
        powers = np.zeros((len(meas_df), len(parameters)), int)
        for i, (measured, adjustment) in enumerate(zip(meas_df['node'], adjust_data)):
            powers[i, parameters.index(measured)] = 1
            if adjustment is not None:
                for parameter, operator in zip(adjustment[0], adjustment[1]):
                    if operator not in ('/', '*'):
                        raise ValueError(f'Unknown average adjustment: {operator}')
                    key = parameter if parameter in parameters else 'nuisance_'+parameter
                    powers[i, parameters.index(key)] += -1 if operator == '/' else 1
        chi2, chi2_grad, chi2_val, chi2_open = build_product_chi2(y, error_n, error_p, corr_mat_inv, powers)
    else:
        chi2, chi2_grad, chi2_val, chi2_open = build_chi2(
            y, mu_adjust, error_n, error_p, corr_mat_inv, lambda x: x, use_jit=not direct)

    if not direct and not np.isfinite(chi2_val(param_init)):
        raise ValueError(f"Non-finite initial chi2 for {node}")

    if skip_avg:
        return None
    t0 = time.perf_counter()
    if direct:
        scalar = scalar_average(y, error_n, error_p, corr_mat_inv)
        param_values, chi2_min = np.array([scalar['value']]), scalar['q']
    else:
        prepared = {'fitted_values': param_init, 'covariance': None,
                    'chi2_open': chi2_open, 'meas_df': meas_df,
                    'corr_mat': corr_mat, 'correlation_blocks': blocks,
                    'mu_adjust': mu_adjust, 'fitted_params_to_params': lambda p: p}
        optimized = prepare_refit(prepared)(full_output=False)
        param_values = np.asarray(optimized['fitted_values'])
        chi2_min = optimized['chi2_min']
    t1 = time.perf_counter()

    val = float(param_values[primary_idx])

    t2 = time.perf_counter()
    if direct:
        error_p_result, error_n_result = scalar['error_p'], scalar['error_n']
        diagnostics = scalar['diagnostics']
        profile_diagnostics = []
    else:
        # Start with spans of the corrected primary measurements, then profile
        # auxiliary inputs at each trial value and check Delta Q = 1 endpoints.
        adjustment = np.asarray(adjust(param_values))
        primary_error_p = np.asarray((adjustment * meas_df['error_p'])[primary_mask], dtype=np.float64)
        primary_error_n = np.asarray((adjustment * meas_df['error_n'])[primary_mask], dtype=np.float64)
        lower = val - 4 * float(np.max(primary_error_n))
        upper = val + 4 * float(np.max(primary_error_p))
        profile_chi2 = build_coordinate_profile_chi2(
            chi2, param_values, primary_idx, node, covariance=optimized['preconditioner'],
        )
        base_chi2 = float(chi2_val(jnp.array(param_values, dtype=jnp.float64)))
        if base_chi2 > chi2_min + 1e-5:
            raise RuntimeError(f'Chi2 at fitted optimum for {node} is {base_chi2}, above {chi2_min}')
        root = find_profile_root(profile_chi2, val, chi2_min, lower, upper)
        error_p_result, error_n_result = root.upper.error, root.lower.error
        diagnostics = root.diagnostics()
        profile_diagnostics = profile_chi2.diagnostics
    t3 = time.perf_counter()

    print(f"{node}: {val:.8g} +{error_p_result:.6g} -{error_n_result:.6g}")


    if contours:
        t4 = time.perf_counter()
        m = Minuit(chi2_val, param_values, name=parameters, grad=chi2_grad)
        m.errordef = 1
        m.migrad()
        m.minos()

        t5 = time.perf_counter()
        print(f'minuit: {(t5-t4)*1000:.1f} ms')
        filepath = os.path.join(contours_dir, f'{node}.png') if contours_dir is not None else None
        from pdgfits.plotting import plot_mnmatrix
        plot_mnmatrix(m, filepath=filepath)

    print(f'optimization {(t1-t0)*1000:.1f} ms, error estimation {(t3-t2)*1000:.1f} ms')

    return {
        'node': node,
        'label': None,
        'algorithm': None,
        'parameters': parameters,
        'nodes': nodes_list,
        'param_values': param_values,
        'fitted_values': param_values,  # identity param map, so same as param_values
        'chi2_min': chi2_min,
        'n_meas': int(len(meas_df)),
        'n_primary_meas': int((meas_df['node'] == node).sum()),
        'ndof': int(len(meas_df) - len(parameters)),
        'error_n': error_n_result,
        'error_p': error_p_result,
        'asym_error_diagnostics': diagnostics,
        'profile_diagnostics': profile_diagnostics,
        'chi2': chi2,
        'chi2_open': chi2_open,
        'chi2_grad': chi2_grad,
        'fitted_params_to_params': lambda x: x,
        'params_to_fitted_params': lambda x: x,
        'node_funcs': node_funcs,
        'parameter_funcs': parameter_funcs,
        'mu': mu,
        'mu_adjust': mu_adjust,
        'corr_mat': corr_mat,
        'correlation_blocks': blocks,
        'meas_df': meas_df,
        'rel_df': rel_df,
        'fit_df': fit_df,
        'covariance': None,
        'fit_valid': None,
        'hesse_accurate': None,
    }
