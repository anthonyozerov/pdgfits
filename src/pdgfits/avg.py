import io
import contextlib
import os

import numpy as np
import pandas as pd
import jax
from scipy.optimize import minimize as scipy_minimize
import time

from jax import numpy as jnp
from iminuit import Minuit

from pdgfits.preprocess import preprocess
from pdgfits.build_funcs import get_mu_vectorized, get_translate_dep, get_adjust, get_node_funcs, get_parameter_funcs
from pdgfits.corr_mat import get_corr_mat
from pdgfits.build_chi2 import build_chi2
from pdgfits.asym_errors import binary_search_error
from pdgfits.plotting import plot_mnmatrix


def run_avg(node, meas_df_node, corr_df_node, skip_avg=False, contours=False, contours_dir=None):
    """
    Run a weighted average for a single node.

    Returns a dict with keys:
        node, status, parameters, nodes,
        param_values, chi2_min, n_meas, covariance,
        meas_df, rel_df, fit_df
    On failure: node, status, error.
    """
    print(node)
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


    mu = get_mu_vectorized(parameters, nodes_list, meas_df, fit_df, rel_df, jit=False)
    node_funcs = get_node_funcs(nodes_list, parameters, fit_df, rel_df, jit=False)
    parameter_funcs = get_parameter_funcs(parameters)
    translate_dep = get_translate_dep(dep_meas_data, parameters, nodes_list, parameter_funcs, node_funcs)
    adjust = get_adjust(adjust_data, parameters, nodes_list, parameter_funcs, node_funcs)

    # Use the mean of the main node's parsed measurements as the starting seed
    fit_seed_df.loc[fit_seed_df['parameter_key'] == node, 'seed'] = np.nan
    param_init = np.array(
        [fit_seed_df[fit_seed_df['parameter_key'] == p]['seed'].iloc[0] for p in parameters],
        dtype=np.float64,
    )
    seed_val = np.mean((meas_df['value']*adjust(param_init)+translate_dep(param_init))[meas_df['node'] == node])
    # seed_val = float(main_meas['value'].mean())
    fit_seed_df = fit_seed_df.copy()
    fit_seed_df.loc[fit_seed_df['parameter_key'] == node, 'seed'] = seed_val

    y = jnp.array(meas_df['value'], dtype=jnp.float64)
    error_n = jnp.array(meas_df['error_n'], dtype=jnp.float64)
    error_p = jnp.array(meas_df['error_p'], dtype=jnp.float64)

    corr_mat = get_corr_mat(meas_df, corr_df)
    corr_mat_inv = jnp.linalg.pinv(corr_mat)

    chi2, chi2_grad, chi2_val, _ = build_chi2(
        y, mu, error_n, error_p, corr_mat_inv,
        lambda x: x,
        translate_dep=translate_dep,
        adjust=adjust,
        use_jit=True,
    )
    chi2_val_and_grad = jax.value_and_grad(chi2)

    param_init = np.array(
        [fit_seed_df[fit_seed_df['parameter_key'] == p]['seed'].iloc[0] for p in parameters],
        dtype=np.float64,
    )

    # time the below step
    # t0 = time.perf_counter()
    # chi2_hess = jax.hessian(chi2)
    # hess = lambda x: np.array(chi2_hess(jnp.array(x, dtype=jnp.float64)), dtype=np.float64)
    # hess(param_init)
    # t1 = time.perf_counter()
    # print(f"{1000*(t1-t0):.1f} ms")

    def scipy_obj(x):
        return chi2_val(jnp.array(x, dtype=jnp.float64))
        # val, grad = chi2_val_and_grad(jnp.array(x, dtype=jnp.float64))
        # return float(val), np.array(grad, dtype=np.float64)
    assert ~np.isnan(scipy_obj(param_init)), f"Initial chi2 is NaN for node {node} with seed {param_init}, measurements {y}, errors {error_n}, {error_p}"

    if skip_avg:
        return None
    t0 = time.perf_counter()
    result = scipy_minimize(fun=scipy_obj, x0=param_init, method='Nelder-Mead', options={'maxiter': 1000, 'fatol': 1e-4, 'xatol': np.inf})
    t1 = time.perf_counter()
    # result = scipy_minimize(fun=scipy_obj, x0=param_init, jac=True, method='trust-ncg', hess=hess)
    if not result.success:
        if len(parameters) == 1:
            # plot the chi2 landscape for debugging
            xspace = np.linspace(min(y), max(y), 100)
            chi2_space = [chi2_val(jnp.array([x], dtype=jnp.float64)) for x in xspace]
            import matplotlib.pyplot as plt
            plt.plot(xspace, chi2_space)
            plt.scatter([result.x], [result.fun], color='red')
            plt.show()
    assert result.success, f"Chi2 minimization failed for node {node}: \n{result}"
    chi2_min = float(result.fun)
    param_values = np.array(result.x, dtype=np.float64)

    primary_idx = parameters.index(node)
    val = float(param_values[primary_idx])

    # Profile chi2 over nuisance parameters at a fixed value of the primary node.
    # For the common 1-parameter case this is just a direct evaluation.
    def profile_chi2(x_primary):
        if len(parameters) == 1:
            return chi2_val(jnp.array([x_primary], dtype=jnp.float64))
        else:
            x0_nuisance = np.delete(param_values, primary_idx)
            def obj(x_nuisance):
                x = np.insert(x_nuisance, primary_idx, x_primary)
                return chi2_val(jnp.array(x, dtype=jnp.float64))
            def grad(x_nuisance):
                x = np.insert(x_nuisance, primary_idx, x_primary)
                return np.delete(chi2_grad(jnp.array(x, dtype=jnp.float64)), primary_idx)
            # result = scipy_minimize(fun=obj, x0=x0_nuisance, jac=grad)
            result = scipy_minimize(fun=obj, x0=x0_nuisance, method='Nelder-Mead', options={'maxiter': 1000, 'fatol': 1e-4, 'xatol': np.inf})
            assert result.success, f"Profile chi2 minimization failed for node {node}. \n{result}"
            return float(result.fun)

    # Use heuristic measurement spans as the initial bracket for the binary search.
    ub = val + max(4*(adjust(param_values)*meas_df['error_p'])[meas_df['node'] == node])
    lb = val - max(4*(adjust(param_values)*meas_df['error_n'])[meas_df['node'] == node])
    err_scale = (ub - lb) / 2

    t2 = time.perf_counter()
    error_p_result, error_n_result = binary_search_error(
        profile_chi2, val, chi2_min, err_scale, lb, ub
    )
    t3 = time.perf_counter()


    if contours:
        t4 = time.perf_counter()
        m = Minuit(chi2_val, param_values, name=parameters, grad=chi2_grad)
        m.errordef = 1
        m.migrad()
        m.minos()

        t5 = time.perf_counter()
        print(f'minuit: {(t5-t4)*1000:.1f} ms')
        filepath = os.path.join(contours_dir, f'{node}.png') if contours_dir is not None else None
        plot_mnmatrix(m, filepath=filepath)

    print(f'optimization {(t1-t0)*1000:.1f} ms, error estimation {(t3-t2)*1000:.1f} ms')

    return {
        'node': node,
        'parameters': parameters,
        'nodes': nodes_list,
        'param_values': param_values,
        'chi2_min': chi2_min,
        'n_meas': int(len(meas_df_node)),
        'error_n': error_n_result,
        'error_p': error_p_result,
        'meas_df': meas_df,
        'rel_df': rel_df,
        'fit_df': fit_df,
    }
