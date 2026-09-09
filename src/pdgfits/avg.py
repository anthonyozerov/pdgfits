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
from pdgfits.build_funcs import get_mu_vectorized, get_translate_dep, get_adjust, get_node_funcs, get_parameter_funcs, get_mu_adjust
from pdgfits.corr_mat import get_corr_mat
from pdgfits.build_chi2 import build_chi2
from pdgfits.asym_errors import CallableProfileProblem, ProfilePoint, binary_search_error, symmetrize
from pdgfits.birge import block_birge


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


    mu = get_mu_vectorized(parameters, nodes_list, meas_df, fit_df, rel_df, jit=False)
    node_funcs = get_node_funcs(nodes_list, parameters, fit_df, rel_df, jit=False)
    parameter_funcs = get_parameter_funcs(parameters)
    translate_dep = get_translate_dep(dep_meas_data, parameters, nodes_list, parameter_funcs, node_funcs)
    adjust = get_adjust(adjust_data, parameters, nodes_list, parameter_funcs, node_funcs)
    mu_adjust = get_mu_adjust(mu, adjust, translate_dep)

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
    if corr_floor > 0.0:
        off_diag = 1.0 - jnp.eye(corr_mat.shape[0])
        corr_mat = jnp.maximum(corr_mat, corr_floor * off_diag)
    corr_mat_inv = jnp.linalg.pinv(corr_mat)

    chi2, chi2_grad, chi2_val, chi2_open = build_chi2(
        y, mu_adjust, error_n, error_p, corr_mat_inv,
        lambda x: x,
        use_jit=True,
    )
    chi2_val_and_grad = jax.jit(jax.value_and_grad(chi2))

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

    error = symmetrize(y, mu_adjust(param_values), error_n, error_p)
    # print(meas_df['node'])
    all_nodes = list(meas_df['node'].unique())
    node_idx = all_nodes.index(node)
    blocks = [np.where(meas_df['node'] == node)[0] for node in all_nodes]
    # print(blocks)
    chi2_b, exp_chi2_b, birge_b = block_birge(y, corr_mat_inv, error, param_values, chi2, mu_adjust, jax.hessian(chi2), blocks)
    print(birge_b[node_idx])

    # Use heuristic measurement spans as the initial bracket for the binary search.
    primary_mask = meas_df['node'] == node
    primary_error_p = np.asarray((adjust(param_values)*meas_df['error_p'])[primary_mask], dtype=np.float64)
    primary_error_n = np.asarray((adjust(param_values)*meas_df['error_n'])[primary_mask], dtype=np.float64)
    ub = val + 4 * float(np.max(primary_error_p))
    lb = val - 4 * float(np.max(primary_error_n))

    # Averages target a direct primary parameter. Fix that coordinate explicitly
    # and minimize over nuisance coordinates; this avoids equality-constraint
    # scaling and path-dependence for tiny lifetime/branching-ratio averages.
    profile_diagnostics = []
    base_nuisance = np.delete(param_values, primary_idx)
    last_nuisance = base_nuisance.copy()

    def profile_chi2(x_primary):
        nonlocal last_nuisance
        x_primary = float(x_primary)
        if len(parameters) == 1:
            x = np.array([x_primary], dtype=np.float64)
            prof_chi2 = float(chi2_val(jnp.array(x, dtype=jnp.float64)))
            point = ProfilePoint(x_primary, prof_chi2, True, 0.0, "direct")
            profile_diagnostics.append(point)
            profile_chi2.last_point = point
            return prof_chi2

        def full_params(x_nuisance):
            return np.insert(np.asarray(x_nuisance, dtype=np.float64), primary_idx, x_primary)

        def obj(x_nuisance):
            return float(chi2_val(jnp.array(full_params(x_nuisance), dtype=jnp.float64)))

        def obj_and_grad(x_nuisance):
            val_jax, grad_jax = chi2_val_and_grad(jnp.array(full_params(x_nuisance), dtype=jnp.float64))
            return float(val_jax), np.delete(np.asarray(grad_jax, dtype=np.float64), primary_idx)

        def grad(x_nuisance):
            return obj_and_grad(x_nuisance)[1]

        starts = [last_nuisance]
        if np.linalg.norm(base_nuisance - last_nuisance) > 1e-12 * max(np.linalg.norm(base_nuisance), np.linalg.norm(last_nuisance), 1.0):
            starts.append(base_nuisance)

        candidates = []
        messages = []
        for start in starts:
            start = np.asarray(start, dtype=np.float64)
            start_fun = obj(start)
            if np.isfinite(start_fun):
                candidates.append((start_fun, start, True, "feasible start"))
            run_nelder_mead = True
            try:
                bfgs = scipy_minimize(
                    fun=obj_and_grad,
                    x0=start,
                    jac=True,
                    method='BFGS',
                    options={'maxiter': max(1000, 500 * len(start)), 'gtol': 1e-8},
                )
                messages.append(f"BFGS success={bfgs.success}: {bfgs.message}")
                if np.isfinite(bfgs.fun):
                    bfgs_x = np.asarray(bfgs.x, dtype=np.float64)
                    candidates.append((float(bfgs.fun), bfgs_x, bool(bfgs.success), str(bfgs.message)))
                    if bfgs.success:
                        run_nelder_mead = False
                    else:
                        bfgs_grad_norm = float(np.linalg.norm(grad(bfgs_x)))
                        if bfgs_grad_norm <= 1e-5 * max(abs(float(bfgs.fun)), 1.0):
                            run_nelder_mead = False
            except Exception as exc:  # noqa: BLE001 - fallback below records failure
                messages.append(f"BFGS exception: {type(exc).__name__}: {exc}")

            if run_nelder_mead:
                nm_start = candidates[-1][1] if candidates else start
                xatol = max(np.linalg.norm(nm_start) * 1e-10, np.finfo(float).eps)
                try:
                    nm = scipy_minimize(
                        fun=obj,
                        x0=nm_start,
                        method='Nelder-Mead',
                        options={
                            'maxiter': max(1000, 500 * len(start)),
                            'xatol': xatol,
                            'fatol': 1e-8,
                            'adaptive': True,
                        },
                    )
                    messages.append(f"Nelder-Mead success={nm.success}: {nm.message}")
                    if np.isfinite(nm.fun):
                        candidates.append((float(nm.fun), np.asarray(nm.x, dtype=np.float64), bool(nm.success), str(nm.message)))
                except Exception as exc:  # noqa: BLE001 - handled by candidate check
                    messages.append(f"Nelder-Mead exception: {type(exc).__name__}: {exc}")

        if not candidates:
            raise RuntimeError(f"Profile chi2 minimization failed for node {node}: {'; '.join(messages)}")

        min_fun = min(c[0] for c in candidates)
        fun_tol = max(1e-6, 1e-8 * abs(min_fun))
        successful_ties = [c for c in candidates if c[2] and c[0] <= min_fun + fun_tol]
        if successful_ties:
            best_fun, best_x, best_success, best_message = min(successful_ties, key=lambda c: c[0])
        else:
            best_fun, best_x, best_success, best_message = min(candidates, key=lambda c: c[0])
        nuisance_grad_norm = float(np.linalg.norm(grad(best_x)))
        ok = bool(best_success or nuisance_grad_norm <= 1e-5 * max(abs(best_fun), 1.0))
        point = ProfilePoint(
            x_primary,
            best_fun,
            ok,
            0.0,
            f"{best_message}; nuisance_grad_norm={nuisance_grad_norm:.3g}",
        )
        profile_diagnostics.append(point)
        profile_chi2.last_point = point
        if not ok:
            raise RuntimeError(
                f"Profile chi2 minimization failed for node {node}: "
                f"best_fun={best_fun}, nuisance_grad_norm={nuisance_grad_norm}, messages={'; '.join(messages)}"
            )
        last_nuisance = best_x
        return best_fun

    profile_chi2.diagnostics = profile_diagnostics
    profile_chi2.last_point = None
    # The MLE point is feasible for the fixed-primary profile at val. Endpoint
    # verification below exercises the profile solver where it matters.
    base_chi2 = float(chi2_val(jnp.array(param_values, dtype=jnp.float64)))
    if base_chi2 > chi2_min + 1e-5:
        raise RuntimeError(
            f"Chi2 at fitted optimum for node {node} is {base_chi2}, "
            f"above chi2_min {chi2_min}"
        )

    # err_scale = (ub - lb) / 2

    t2 = time.perf_counter()
    profile_problem = CallableProfileProblem(
        profile_chi2=profile_chi2,
        target_name=node,
        target_value=val,
        chi2_min=chi2_min,
        lower_initial=lb,
        upper_initial=ub,
    )
    error_p_result, error_n_result = binary_search_error(
        profile_problem, val, chi2_min, lb, ub, verbose=False
    )
    t3 = time.perf_counter()

    print(birge_b[node_idx]*error_n_result, birge_b[node_idx]*error_p_result)


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
        'n_meas': int(len(meas_df_node)),
        'error_n': error_n_result,
        'error_p': error_p_result,
        'asym_error_diagnostics': binary_search_error.last_diagnostics,
        'profile_diagnostics': profile_chi2.diagnostics,
        'chi2': chi2,
        'chi2_open': chi2_open,
        'chi2_grad': chi2_grad,
        'fitted_params_to_params': lambda x: x,
        'params_to_fitted_params': lambda x: x,
        'node_funcs': node_funcs,
        'parameter_funcs': parameter_funcs,
        'mu': mu,
        'meas_df': meas_df,
        'rel_df': rel_df,
        'fit_df': fit_df,
        'covariance': None,
        'fit_valid': None,
        'hesse_accurate': None,
    }
