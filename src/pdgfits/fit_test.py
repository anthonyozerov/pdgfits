import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
from jax import numpy as jnp
import jax.scipy.optimize
from iminuit import Minuit
import warnings
import argparse
from decimal import Decimal

from pdgfits.fit_query import all_fits, query_db, pdg_most_precise_value
from pdgfits.preprocess import preprocess
from pdgfits.build_funcs import get_node_funcs, get_parameter_funcs, get_meas_funcs, get_mu, get_translate_dep, get_adjust
from pdgfits.func_factory import ALLOWED_EQUATION_TYPES
from pdgfits.corr_mat import get_corr_mat
from pdgfits.build_chi2 import build_chi2, build_param_map_softmax, build_param_map_sigmoid, build_chi2_c

if __name__ == '__main__':

    # use argparse to get the start_from, fit_type, fit_label, and measurement_type
    parser = argparse.ArgumentParser()
    parser.add_argument('--start_from', type=str, default=None)
    parser.add_argument('--fit_type', type=str, default=None)
    parser.add_argument('--fit_label', type=str, default=None)
    parser.add_argument('--measurement_type', type=str, default=None)
    parser.add_argument('--compare_to_pdg', action='store_true', default=False)
    parser.add_argument('--calc_asym_errors', action='store_true', default=False)
    parser.add_argument('--meas_diagnostics', action='store_true', default=False)
    args = parser.parse_args()
    start_from = args.start_from
    fit_type = args.fit_type
    fit_label = args.fit_label
    measurement_type = args.measurement_type
    compare_to_pdg = args.compare_to_pdg
    calc_asym_errors = args.calc_asym_errors
    meas_diagnostics = args.meas_diagnostics

    fits_df = all_fits()
    fits_df_sub = fits_df[fits_df['algorithm'] != 'IGNORE']    

    if fit_type is not None:
        fits_df_sub = fits_df_sub[fits_df_sub['algorithm'] == fit_type]
    if fit_label is not None:
        fits_df_sub = fits_df_sub[fits_df_sub['label'] == fit_label]
    if measurement_type is not None:
        fits_df_sub = fits_df_sub[fits_df_sub['measurement_type'] == measurement_type]

    labels = list(fits_df_sub['label'])
    algorithms = list(fits_df_sub['algorithm'])
    chi2s = list(fits_df_sub['chi_square'])
    if start_from is not None:
        start_from_idx = labels.index(start_from)
    else:
        start_from_idx = 0


    # iterate through each fit in the filtered fits
    for i in range(start_from_idx, len(labels)):
        label = labels[i]
        algorithm = algorithms[i]
        chi2_pdg = chi2s[i]
        print('='*100)
        print(f'{label}: {algorithm}')
        if label == 'tauhflav':
            print('skipping tauhflav')
            continue
        _, measurement_type, fit_df, rel_df, meas_df, corr_df, fit_seed_df, tree_df = query_db(label, verbose=False)
        fit_df, rel_df, meas_df, corr_df, fit_seed_df, dep_meas_data, adjust_data = preprocess(fit_df, rel_df, meas_df, corr_df, fit_seed_df, tree_df, algorithm, measurement_type)

        # parameters are anything with a fit_seed
        parameters = list(fit_seed_df['parameter_key'].unique())
        # nodes are any nodes in the relationship table
        nodes = list(np.unique(rel_df['node']))
        # particles are the particles in the fit (excluding nuisance parameters)
        particles = list(np.unique([node[:4] for node in nodes+parameters if not node.startswith('nuisance_')]))

        print(f'{len(parameters)} parameters, {len(nodes)} nodes, {len(particles)} particles')
        print('Parameters:', ', '.join(parameters))
        # check that all measurements are of a node or parameter in the fit
        assert all(meas_df['node'].isin(nodes+parameters))

        # check that all nodes have an allowed equation type
        if any(~fit_df['type'].isin(ALLOWED_EQUATION_TYPES)):
            print(list(fit_df['type'][~fit_df['type'].isin(ALLOWED_EQUATION_TYPES)]))
            warnings.warn("Some nodes have an equation type which is not supported. Fit skipped.")
            continue
        
        # get the functions which map the param vector to node values
        node_funcs = get_node_funcs(nodes, parameters, fit_df, rel_df)
        # get the functions which map the param vector to parameter values
        # (these just select one coordinate of the param vector)
        parameter_funcs = get_parameter_funcs(parameters)
        # get the functions which map the param vector to measurement values
        # (these are just the corresponding node/parameter functions for each measurement)
        meas_funcs = get_meas_funcs(dict(zip(nodes, node_funcs)), dict(zip(parameters, parameter_funcs)), meas_df)
        # function which maps the param vector to a vector of measurement values
        mu = get_mu(meas_funcs)

        # 
        if any([d is not None for d in dep_meas_data]):
            print('Dependent measurements found, these will be accounted for.')
        translate_dep = get_translate_dep(dep_meas_data, parameters, nodes, parameter_funcs, node_funcs)
        adjust = get_adjust(adjust_data, parameters, nodes, parameter_funcs, node_funcs)

        # array of measured values
        y = jnp.array(meas_df['value'], dtype=jnp.float64)

        # arrays of negative and positive errors
        error_n = jnp.array(meas_df['error_n'], dtype=jnp.float64)
        error_p = jnp.array(meas_df['error_p'], dtype=jnp.float64)

        # turn the correlation dataframe into a matrix of correlations between all measurements
        corr_mat = get_corr_mat(meas_df, corr_df)
        # check if corr_mat is PSD
        if not jnp.all(jnp.linalg.eigvals(corr_mat) >= 0):
            print('!!!correlation matrix is not PSD!!!')
            print('!!!this is probably bad!!!')
        corr_mat_inv = jnp.linalg.pinv(corr_mat)

        # first case: decay parameters constrained to [0, 1]
        if algorithm == 'BRU' or (algorithm in ['BR', 'BR (NO MATRIX)', 'BR PRINT'] and len(particles) > 1):
            fitted_params_to_params, params_to_fitted_params, decay_param_idxs = build_param_map_sigmoid(parameters)
            fixed_idx = None
        # second case: decay parameters constrained to sum to 1
        elif algorithm in ['BR', 'BR (NO MATRIX)', 'BR PRINT'] and len(particles) == 1:
            # fitted_params_to_params, fitted_parameters, excluded_param_idxs = build_fitted_params_to_params(parameters, fit_seed_df)
            fitted_params_to_params, params_to_fitted_params, decay_param_idxs, fixed_idx = build_param_map_softmax(parameters)
        # third case: no constraints on any parameters
        else:
            fitted_params_to_params = lambda x: x
            params_to_fitted_params = lambda x: x
            fixed_idx = None
            decay_param_idxs = None

        # build the chi2 function (a function of fitted parameters)
        chi2, chi2_grad, chi2_val = build_chi2(y, mu, error_n, error_p, corr_mat_inv, fitted_params_to_params, translate_dep=translate_dep, adjust=adjust)

        # get initial values for the parameters from the fit_seed table
        param_init = jnp.array([fit_seed_df[fit_seed_df['parameter_key'] == param]['seed'].iloc[0] for param in parameters], dtype=jnp.float64)

        # print(fit_seed_df)
        # for any decay parameters, make the fit seed sum to <= 1, and clip to 1e-6, 1
        if decay_param_idxs is not None:
            for particle in particles:
                bool_select = np.array([p.startswith(particle+'.') for p in parameters])
                decay_param_idxs_particle = np.where(bool_select)[0]
                decay_seed_sum = param_init[bool_select].sum()
                if decay_seed_sum >= 1:
                    param_init = param_init.at[bool_select].set(param_init[bool_select] / decay_seed_sum)
                clip_val = 1e-6
                param_init = param_init.at[bool_select].set(jnp.clip(param_init[bool_select], clip_val, 1))

        # convert the init param vector to a fitted param vector
        fitted_param_init = params_to_fitted_params(param_init)

        # check that none of the fitted parameters are nan
        assert not jnp.any(jnp.isnan(fitted_param_init))
        # test that the fitted_params_to_params function is working
        fitted_params_to_params(fitted_param_init)

        # nodes_with_measurements = list(np.unique(meas_df['node']))
        # n_nodes_with_measurements = len(nodes_with_measurements)
        # meas_node_idxs = np.array([nodes_with_measurements.index(node) for node in list(meas_df['node'])])

        # @jax.jit
        # def c_map(c_params):
        #     return 1 + jnp.exp(c_params[meas_node_idxs])
        # @jax.jit
        # def c_map(c_params):
        #     return 1 + jnp.clip(c_params[meas_node_idxs], 0, 1e2)

        # chi2, chi2_grad, chi2_val = build_chi2_c(y, mu, error_n, error_p, corr_mat_inv, fitted_params_to_params, translate_dep=translate_dep, adjust=adjust, len_fitted_params=len(fitted_param_init), c_map=c_map)
        # c_init = -2*jnp.ones(n_nodes_with_measurements)
        # c_init = jnp.zeros(n_nodes_with_measurements)
        # fitted_param_and_c_init = jnp.concatenate([fitted_param_init, c_init])

        # set up the Minuit object and perform the fit
        print('performing fit...')
        m = Minuit(chi2_val, fitted_param_init, grad=chi2_grad)

        m.errordef = 1
        m.strategy = 0

        # long sequence of migrad and simplex calls to hopefully converge!
        # need to experiment to see if this is really necessary.

        m.migrad()
        m.simplex()
        m.migrad()
        m.simplex()
        m.migrad()
        m.simplex()
        m.migrad()
        m.hesse()

        # resolve the redundancy in decay parameters (if it exists)
        # by fixing one of them.
        if fixed_idx is not None:
            m.fixed[fixed_idx] = True
            m.hesse()

        chi2_min = float(m.fval)
        print(f'chi2 obtained: {Decimal(chi2_min):.2E}')
        print(f'chi2 obtained by PDG: {Decimal(chi2_pdg):.2E}')
        print(f'fit valid: {m.valid}, errors from hessian accurate: {m.accurate}')
        fitted_values = jnp.array(m.values)
        # fitted_values = jnp.array(m.values)[:len(fitted_param_init)]
        # c_values = jnp.array(m.values)[len(fitted_param_init):]
        # print(list(zip(nodes_with_measurements, 1+jnp.clip(c_values, 0, 1e2))))
        param_values = fitted_params_to_params(fitted_values)

        if compare_to_pdg:
            pdg_params = []
            values_found = True
            for i, p in enumerate(parameters):
                if p.startswith('nuisance_'):
                    p = p.removeprefix('nuisance_')
                pdg_param_value, pdg_param_error_p, pdg_param_error_n = pdg_most_precise_value(p)
                if pdg_param_value is None:
                    values_found = False
                    break
                pdg_param_error = (pdg_param_error_p + pdg_param_error_n) / 2
                pdg_params.append(pdg_param_value)
                print(p, param_values[i], pdg_param_value, np.abs(param_values[i] - pdg_param_value)/pdg_param_error)
                meas_df_param = meas_df[meas_df['node'] == p]
                if len(meas_df_param) > 0:
                    meas_value = np.array(meas_df_param['value'])
                    meas_error = np.array(meas_df_param['error'])
                    print(np.sum((meas_value - param_values[i])**2/meas_error**2))
            if values_found:
                print('--------')
                print('PDG obtains chi2:', chi2(params_to_fitted_params(jnp.array(pdg_params))))

        if calc_asym_errors:
            # TODO: check that this method works for all fits

            # calculate the covariance matrix of the parameters
            J = jax.jacobian(fitted_params_to_params)(fitted_values)
            param_cov = J @ jnp.array(m.covariance) @ J.T
            for node in nodes:
                # print(node)

                # propagate the covariance matrix of the parameters
                # to the variance of the node
                # (we will use this to set bounds for the binary search)
                node_func = node_funcs[nodes.index(node)]
                node_value = node_func(param_values)
                J_node = jax.jacobian(node_func)(param_values)
                node_cov = float(J_node @ param_cov @ J_node.T)
                node_std = np.sqrt(node_cov)
                # print(node, node_value, node_std)

                # define a constraint function which is just
                # the jit'ed node function
                @jax.jit
                def constraint_jax(fitted_params):
                    return node_func(fitted_params_to_params(fitted_params))

                def constraint(fitted_params):
                    return constraint_jax(jnp.array(fitted_params))
                from scipy.optimize import NonlinearConstraint, minimize

                # define a function which calculates the chi2 at a given value of the node
                # this is just the chi2 minimized with the constraint of a fixed node value.
                def chi2_at_node_val(node_val):
                    constraint_scipy = NonlinearConstraint(constraint, lb=node_val, ub=node_val)
                    res = minimize(fun=chi2, x0=fitted_values, jac=chi2_grad, constraints=[constraint_scipy])
                    return res

                # print('starting binary search above')
                lb = node_value
                ub = node_value + 10*node_std
                # search between lb and ub to find where chi2_at_node_val = chi2_min + 1
                while (ub - lb)/node_std > 1e-3:
                    mid = (lb + ub) / 2
                    res = chi2_at_node_val(mid)
                    assert res.success
                    if res.fun < chi2_min + 1:
                        lb = mid
                    else:
                        ub = mid
                upper_err = ub - node_value
                # print(lb-node_value)
                # print(chi2_at_node_val(lb).fun)
                # print('starting binary search below')
                lb = node_value - 10*node_std
                ub = node_value
                while (ub - lb)/node_std > 1e-3:
                    mid = (lb + ub) / 2
                    res = chi2_at_node_val(mid)
                    # assert res.success
                    if res.fun < chi2_min + 1:
                        ub = mid
                    else:
                        lb = mid
                lower_error = node_value - lb
                node_value = float(node_value)
                upper_err = float(upper_err)
                lower_error = float(lower_error)
                print(f'{node}: {Decimal(node_value):.5E} + {Decimal(upper_err):.2E} - {Decimal(lower_error):.2E}')

        if meas_diagnostics:

            for i, node in enumerate(nodes):
                pdg_value, pdg_error_p, pdg_error_n = pdg_most_precise_value(node)
                if pdg_value is None:
                    continue
                pdg_error = (pdg_error_p + pdg_error_n) / 2
                fitted_value = node_funcs[i](param_values)
                print('------'+node)
                print(rel_df[rel_df['node'] == node])
                print(fit_df[fit_df['node'] == node]['type'].iloc[0])
                print(node, fitted_value, pdg_value, np.abs(fitted_value - pdg_value)/pdg_error)
                meas_df_node = meas_df[meas_df['node'] == node]
                meas_value = np.array(meas_df_node['value'])
                meas_error_n = np.array(meas_df_node['error_n'])
                meas_error_p = np.array(meas_df_node['error_p'])
                meas_error_min = np.minimum(meas_error_n, meas_error_p)
                meas_error_max = np.maximum(meas_error_n, meas_error_p)
                meas_resid = meas_value - fitted_value
                meas_error_between = (2 * meas_error_n * meas_error_p - meas_resid * (meas_error_p - meas_error_n)) / (
                    meas_error_n + meas_error_p
                )
                meas_error = np.clip(meas_error_between, meas_error_min, meas_error_max)
                # print(meas_error)
                # print(meas_error_n)
                # print(meas_error_p)
                print('our chi2:', np.sum(meas_resid**2/meas_error**2))
                meas_resid = meas_value - pdg_value
                meas_error_between = (2 * meas_error_n * meas_error_p - meas_resid * (meas_error_p - meas_error_n)) / (
                    meas_error_n + meas_error_p
                )
                meas_error = np.clip(meas_error_between, meas_error_min, meas_error_max)
                print('PDG chi2:', np.sum(meas_resid**2/meas_error**2))
            meas_error_min = np.minimum(np.array(meas_df['error_p']), np.array(meas_df['error_n']))
            print(mu(param_values))
            print(np.sum((np.array(meas_df['value']) - np.array(mu(param_values)))**2/meas_error_min**2))