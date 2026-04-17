import numpy as np
from jax import numpy as jnp

from pdgfits.fit_query import all_fits, pdg_most_precise_value


def get_pdg_chi2(label):
    """Return the PDG-reported chi2 for a fit label from fit_control1."""
    fits_df = all_fits()
    row = fits_df[fits_df['label'] == label]
    if len(row) == 0:
        raise ValueError(f'Label {label} not found in fit_control1')
    return float(row['chi_square'].iloc[0])


def compare_to_pdg(fit):
    """Compare fitted parameter values to PDG summary values and print residuals."""
    parameters = fit['parameters']
    param_values = fit['param_values']
    chi2 = fit['chi2']
    params_to_fitted_params = fit['params_to_fitted_params']
    meas_df = fit['meas_df']

    pdg_params = []
    for idx, p in enumerate(parameters):
        p_query = p.removeprefix('nuisance_') if p.startswith('nuisance_') else p
        pdg_value, pdg_error_p, pdg_error_n = pdg_most_precise_value(p_query)
        if pdg_value is None:
            print(f'No PDG value found for {p_query}, stopping comparison.')
            return
        pdg_error = (pdg_error_p + pdg_error_n) / 2
        pdg_params.append(pdg_value)
        pull = np.abs(param_values[idx] - pdg_value) / pdg_error
        print(p, param_values[idx], pdg_value, pull)
        meas_df_param = meas_df[meas_df['node'] == p]
        if len(meas_df_param) > 0:
            meas_value = np.array(meas_df_param['value'])
            meas_error = np.array(meas_df_param['error'])
            print(np.sum((meas_value - param_values[idx])**2 / meas_error**2))

    print('--------')
    print('PDG obtains chi2:', chi2(params_to_fitted_params(jnp.array(pdg_params))))


def meas_diagnostics(fit):
    """Print per-node measurement chi2 diagnostics comparing our fit to PDG values."""
    nodes = fit['nodes']
    node_funcs = fit['node_funcs']
    param_values = fit['param_values']
    rel_df = fit['rel_df']
    fit_df = fit['fit_df']
    meas_df = fit['meas_df']
    mu = fit['mu']

    for i, node in enumerate(nodes):
        pdg_value, pdg_error_p, pdg_error_n = pdg_most_precise_value(node)
        if pdg_value is None:
            continue
        pdg_error = (pdg_error_p + pdg_error_n) / 2
        fitted_value = node_funcs[i](param_values)
        print('------' + node)
        print(rel_df[rel_df['node'] == node])
        print(fit_df[fit_df['node'] == node]['type'].iloc[0])
        print(node, fitted_value, pdg_value, np.abs(fitted_value - pdg_value) / pdg_error)

        meas_df_node = meas_df[meas_df['node'] == node]
        meas_value = np.array(meas_df_node['value'])
        meas_error_n = np.array(meas_df_node['error_n'])
        meas_error_p = np.array(meas_df_node['error_p'])
        meas_error_min = np.minimum(meas_error_n, meas_error_p)
        meas_error_max = np.maximum(meas_error_n, meas_error_p)

        def asymmetric_error(resid):
            between = (2 * meas_error_n * meas_error_p - resid * (meas_error_p - meas_error_n)) / (
                meas_error_n + meas_error_p
            )
            return np.clip(between, meas_error_min, meas_error_max)

        resid_our = meas_value - fitted_value
        print('our chi2:', np.sum(resid_our**2 / asymmetric_error(resid_our)**2))

        resid_pdg = meas_value - pdg_value
        print('PDG chi2:', np.sum(resid_pdg**2 / asymmetric_error(resid_pdg)**2))

    meas_error_min = np.minimum(np.array(meas_df['error_p']), np.array(meas_df['error_n']))
    mu_vals = mu(param_values)
    print(mu_vals)
    print(np.sum((np.array(meas_df['value']) - np.array(mu_vals))**2 / meas_error_min**2))
