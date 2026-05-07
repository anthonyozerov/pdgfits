import numpy as np
import jax
from jax import numpy as jnp

from pdgfits.query import all_fits, pdg_most_precise_value


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
    print(f"\n{'Parameter':<16} {'Fitted':>14} {'PDG':>14} {'Pull':>8}")
    print('-' * 60)
    for idx, p in enumerate(parameters):
        p_query = p.removeprefix('nuisance_') if p.startswith('nuisance_') else p
        pdg_value, pdg_error_p, pdg_error_n = pdg_most_precise_value(p_query)
        if pdg_value is None:
            print(f'No PDG value found for {p_query}, stopping comparison.')
            return
        pdg_error = (pdg_error_p + pdg_error_n) / 2
        pdg_params.append(pdg_value)
        pull = np.abs(param_values[idx] - pdg_value) / pdg_error
        print(f"{p:<16} {param_values[idx]:>14.6g} {pdg_value:>14.6g} {pull:>8.4f}")
        meas_df_param = meas_df[meas_df['node'] == p]
        if len(meas_df_param) > 0:
            meas_value = np.array(meas_df_param['value'])
            meas_error = np.array(meas_df_param['error'])
            meas_chi2 = np.sum((meas_value - param_values[idx])**2 / meas_error**2)
            print(f"  meas chi2: {meas_chi2:.4g}")

    print('-' * 60)
    print(f"PDG chi2 at PDG params: {chi2(params_to_fitted_params(jnp.array(pdg_params))):.4g}")


def meas_diagnostics(fit):
    """Print per-node measurement chi2 diagnostics comparing our fit to PDG values."""
    nodes = fit['nodes']
    node_funcs = fit['node_funcs']
    param_values = fit['param_values']
    rel_df = fit['rel_df']
    fit_df = fit['fit_df']
    meas_df = fit['meas_df']
    mu = fit['mu']

    print(f"\n{'Node':<16} {'Fitted':>14} {'PDG':>14} {'Pull':>8} {'Our chi2':>10} {'PDG chi2':>10}")
    print('-' * 76)
    for i, node in enumerate(nodes):
        pdg_value, pdg_error_p, pdg_error_n = pdg_most_precise_value(node)
        if pdg_value is None:
            continue
        pdg_error = (pdg_error_p + pdg_error_n) / 2
        fitted_value = node_funcs[i](param_values)
        pull = np.abs(fitted_value - pdg_value) / pdg_error

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
        chi2_our = np.sum(resid_our**2 / asymmetric_error(resid_our)**2)

        resid_pdg = meas_value - pdg_value
        chi2_pdg = np.sum(resid_pdg**2 / asymmetric_error(resid_pdg)**2)

        node_type = fit_df[fit_df['node'] == node]['type'].iloc[0]
        print(f"{node:<16} {float(fitted_value):>14.6g} {pdg_value:>14.6g} {float(pull):>8.4f} {chi2_our:>10.4g} {chi2_pdg:>10.4g}")
        # print(f"  type={node_type}  relations: {}")

    meas_error_min = np.minimum(np.array(meas_df['error_p']), np.array(meas_df['error_n']))
    mu_vals = mu(param_values)
    total_chi2 = np.sum((np.array(meas_df['value']) - np.array(mu_vals))**2 / meas_error_min**2)
    print('-' * 76)
    print(f"Total chi2 (sym errors): {total_chi2:.4g}")


def meas_sensitivity(fit):
    """
    For each measurement j, compute and print:
        d(mu_j)/d(y_j) * sigma_meas_j / sigma_fitted_j

    where mu_j is the predicted value for measurement j, sigma_meas_j is the
    average of its asymmetric errors, and sigma_fitted_j is the propagated
    uncertainty on mu_j from the parameter covariance.

    Also returns d_params_dy, the full (n_params, n_meas) derivative matrix
    of fitted parameter values with respect to measurements.
    """
    chi2_open = fit['chi2_open']
    fitted_values = fit['fitted_values']
    params_opt = fit['param_values']
    meas_df = fit['meas_df']
    covariance = fit['covariance']
    mu = fit['mu']
    fitted_params_to_params = fit['fitted_params_to_params']

    y = jnp.array(meas_df['value'], dtype=jnp.float64)
    error_n = jnp.array(meas_df['error_n'], dtype=jnp.float64)
    error_p = jnp.array(meas_df['error_p'], dtype=jnp.float64)

    # H_fp = d²χ²/d(fp)²  at optimum, shape (n_fp, n_fp)
    H_fp = jax.hessian(chi2_open, argnums=0)(fitted_values, y)

    # J_cross[i,j] = d²χ²/d(fp_i)d(y_j), shape (n_fp, n_meas)
    J_cross = jax.jacobian(jax.grad(chi2_open, argnums=0), argnums=1)(fitted_values, y)

    # d(fp*)/d(y) = -H_fp^{-1} J_cross, shape (n_fp, n_meas)
    d_fp_dy = -jnp.linalg.solve(H_fp, J_cross)

    # J_f2p = d(params)/d(fp), shape (n_params, n_fp)
    J_f2p = jax.jacobian(fitted_params_to_params)(fitted_values)

    # d(params*)/d(y), shape (n_params, n_meas)
    d_params_dy = J_f2p @ d_fp_dy

    # J_mu = d(mu)/d(params), shape (n_meas, n_params)
    J_mu = jax.jacobian(mu)(params_opt)

    # J_mu_fp = d(mu)/d(fp), shape (n_meas, n_fp)
    J_mu_fp = J_mu @ J_f2p

    # d(mu_j)/d(y_j): diagonal of J_mu @ d_params_dy, shape (n_meas,)
    d_mu_dy_diag = jnp.einsum('ij,ji->i', J_mu, d_params_dy)

    # sigma_fitted_j = sqrt(Var(mu_j)) via error propagation through covariance (fp space)
    var_mu = jnp.einsum('ij,jk,ik->i', J_mu_fp, covariance, J_mu_fp)
    sigma_fitted = jnp.sqrt(jnp.clip(var_mu, 0.0))

    sigma_meas = (error_n + error_p) / 2
    sensitivity = d_mu_dy_diag * sigma_meas / sigma_fitted

    print(f"\n{'Node':<16} {'Value':>14} {'±Error':>12} {'Sensitivity':>12}")
    print('-' * 70)
    for j, row in enumerate(meas_df.itertuples(index=False)):
        print(f"{row.node:<16} {row.value:>14.6g} {float(sigma_meas[j]):>12.4g} {float(sensitivity[j]):>12.4f}")

    return d_params_dy
