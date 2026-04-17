import numpy as np
import jax.numpy as jnp
import jax


def build_chi2(y, mu, error_n, error_p, corr_mat_inv, fitted_params_to_params, translate_dep=None, adjust=None):
    if translate_dep is None:
        translate_dep = lambda x: 0
    if adjust is None:
        adjust = lambda x: 1
    error_min = jnp.minimum(error_n, error_p)
    error_max = jnp.maximum(error_n, error_p)

    def get_sigma(error_n, error_p, resid, adjustment):
        resid_ua = resid/adjustment
        error_between = (2 * error_n * error_p - resid_ua * (error_p - error_n)) / (
            error_n + error_p
        )
        return jnp.clip(error_between, error_min, error_max) * adjustment

    @jax.jit
    def chi2(fitted_params):
        params = fitted_params_to_params(fitted_params)
        adjustment = adjust(params)
        resid = (y*adjustment+translate_dep(params)) - mu(params)
        error = get_sigma(error_n, error_p, resid, adjustment)
        cov_mat_inv = corr_mat_inv * 1/jnp.outer(error, error)
        chi2 = jnp.sum(resid @ cov_mat_inv @ resid)
        return chi2

    chi2_grad_jax = jax.jit(jax.grad(chi2))
    def chi2_grad(fitted_params):
        return np.asarray(chi2_grad_jax(fitted_params))
    def chi2_val(fitted_params):
        return float(chi2(fitted_params))

    return chi2, chi2_grad, chi2_val


def build_chi2_c(y, mu, error_n, error_p, corr_mat_inv, fitted_params_to_params, translate_dep=None, adjust=None, len_fitted_params=None, c_map=None):
    if translate_dep is None:
        translate_dep = lambda x: 0
    if adjust is None:
        adjust = lambda x: 1
    error_min = jnp.minimum(error_n, error_p)
    error_max = jnp.maximum(error_n, error_p)

    def get_sigma(error_n, error_p, resid, adjustment):
        resid_ua = resid/adjustment
        error_between = (2 * error_n * error_p - resid_ua * (error_p - error_n)) / (
            error_n + error_p
        )
        return jnp.clip(error_between, error_min, error_max) * adjustment

    det_corr_mat = jnp.linalg.det(jnp.linalg.pinv(corr_mat_inv))
    print('Determinant of the correlation matrix:', det_corr_mat)

    @jax.jit
    def chi2(fitted_params_and_c):
        fitted_params = fitted_params_and_c[:len_fitted_params]
        c_params = fitted_params_and_c[len_fitted_params:]
        c = c_map(c_params)
        params = fitted_params_to_params(fitted_params)
        adjustment = adjust(params)
        resid = (y*adjustment+translate_dep(params)) - mu(params)
        error = get_sigma(error_n, error_p, resid, adjustment)*c
        cov_mat_inv = corr_mat_inv * 1/jnp.outer(error, error)
        chi2 = jnp.sum(resid @ cov_mat_inv @ resid)
        # det = det_corr_mat * jnp.prod(error)**2
        return chi2 + jnp.log(det_corr_mat) + 2*jnp.sum(jnp.log(error))

    chi2_grad_jax = jax.jit(jax.grad(chi2))
    def chi2_grad(fitted_params):
        return np.asarray(chi2_grad_jax(fitted_params))
    def chi2_val(fitted_params):
        return float(chi2(fitted_params))

    return chi2, chi2_grad, chi2_val
