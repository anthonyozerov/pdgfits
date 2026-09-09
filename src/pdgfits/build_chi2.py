import numpy as np
import jax.numpy as jnp
import jax


def build_chi2(y, mu, error_n, error_p, corr_mat_inv, fitted_params_to_params, translate_dep=None, adjust=None, use_jit=True):

    if translate_dep is None:
        translate_dep = lambda x: 0
    if adjust is None:
        adjust = lambda x: 1

    error_min = jnp.minimum(error_n, error_p)
    error_max = jnp.maximum(error_n, error_p)
    sum_e = error_n + error_p
    diff_e = error_p - error_n
    prod_e2 = 2 * error_n * error_p

    def get_sigma(resid):
        error_between = (prod_e2 - resid * diff_e) / sum_e
        return jnp.clip(error_between, error_min, error_max)

    maybe_jit = jax.jit if use_jit else (lambda f: f)

    @maybe_jit
    def chi2_open(fitted_params, y_arg):
        params = fitted_params_to_params(fitted_params)
        adjustment = adjust(params)
        resid = (y_arg * adjustment + translate_dep(params)) - mu(params)
        error = get_sigma(resid / adjustment) * adjustment
        normed_resid = resid / error
        return normed_resid @ corr_mat_inv @ normed_resid


    @maybe_jit
    def chi2(fitted_params):
        return chi2_open(fitted_params, y)

    chi2_grad_jax = (jax.jit if use_jit else (lambda f: f))(jax.grad(chi2))
    def chi2_grad(fitted_params):
        return np.asarray(chi2_grad_jax(fitted_params))
    def chi2_val(fitted_params):
        return float(chi2(fitted_params))

    return chi2, chi2_grad, chi2_val, chi2_open
