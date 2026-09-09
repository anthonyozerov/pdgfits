import numpy as np
import jax.numpy as jnp
import jax


def _whitener(precision):
    precision = np.asarray(precision)
    diagonal = np.diag(precision)
    if np.array_equal(precision, np.diag(diagonal)):
        return np.sqrt(np.maximum(diagonal, 0))
    eigenvalues, eigenvectors = np.linalg.eigh(precision)
    return np.sqrt(np.maximum(eigenvalues, 0))[:, None]*eigenvectors.T


def _whiten(whitener, residual):
    return whitener*residual if whitener.ndim == 1 else whitener@residual


def _product_residuals(params, values, scales, lower, upper, whitener, powers):
    bases = jnp.where(powers != 0, params[None, :], 1.)
    prediction = jnp.prod(bases**powers, axis=1)
    residual = (values-prediction)/scales
    sigma = jnp.clip((2*lower*upper-residual*(upper-lower))/(lower+upper),
                     jnp.minimum(lower, upper), jnp.maximum(lower, upper))
    return _whiten(whitener, residual/sigma)


def _product_q(*args):
    residual = _product_residuals(*args)
    return residual @ residual


_product_value = jax.jit(_product_q)
_product_residual_value = jax.jit(_product_residuals)
_product_value_grad = jax.jit(jax.value_and_grad(_product_q))
_product_residual_jac = jax.jit(lambda *args: (_product_residuals(*args), jax.jacfwd(_product_residuals)(*args)))
_product_hessian = jax.jit(jax.hessian(_product_q))


def build_product_chi2(values, lower, upper, precision, powers):
    """Share compiled averages of products/ratios across different node data.

    All data and exponent arrays are runtime arguments, so equal-sized models
    reuse a compiled function. This includes ordinary averages and auxiliary
    product/ratio corrections; dependent additive corrections use build_chi2.
    """
    whitener = _whitener(precision)
    fixed = (np.asarray(lower), np.asarray(upper), whitener, np.asarray(powers, int))
    def opened(params, data, scales=1.):
        return _product_value(params, data, scales, *fixed)
    opened.residual_value_jac = lambda p, y, s=1.: _product_residual_jac(p, y, s, *fixed)
    opened.residuals = lambda p, y, s=1.: _product_residual_value(p, y, s, *fixed)
    opened.hessian = lambda p, y, s=1.: _product_hessian(p, y, s, *fixed)
    def chi2(params):
        return opened(params, values)
    chi2.value_and_grad = lambda p: _product_value_grad(p, values, 1., *fixed)
    return chi2, lambda p: np.asarray(chi2.value_and_grad(p)[1]), lambda p: float(chi2(p)), opened


def build_chi2(y, mu, error_n, error_p, corr_mat_inv, fitted_params_to_params, translate_dep=None, adjust=None, use_jit=True):

    if translate_dep is None:
        translate_dep = lambda x: 0
    if adjust is None:
        adjust = lambda x: 1

    # These are fixed input data. Computing constants with NumPy avoids
    # compiling separate JAX kernels before the objective is even traced.
    error_n, error_p = np.asarray(error_n), np.asarray(error_p)
    error_min = np.minimum(error_n, error_p)
    error_max = np.maximum(error_n, error_p)
    sum_e = error_n + error_p
    diff_e = error_p - error_n
    prod_e2 = 2 * error_n * error_p

    def get_sigma(resid):
        error_between = (prod_e2 - resid * diff_e) / sum_e
        return jnp.clip(error_between, error_min, error_max)

    maybe_jit = jax.jit if use_jit else (lambda f: f)

    @maybe_jit
    def normalized_residuals(fitted_params, y_arg, scales=1.0):
        params = fitted_params_to_params(fitted_params)
        adjustment = adjust(params)
        resid = (y_arg * adjustment + translate_dep(params)) - mu(params)
        error = scales * get_sigma(resid / (adjustment * scales)) * adjustment
        return resid / error

    @maybe_jit
    def chi2_open(fitted_params, y_arg, scales=1.0):
        normed_resid = normalized_residuals(fitted_params, y_arg, scales)
        return normed_resid @ corr_mat_inv @ normed_resid

    # Residuals permit a least-squares solver and repeated fits without
    # recompiling the measurement model for each simulated dataset.
    whitener = jnp.asarray(_whitener(corr_mat_inv))
    chi2_open.residuals = maybe_jit(lambda fp, data, scales=1.0:
                                  _whiten(whitener, normalized_residuals(fp, data, scales)))
    chi2_open.residual_value_jac = maybe_jit(lambda fp, data, scales=1.0:
        (chi2_open.residuals(fp, data, scales), jax.jacfwd(chi2_open.residuals)(fp, data, scales)))
    chi2_open.hessian = maybe_jit(jax.hessian(chi2_open))


    @maybe_jit
    def chi2(fitted_params):
        return chi2_open(fitted_params, y)

    chi2_grad_jax = (jax.jit if use_jit else (lambda f: f))(jax.grad(chi2))
    def chi2_grad(fitted_params):
        return np.asarray(chi2_grad_jax(fitted_params))
    def chi2_val(fitted_params):
        return float(chi2(fitted_params))

    return chi2, chi2_grad, chi2_val, chi2_open
