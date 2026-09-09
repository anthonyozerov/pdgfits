import jax.numpy as jnp
import numpy as np
import jax

def block_birge(y, corr_inv, error, theta, chi2, mu, hess, blocks):
    """
    Experimental block residual/leverage diagnostic.

    The expected residual formula is justified for linear Gaussian least
    squares with fixed covariance. With the observed Hessian of a nonlinear
    asymmetric objective, this is only a local diagnostic and can be negative.
    It must not automatically rescale reported confidence intervals.
    """
    # W = D^{-1} corr_inv D^{-1}
    W = corr_inv * jnp.outer(1/error, 1/error) # (n, n)

    # Per-index contribution to chi^2
    r = y - mu(theta) # (n,)
    contrib = r * (W @ r) # (n,)

    # Fisher info and Jacobian
    F = 0.5 * hess(theta) # (p, p)
    J = jax.jacfwd(mu)(theta) # (n, p)

    # diag of W J F^{-1} J^T, computed as row-wise dot of (W J) and (J F^{-T})
    F_inv = jnp.linalg.inv(F) # (p, p)
    diag_M = jnp.sum((W @ J) * (J @ F_inv), axis=1) # (n,)

    chi2_b = jnp.array([jnp.sum(contrib[idx]) for idx in blocks]) # (n_blocks,)
    exp_chi2_b = jnp.array([len(idx) - jnp.sum(diag_M[idx]) for idx in blocks]) # (n_blocks,)

    assert np.allclose(np.sum(chi2_b), np.sum(contrib))
    assert np.allclose(np.sum(chi2_b), chi2(theta)), f'chi2_b: {chi2_b}, np.sum(contrib): {np.sum(contrib)}, chi2(theta): {chi2(theta)}'
    # assert np.allclose(exp_chi2_b, len(y) - len(theta), atol=0.1), f'exp_chi2_b: {exp_chi2_b}, len(y): {len(y)}, len(theta): {len(theta)}'

    return chi2_b, exp_chi2_b, np.sqrt(chi2_b / exp_chi2_b)
