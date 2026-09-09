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


def linear_birge(y, design, covariance, nodes):
    """Fit y = design @ beta with FIXED positive-definite Gaussian covariance.

    Return residual scale diagnostics for independent nodes / correlated blocks.
    Correlation-connected measurements form one block, even across nodes;
    remaining independent measurements are grouped by node. No weak-input
    exclusion is applied. This implements the expectation in Bonventre's 2024
    proposal for the linear, unconstrained case, not the legacy PDG procedure.

    `expected_q` is exact under the supplied model. With a scale s[h] on each
    block, E[Q[g]] = sum_h response[g,h] * s[h]**2. Consequently Q[g]/expected_q[g]
    is NOT in general an unbiased estimate of that node's unknown variance scale.
    A block with zero residual degrees of freedom has no estimable scale (NaN).
    The clipped scales are diagnostics only: this function does not refit or
    rescale intervals, and must not be fed data-dependent asymmetric covariances
    as though its expectation remained exact.
    """
    y = np.asarray(y, float)
    design = np.asarray(design, float)
    covariance = np.asarray(covariance, float)
    nodes = np.asarray(nodes)
    n = len(y)
    if (y.shape != (n,) or design.ndim != 2 or design.shape[0] != n
            or covariance.shape != (n, n) or nodes.shape != (n,)):
        raise ValueError('Incompatible measurement, design, covariance or node shapes')
    if not all(np.isfinite(a).all() for a in (y, design, covariance)):
        raise ValueError('Inputs must be finite')
    if not np.allclose(covariance, covariance.T, rtol=1e-12, atol=0):
        raise ValueError('Covariance must be symmetric')
    # Work in standardized units before whitening, so rank and block discovery
    # do not depend on the different physical units of the measured nodes.
    errors = np.sqrt(np.diag(covariance))
    if np.any(errors <= 0) or not np.isfinite(errors).all():
        raise ValueError('Measurement variances must be positive')
    correlation = covariance / np.outer(errors, errors)
    try:
        chol = np.linalg.cholesky(correlation)
    except np.linalg.LinAlgError as exc:
        raise ValueError('Covariance must be positive definite') from exc
    a = np.linalg.solve(chol, design / errors[:, None])
    b = np.linalg.solve(chol, y / errors)
    units = np.linalg.norm(a, axis=0)
    units[units == 0] = 1
    a = a / units
    coefficients, _, rank, _ = np.linalg.lstsq(a, b, rcond=None)
    u = np.linalg.svd(a, full_matrices=False)[0][:, :rank]
    residual_projection = np.eye(n) - u @ u.T
    residual = b - a @ coefficients

    # Do not split correlated quadratic cross terms between individual nodes.
    remaining = set(range(n))
    groups, independent = [], []
    while remaining:
        component, pending = [], [min(remaining)]
        while pending:
            i = pending.pop()
            if i not in remaining:
                continue
            remaining.remove(i)
            component.append(i)
            pending.extend(j for j in remaining if correlation[i, j] != 0)
        if len(component) > 1:
            groups.append(sorted(component))
        else:
            independent.extend(component)
    for node in dict.fromkeys(nodes[independent]):
        groups.append([i for i in independent if nodes[i] == node])
    groups.sort(key=min)
    response = np.array([[np.sum(residual_projection[np.ix_(g, h)]**2)
                          for h in groups] for g in groups])
    blocks = []
    for g, indices in enumerate(groups):
        q = float(np.sum(residual[indices]**2))
        expected = float(response[g].sum())
        estimable = expected > 1e-10
        blocks.append({'indices': indices, 'nodes': list(dict.fromkeys(nodes[indices])),
                       'q': q, 'expected_q': expected, 'estimable': estimable,
                       'scale': max(1., np.sqrt(q / expected)) if estimable else np.nan})
    return {'parameters': coefficients / units, 'rank': rank, 'q': float(residual @ residual),
            'blocks': blocks, 'response': response}


def fit_linear_scales(y, design, covariance, nodes):
    """Experimental Gaussian REML with one inflation-only scale per node.

    Model: y = X beta + error, Cov(error) = D_s V D_s, with fixed symmetric V
    and s[node] >= 1. Known correlations are preserved, including across nodes.
    Optimize the restricted likelihood; refit beta at each covariance update.
    This recovers the clipped Birge ratio for a single ordinary average.

    Node contributions are derivatives of this specific covariance model. With
    cross-node correlations they can be negative: never take their square roots
    as an update rule. The optimizer uses the restricted-likelihood gradient.
    Returned covariance is conditional on the estimated scales, not a calibrated
    confidence interval. This does not justify an asymmetric/nonlinear extension
    or reproduce the full legacy PDG procedure. No weak-input exclusion is used.
    """
    from scipy.optimize import minimize

    y, design, covariance = map(lambda x: np.asarray(x, float), (y, design, covariance))
    initial = linear_birge(y, design, covariance, nodes)  # validate the inputs
    node_names = list(dict.fromkeys(nodes))
    groups = [np.flatnonzero(np.asarray(nodes) == node) for node in node_names]
    errors = np.sqrt(np.diag(covariance))
    correlation = covariance / np.outer(errors, errors)
    standardized_design = design / errors[:, None]
    units = np.linalg.norm(standardized_design, axis=0)
    units[units == 0] = 1
    # A fixed full-rank basis for the mean space also handles redundant columns.
    basis = np.linalg.svd(standardized_design / units, full_matrices=False)[0][:, :initial['rank']]
    values = y / errors

    def evaluate(log_variances):
        scales = np.ones(len(y))
        for indices, log_variance in zip(groups, log_variances):
            scales[indices] = np.exp(log_variance / 2)
        current = correlation * np.outer(scales, scales)
        precision_basis = np.linalg.solve(current, basis)
        information = basis.T @ precision_basis
        coefficient = np.linalg.solve(information, precision_basis.T @ values)
        residual = values - basis @ coefficient
        contributions = residual * np.linalg.solve(current, residual)
        expected = 1 - np.sum(np.linalg.solve(information, precision_basis.T).T * basis, axis=1)
        q = np.array([contributions[g].sum() for g in groups])
        degrees = np.array([expected[g].sum() for g in groups])
        value = (np.linalg.slogdet(current)[1] + np.linalg.slogdet(information)[1] + q.sum()) / 2
        return value, (degrees-q)/2, q, degrees, current

    def objective(log_variances):
        return evaluate(log_variances)[:2]

    fit = minimize(objective, np.zeros(len(groups)), jac=True, method='L-BFGS-B',
                   bounds=[(0, None)] * len(groups),
                   options={'ftol': 1e-12, 'gtol': 1e-7, 'maxiter': 500})
    value, gradient, q, degrees, current = evaluate(fit.x)
    projected = np.where(fit.x <= 1e-8, np.minimum(gradient, 0), gradient)
    if not fit.success or np.max(np.abs(projected), initial=0) > 1e-5:
        raise RuntimeError(f'Node-scale REML did not converge: {fit.message}')
    current_covariance = current * np.outer(errors, errors)
    fitted = linear_birge(y, design, current_covariance, nodes)
    return {'parameters': fitted['parameters'], 'nodes': node_names,
            'scales': np.exp(fit.x/2), 'measurement_covariance': current_covariance,
            'q': fitted['q'], 'observed_q': q, 'expected_q': degrees,
            'restricted_nll': value, 'iterations': fit.nit,
            'score_residual': float(np.max(np.abs(projected), initial=0)),
            'initial': initial}
