"""One local-geometry Birge-scale pass, followed by one scaled mean fit.

The local Gaussian tangent model supplies residual degrees of freedom. It is
exact for an unconstrained linear Gaussian fit and an approximation otherwise.
There is no simulated expectation and no iteration over scale estimates.
"""

import jax
import jax.numpy as jnp
import numpy as np
from scipy.linalg import null_space

from pdgfits.pdg_scaling import _correlated_blocks
from pdgfits.refit import physical_refit_model, prepare_refit


def node_scale_geometry(residual, jacobian, correlation, nodes):
    """Compare each node's marginal quadratic residual with its local expectation.

    Inputs are normalized measurement residuals z, their parameter Jacobian A,
    and the supplied correlation R. In the local Gaussian model the residual
    covariance is L (I - U U.T) L.T, where L L.T = R and U spans L^+ A.
    For group g use Q_g = z_g.T R_gg^+ z_g and E_g = tr(R_gg^+ Cov(z)_gg).
    S_g = max(1, sqrt(Q_g/E_g)); zero residual degrees of freedom are flagged.

    Cross-node correlations enter the global projection. Marginal node Q's
    need not sum to the global Q when nodes are correlated. This avoids signed
    cross-term allocations or assigning rotated whitened coordinates to nodes.
    Exactly dependent blocks tie their nodes to a common scale. Distinct scale
    estimates are a single-pass diagnostic, not joint variance-component MLEs.
    """
    z, a, corr = map(lambda x: np.asarray(x, float), (residual, jacobian, correlation))
    nodes = np.asarray(nodes)
    n = len(z)
    if z.shape != (n,) or a.ndim != 2 or a.shape[0] != n or corr.shape != (n, n) or nodes.shape != (n,):
        raise ValueError('Incompatible residual, Jacobian, correlation or node shapes')
    if not all(np.isfinite(x).all() for x in (z, a, corr)):
        raise ValueError('Local geometry inputs must be finite')
    if not np.allclose(corr, corr.T, atol=1e-12) or not np.allclose(np.diag(corr), 1):
        raise ValueError('Correlation must be symmetric with unit diagonal')
    eigenvalues, eigenvectors = np.linalg.eigh(corr)
    if eigenvalues.min() < -1e-10:
        raise ValueError('Correlation must be positive semidefinite')
    supported = eigenvalues > 1e-15*eigenvalues.max()
    basis = eigenvectors[:, supported]
    factor = basis*np.sqrt(eigenvalues[supported])
    whitener = (basis/np.sqrt(eigenvalues[supported])).T
    whitened = whitener@a
    units = np.linalg.norm(whitened, axis=0)
    whitened = whitened/np.where(units > 0, units, 1.)
    u, singular, _ = np.linalg.svd(whitened, full_matrices=False)
    rank = int(np.sum(singular > 1e-10*singular.max(initial=0)))
    residual_factor = factor-(factor@u[:, :rank])@u[:, :rank].T
    residual_covariance = residual_factor@residual_factor.T
    # A singular precision kernel only measures the supported component.
    z = basis@(basis.T@z)
    names = list(dict.fromkeys(nodes))
    groups = [{node} for node in names]
    for block in _correlated_blocks(corr):
        block_eigenvalues = np.linalg.eigvalsh(corr[np.ix_(block, block)])
        if block_eigenvalues.min() <= 1e-15*block_eigenvalues.max():
            tied = set(nodes[block])
            merged = set().union(*(group for group in groups if group & tied))
            groups = [group for group in groups if not group & tied]+[merged]
    groups.sort(key=lambda group: min(names.index(node) for node in group))
    rows = []
    scales = np.ones(n)
    for group in groups:
        indices = np.flatnonzero(np.isin(nodes, list(group)))
        marginal_precision = np.linalg.pinv(corr[np.ix_(indices, indices)])
        observed = float(z[indices]@marginal_precision@z[indices])
        expected = max(0., float(np.trace(marginal_precision@residual_covariance[np.ix_(indices, indices)])))
        estimable = expected > 1e-10*len(indices)
        scale = max(1., np.sqrt(observed/expected)) if estimable else 1.
        scales[indices] = scale
        rows.append({'nodes': [name for name in names if name in group], 'indices': indices.tolist(),
                     'observed': observed, 'expected': expected, 'scale': scale,
                     'estimable': estimable})
    return {'nodes': names, 'scales': np.array([scales[np.flatnonzero(nodes == name)[0]] for name in names]),
            'measurement_scales': scales, 'groups': rows, 'mean_rank': rank,
            'measurement_rank': int(supported.sum()), 'residual_df': int(supported.sum())-rank,
            'q': float(np.sum((whitener@z)**2)), 'residual_covariance': residual_covariance}


def fit_node_scales(fit, *, verbose=False):
    """Calculate local node scales once, then refit the original objective.

    Linearize the normalized residual map at the unscaled optimum, including
    the error-interpolation slope. At an asymmetric knot use the mean of the
    two one-sided slopes and flag it. Active bounds use the tangent space of
    the current face, which is not a boundary-mixture calibration theorem.
    Scales multiply both quoted errors and retain correlation coefficients.
    No automatic precision exclusion is applied. Returned intervals condition
    on these scale estimates; their frequentist coverage is not established.
    """
    fit = physical_refit_model(fit)
    refit = prepare_refit(fit)
    original = refit()
    values = original['fitted_values']
    predict = lambda p: original['mu_adjust'](original['fitted_params_to_params'](p))
    raw = original['meas_df']['value'].to_numpy()-np.asarray(predict(jnp.asarray(values)))
    lower, upper = [original['meas_df'][key].to_numpy() for key in ['error_n', 'error_p']]
    midpoint = 2*lower*upper/(lower+upper)
    sigma = np.clip(midpoint-raw*(upper-lower)/(lower+upper), np.minimum(lower, upper), np.maximum(lower, upper))
    slope = np.where(raw < -upper, 1/upper, np.where(raw > lower, 1/lower, midpoint/sigma**2))
    knots = (lower != upper) & (np.abs(np.abs(raw/sigma)-1) < 1e-7)
    slope[knots] = (midpoint[knots]/sigma[knots]**2+1/sigma[knots])/2
    jacobian = -slope[:, None]*np.asarray(jax.jacfwd(predict)(jnp.asarray(values)))
    # The prepared physical chart already eliminates exact equalities.
    constraint = original.get('mean_constraint')
    active = np.array([], int)
    if constraint is not None:
        at = constraint.A@values
        active = np.flatnonzero((at-constraint.lb < 1e-7) | (constraint.ub-at < 1e-7))
        if len(active):
            jacobian = jacobian@null_space(constraint.A[active])
    geometry = node_scale_geometry(raw/sigma, jacobian, original['corr_mat'], original['meas_df']['node'])
    if abs(geometry['q']-original['chi2_min']) > 1e-7*max(1., abs(original['chi2_min'])):
        raise ValueError('Local residual geometry does not reproduce the fitted objective')
    scales = geometry.pop('measurement_scales')
    geometry.pop('residual_covariance')
    result = refit(scales=scales, start=original['x'])
    result['node_scaling'] = {**geometry, 'method': 'local geometry, one pass',
                              'asymmetric_rows': np.flatnonzero(lower != upper).tolist(),
                              'knot_rows': np.flatnonzero(knots).tolist(), 'active_bounds': active.tolist(),
                              'original_chi2': original['chi2_min'], 'original_values': original['param_values']}
    if verbose:
        for group in geometry['groups']:
            print(f"{', '.join(group['nodes'])}: Q={group['observed']:.6g}, "
                  f"local expectation={group['expected']:.6g}, S={group['scale']:.6g}")
    return result
