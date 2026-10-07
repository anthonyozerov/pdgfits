"""One local-geometry Birge-scale pass, followed by one scaled mean fit.

The local Gaussian tangent model supplies residual degrees of freedom. It is
exact for an unconstrained linear Gaussian fit and an approximation otherwise.
There is no simulated expectation and no iteration over scale estimates.
"""

import jax
import jax.numpy as jnp
import numpy as np
from scipy.linalg import null_space

from pdgfits.corr_mat import correlation_blocks
from pdgfits.refit import physical_refit_model, prepare_refit


def node_scale_geometry(residual, jacobian, correlation, nodes, *,
                        group_correlated=True, correlated_blocks=None):
    """Compare each scale group's quadratic residual with its local expectation.

    Inputs are normalized residuals z, their parameter Jacobian A, and supplied
    correlation R. The local residual covariance is L (I - U U.T) L.T, where
    L L.T = R and U spans L^+ A. For group g calculate
    Q_g = z_g.T R_gg^+ z_g and E_g = tr(R_gg^+ Cov(z)_gg), then
    S_g = max(1, sqrt(Q_g/E_g)). Flag groups with no residual information.

    By default each declared correlation block has one scale; the remaining
    measurements are grouped by node. Blocks can span nodes without bringing
    in their other measurements. In the absence of explicit blocks, infer them
    from nonzero correlations. This is the grouping of SDOFIT/SSCAFAC, using
    Richie's proposed observed/expected formula rather than the old pull rule.

    group_correlated=False restores marginal node groups, tying nodes only for
    exactly dependent blocks. Such marginal Q's need not sum to the global Q.
    Measurement scales and group records are authoritative: a node can now
    participate in several different scale groups.
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
    blocks = correlation_blocks(corr, correlated_blocks)
    if group_correlated:
        grouped = np.zeros(n, bool)
        groups = [np.asarray(block, dtype=int) for block in blocks]
        for indices in groups:
            grouped[indices] = True
        groups.extend(np.flatnonzero((nodes == name) & ~grouped) for name in names)
        groups = sorted((g for g in groups if len(g)), key=lambda g: int(g[0]))
    else:
        # Preserve exact dependencies in the optional previous node-only rule.
        node_groups = [{node} for node in names]
        for block in correlation_blocks(corr):
            eigenvalues = np.linalg.eigvalsh(corr[np.ix_(block, block)])
            if eigenvalues.min() <= 1e-15*eigenvalues.max():
                tied = set(nodes[block])
                merged = set().union(*(group for group in node_groups if group & tied))
                node_groups = [group for group in node_groups if not group & tied]+[merged]
        node_groups.sort(key=lambda group: min(names.index(node) for node in group))
        groups = [np.flatnonzero(np.isin(nodes, list(group))) for group in node_groups]
    rows = []
    scales = np.ones(n)
    for indices in groups:
        marginal_precision = np.linalg.pinv(corr[np.ix_(indices, indices)])
        observed = float(z[indices]@marginal_precision@z[indices])
        expected = max(0., float(np.trace(marginal_precision@residual_covariance[np.ix_(indices, indices)])))
        estimable = expected > 1e-10*len(indices)
        scale = max(1., np.sqrt(observed/expected)) if estimable else 1.
        scales[indices] = scale
        rows.append({'nodes': list(dict.fromkeys(nodes[indices])), 'indices': indices.tolist(),
                     'observed': observed, 'expected': expected, 'scale': scale,
                     'estimable': estimable})
    return {'nodes': names, 'group_correlated': group_correlated,
            'measurement_scales': scales, 'groups': rows, 'mean_rank': rank,
            'measurement_rank': int(supported.sum()), 'residual_df': int(supported.sum())-rank,
            'q': float(np.sum((whitener@z)**2)), 'residual_covariance': residual_covariance}


def fit_node_scales(fit, *, group_correlated=True, verbose=False):
    """Calculate node/block scales once, then refit the original objective.

    Linearize the normalized residual map at the unscaled optimum, including
    the error-interpolation slope. At an asymmetric knot use the mean of the
    two one-sided slopes and flag it. Active bounds use the tangent space of
    the current face, which is not a boundary-mixture calibration theorem.
    Correlation blocks share a scale by default; pass group_correlated=False
    for the previous node-only groups. Scales multiply both quoted errors and retain correlation coefficients.
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
    geometry = node_scale_geometry(
        raw/sigma, jacobian, original['corr_mat'], original['meas_df']['node'],
        group_correlated=group_correlated, correlated_blocks=original.get('correlation_blocks'),
    )
    if abs(geometry['q']-original['chi2_min']) > 1e-7*max(1., abs(original['chi2_min'])):
        raise ValueError('Local residual geometry does not reproduce the fitted objective')
    scales = geometry['measurement_scales']
    geometry.pop('residual_covariance')
    result = refit(scales=scales, start=original['x'])
    result['node_scaling'] = {**geometry, 'method': 'local geometry, one pass',
                              'asymmetric_rows': np.flatnonzero(lower != upper).tolist(),
                              'knot_rows': np.flatnonzero(knots).tolist(), 'active_bounds': active.tolist(),
                              'original_chi2': original['chi2_min'], 'original_values': original['param_values']}
    if verbose:
        for group in geometry['groups']:
            print(f"{', '.join(group['nodes'])}, rows {group['indices']}: Q={group['observed']:.6g}, "
                  f"local expectation={group['expected']:.6g}, S={group['scale']:.6g}")
    return result
