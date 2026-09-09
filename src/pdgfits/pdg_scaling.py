"""Reference implementations of documented PDG scale procedures.

The average follows savg.f, including its asymmetric iteration. The joint-fit
comparator covers linear means, fixed symmetric errors and nonsingular explicit
correlation blocks. It follows sscafac.f/sbrfit.f (one scale pass, threshold
1.001, epsilon 1e-5). Dependent singular inputs and nonlinear asymmetric fit
error propagation are outside this comparator's scope.
"""

import numpy as np


def pdg_average(values, error_n, error_p=None, exclude_weak=True):
    y, en = np.asarray(values, float), np.asarray(error_n, float)
    ep = en if error_p is None else np.asarray(error_p, float)
    if len(y) < 2 or en.shape != y.shape or ep.shape != y.shape or np.any(en <= 0) or np.any(ep <= 0):
        raise ValueError('Need at least two measurements with positive errors')
    if not all(np.isfinite(a).all() for a in (y, en, ep)):
        raise ValueError('Measurements and errors must be finite')
    errors = 2*en*ep/(en+ep)
    previous = -999.
    for iteration in range(102):
        weights = 1/errors**2
        mean = np.sum(weights*y)/np.sum(weights)
        variance = 1/np.sum(weights)
        contributions = ((y-mean)/errors)**2
        q = contributions.sum()
        if abs(q-previous) <= .01:
            break
        if iteration == 101:
            raise RuntimeError('PDG asymmetric average did not converge')
        previous = q
        delta = mean-y
        errors = np.where(delta < -en, en, np.where(delta > ep, ep, (2*en*ep+delta*(ep-en))/(en+ep)))
    keep = errors < 3*np.sqrt(len(y)*variance) if exclude_weak else np.ones(len(y), bool)
    scale = np.sqrt(max(1., contributions[keep].sum()/(keep.sum()-1))) if keep.sum() > 1 else 1.
    return {'value': mean, 'error_n': scale/np.sqrt(np.sum(1/en**2)),
            'error_p': scale/np.sqrt(np.sum(1/ep**2)), 'scale': scale,
            'scale_inputs': keep, 'effective_errors': errors, 'q': q}


def _gls(y, x, v):
    chol = np.linalg.cholesky(v)
    a, b = np.linalg.solve(chol, x), np.linalg.solve(chol, y)
    units = np.linalg.norm(a, axis=0)
    if np.any(units == 0):
        raise ValueError('Mean parameters are not identifiable')
    a = a/units
    coefficient, _, rank, _ = np.linalg.lstsq(a, b, rcond=None)
    if rank != x.shape[1]:
        raise ValueError('Mean parameters are not identifiable after exclusion')
    covariance = np.linalg.inv(a.T@a)/np.outer(units, units)
    parameters = coefficient/units
    return parameters, covariance, y-x@parameters, x@covariance@x.T


def _correlated_blocks(v):
    remaining, blocks = set(range(len(v))), []
    while remaining:
        component, pending = [], [min(remaining)]
        while pending:
            i = pending.pop()
            if i not in remaining:
                continue
            remaining.remove(i)
            component.append(i)
            pending.extend(j for j in remaining if v[i, j] != 0)
        if len(component) > 1:
            blocks.append(sorted(component))
    return blocks


def pdg_linear_fit(values, design, covariance, nodes, exclude_weak=True, correlated_blocks=None):
    """PDG separate pull scales, with the original central value retained.

    Weak inputs are selected ONCE using original fitted node uncertainties;
    refit without them, compute pulls, scale and refit again. The reported
    parameters are from the original fit; `refitted_parameters` shows the center
    belonging to the final covariance. Explicit correlation blocks use the old
    rank-one covariance adjustment, which can change correlation coefficients.
    Passing an explicit identity block deliberately differs from no block.

    Correlation blocks must be disjoint and include every nonzero correlation.
    Omitted blocks are inferred from nonzero covariances. Measurements marked
    weak in Fortran are represented as removed rather than given enormous errors.
    """
    y, x, v = map(lambda a: np.asarray(a, float), (values, design, covariance))
    nodes = np.asarray(nodes)
    if (y.ndim != 1 or x.ndim != 2 or x.shape[0] != len(y)
            or v.shape != (len(y), len(y)) or nodes.shape != y.shape):
        raise ValueError('Incompatible measurement, design, covariance or node shapes')
    if not all(np.isfinite(a).all() for a in (y, x, v)) or not np.allclose(v, v.T, rtol=1e-12, atol=0):
        raise ValueError('Inputs must be finite and covariance symmetric')
    original, original_covariance, _, fitted_covariance = _gls(y, x, v)
    blocks = _correlated_blocks(v) if correlated_blocks is None else correlated_blocks
    block_ids = np.full(len(y), -1, int)
    for b, indices in enumerate(blocks):
        if any(i < 0 or i >= len(y) for i in indices):
            raise ValueError('Correlation block index outside measurement range')
        if len(indices) < 2 or len(set(indices)) != len(indices) or np.any(block_ids[indices] >= 0):
            raise ValueError('Correlation blocks must be disjoint sets of at least two inputs')
        block_ids[indices] = b
    rows, columns = np.nonzero(v - np.diag(np.diag(v)))
    if np.any((block_ids[rows] < 0) | (block_ids[rows] != block_ids[columns])):
        raise ValueError('Every nonzero correlation must belong to one explicit block')
    keep = np.ones(len(y), bool)
    if exclude_weak:
        for node in dict.fromkeys(nodes):
            selected = (nodes == node) & (block_ids < 0)
            cutoff = 3*np.sqrt(selected.sum()*np.diag(fitted_covariance)[selected])
            keep[selected] = np.sqrt(np.diag(v)[selected]) <= cutoff
    indices = np.flatnonzero(keep)
    reduced_y, reduced_x, reduced_v = y[keep], x[keep], v[np.ix_(keep, keep)]
    before, _, residual, prediction_covariance = _gls(reduced_y, reduced_x, reduced_v)
    node_scales = {}
    scaled_v = reduced_v.copy()
    for node in dict.fromkeys(nodes):
        selected = (nodes[keep] == node) & (block_ids[keep] < 0)
        if not selected.any():
            continue
        denominator = np.diag(reduced_v)[selected] - (1-1e-5)**2*np.diag(prediction_covariance)[selected]
        pulls = residual[selected]**2/denominator
        magnitude = np.sqrt(np.sum(np.maximum(pulls, 0))/selected.sum())
        scale = magnitude if magnitude > 1.001 else 1.
        node_scales[str(node)] = scale
        locations = np.flatnonzero(selected)
        scaled_v[locations, locations] *= scale**2
    block_results = []
    for block in blocks:
        local = np.flatnonzero(np.isin(indices, block))
        measurement_v = reduced_v[np.ix_(local, local)]
        expected_residual_v = measurement_v - (1-1e-5)*prediction_covariance[np.ix_(local, local)]
        correction_precision = np.linalg.inv(expected_residual_v)
        eigenvalues, eigenvectors = np.linalg.eigh(correction_precision)
        pull = (eigenvectors*np.sqrt(np.maximum(eigenvalues, 0)))@eigenvectors.T@residual[local]
        magnitude = np.sqrt(max(0., residual[local]@correction_precision@residual[local]))
        norm = np.linalg.norm(pull)
        if magnitude > 1.001 and norm > 0:
            direction = pull/norm
            errors = np.sqrt(np.diag(measurement_v))
            along = (direction/errors)@measurement_v@(direction/errors)
            measurement_v = measurement_v + np.outer(direction*errors, direction*errors)*(magnitude**2-1)*along
            scaled_v[np.ix_(local, local)] = measurement_v
        block_results.append({'indices': list(block), 'pull_magnitude': magnitude})
    final, final_covariance, final_residual, _ = _gls(reduced_y, reduced_x, scaled_v)
    return {'parameters': original, 'covariance': final_covariance,
            'unscaled_covariance': original_covariance, 'refitted_parameters': final,
            'before_scaling_parameters': before, 'retained': keep,
            'node_scales': node_scales, 'correlated_blocks': block_results,
            'measurement_covariance': scaled_v,
            'q_final': final_residual@np.linalg.solve(scaled_v, final_residual)}
