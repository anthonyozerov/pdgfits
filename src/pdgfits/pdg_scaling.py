"""Reference implementations of documented PDG scale procedures.

The average follows savg.f, including its asymmetric iteration. The linear
reference follows sscafac.f/sbrfit.f with fixed symmetric errors (one scale
pass, threshold 1.001, epsilon 1e-5). The general wrapper applies that scaling
pass to the Python mean-fit objective; it does not port every legacy Fortran
asymmetric-error propagation step.
"""

import numpy as np

from pdgfits.corr_mat import correlation_blocks


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
    blocks = correlation_blocks(v, correlated_blocks)
    block_ids = np.full(len(y), -1, int)
    for b, indices in enumerate(blocks):
        block_ids[indices] = b
    keep = np.ones(len(y), bool)
    if exclude_weak:
        for node in dict.fromkeys(nodes):
            selected = (nodes == node) & (block_ids < 0)
            cutoff = 3*np.sqrt(selected.sum()*np.diag(fitted_covariance)[selected])
            keep[selected] = np.sqrt(np.diag(v)[selected]) <= cutoff
    indices = np.flatnonzero(keep)
    reduced_y, reduced_x, reduced_v = y[keep], x[keep], v[np.ix_(keep, keep)]
    before, _, residual, prediction_covariance = _gls(reduced_y, reduced_x, reduced_v)
    reduced_blocks = [np.flatnonzero(np.isin(indices, block)).tolist() for block in blocks]
    scaled_v, node_scales, block_results = _pull_update(
        reduced_y, reduced_v, residual, prediction_covariance, nodes[keep], block_ids[keep], reduced_blocks)
    for row, block in zip(block_results, blocks):
        row['indices'] = list(block)
    final, final_covariance, final_residual, _ = _gls(reduced_y, reduced_x, scaled_v)
    return {'parameters': original, 'covariance': final_covariance,
            'unscaled_covariance': original_covariance, 'refitted_parameters': final,
            'before_scaling_parameters': before, 'retained': keep,
            'node_scales': node_scales, 'correlated_blocks': block_results,
            'measurement_covariance': scaled_v,
            'q_final': final_residual@np.linalg.solve(scaled_v, final_residual)}


def _pull_update(y, v, residual, prediction_covariance, nodes, block_ids, blocks):
    node_scales = {}
    scaled_v = v.copy()
    for node in dict.fromkeys(nodes):
        selected = (nodes == node) & (block_ids < 0)
        if not selected.any():
            continue
        denominator = np.diag(v)[selected] - (1-1e-5)**2*np.diag(prediction_covariance)[selected]
        pulls = residual[selected]**2/denominator
        magnitude = np.sqrt(np.sum(np.maximum(pulls, 0))/selected.sum())
        scale = magnitude if magnitude > 1.001 else 1.
        node_scales[str(node)] = scale
        locations = np.flatnonzero(selected)
        scaled_v[locations, locations] *= scale**2
    block_results = []
    for block in blocks:
        local = np.flatnonzero(np.isin(np.arange(len(y)), block))
        measurement_v = v[np.ix_(local, local)]
        measurement_precision = np.linalg.pinv(measurement_v)
        # The Fortran form also accepts exactly dependent summaries: it does
        # not require an inverse of their singular measurement covariance.
        correction_precision = measurement_precision@np.linalg.inv(
            np.eye(len(local))-(1-1e-5)*prediction_covariance[np.ix_(local, local)]@measurement_precision)
        correction_precision = (correction_precision+correction_precision.T)/2
        square = float(residual[local]@correction_precision@residual[local])
        if square < 0:
            block_results.append({'indices': list(block), 'pull_magnitude': 0., 'invalid_pull': True})
            continue
        eigenvalues, eigenvectors = np.linalg.eigh(correction_precision)
        pull = (eigenvectors*np.sqrt(np.maximum(eigenvalues, 0)))@eigenvectors.T@residual[local]
        magnitude = np.sqrt(max(0., residual[local]@correction_precision@residual[local]))
        norm = np.linalg.norm(pull)
        errors = np.sqrt(np.diag(measurement_v))
        dependent = np.linalg.eigvalsh(measurement_v/np.outer(errors, errors)).min() < 1e-10
        if magnitude > 1.001 and norm > 0:
            if dependent:
                # SBRFIT uses a common factor for a dependency block, retaining
                # the exact relation between its reported summaries.
                measurement_v = measurement_v*magnitude**2
            else:
                direction = pull/norm
                along = (direction/errors)@measurement_v@(direction/errors)
                measurement_v = measurement_v + np.outer(direction*errors, direction*errors)*(magnitude**2-1)*along
            scaled_v[np.ix_(local, local)] = measurement_v
        block_results.append({'indices': list(block), 'pull_magnitude': magnitude, 'dependent': bool(dependent)})
    return scaled_v, node_scales, block_results


def pdg_fit_scales(fit, exclude_weak=True):
    """Apply the PDG separate-pull scaling pass to a prepared general fit.

    The supplied Python Q defines the nonlinear/asymmetric mean fits and their
    local covariances. The scaling pass follows SSCAFAC/SBRFIT: select weak
    independent inputs once, refit, compute pulls, inflate, and refit. This
    isolates the scale prescription; it is not a port of Fortran's separate
    asymmetric-error propagation or every upstream input-selection convention.

    The result separates the original reported center from the final refit.
    Profile the latter and attach its errors to either center for a comparison.
    No additional precision cut is made when exclude_weak=False.
    """
    import jax
    import jax.numpy as jnp
    from pdgfits.build_chi2 import build_chi2
    from pdgfits.refit import physical_refit_model, prepare_refit

    original = fit
    fit = physical_refit_model(fit)
    fit = prepare_refit(fit)()
    data = fit['meas_df']
    nodes = data['node'].to_numpy()
    correlation = np.asarray(fit['corr_mat'])
    blocks = correlation_blocks(correlation, fit.get('correlation_blocks'))
    block_ids = np.full(len(data), -1, int)
    for i, block in enumerate(blocks):
        block_ids[block] = i

    def information(current):
        prediction = lambda fp: current['mu_adjust'](current['fitted_params_to_params'](fp))
        fp = jnp.asarray(current['fitted_values'])
        jacobian = np.asarray(jax.jacfwd(prediction)(fp))
        residual = current['meas_df']['value'].to_numpy()-np.asarray(prediction(fp))
        lower, upper = [current['meas_df'][key].to_numpy() for key in ['error_n', 'error_p']]
        errors = np.clip((2*lower*upper-residual*(upper-lower))/(lower+upper),
                          np.minimum(lower, upper), np.maximum(lower, upper))
        return residual, errors, jacobian@current['covariance']@jacobian.T

    def rebuild(current, retained, lower, upper, corr):
        answer = dict(current)
        indices = np.flatnonzero(retained)
        prediction = lambda p: current['mu_adjust'](p)[indices]
        measurements = current['meas_df'].iloc[indices].copy()
        measurements['error_n'], measurements['error_p'] = lower, upper
        q, gradient, _, opened = build_chi2(measurements['value'].to_numpy(), prediction,
            lower, upper, np.linalg.pinv(corr), current['fitted_params_to_params'])
        answer.update(meas_df=measurements, mu_adjust=prediction, corr_mat=corr,
                      chi2=q, chi2_grad=gradient, chi2_open=opened)
        if 'correlation_blocks' in current:
            positions = {old: new for new, old in enumerate(indices)}
            blocks = [[positions[i] for i in block if i in positions]
                      for block in current['correlation_blocks']]
            answer['correlation_blocks'] = [block for block in blocks if len(block) > 1]
        answer.pop('input_scales', None)
        return prepare_refit(answer)()

    residual, errors, prediction_covariance = information(fit)
    keep = np.ones(len(data), bool)
    if exclude_weak:
        for node in dict.fromkeys(nodes):
            selected = (nodes == node) & (block_ids < 0)
            cutoff = 3*np.sqrt(selected.sum()*np.maximum(np.diag(prediction_covariance)[selected], 0))
            keep[selected] = errors[selected] <= cutoff
    if not keep.all():
        fit = rebuild(fit, keep, data['error_n'].to_numpy()[keep], data['error_p'].to_numpy()[keep],
                      correlation[np.ix_(keep, keep)])
        residual, errors, prediction_covariance = information(fit)
    indices = np.flatnonzero(keep)
    local_blocks = [np.flatnonzero(np.isin(indices, block)).tolist() for block in blocks]
    covariance = correlation[np.ix_(keep, keep)]*np.outer(errors, errors)
    scaled, node_scales, block_results = _pull_update(data['value'].to_numpy()[keep], covariance,
        residual, prediction_covariance, nodes[keep], block_ids[keep], local_blocks)
    scaled_errors = np.sqrt(np.diag(scaled))
    factors = scaled_errors/errors
    result = rebuild(fit, np.ones(keep.sum(), bool), fit['meas_df']['error_n'].to_numpy()*factors,
                     fit['meas_df']['error_p'].to_numpy()*factors, scaled/np.outer(scaled_errors, scaled_errors))
    return {'original_fit': original, 'refitted_fit': result, 'retained': keep,
            'node_scales': node_scales, 'correlated_blocks': block_results,
            'measurement_scales': factors, 'scope': 'PDG scale pass on Python Q mean fits'}
