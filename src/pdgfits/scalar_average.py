"""Direct one-parameter averages, including asymmetric objective components."""

import numpy as np
from scipy.optimize import brentq, minimize_scalar


def scalar_average(values, lower, upper, precision, grid_size=257):
    values, lower, upper, precision = map(lambda a: np.asarray(a, float), (values, lower, upper, precision))
    shift = float(np.median(values))
    scale = float(np.median((lower+upper)/2))
    y, en, ep = (values-shift)/scale, lower/scale, upper/scale
    independent = np.array_equal(precision, np.eye(len(y)))
    calls = 0

    def objective(theta):
        nonlocal calls
        calls += np.size(theta)
        r = y-np.asarray(theta)[..., None]
        errors = np.clip((2*en*ep-r*(ep-en))/(en+ep), np.minimum(en, ep), np.maximum(en, ep))
        z = r/errors
        return np.sum(z*z, axis=-1) if independent else np.sum((z@precision)*z, axis=-1)

    def quadratic(errors):
        weight = precision/np.outer(errors, errors)
        variance = 1/weight.sum()
        return float(np.sum(weight@y)*variance), np.sqrt(variance)

    if np.array_equal(en, ep):
        mean, se = quadratic(en)
        qmin = float(objective(mean))
        roots = np.array([mean-se, mean+se])
    else:
        left_mean, left_se = quadratic(en)
        right_mean, right_se = quadratic(ep)
        lo = min(float(np.min(y-en)), left_mean)-4*left_se
        hi = max(float(np.max(y+ep)), right_mean)+4*right_se
        knots = np.unique(np.r_[lo, y-en, y, y+ep, hi])
        grid = np.unique(np.r_[knots, np.linspace(lo, hi, grid_size)])
        q = objective(grid)
        candidates = [(float(q.min()), float(grid[q.argmin()]))]
        pieces = list(zip(knots[:-1], knots[1:]))
        pieces.extend((grid[i-1], grid[i+1]) for i in range(1, len(grid)-1)
                      if q[i] <= min(q[i-1], q[i+1]))
        for left, right in pieces:
            fit = minimize_scalar(objective, bounds=(left, right), method='bounded',
                                  options={'xatol': 1e-12})
            if not fit.success:
                raise RuntimeError(f'Scalar average did not converge: {fit.message}')
            candidates.append((float(fit.fun), float(fit.x)))
        qmin, mean = min(candidates)
        while objective(lo) <= qmin+1:
            lo = mean+2*(lo-mean)
        while objective(hi) <= qmin+1:
            hi = mean+2*(hi-mean)
        grid = np.unique(np.r_[lo, grid, hi, [x for _, x in candidates]])
        difference = objective(grid)-qmin-1
        roots = []
        for i in range(len(grid)-1):
            if difference[i] == 0:
                roots.append(grid[i])
            if difference[i]*difference[i+1] < 0:
                roots.append(brentq(lambda t: float(objective(t)-qmin-1), grid[i], grid[i+1], xtol=1e-12))
        roots = np.array(sorted(set(roots)))
    if len(roots) < 2 or len(roots) % 2:
        raise RuntimeError('Could not resolve the scalar profile confidence set')
    components = roots.reshape(-1, 2)
    containing = [pair for pair in components if pair[0] <= mean <= pair[1]]
    if len(containing) != 1:
        raise RuntimeError('Scalar minimum is not in exactly one confidence component')
    left, right = containing[0]
    diagnostics = {'target': qmin+1, 'target_value': shift+scale*mean,
                   'search_method': 'piecewise-scalar', 'function_evals': calls, 'cache_hits': 0,
                   'components': (shift+scale*components).tolist()}
    for side, endpoint in [('lower', left), ('upper', right)]:
        q = float(objective(endpoint))
        if abs(q-qmin-1) > 5e-6:
            raise RuntimeError('Scalar endpoint failed its objective check')
        diagnostics.update({f'{side}_endpoint': shift+scale*endpoint,
                            f'{side}_error': scale*abs(endpoint-mean),
                            f'{side}_chi2': q, f'{side}_residual': q-qmin-1,
                            f'{side}_is_bound': False,
                            f'{side}_profile_point': {'success': True, 'chi2': q,
                                                      'value': shift+scale*endpoint, 'method': 'direct'}})
    return {'value': shift+scale*mean, 'q': qmin,
            'error_n': scale*(mean-left), 'error_p': scale*(right-mean), 'diagnostics': diagnostics}
