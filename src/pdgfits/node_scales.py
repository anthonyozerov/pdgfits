"""Node scales for nonlinear/asymmetric fits, using refitted simulation scores.

The working sampling density is proportional to exp(-Q/2), where Q is the
existing interpolated-error objective in the original measurement units. For
full-rank inputs its normalization changes by product(s_i). Exactly dependent
summaries instead stay on a fixed physical measurement plane. In either case
the unknown normalizing derivative cancels between observed and expected scores.
This declares a sampling model; quoted asymmetric intervals alone do not do so.
"""

import jax
import jax.numpy as jnp
import numpy as np
from scipy.linalg import null_space
from scipy.optimize import brentq, least_squares, lsq_linear, root, minimize, LinearConstraint

from pdgfits.pdg_scaling import _correlated_blocks
from pdgfits.refit import physical_refit_model, prepare_refit


def sample_errors(error_n, error_p, correlation, draws, seed=0):
    """Independent exact rejection draws from the normalized PDG Q kernel.

    Write z=r/sigma(r). In the middle segment, r=a*z/(1+b*z),
    so its Jacobian is a/(1+b*z)^2. Reweighting a correlated normal
    draw by this Jacobian samples exp(-Q(r)/2) in measurement units.
    Independent correlation blocks are sampled separately for efficiency.
    Symmetric errors recover precisely a multivariate normal distribution.
    """
    en, ep, corr = map(lambda a: np.asarray(a, float), (error_n, error_p, correlation))
    if en.ndim != 1 or ep.shape != en.shape or corr.shape != (len(en), len(en)):
        raise ValueError('Error and correlation shapes do not agree')
    if not all(np.isfinite(a).all() for a in (en, ep, corr)) or np.any(en <= 0) or np.any(ep <= 0):
        raise ValueError('Finite positive errors and finite correlations are required')
    if not np.allclose(corr, corr.T) or not np.allclose(np.diag(corr), 1):
        raise ValueError('A symmetric correlation matrix with unit diagonal is required')
    np.linalg.cholesky(corr)  # A singular Q kernel is not a density on R^n.
    if draws < 2:
        raise ValueError('At least two simulation draws are required')
    blocks = _correlated_blocks(corr)
    assigned = {i for block in blocks for i in block}
    blocks.extend([i] for i in range(len(en)) if i not in assigned)
    rng = np.random.default_rng(seed)
    output = np.empty((draws, len(en)))
    for block in blocks:
        lower, upper = en[block], ep[block]
        a = 2*lower*upper/(lower+upper)
        b = (upper-lower)/(upper+lower)
        maximum_jacobian = np.maximum(np.maximum(lower, upper), a/(1-np.abs(b))**2)
        chol = np.linalg.cholesky(corr[np.ix_(block, block)])
        accepted, proposed = 0, 0
        while accepted < draws:
            count = min(65536, max(256, 4*(draws-accepted)))
            z = rng.standard_normal((count, len(block))) @ chol.T
            middle = np.clip(z, -1, 1)
            jacobian = np.where(z < -1, upper,
                                np.where(z > 1, lower, a/(1+b*middle)**2))
            keep = np.log(rng.random(count)) < np.sum(np.log(jacobian/maximum_jacobian), axis=1)
            z = z[keep][:draws-accepted]
            r = np.where(z < -1, upper*z,
                         np.where(z > 1, lower*z, a*z/(1+b*np.clip(z, -1, 1))))
            output[accepted:accepted+len(z), block] = r
            accepted += len(z)
            proposed += count
            if proposed > 2_000_000 and accepted < draws:
                raise RuntimeError(f'Asymmetric sampling acceptance is too low for block {block}')
    return output


def _error_sampler(fit, draws, seed):
    """Keep exactly dependent summaries on their physical measurement plane.

    Nonsingular blocks use the exact sampler above. A singular Gaussian block
    is sampled on its supported plane. For an asymmetric singular block,
    Student proposals on that plane give importance-weighted expectations;
    their effective sample size and Monte Carlo error remain explicit.
    """
    data = fit['meas_df']
    lower, upper = data['error_n'].to_numpy(), data['error_p'].to_numpy()
    correlation = np.asarray(fit['corr_mat'])
    rng = np.random.default_rng(seed)
    singular, removed = [], set()
    center = jnp.asarray(fit['fitted_values'])
    predict = lambda fp: fit['mu_adjust'](fit['fitted_params_to_params'](fp))
    for block in _correlated_blocks(correlation):
        corr = correlation[np.ix_(block, block)]
        eigenvalues, eigenvectors = np.linalg.eigh(corr)
        if eigenvalues[0] > 1e-10:
            continue
        if eigenvalues[0] < -1e-10:
            raise ValueError('The measurement correlation matrix is not positive semidefinite')
        selected = eigenvalues > 1e-10
        rank = int(selected.sum())
        reference = (lower[block]+upper[block])/2
        asymmetric = np.any(lower[block] != upper[block])
        basis = reference[:, None]*eigenvectors[:, selected]
        if asymmetric:
            # The unequal lower/upper errors do not define a unique linear
            # standardization. Recover the exact raw relation from the model.
            jacobian = np.asarray(jax.jacfwd(predict)(center))[block]/reference[:, None]
            units = np.linalg.norm(jacobian, axis=0)
            jacobian /= np.where(units > 0, units, 1.)
            u, s, _ = np.linalg.svd(jacobian, full_matrices=False)
            if int(np.sum(s > 1e-10*s[0])) != rank:
                raise ValueError('An asymmetric singular block needs explicit independent measurement coordinates')
            basis = reference[:, None]*u[:, :rank]
        standardized = basis/reference[:, None]
        null = np.eye(len(block))-standardized@np.linalg.pinv(standardized)
        nominal_prediction = np.asarray(predict(center))[block]
        observed = (data['value'].to_numpy()[block]-nominal_prediction)/reference
        if np.max(np.abs(null@observed)) > 1e-7:
            raise ValueError('Exactly correlated summaries violate their raw measurement relation')
        parameter_units = (np.sqrt(np.maximum(np.diag(fit['covariance']), 0))
                           if fit['covariance'] is not None else np.maximum(np.abs(center), .01))
        for direction in [np.ones(len(center)), np.sin(np.arange(len(center))+1)]:
            difference = (np.asarray(predict(center+parameter_units*direction))[block]-nominal_prediction)/reference
            if np.max(np.abs(null@difference)) > 1e-7*max(np.linalg.norm(difference), 1.):
                raise ValueError('The model does not preserve a singular block measurement relation')
        normal = rng.standard_normal((draws, rank))
        if asymmetric:
            normal /= np.sqrt(rng.chisquare(5, draws)/5)[:, None]
        singular.append((block, basis, np.linalg.pinv(corr), normal, asymmetric))
        removed.update(block)
    ordinary = np.array([i for i in range(len(data)) if i not in removed], int)
    ordinary_seed = int(rng.integers(0, 2**63-1)) if singular else seed
    base = sample_errors(lower[ordinary], upper[ordinary], correlation[np.ix_(ordinary, ordinary)], draws, ordinary_seed)

    def sample(scales):
        errors = np.zeros((draws, len(data)))
        errors[:, ordinary] = scales[ordinary]*base
        log_weight = np.zeros(draws)
        for block, basis, precision, normal, asymmetric in singular:
            en, ep = scales[block]*lower[block], scales[block]*upper[block]
            reference = 2*en*ep/(en+ep)
            standardized = basis/reference[:, None]
            covariance = np.linalg.inv(standardized.T@precision@standardized)
            coordinates = normal@np.linalg.cholesky(covariance).T
            residual = coordinates@basis.T
            errors[:, block] = residual
            if asymmetric:
                sigma = np.clip((2*en*ep-residual*(ep-en))/(en+ep), np.minimum(en, ep), np.maximum(en, ep))
                z = residual/sigma
                log_weight += -.5*np.sum((z@precision)*z, axis=1) + (5+basis.shape[1])/2*np.log1p(np.sum(normal**2, axis=1)/5)
        weights = np.exp(log_weight-log_weight.max())
        weights /= weights.sum()
        effective = 1/(weights@weights)
        if effective < min(32, draws/4):
            raise RuntimeError('Too few effective simulation draws for a singular asymmetric block; increase draws')
        return errors, weights, effective
    return sample, [block for block, *_ in singular]


def fit_node_scales(fit, *, draws=256, seed=0, tolerance=1e-3, mc_tolerance=0., max_iterations=60, verbose=False):
    """Fit separate node scales >=1 by balancing observed and expected scores.

    At each scale vector: refit the real data; simulate from that fitted model;
    refit every simulated dataset; subtract the mean simulated scale score from
    the observed one. Solve the bounded score equations, including the floor
    at one. Known correlations and nonlinear measurement maps are retained.

    This is a parametric-bootstrap estimating equation, not an exact general
    REML likelihood. Linear Gaussian expected scores recover the REML equations
    as the simulation size grows. Common simulation draws make the outer
    equation deterministic. Returned profile intervals condition on the scales;
    scale uncertainty and frequentist interval calibration remain separate.
    No automatic weak-input cut is applied to the supplied measurement list.
    The default tolerance checks the strict finite-simulation equations.
    Expected-score simulation errors are reported separately. An explicit
    nonzero mc_tolerance allows that fraction of a simulation standard error;
    both the raw and remaining numerical residuals are saved.
    """
    if tolerance <= 0 or not 0 <= mc_tolerance <= 1 or max_iterations < 1:
        raise ValueError('Require positive tolerance/iterations and mc_tolerance in [0, 1]')
    fit = physical_refit_model(fit)
    data = np.asarray(fit['meas_df']['value'], float)
    labels = np.asarray(fit['meas_df']['node'])
    nodes = list(dict.fromkeys(labels))
    group = np.array([nodes.index(node) for node in labels])
    counts = np.bincount(group)
    sample, singular_blocks = _error_sampler(fit, draws, seed)
    refit = prepare_refit(fit)

    def contribution(fp, values, log_scales):
        return -.5*jax.grad(lambda logs: fit['chi2_open'](fp, values, jnp.exp(logs[group])))(log_scales)

    contributions = jax.jit(jax.vmap(contribution, in_axes=(0, 0, None)))
    observed_contribution = jax.jit(contribution)
    predict = jax.jit(lambda fp: fit['mu_adjust'](fit['fitted_params_to_params'](fp)))
    predict_jacobian = jax.jit(jax.jacfwd(predict))
    lower = fit['meas_df']['error_n'].to_numpy()
    upper = fit['meas_df']['error_p'].to_numpy()
    a, b = 2*lower*upper/(lower+upper), (upper-lower)/(lower+upper)
    precision = np.linalg.pinv(fit['corr_mat'])
    units = np.linalg.norm(np.asarray(predict_jacobian(fit['fitted_values']))/((lower+upper)/2)[:, None], axis=0)
    units = np.maximum(units, 1e-100)
    history, cache = [], {}

    def kink_score(fitted, values, scales, fallback, expected=None):
        residual = values-np.asarray(predict(fitted))
        sigma = np.clip(scales*a-b*residual, scales*np.minimum(lower, upper), scales*np.maximum(lower, upper))
        z = residual/sigma
        active = (lower != upper) & (np.abs(np.abs(z)-1) < 2e-7)
        if not active.any():
            return fallback
        weights = precision @ z
        derivative = np.where(residual < -scales*upper, 1/(scales*upper),
                               np.where(residual > scales*lower, 1/(scales*lower), scales*a/sigma**2))
        inside = scales[active]*a[active]/sigma[active]**2
        outside = 1/sigma[active]
        midpoint = (inside+outside)/2
        halfwidth = np.abs(inside-outside)/2
        derivative[active] = midpoint
        jacobian = np.asarray(predict_jacobian(fitted))/units
        matrix = jacobian[active].T*(weights[active]*halfwidth)
        gradient = jacobian.T @ (weights*derivative)
        # At an error-rule knot, the mean optimum uses a convex combination
        # of the two one-sided derivatives. Solve that stationarity condition;
        # an arbitrary AD branch is not the derivative of the profiled Q.
        lower_bounds, upper_bounds = -np.ones(active.sum()), np.ones(active.sum())
        constraint = fit.get('mean_constraint')
        if constraint is not None:
            at = constraint.A@fitted
            normals = np.vstack([-constraint.A[at-constraint.lb < 1e-7],
                                  constraint.A[constraint.ub-at < 1e-7]])
            matrix = np.column_stack([matrix, -(normals/units).T])
            lower_bounds = np.r_[lower_bounds, np.zeros(len(normals))]
            upper_bounds = np.r_[upper_bounds, np.full(len(normals), np.inf)]
        mixture = lsq_linear(matrix, -gradient, bounds=(lower_bounds, upper_bounds), tol=1e-12)
        if expected is not None:
            # At a simultaneous mean boundary and error-rule knot, mean KKT
            # alone may leave a range of valid profile-scale derivatives.
            # Choose the admissible derivative closest to the expected score,
            # retaining the mean stationarity equation exactly.
            freedom = null_space(matrix)
            if freedom.shape[1]:
                score_matrix = np.zeros((len(nodes), len(mixture.x)))
                score_matrix[group[active], np.arange(active.sum())] = residual[active]*weights[active]*halfwidth
                middle = np.bincount(group, weights=residual*weights*derivative, minlength=len(nodes))
                direction = score_matrix@freedom
                base = middle+score_matrix@mixture.x-expected
                at_floor = np.array([scales[group == i][0] <= 1+1e-9 for i in range(len(nodes))])
                def objective(t):
                    score = base+direction@t
                    score = np.where(at_floor, np.maximum(score, 0), score)/(2*counts)
                    return float(score@score), 2*direction.T@(score/(2*counts))
                adjusted = minimize(objective, np.zeros(freedom.shape[1]), jac=True, method='SLSQP',
                    constraints=LinearConstraint(freedom, lower_bounds-mixture.x, upper_bounds-mixture.x),
                    options={'ftol': 1e-14, 'maxiter': 100})
                if adjusted.success:
                    mixture.x = mixture.x+freedom@adjusted.x
        derivative[active] += halfwidth*mixture.x[:active.sum()]
        return np.bincount(group, weights=residual*weights*derivative, minlength=len(nodes))

    def evaluate(log_scales):
        logs = np.maximum(np.asarray(log_scales, float), 0)
        key = tuple(logs)
        if key in cache:
            return cache[key]
        if not np.isfinite(logs).all() or np.max(logs) > 15:
            raise RuntimeError('Node-scale iteration diverged')
        scales = np.exp(logs[group])
        observed = refit(scales=scales, full_output=False)
        errors, weights, effective = sample(scales)
        simulated = np.asarray(predict(observed['fitted_values'])) + errors
        fitted = refit.batch(simulated, scales, start=observed['x'])
        sampled = np.asarray(contributions(fitted, simulated, logs))
        actual = np.asarray(observed_contribution(observed['fitted_values'], data, logs))
        sampled = np.array([kink_score(r, y, scales, score)
                            for r, y, score in zip(fitted, simulated, sampled)])
        expected = weights@sampled
        actual = kink_score(observed['fitted_values'], data, scales, actual, expected=expected)
        score = actual-expected
        row = {'scales': np.exp(logs), 'observed': actual, 'expected': expected,
               'expected_mc_se': np.sqrt(np.sum(weights[:, None]**2*(sampled-expected)**2, axis=0)*effective/(effective-1)),
               'effective_draws': effective, 'score': score, 'fit': observed}
        row['scalar_fallbacks'] = refit.batch.last_fallbacks
        cache[key] = row
        if len(cache) > 128:
            del cache[next(iter(cache))]
        history.append({k: v for k, v in row.items() if k != 'fit'})
        if verbose and len(history) % 25 == 0:
            print(f"Node-scale evaluation {len(history)}: largest scale {np.exp(logs).max():.4g}; "
                  f"score residual {excess(row):.3g}; "
                  f"scalar refit fallbacks {row['scalar_fallbacks']}/{draws}", flush=True)
        return row

    def equation(log_scales, account_for_mc=True):
        row = evaluate(log_scales)
        # Fixed point of a projected score step. At the solution an inflated
        # node has zero adjusted score; a node at one may have a negative score.
        score = row['score']
        if account_for_mc:
            score = np.sign(score)*np.maximum(np.abs(score)-mc_tolerance*row['expected_mc_se'], 0)
        step = np.arcsinh(score/(2*counts))
        return np.maximum(log_scales + step, 0)-log_scales

    def excess(row):
        score = np.sign(row['score'])*np.maximum(np.abs(row['score'])-mc_tolerance*row['expected_mc_se'], 0)
        point = np.log(row['scales'])
        return np.max(np.abs(np.maximum(point+np.arcsinh(score/(2*counts)), 0)-point))

    baseline = evaluate(np.zeros(len(nodes)))
    initial = np.zeros(len(nodes))
    positive = ((baseline['expected'] > 1e-8*counts)
                & (baseline['score'] > 2*counts*tolerance))
    initial[positive] = .5*np.log(baseline['observed'][positive]/baseline['expected'][positive])
    logs = initial
    if np.max(np.abs(equation(logs))) > tolerance:
        # Most fits need only a few inexpensive joint score updates.
        try:
            trial = root(equation, logs, method='broyden1',
                         options={'fatol': tolerance, 'maxiter': min(10, max_iterations),
                                  'jac_options': {'alpha': .5}})
            logs = np.maximum(trial.x, 0)
        except RuntimeError as exc:
            if str(exc) != 'Node-scale iteration diverged':
                raise
        if np.max(np.abs(equation(logs))) > tolerance:
            logs = np.log(min(history, key=excess)['scales'])
    if np.max(np.abs(equation(logs))) > tolerance:
        # Bounded joint steps avoid chasing an almost uninformative node to a
        # huge scale while other nodes are still adjusting. Scalar bracketing
        # below remains useful at nonsmooth corners.
        # A scale already satisfying its floor condition needs no numerical
        # Jacobian column. All node scores are still checked after the update;
        # the following coordinate pass can release a newly active floor.
        free = np.flatnonzero((logs > 0) | (equation(logs) > tolerance))
        fixed = logs.copy()
        def expand(values):
            point = fixed.copy()
            point[free] = values
            return point
        class Balanced(Exception):
            pass
        def checked_equation(values):
            point = expand(values)
            value = equation(point)
            if np.max(np.abs(value)) <= tolerance:
                raise Balanced(point.copy())
            return value
        def jacobian(values):
            # A relative step becomes tiny near the scale floor and resolves
            # inner-fit numerical error instead of the scale-score derivative.
            # Work in log-scale units, with the same finite step near zero.
            base = checked_equation(values)
            columns = []
            for i in range(len(values)):
                point = values.copy()
                step = 1e-3*max(1., abs(values[i]))
                point[i] += step
                columns.append((checked_equation(point)-base)/step)
            return np.column_stack(columns)
        try:
            trial = least_squares(checked_equation, np.maximum(logs[free], 1e-6), bounds=(0, np.inf),
                                  jac=jacobian, ftol=1e-6, xtol=1e-6, gtol=1e-6,
                                  max_nfev=max_iterations)
            logs = expand(trial.x)
        except Balanced as completed:
            logs = completed.args[0]
        except RuntimeError as exc:
            if str(exc) != 'Node-scale iteration diverged':
                raise
        if np.max(np.abs(equation(logs))) > tolerance:
            logs = np.log(min(history, key=excess)['scales'])
    if np.max(np.abs(equation(logs))) > tolerance:
        # A bounded coordinate solve avoids a nearly singular joint Jacobian
        # when a node has almost no residual information. Each update balances
        # one node after refitting the complete coupled model.
        logs = np.maximum(logs, 0.)
        for _ in range(max_iterations):
            for i in np.argsort(-np.abs(equation(logs))):
                if abs(equation(logs)[i]) <= tolerance:
                    continue
                def score(value):
                    point = logs.copy()
                    point[i] = value
                    return evaluate(point)['score'][i]
                # Bracket from the current scale. Testing the floor first can
                # undo nearly all inflation and make a near-solved mean fit
                # unnecessarily difficult. Expand only in the needed direction.
                bracket_lower = bracket_upper = logs[i]
                step = .25
                if score(logs[i]) > 0:
                    while score(bracket_upper) > 0 and bracket_upper < 10:
                        bracket_upper = min(10., bracket_upper+step)
                        step *= 2
                    if score(bracket_upper) > 0:
                        continue
                else:
                    while score(bracket_lower) < 0 and bracket_lower > 0:
                        bracket_lower = max(0., bracket_lower-step)
                        step *= 2
                    if score(bracket_lower) <= 0:
                        logs[i] = 0.
                        continue
                logs[i] = brentq(score, bracket_lower, bracket_upper, xtol=1e-7)
            if np.max(np.abs(equation(logs))) <= tolerance:
                break
            if verbose:
                print(f'Node-scale pass {_+1}: excess score residual {np.max(np.abs(equation(logs))):.4g}', flush=True)
    at_floor = (logs < tolerance) & (evaluate(logs)['score'] <= 0)
    logs[at_floor] = 0.
    residual = float(np.max(np.abs(equation(logs))))
    if residual > tolerance:
        failure = RuntimeError(f'Node-scale score equations did not converge after {max_iterations} passes; residual={residual:.3g}')
        failure.history = history
        raise failure
    final = evaluate(logs)
    result = refit(scales=final['scales'][group])
    result['node_scaling'] = {'method': 'refitted-simulation-score', 'sampling_model': 'normalized-pdg-Q',
                              'nodes': nodes, 'scales': final['scales'],
                              'observed': final['observed'], 'expected': final['expected'],
                              'expected_mc_se': final['expected_mc_se'],
                              'score_residual': residual, 'draws': draws, 'seed': seed,
                              'raw_score_residual': float(np.max(np.abs(equation(logs, account_for_mc=False)))),
                              'mc_tolerance': mc_tolerance, 'tolerance': tolerance,
                              'effective_draws': final['effective_draws'], 'singular_blocks': singular_blocks,
                              'evaluations': len(history), 'history': history}
    return result
