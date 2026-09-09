"""Repeated least-squares fits of an already prepared measurement model."""

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import least_squares, lsq_linear, minimize, LinearConstraint, OptimizeResult
from pdgfits.build_chi2 import build_chi2
from pdgfits.param_maps import covariance_factor, physical_coordinates


def physical_refit_model(fit):
    """Keep the same Q while making exact physical boundaries accessible."""
    chart = physical_coordinates(fit)
    if chart is None:
        return fit
    center, transform, bounds = chart
    mapping = jax.jit(lambda z: jnp.asarray(center)+jnp.asarray(transform)@z)
    data = fit['meas_df']
    q, gradient, _, opened = build_chi2(data['value'].to_numpy(), fit['mu_adjust'],
        data['error_n'].to_numpy(), data['error_p'].to_numpy(), np.linalg.pinv(fit['corr_mat']), mapping)
    answer = dict(fit)
    answer.update(fitted_values=np.zeros(transform.shape[1]), covariance=np.eye(transform.shape[1]),
                  fitted_params_to_params=mapping, params_to_fitted_params=lambda p: np.linalg.pinv(transform)@(p-center),
                  chi2=q, chi2_grad=gradient, chi2_open=opened, mean_constraint=bounds)
    return answer


def prepare_refit(fit):
    """Compile once; refit new data and input-error scales without re-querying.

    The physical parameter map, asymmetric interpolation and correlations are
    inherited from the original fit. Returned covariances condition on the
    supplied scales. This is also usable for averages with nuisance parameters.
    """
    center = jnp.asarray(fit['fitted_values'])
    data = np.asarray(fit['meas_df']['value'], float)
    precision = np.linalg.pinv(fit['corr_mat']) if 'corr_mat' in fit else None
    covariance = fit['covariance']
    if covariance is None:
        # Column norms of the residual Jacobian provide units even when the
        # observed Hessian at the starting point is not positive definite.
        jacobian = np.asarray(fit['chi2_open'].residual_value_jac(center, data, np.ones(len(data)))[1])
        norms = np.linalg.norm(jacobian, axis=0)
        if np.any(norms == 0):
            raise ValueError('A mean parameter has no local measurement sensitivity')
        transform = jnp.diag(1/norms)
    else:
        transform = jnp.asarray(covariance_factor(covariance))

    # Keep this affine change outside the compiled model. Shared average
    # kernels then remain shared across nodes, rather than recompiling each
    # node's fixed center and scale matrix into a new closure.
    center, transform = np.asarray(center), np.asarray(transform)
    constraint = fit.get('mean_constraint')
    if constraint is not None:
        constraint = LinearConstraint(constraint.A@transform,
                                       constraint.lb-constraint.A@center,
                                       constraint.ub-constraint.A@center)
    def evaluate(z, values, scales):
        residual, jacobian = fit['chi2_open'].residual_value_jac(center + transform @ z, values, scales)
        return np.asarray(residual), np.asarray(jacobian) @ transform
    def hessian(z, values, scales):
        return transform.T @ np.asarray(fit['chi2_open'].hessian(center + transform @ z, values, scales)) @ transform
    def residual(z, values, scales):
        return fit['chi2_open'].residuals(jnp.asarray(center) + jnp.asarray(transform) @ z, values, scales)
    batch_objective = jax.jit(lambda z, values, scales:
                              jnp.sum(jax.vmap(residual, in_axes=(0, None, None))(z, values, scales)**2, axis=1))
    if 'mu_adjust' in fit:
        predict = lambda z: fit['mu_adjust'](fit['fitted_params_to_params'](jnp.asarray(center)+jnp.asarray(transform)@z))
        prediction_jacobian = jax.jit(lambda z: (predict(z), jax.jacfwd(predict)(z)))
    else:
        prediction_jacobian = None

    def refit(values=None, scales=None, start=None, full_output=True):
        values = data if values is None else np.asarray(values, float)
        scales = np.ones(len(values)) if scales is None else np.asarray(scales, float)
        if values.shape != data.shape or scales.shape != data.shape or np.any(scales <= 0):
            raise ValueError('New data and positive scales must match the measurement rows')
        if not np.isfinite(values).all() or not np.isfinite(scales).all():
            raise ValueError('Data and scales must be finite')
        initial = np.zeros(transform.shape[1]) if start is None else np.asarray(start, float)
        cache_x, cache_result = None, None
        evaluations = 0

        def compute(z):
            nonlocal cache_x, cache_result, evaluations
            if cache_x is None or not np.array_equal(z, cache_x):
                cache_x = np.array(z)
                cache_result = tuple(np.asarray(a) for a in evaluate(z, values, scales))
                evaluations += 1
            return cache_result

        def objective(z):
            r = compute(z)[0]
            return float(r@r)

        def projected_gradient(x, r, jac):
            gradient = jac.T@r
            if constraint is not None:
                values = constraint.A@x
                normals = np.vstack([-constraint.A[values-constraint.lb < 1e-7],
                                      constraint.A[constraint.ub-values < 1e-7]])
                if len(normals):
                    multipliers = lsq_linear(normals.T, -gradient, bounds=(0, np.inf), tol=1e-12)
                    gradient += normals.T@multipliers.x
            return gradient

        def optimality(x, r, jac):
            return float(np.max(np.abs(projected_gradient(x, r, jac))))

        def knot_optimality(x):
            """Check the convex one-sided gradients at an asymmetric knot."""
            if prediction_jacobian is None or precision is None:
                return np.inf
            prediction, jacobian = map(np.asarray, prediction_jacobian(x))
            raw = values-prediction
            en = scales*fit['meas_df']['error_n'].to_numpy()
            ep = scales*fit['meas_df']['error_p'].to_numpy()
            a = 2*en*ep/(en+ep)
            sigma = np.clip(a-raw*(ep-en)/(en+ep), np.minimum(en, ep), np.maximum(en, ep))
            z = raw/sigma
            active = (en != ep) & (np.abs(np.abs(z)-1) < 2e-7)
            if not active.any():
                return np.inf
            weights = precision@z
            derivative = np.where(raw < -ep, 1/ep, np.where(raw > en, 1/en, a/sigma**2))
            halfwidth = np.abs(a[active]/sigma[active]**2-1/sigma[active])/2
            derivative[active] = (a[active]/sigma[active]**2+1/sigma[active])/2
            gradient = -jacobian.T@(weights*derivative)
            matrix = -jacobian[active].T*(weights[active]*halfwidth)
            lo, hi = -np.ones(active.sum()), np.ones(active.sum())
            if constraint is not None:
                at = constraint.A@x
                normals = np.vstack([-constraint.A[at-constraint.lb < 1e-7],
                                      constraint.A[constraint.ub-at < 1e-7]])
                matrix = np.column_stack([matrix, normals.T])
                lo, hi = np.r_[lo, np.zeros(len(normals))], np.r_[hi, np.full(len(normals), np.inf)]
            mixture = lsq_linear(matrix, -gradient, bounds=(lo, hi), tol=1e-12)
            return float(np.max(np.abs(gradient+matrix@mixture.x)))

        if constraint is None:
            result = least_squares(lambda z: compute(z)[0], initial,
                                   jac=lambda z: compute(z)[1], method='trf',
                                   ftol=1e-11, xtol=1e-11, gtol=1e-8, max_nfev=1000)
        else:
            class MeanMinimum(Exception):
                pass
            def stop_at_mean(z):
                at = constraint.A@z
                if np.any(at < constraint.lb-1e-9) or np.any(at > constraint.ub+1e-9):
                    return
                r, jac = compute(z)
                if optimality(z, r, jac) < 1e-6 or knot_optimality(z) < 1e-6:
                    raise MeanMinimum(z.copy())
            try:
                result = minimize(objective, initial, jac=lambda z: 2*compute(z)[1].T@compute(z)[0],
                                  method='SLSQP', constraints=constraint, callback=stop_at_mean,
                                  options={'ftol': 1e-11, 'maxiter': 1000})
            except MeanMinimum as completed:
                result = OptimizeResult(x=completed.args[0], success=True, nfev=evaluations,
                                        message='Feasible stationary mean')
            result.fun, result.jac = compute(result.x)
            result.optimality = optimality(result.x, result.fun, result.jac)
        # A small step alone can signal stagnation. The scaled gradient must
        # also be small compared with the objective's numerical tolerance.
        descent_start = initial
        def descent_check(candidate):
            nonlocal descent_start
            gradient = projected_gradient(candidate.x, candidate.fun, candidate.jac)
            directions = np.vstack([np.eye(len(initial)), -np.eye(len(initial)),
                                     -gradient/max(np.linalg.norm(gradient), 1e-100)])
            steps = 10.**np.arange(-10, -1)
            trials = candidate.x + (steps[:, None, None]*directions[None, :, :]).reshape(-1, len(initial))
            feasible = np.ones(len(trials), bool)
            if constraint is not None:
                values_at_trial = trials@constraint.A.T
                feasible = np.all((values_at_trial >= constraint.lb)
                                   & (values_at_trial <= constraint.ub), axis=1)
            if not feasible.any():
                return 0.
            # Keep a fixed batch shape: slicing to the feasible subset would
            # compile another JAX program for every active-boundary pattern.
            trial_values = np.where(feasible, np.asarray(batch_objective(trials, values, scales)), np.inf)
            descent_start = trials[np.nanargmin(trial_values)]
            return float(candidate.fun @ candidate.fun - np.min(trial_values))

        descent = None
        if np.isfinite(result.fun).all() and result.optimality > 1e-4:
            descent = descent_check(result)
        if not result.success or (descent is not None and descent > 1e-7):
            # Scale changes can move a simulated fit far from the original
            # covariance chart. Recondition at the stalled point using the
            # residual Jacobian, keeping the same objective and constraints.
            origin = result.x.copy()
            norms = np.maximum(np.linalg.norm(result.jac, axis=0), 1e-100)
            _, singular, vectors = np.linalg.svd(result.jac/norms, full_matrices=False)
            local = (vectors.T/np.maximum(singular, 1e-12*singular[0]))/norms[:, None]
            local_constraint = (() if constraint is None else
                LinearConstraint(constraint.A@local, constraint.lb-constraint.A@origin,
                                 constraint.ub-constraint.A@origin))
            polished = minimize(lambda t: objective(origin+local@t), np.zeros(len(initial)),
                jac=lambda t: 2*local.T@compute(origin+local@t)[1].T@compute(origin+local@t)[0],
                method='SLSQP', constraints=local_constraint, options={'ftol': 1e-11, 'maxiter': 300})
            point = origin+local@polished.x
            if np.isfinite(polished.fun) and polished.fun <= result.fun@result.fun+1e-10:
                result.x = point
                result.fun, result.jac = compute(point)
                result.optimality = optimality(point, result.fun, result.jac)
                result.success, result.message = polished.success, polished.message
                result.nfev += polished.nfev
                descent = descent_check(result) if result.optimality > 1e-4 else None
        def polish_knots(result, descent):
            if descent is not None and prediction_jacobian is not None:
                # A derivative jump can pin a minimum to a quoted-error knot.
                # Optimize on the nearby knot, then check descent in the original
                # unconstrained objective before accepting it.
                prediction, _ = prediction_jacobian(result.x)
                raw = values-np.asarray(prediction)
                lower = scales*fit['meas_df']['error_n'].to_numpy()
                upper = scales*fit['meas_df']['error_p'].to_numpy()
                knot = np.where(raw > 0, lower, -upper)
                active = (lower != upper) & (np.abs(raw-knot) < 1e-3*np.minimum(lower, upper))
                if active.any():
                    target = (values-knot)[active]
                    units = np.minimum(lower, upper)[active]
                    equality = {'type': 'eq',
                                'fun': lambda z: (np.asarray(prediction_jacobian(z)[0])[active]-target)/units,
                                'jac': lambda z: np.asarray(prediction_jacobian(z)[1])[active]/units[:, None]}
                    constraints = [equality] if constraint is None else [constraint, equality]
                    class KnotMinimum(Exception):
                        pass
                    def stop_on_knot(z):
                        if np.max(np.abs(equality['fun'](z))) > 1e-9:
                            return
                        matrix = equality['jac'](z).T
                        lo, hi = np.full(active.sum(), -np.inf), np.full(active.sum(), np.inf)
                        if constraint is not None:
                            at = constraint.A@z
                            if np.any(at < constraint.lb-1e-9) or np.any(at > constraint.ub+1e-9):
                                return
                            normals = np.vstack([-constraint.A[at-constraint.lb < 1e-7],
                                                  constraint.A[constraint.ub-at < 1e-7]])
                            matrix = np.column_stack([matrix, normals.T])
                            lo, hi = np.r_[lo, np.zeros(len(normals))], np.r_[hi, np.full(len(normals), np.inf)]
                        r, jac = compute(z)
                        gradient = jac.T@r
                        multipliers = lsq_linear(matrix, -gradient, bounds=(lo, hi), tol=1e-12)
                        if np.max(np.abs(gradient+matrix@multipliers.x)) < 1e-5:
                            raise KnotMinimum(z.copy())
                    before = evaluations
                    try:
                        polished = minimize(objective, result.x, jac=lambda z: 2*compute(z)[1].T@compute(z)[0],
                                             method='SLSQP', constraints=constraints, callback=stop_on_knot,
                                             options={'ftol': 1e-12, 'maxiter': 1000})
                    except KnotMinimum as completed:
                        point = completed.args[0]
                        polished = OptimizeResult(x=point, fun=objective(point), success=True,
                                                   nfev=evaluations-before)
                    if polished.success and polished.fun <= result.fun@result.fun:
                        result.x = polished.x
                        result.fun, result.jac = compute(polished.x)
                        result.optimality = optimality(result.x, result.fun, result.jac)
                        result.success = True
                        result.message = 'Asymmetric-knot constrained polish'
                        result.nfev += polished.nfev
                        descent = descent_check(result) if result.optimality > 1e-4 else None
            return descent

        descent = polish_knots(result, descent)
        for _ in range(8):
            if np.isfinite(result.fun).all() and result.optimality < 1e-6:
                result.success = True
                result.message = 'Mean stationarity check passed'
            if descent is not None and descent <= 1e-7 and knot_optimality(result.x) < 1e-4:
                result.success = True
                result.message = 'Asymmetric-knot stationarity and descent checks passed'
            if result.success and (descent is None or descent <= 1e-7):
                break
            # Clipped asymmetric errors create derivative jumps. A directional
            # minimizer can reach such a corner when a smooth least-squares
            # stopping rule stalls on one side of it.
            start = descent_start if descent is not None and descent > 0 else result.x
            polished = (minimize(objective, start, method='Powell',
                                  options={'xtol': 1e-10, 'ftol': 1e-11, 'maxiter': 1000})
                        if constraint is None else
                        minimize(objective, start, method='COBYQA', constraints=constraint,
                                 options={'initial_tr_radius': .1, 'final_tr_radius': 1e-9, 'maxfev': 10000}))
            if objective(start) < polished.fun:
                polished.x, polished.fun = start.copy(), objective(start)
                polished.success = False
                polished.message = 'Retained a better feasible descent point'
            if np.isfinite(polished.fun) and polished.fun <= result.fun @ result.fun:
                result.x = polished.x
                result.fun, result.jac = compute(polished.x)
                result.optimality = optimality(result.x, result.fun, result.jac)
                result.success = polished.success
                result.message = str(polished.message)
                result.nfev += polished.nfev
                descent = descent_check(result) if result.optimality > 1e-4 else None
                descent = polish_knots(result, descent)
        if constraint is not None:
            at = constraint.A@result.x
            result.success = result.success and np.all(at >= constraint.lb-1e-7) and np.all(at <= constraint.ub+1e-7)
        if (not result.success or not np.isfinite(result.fun).all()
                or (result.optimality > 1e-4 and (descent is None or not np.isfinite(descent) or descent > 1e-7))):
            failure = RuntimeError(f'Refit did not converge: {result.message}; gradient={result.optimality:.3g}; descent={descent}')
            failure.values, failure.scales, failure.fitted_values = values, scales, center+transform@result.x
            raise failure
        fitted_values = center + transform @ result.x
        answer = {'x': result.x, 'fitted_values': fitted_values,
                  'chi2_min': float(result.fun @ result.fun), 'optimality': result.optimality,
                  'nfev': evaluations, 'descent_improvement': descent,
                  'preconditioner': transform @ np.linalg.pinv(result.jac.T @ result.jac) @ transform.T}
        if not full_output:
            return answer
        curvature = np.asarray(hessian(result.x, values, scales))/2
        if np.linalg.eigvalsh(curvature).min() <= 0:
            raise RuntimeError('Fitted mean has nonpositive curvature')
        covariance = transform @ np.linalg.inv(curvature) @ transform.T
        smooth_interior = result.optimality < 1e-4 and not np.isfinite(knot_optimality(result.x))
        if constraint is not None:
            at = constraint.A@result.x
            smooth_interior = smooth_interior and np.all(at > constraint.lb+1e-7) and np.all(at < constraint.ub-1e-7)
        chi2 = jax.jit(lambda fp: fit['chi2_open'](fp, values, scales))
        answer.update(fit)
        answer.update(chi2=chi2, chi2_grad=jax.jit(jax.grad(chi2)),
                      chi2_min=float(result.fun @ result.fun), fitted_values=fitted_values,
                      param_values=fit['fitted_params_to_params'](fitted_values),
                      covariance=covariance, fit_valid=True, hesse_accurate=bool(smooth_interior),
                      input_scales=scales, refit_optimality=result.optimality,
                      refit_evaluations=evaluations, refit_descent_improvement=descent)
        answer['meas_df'] = fit['meas_df'].assign(value=values)
        return answer

    def batch_residual_jacobian(z, values, scales):
        r, jac = jax.vmap(fit['chi2_open'].residual_value_jac, in_axes=(0, 0, None))(
            jnp.asarray(center)+z@jnp.asarray(transform).T, values, scales)
        return r, jac@jnp.asarray(transform)

    @jax.jit
    def batch_step(z, values, scales, damping):
        r, jac = batch_residual_jacobian(z, values, scales)
        gradient = jnp.einsum('bnp,bn->bp', jac, r)
        curvature = jnp.einsum('bnp,bnq->bpq', jac, jac)
        units = jnp.sqrt(jnp.maximum(jnp.diagonal(curvature, axis1=1, axis2=2), 1e-100))
        scaled = curvature/(units[:, :, None]*units[:, None, :])
        direction = -jnp.linalg.solve(scaled+damping[:, None, None]*jnp.eye(z.shape[1]),
                                       (gradient/units)[..., None])[..., 0]/units
        if constraint is not None:
            at, delta = z@jnp.asarray(constraint.A).T, direction@jnp.asarray(constraint.A).T
            limit = jnp.where(delta > 0, (constraint.ub-at)/delta,
                              jnp.where(delta < 0, (constraint.lb-at)/delta, jnp.inf))
            direction *= jnp.minimum(1., .999999*jnp.maximum(0., jnp.min(limit, axis=1)))[:, None]
        proposal = z+direction
        trial = jax.vmap(residual, in_axes=(0, 0, None))(proposal, values, scales)
        improves = jnp.sum(trial**2, axis=1) < jnp.sum(r**2, axis=1)
        return (jnp.where(improves[:, None], proposal, z),
                jnp.clip(damping*jnp.where(improves, .3, 10.), 1e-12, 1e12),
                jnp.max(jnp.abs(gradient), axis=1))

    def batch(values, scales, start):
        """Batched mean-fit proposals, with scalar safeguarded fallbacks.

        The fast path accepts only finite, feasible, stationary solutions.
        Kinks and uncertified boundary solutions go through the scalar solver.
        """
        values, scales = np.asarray(values), np.asarray(scales)
        if values.ndim != 2 or values.shape[1] != len(data) or not np.isfinite(values).all():
            raise ValueError('Simulation data must be a finite draws-by-measurements array')
        if scales.shape != data.shape or np.any(scales <= 0) or not np.isfinite(scales).all():
            raise ValueError('Positive finite scales must match measurement rows')
        z = np.broadcast_to(start, (len(values), len(start))).copy()
        damping = np.full(len(values), 1e-5)
        # Once almost all draws are stationary, hand the difficult tail to the
        # scalar solver. Keep iterating when a substantial fraction still needs
        # the batch: small nonlinear fits can need more steps than large ones.
        for _ in range(32):
            z, damping, gradient = batch_step(z, values, scales, damping)
            stationary = np.asarray(gradient) < 1e-6
            if stationary.all() or (_ >= 7 and stationary.mean() >= .9):
                break
        z = np.asarray(z)
        r, jac = batch_residual_jacobian(z, values, scales)
        gradients = np.einsum('bnp,bn->bp', jac, r)
        gradient = np.max(np.abs(gradients), axis=1)
        finite = np.isfinite(z).all(axis=1) & np.isfinite(r).all(axis=1)
        if constraint is not None:
            at = z@constraint.A.T
            for i in np.flatnonzero(finite & (gradient >= 1e-6)):
                normals = np.vstack([-constraint.A[at[i]-constraint.lb < 1e-7],
                                      constraint.A[constraint.ub-at[i] < 1e-7]])
                if len(normals):
                    multipliers = lsq_linear(normals.T, -gradients[i], bounds=(0, np.inf), tol=1e-12)
                    gradient[i] = np.max(np.abs(gradients[i]+normals.T@multipliers.x))
        good = finite & (gradient < 1e-6)
        if constraint is not None:
            at = z@constraint.A.T
            good &= np.all((at >= constraint.lb-1e-8) & (at <= constraint.ub+1e-8), axis=1)
        fitted = center+z@transform.T
        for i in np.flatnonzero(~good):
            # An unproductive proposal must not lose the original valid start.
            seed = z[i] if np.isfinite(z[i]).all() else start
            fitted[i] = refit(values[i], scales, start=seed, full_output=False)['fitted_values']
        batch.last_fallbacks = int((~good).sum())
        return fitted

    refit.batch = batch
    return refit
