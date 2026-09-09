"""Repeated least-squares fits of an already prepared measurement model."""

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import least_squares, lsq_linear, minimize, LinearConstraint
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
                                   jac=lambda z: compute(z)[1], x_scale='jac',
                                   ftol=1e-11, xtol=1e-11, gtol=1e-8, max_nfev=1000)
        else:
            result = minimize(objective, initial, jac=lambda z: 2*compute(z)[1].T@compute(z)[0],
                              method='SLSQP', constraints=constraint,
                              options={'ftol': 1e-11, 'maxiter': 1000})
        descent_start = initial
        descent = None

        def check(candidate):
            nonlocal descent, descent_start
            r, jac = compute(candidate.x)
            candidate.fun, candidate.jac = r, jac
            corner = knot_optimality(candidate.x)
            candidate.optimality = min(optimality(candidate.x, r, jac), corner)
            if not np.isfinite(r).all():
                return False
            if constraint is not None:
                at = constraint.A@candidate.x
                if np.any(at < constraint.lb-1e-7) or np.any(at > constraint.ub+1e-7):
                    return False
            if candidate.optimality < 1e-4:
                descent = None
                return True
            # A flat or nonsmooth objective can stop with a large coordinate
            # gradient. Check feasible descent in uncertainty units before
            # accepting a solver's small-step stopping condition.
            gradient = projected_gradient(candidate.x, r, jac)
            directions = np.vstack([np.eye(len(initial)), -np.eye(len(initial)),
                                     -gradient/max(np.linalg.norm(gradient), 1e-100)])
            steps = 10.**np.arange(-10, -1)
            trials = candidate.x + (steps[:, None, None]*directions[None, :, :]).reshape(-1, len(initial))
            feasible = np.ones(len(trials), bool)
            if constraint is not None:
                at = trials@constraint.A.T
                feasible = np.all((at >= constraint.lb) & (at <= constraint.ub), axis=1)
            if not feasible.any():
                return False
            trial_values = np.where(feasible, np.asarray(batch_objective(trials, values, scales)), np.inf)
            descent_start = trials[np.nanargmin(trial_values)]
            descent = float(r@r-np.min(trial_values))
            return bool(candidate.success and not np.isfinite(corner)
                        and np.isfinite(descent) and descent <= 1e-7)

        if not check(result) and prediction_jacobian is not None:
            # Coordinate searches can stall at an error-rule corner even when
            # there is descent ALONG the corner. Minimize on that surface once,
            # then check the full one-sided stationarity conditions.
            raw = values-np.asarray(prediction_jacobian(result.x)[0])
            lower = scales*fit['meas_df']['error_n'].to_numpy()
            upper = scales*fit['meas_df']['error_p'].to_numpy()
            knot = np.where(raw > 0, lower, -upper)
            active = (lower != upper) & (np.abs(raw-knot) < 1e-3*np.minimum(lower, upper))
            if active.any():
                target, units = (values-knot)[active], np.minimum(lower, upper)[active]
                equality = {'type': 'eq',
                            'fun': lambda z: (np.asarray(prediction_jacobian(z)[0])[active]-target)/units,
                            'jac': lambda z: np.asarray(prediction_jacobian(z)[1])[active]/units[:, None]}
                constraints = [equality] if constraint is None else [constraint, equality]
                polished = minimize(objective, result.x, jac=lambda z: 2*compute(z)[1].T@compute(z)[0],
                                     method='SLSQP', constraints=constraints,
                                     options={'ftol': 1e-11, 'maxiter': 1000})
                if np.isfinite(polished.fun) and polished.fun <= result.fun@result.fun+1e-10:
                    check(polished)
                    result = polished
        if not check(result):
            # One derivative-free fallback handles interpolation corners. Keep
            # the better feasible point; never accept a failed or descending fit.
            start = descent_start if descent is not None and descent > 0 else result.x
            fallback = (minimize(objective, start, method='Powell',
                                 options={'xtol': 1e-10, 'ftol': 1e-11, 'maxiter': 1000})
                        if constraint is None else
                        minimize(objective, start, method='COBYQA', constraints=constraint,
                                 options={'initial_tr_radius': .1, 'final_tr_radius': 1e-9, 'maxfev': 10000}))
            if not check(fallback) or fallback.fun@fallback.fun > result.fun@result.fun+1e-7:
                failure = RuntimeError(f'Refit did not converge: {fallback.message}; '
                                       f'gradient={fallback.optimality:.3g}; descent={descent}')
                failure.values, failure.scales = values, scales
                failure.fitted_values = center+transform@fallback.x
                raise failure
            result = fallback
        fitted_values = center + transform @ result.x
        result.success = True
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

    return refit
