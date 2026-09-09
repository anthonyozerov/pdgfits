import numpy as np
import jax
import jax.numpy as jnp
from scipy.optimize import LinearConstraint


def covariance_factor(covariance):
    """A scale map omitting explicitly fixed coordinates."""
    covariance = np.asarray(covariance, float)
    active = np.flatnonzero(np.diag(covariance) > 0)
    sigma = np.sqrt(np.diag(covariance)[active])
    correlation = covariance[np.ix_(active, active)]/np.outer(sigma, sigma)
    try:
        factor = sigma[:, None]*np.linalg.cholesky(correlation)
    except np.linalg.LinAlgError:
        factor = np.diag(sigma)
    result = np.zeros((len(covariance), len(active)))
    result[active] = factor
    return result


def physical_coordinates(fit):
    """An affine chart reaching the closed branching-fraction domain exactly."""
    if fit.get('algorithm') not in ('BR', 'BRU', 'BR (NO MATRIX)', 'BR PRINT'):
        return None
    particles, decay = get_decay_info(fit['parameters'])
    if not len(decay):
        return None
    center = np.asarray(fit['param_values'])
    jac = np.asarray(jax.jacfwd(fit['fitted_params_to_params'])(fit['fitted_values']))
    covariance = jac@np.asarray(fit['covariance'])@jac.T
    sum_one = fit['algorithm'] != 'BRU' and len(particles) == 1
    free = np.delete(np.arange(len(center)), decay[0]) if sum_one else np.arange(len(center))
    basis = np.eye(len(center))[:, free]
    if sum_one:
        basis[decay[0], np.isin(free, decay)] = -1
    transform = basis@covariance_factor(covariance[np.ix_(free, free)])
    row_scale = np.maximum(np.linalg.norm(transform[decay], axis=1), np.finfo(float).tiny)
    bounds = LinearConstraint(transform[decay]/row_scale[:, None],
                              -center[decay]/row_scale, (1-center[decay])/row_scale)
    return center, transform, bounds


# Keep the original arctan map that Minuit handles well, but shrink fitted-space
# coordinates so tiny branching fractions do not start in O(1e4) saturation.
SIGMOID_ATAN_SCALE = 1e-4


def get_decay_info(parameters):
    indices = np.array([i for i, p in enumerate(parameters) if p[4:5] == '.'], dtype=int)
    particles = np.array(sorted({parameters[i][:4] for i in indices}))
    return particles, indices


def build_param_map_sigmoid(parameters):
    """Unconstrained fit: decay params reparametrized to enforce (0, 1)."""
    decay_particles, decay_param_idxs = get_decay_info(parameters)
    print('particles with decays:', decay_particles)

    @jax.jit
    def fitted_params_to_params(fitted_params):
        return fitted_params.at[decay_param_idxs].set(
            0.5 + jax.lax.atan(fitted_params[decay_param_idxs] / SIGMOID_ATAN_SCALE) / jnp.pi
        )

    @jax.jit
    def params_to_fitted_params(params):
        decay_params = params[decay_param_idxs]
        decay_params = jnp.clip(decay_params, 1e-12, 1 - 1e-12)
        return params.at[decay_param_idxs].set(
            SIGMOID_ATAN_SCALE * jax.lax.tan(jnp.pi * (decay_params - 0.5))
        )

    return fitted_params_to_params, params_to_fitted_params, decay_param_idxs


def build_param_map_softmax(parameters):
    """Unconstrained fit: decay params reparametrized via log/softmax to enforce sum-to-one."""
    decay_particles, decay_param_idxs = get_decay_info(parameters)
    print('particles with decays:', decay_particles)
    assert len(decay_particles) == 1, "Softmax reparametrization only supported for one particle with decays"

    fixed_idx = decay_param_idxs[0]

    @jax.jit
    def fitted_params_to_params(fitted_params):
        weights = jnp.exp(fitted_params[decay_param_idxs])
        return fitted_params.at[decay_param_idxs].set(weights / jnp.sum(weights))

    @jax.jit
    def params_to_fitted_params(params):
        return params.at[decay_param_idxs].set(jnp.log(params[decay_param_idxs]))

    return fitted_params_to_params, params_to_fitted_params, decay_param_idxs, fixed_idx


def build_param_map_scaled(fitted_params_to_params_base, params_to_fitted_params_base,
                           scale_idxs, scales):
    """Compose a linear preconditioner into fitted parameter coordinates.

    ``scale_idxs`` are indices in fitted space. The optimizer sees
    ``base_fitted[idx] / scale`` while the scientific parameter map still sees
    the original base fitted coordinate. This preserves the chi2 model and its
    optimum in parameter space, but improves optimizer conditioning for raw
    coordinates with very large natural scales.
    """
    scale_idxs = jnp.array(scale_idxs, dtype=jnp.int32)
    scales = jnp.array(scales, dtype=jnp.float64)

    @jax.jit
    def fitted_params_to_params(fitted_params):
        base_fitted_params = fitted_params.at[scale_idxs].set(
            fitted_params[scale_idxs] * scales
        )
        return fitted_params_to_params_base(base_fitted_params)

    @jax.jit
    def params_to_fitted_params(params):
        fitted_params = params_to_fitted_params_base(params)
        return fitted_params.at[scale_idxs].set(
            fitted_params[scale_idxs] / scales
        )

    return fitted_params_to_params, params_to_fitted_params
