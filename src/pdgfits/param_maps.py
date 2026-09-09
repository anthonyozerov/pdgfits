import numpy as np
import jax
import jax.numpy as jnp


# Keep the original arctan map that Minuit handles well, but shrink fitted-space
# coordinates so tiny branching fractions do not start in O(1e4) saturation.
SIGMOID_ATAN_SCALE = 1e-4


def get_decay_info(parameters):
    decay_particles = np.unique([param[:4] for param in parameters if any(np.strings.startswith(np.array(parameters), param[:4]+'.'))])
    decay_param_idxs = np.where([any(p.startswith(particle + '.') for particle in decay_particles) for p in parameters])[0]
    return decay_particles, decay_param_idxs


"""
Deprecated: set one of the decay params to 1 - (sum of the others)
"""
def build_param_map(parameters, fit_seed_df):
    print('enforcing sum to one constraint')
    decay_particles = np.unique([param[:4] for param in parameters if any(np.strings.startswith(np.array(parameters), param[:4]+'.'))])
    print('particles with decays:', decay_particles)

    excluded_param_idxs = []
    for particle in decay_particles:
        fit_seed_sub = fit_seed_df[fit_seed_df['parameter_key'].str.startswith(particle+'.').fillna(False)]
        fit_seeds = np.array(fit_seed_sub['seed'])
        excluded_param_seed_idx = np.argsort(fit_seeds)[-1]
        excluded_param = fit_seed_sub['parameter_key'].iloc[excluded_param_seed_idx]
        excluded_param_idxs.append(list(parameters).index(excluded_param))

    fitted_parameters = np.delete(parameters, excluded_param_idxs)
    fitted_param_idxs = np.where(np.isin(parameters, fitted_parameters))[0]

    summation_idxs = []
    for particle in decay_particles:
        summation_idxs.append(np.where(np.strings.startswith(np.array(fitted_parameters), particle+'.'))[0])

    def fitted_params_to_params(fitted_params):
        params = jnp.full(len(parameters), jnp.nan, dtype=jnp.float64)
        for i in range(len(excluded_param_idxs)):
            params = params.at[excluded_param_idxs[i]].set(
                1 - jnp.sum(fitted_params[summation_idxs[i]])
            )
        params = params.at[jnp.array(fitted_param_idxs)].set(fitted_params)
        return params

    def params_to_fitted_params(params):
        return params[fitted_param_idxs]

    return fitted_params_to_params, params_to_fitted_params, fitted_parameters, excluded_param_idxs


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
        params = fitted_params
        denom = jnp.sum(jnp.exp(fitted_params[decay_param_idxs]))
        for i in range(len(decay_param_idxs)):
            param_val = jnp.exp(fitted_params[decay_param_idxs[i]]) / denom
            params = params.at[decay_param_idxs[i]].set(param_val)
        return params

    @jax.jit
    def params_to_fitted_params(params):
        fitted_params = params
        for i in range(len(decay_param_idxs)):
            val = jnp.log(params[decay_param_idxs[i]])
            fitted_params = fitted_params.at[decay_param_idxs[i]].set(val)
        return fitted_params

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


def build_soft_pos_constraint(excluded_param_idxs):
    def constraint(params):
        return (-jnp.sum(jnp.clip(params[np.array(excluded_param_idxs)], -np.inf, 0)*1e3))**2
    return constraint
