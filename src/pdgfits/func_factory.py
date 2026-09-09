import jax.numpy as jnp
import jax
import numpy as np

ALLOWED_EQUATION_TYPES = ['+', 'G+', 'R+', 'lifetime', '/', 'P', 'G*', 'P/', 'SR', 'SQ']

# coeff_params indexes into [1, param0, param1, ...]: index 0 means multiply by 1 (no param coefficient),
# otherwise multiply by the referenced parameter.
def func_factory(equation_type, coefficients, coeff_params, jit=True):
    coefficients = jnp.array(coefficients, dtype=np.float64)
    coeff_params = jnp.array(coeff_params)
    simple = not jnp.any(coeff_params)  # static: True when no parameter-valued coefficients

    if simple:
        def t(params, i):
            return jnp.dot(params, coefficients[i])
    else:
        has_coeff = coeff_params > 0  # (n_terms, n_params) static boolean mask
        # Split coefficient array into two static masked arrays so no jnp.where is needed
        # at trace/differentiation time. coeff_param_parts[i,j] is non-zero only where
        # element j uses a parameter multiplier; coeff_static_parts[i,j] covers the rest.
        coeff_param_parts = jnp.array(jnp.where(has_coeff, coefficients, 0.0))
        coeff_static_parts = jnp.array(jnp.where(~has_coeff, coefficients, 0.0))
        param_idxs = jnp.array(jnp.maximum(coeff_params - 1, 0))

        # i is always a static Python int at every call site (0, 1, 2).
        def t(params, i):
            params_gathered = params[param_idxs[i]]
            return (jnp.dot(params * params_gathered, coeff_param_parts[i]) +
                    jnp.dot(params, coeff_static_parts[i]))

    maybe_jit = jax.jit if jit else (lambda f: f)

    @maybe_jit
    def f(params):
        if equation_type == '+':
            return t(params, 0)
        elif equation_type in ('G+', 'R+', 'P'):
            return t(params, 0) * t(params, 1)
        elif equation_type == 'lifetime':
            return 1.0 / t(params, 0)
        elif equation_type == '/':
            return t(params, 0) / t(params, 1)
        elif equation_type == 'G*':
            return t(params, 0) * t(params, 1) * t(params, 2)
        elif equation_type == 'P/':
            return t(params, 0) * t(params, 1) / t(params, 2)
        elif equation_type == 'SR':
            return jnp.sqrt(t(params, 0) * t(params, 1))
        elif equation_type == 'SQ':
            return jnp.sqrt(t(params, 0) * t(params, 1)) * t(params, 2)
        else:
            raise ValueError(f"Unknown equation type: {equation_type}")

    return f
