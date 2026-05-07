import jax.numpy as jnp
import jax

ALLOWED_EQUATION_TYPES = ['+', 'G+', 'R+', 'lifetime', '/', 'P', 'G*', 'P/', 'SR', 'SQ']

# coeff_params indexes into [1, param0, param1, ...]: index 0 means multiply by 1 (no param coefficient),
# otherwise multiply by the referenced parameter.
def func_factory(equation_type, coefficients, coeff_params, jit=True):
    coefficients = jnp.array(coefficients, dtype=jnp.float64)
    coeff_params = jnp.array(coeff_params)
    simple = not bool(jnp.any(coeff_params))  # static: True when no parameter-valued coefficients

    if simple:
        def t(params, i):
            return jnp.dot(params, coefficients[i])
    else:
        param_idxs = jnp.maximum(coeff_params - 1, 0)
        def t(params, i):
            coeff_mult = jnp.where(coeff_params[i] > 0, params[param_idxs[i]], 1.0)
            return jnp.dot(params, coefficients[i] * coeff_mult)

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
