"""Compile PDG relationships and measurement corrections into JAX functions."""

import numpy as np
import pandas as pd
import jax.numpy as jnp
from pdgfits.func_factory import func_factory

def get_node_funcs(nodes, parameters, fit_df, rel_df, jit=True):
    """Compile each relationship once, using the same builder as predictions."""
    equations = dict(zip(fit_df['node'], fit_df['type']))
    return [_build_node_func(node, parameters, equations, rel_df, jit) for node in nodes]


def get_parameter_funcs(parameters):
    return [lambda params, i=i: params[i] for i in range(len(parameters))]


def get_meas_funcs(node_func_dict, parameter_func_dict, meas_df):
    """Select a prediction for each measurement, giving relationships priority."""
    functions = {**parameter_func_dict, **node_func_dict}
    missing = set(meas_df['node']) - functions.keys()
    if missing:
        raise ValueError(f'Measurement nodes have no prediction: {sorted(missing)}')
    return [functions[node] for node in meas_df['node']]


def get_translate_dep(dep_meas_data, parameters, nodes, parameter_funcs, node_funcs):
    """Compile additive corrections sum(a_j * (parameter_j - reference_j))."""
    if all(data is None for data in dep_meas_data):
        return lambda params: jnp.zeros(len(dep_meas_data), dtype=jnp.float64)

    # Corrections name physical parameters first, then derived relationships.
    functions = {**dict(zip(nodes, node_funcs)), **dict(zip(parameters, parameter_funcs))}
    terms = []
    for data in dep_meas_data:
        if data is None:
            terms.append(None)
            continue
        names, coefficients, references = data
        getters = [functions[name if name in functions else f'nuisance_{name}'] for name in names]
        coefficients = np.asarray(coefficients, dtype=np.float64)
        constant = -np.sum(coefficients * np.asarray(references))
        terms.append((getters, coefficients, constant))

    def translate_dep(params):
        values = []
        for term in terms:
            if term is None:
                values.append(0)
            else:
                getters, coefficients, constant = term
                predictions = jnp.stack([function(params) for function in getters])
                values.append(jnp.sum(coefficients * predictions) + constant)
        return jnp.stack(values)

    return translate_dep


def get_adjust(adjust_data, parameters, nodes, parameter_funcs, node_funcs):
    """Compile multiplicative corrections in the measurement convention.

    A stored division multiplies the measured value by its auxiliary input;
    a stored multiplication divides it. Prediction construction reverses this
    convention in get_mu_adjust.
    """
    if all(data is None for data in adjust_data):
        return lambda params: jnp.ones(len(adjust_data), dtype=jnp.float64)

    functions = {**dict(zip(nodes, node_funcs)), **dict(zip(parameters, parameter_funcs))}
    terms = []
    for data in adjust_data:
        factors = []
        if data is not None:
            names, operators = data
            for name, operator in zip(names, operators):
                if operator not in ('/', '*'):
                    raise ValueError(f'Unknown relationship: {operator}')
                function = functions[name if name in functions else f'nuisance_{name}']
                factors.append((function, operator))
        terms.append(factors)

    def adjust(params):
        values = []
        for factors in terms:
            if not factors:
                values.append(1)
            else:
                values.append(jnp.prod(jnp.stack([
                    function(params) if operator == '/' else 1/function(params)
                    for function, operator in factors
                ])))
        return jnp.stack(values)

    return adjust


def get_mu(meas_funcs):
    """Stack scalar measurement predictions into a vector-valued prediction."""
    return lambda params: jnp.stack([function(params) for function in meas_funcs])


def _build_node_func(node, parameters, eq_type_map, rel_df, jit=True):
    """Build a func_factory callable for a single node (used as nonlinear fallback)."""
    eq_type = eq_type_map[node]
    rel_sub = rel_df[rel_df['node'] == node]
    n_summations = int(np.max(rel_sub['summation']))
    n_params = len(parameters)
    param_idx = {p: i for i, p in enumerate(parameters)}

    coefficients = []
    coeff_params_list = []
    for s in range(1, n_summations + 1):
        sub = rel_sub[rel_sub['summation'] == s].drop_duplicates('parameter_key', keep='first')
        coeff_vec = np.zeros(n_params, dtype=np.float64)
        cp_vec = np.zeros(n_params, dtype=np.int32)
        for _, row in sub.iterrows():
            p_key = row['parameter_key']
            if p_key in param_idx:
                j = param_idx[p_key]
                coeff_vec[j] = row['coefficient']
                if pd.notna(row['coeff_parameter_key']):
                    cpk = row['coeff_parameter_key']
                    if cpk not in param_idx and f'nuisance_{cpk}' in param_idx:
                        cpk = f'nuisance_{cpk}'
                    assert cpk in param_idx, (
                        f"Coefficient parameter {cpk} for node {node} not found in parameters"
                    )
                    cp_vec[j] = param_idx[cpk] + 1
        coefficients.append(coeff_vec)
        coeff_params_list.append(cp_vec)

    return func_factory(eq_type, coefficients, coeff_params_list, jit=jit)


def get_mu_vectorized(parameters, nodes, meas_df, fit_df, rel_df, jit=True):
    """
    Return mu(params) built as a single matrix multiply for linear-simple '+' nodes,
    with a per-function fallback for non-linear nodes.

    parameters : list of parameter_key strings (order matches params array)
    nodes      : list of node names that appear in rel_df (rel_df['node'].unique())
    meas_df    : preprocessed measurement DataFrame, with 'node' column
    fit_df     : has 'node' and 'type' columns (equation type per node)
    rel_df     : has 'node', 'parameter_key', 'coefficient', 'summation',
                 'coeff_parameter_key' columns
    jit        : passed through to func_factory for the nonlinear fallback callables
    """
    n_meas = len(meas_df)
    n_params = len(parameters)
    param_idx = {p: i for i, p in enumerate(parameters)}
    eq_type_map = dict(zip(fit_df['node'], fit_df['type']))

    # Build coefficient vector for each linear-simple '+' node.
    node_coeff = {}

    for node in nodes:
        eq_type = eq_type_map[node]
        if eq_type != '+':
            continue
        rel_sub = rel_df[(rel_df['node'] == node) & (rel_df['summation'] == 1)]
        if rel_sub['coeff_parameter_key'].notna().any():
            continue  # '+' with parameter-valued coefficients: use func_factory fallback
        coeff_vec = np.zeros(n_params, dtype=np.float64)
        for _, row in rel_sub.iterrows():
            p_key = row['parameter_key']
            if p_key in param_idx:
                coeff_vec[param_idx[p_key]] = row['coefficient']
        node_coeff[node] = coeff_vec

    # Parameters not in nodes are measured directly; coefficient = standard basis vector e_j.
    for p in parameters:
        if p not in nodes:
            coeff_vec = np.zeros(n_params, dtype=np.float64)
            coeff_vec[param_idx[p]] = 1.0
            node_coeff[p] = coeff_vec

    # Classify each measurement row.
    C = np.zeros((n_meas, n_params), dtype=np.float64)
    nonlinear_idxs = []
    nonlinear_funcs = []
    node_func_cache = {}

    for meas_i, meas_node in enumerate(meas_df['node']):
        if meas_node in node_coeff:
            C[meas_i, :] = node_coeff[meas_node]
        else:
            if meas_node not in node_func_cache:
                node_func_cache[meas_node] = _build_node_func(
                    meas_node, parameters, eq_type_map, rel_df, jit=jit
                )
            nonlinear_idxs.append(meas_i)
            nonlinear_funcs.append(node_func_cache[meas_node])

    C_jnp = jnp.array(C)

    if not nonlinear_idxs:
        def mu(params):
            return C_jnp @ params
    else:
        nl_idxs_arr = jnp.array(nonlinear_idxs, dtype=jnp.int32)

        def mu(params):
            result = C_jnp @ params
            nl_vals = jnp.array([f(params) for f in nonlinear_funcs])
            return result.at[nl_idxs_arr].set(nl_vals)

    return mu

def get_mu_adjust(mu, adjust, translate_dep):

    def mu_adjust(params):
        return (mu(params) - translate_dep(params))/adjust(params)

    return mu_adjust
