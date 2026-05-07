import numpy as np
import pandas as pd
import jax.numpy as jnp
from pdgfits.func_factory import func_factory

def get_node_funcs(nodes, parameters, fit_df, rel_df, jit=True):
    # dict node -> equation_type
    equation_type_dict = dict(zip(fit_df['node'], fit_df['type']))

    # form the list of functions which maps parameters -> node values
    # uses the relationship table
    node_funcs = []
    for node in nodes:

        # the rows of the table which specify how the node is determined
        rel_df_sub = rel_df[rel_df['node'] == node]

        # the type of the equation which determines the node
        equation_type = equation_type_dict[node]

        n_summations = np.max(rel_df_sub['summation'])

        # form a list of coefficients for each summation
        # (parameters not in the summation get a coefficient of 0)
        # and a list of coefficient parameter indices (this will be mostly be a list of arrays ofzeros)
        coefficients = []
        coeff_params = []
        for i in range(n_summations):
            # coefficients and coefficient parameter indicesfor the ith summation
            coefficients_i = np.zeros(len(parameters), dtype=np.float64)
            coeff_params_i = np.zeros(len(parameters), dtype=np.int32)

            # for each of the parameters, if it is in the ith summation, put in its coefficient
            for j, param in enumerate(parameters):
                # if the parameter has an entry with summation i+i, then put it in
                entry = rel_df_sub[(rel_df_sub['parameter_key'] == param) & (rel_df_sub['summation'] == (i+1))]
                if len(entry) > 0:
                    coefficients_i[j] = entry['coefficient'].iloc[0]
                    # if there are coefficient parameters, put in their indices (offset by 1 for the trick
                    # in the term() function in func_factory.py)
                    if any(~entry['coeff_parameter_key'].isna()):
                        coeff_parameter_key = entry['coeff_parameter_key'].iloc[0]
                        if coeff_parameter_key not in parameters:
                            coeff_parameter_key = f'nuisance_{coeff_parameter_key}'
                        assert coeff_parameter_key in parameters, f"Coefficient parameter {coeff_parameter_key} for node {node} not found in parameters {parameters}"
                        coeff_params_i[j] = list(parameters).index(coeff_parameter_key)+1
            coefficients.append(coefficients_i)
            coeff_params.append(coeff_params_i)

        # create the function using the equation type, coefficients, and coefficient parameter indices
        node_funcs.append(func_factory(equation_type, coefficients, coeff_params, jit=jit))

    return node_funcs

def get_parameter_funcs(parameters):
    # parameter funcs just select one element from the params array
    parameter_funcs = []
    for i in range(len(parameters)):
        parameter_funcs.append(lambda params, i=i: params[i])
    return parameter_funcs

def get_meas_funcs(node_func_dict, parameter_func_dict, meas_df):
    # list of functions which map parameters to the value of each measurement's measurand
    # these come either from node_funcs or parameter_funcs
    meas_funcs = []
    for meas_node in list(meas_df['node']):
        if meas_node in node_func_dict:
            meas_funcs.append(node_func_dict[meas_node])
        elif meas_node in parameter_func_dict:
            meas_funcs.append(parameter_func_dict[meas_node])
        else:
            raise ValueError(f"Node {meas_node} not found in node_func_dict or parameter_func_dict")
    return meas_funcs

# make a function which will be added to the measurement vector to account for dep_meas measurements
def get_translate_dep(dep_meas_data, parameters, nodes, parameter_funcs, node_funcs):
    n_meas = len(dep_meas_data)
    if all(d is None for d in dep_meas_data):
        return lambda params: jnp.zeros(n_meas, dtype=jnp.float64)
    functions = []
    for data in dep_meas_data:
        if data is not None:
            dep_nodes = data[0]
            for i in range(len(dep_nodes)):
                if dep_nodes[i] not in parameters+nodes:
                    dep_nodes[i] = f'nuisance_{dep_nodes[i]}'
            assert all(node in parameters+nodes for node in dep_nodes)
            constant = -np.sum(np.array(data[1])*np.array(data[2]))
            coefficients = np.array(data[1], dtype=np.float64)
            dep_node_funcs = [(parameter_funcs+node_funcs)[list(parameters+nodes).index(node)] for node in dep_nodes]
            functions.append(
                lambda params, coefficients=coefficients, dep_node_funcs=dep_node_funcs, constant=constant:
                    jnp.sum(coefficients * jnp.stack([dep_node_func(params) for dep_node_func in dep_node_funcs])) + constant
            )
        else:
            functions.append(lambda params: 0)
    # this function just stacks all the functions into one vector-valued function
    def translate_dep(params):
        return jnp.stack([func(params) for func in functions])
    return translate_dep

# make a function which will be multiplied with the measurement vector to account for br_adjust measurements
def get_adjust(adjust_data, parameters, nodes, parameter_funcs, node_funcs):
    n_meas = len(adjust_data)
    if all(d is None for d in adjust_data):
        return lambda params: jnp.ones(n_meas, dtype=jnp.float64)
    functions = []
    for data in adjust_data:
        if data is None:
            functions.append(lambda params: 1)
        else:
            rels = data[1]
            dep_nodes = data[0]

            node_functions = []
            for i in range(len(dep_nodes)):
                rel = rels[i]
                dep_node = dep_nodes[i]
                if dep_node not in parameters+nodes:
                    dep_node = f'nuisance_{dep_node}'
                # print(dep_node)
                if rel == '/':
                    node_functions.append(lambda params, dep_node_func=(parameter_funcs+node_funcs)[list(parameters+nodes).index(dep_node)]: dep_node_func(params))
                elif rel == '*':
                    node_functions.append(lambda params, dep_node_func=(parameter_funcs+node_funcs)[list(parameters+nodes).index(dep_node)]: 1/dep_node_func(params))
                else:
                    raise ValueError(f"Unknown relationship: {rel}")
            functions.append(lambda params, node_functions=node_functions: jnp.prod(jnp.stack([func(params) for func in node_functions])))
    # this function just stacks all the functions into one vector-valued function
    def adjust(params):
        return jnp.stack([func(params) for func in functions])
    return adjust

def get_mu(meas_funcs):
    # function mapping the parameters to an array of measurand values
    # essentially just puts the results from all the meas_funcs into an array
    def mu(params):
        return jnp.stack([meas_func(params) for meas_func in meas_funcs])
    return mu

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
        sub = rel_sub[rel_sub['summation'] == s]
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
