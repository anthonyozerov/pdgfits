import time
import io
import contextlib

import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
from scipy.optimize import minimize as scipy_minimize

from pdgfits.query import avg_queries
from pdgfits.preprocess import preprocess
from pdgfits.build_funcs import get_node_funcs, get_parameter_funcs, get_meas_funcs, get_mu, get_translate_dep, get_adjust
from pdgfits.corr_mat import get_corr_mat
from pdgfits.build_chi2 import build_chi2

avg_df, corr_df_dict = avg_queries()
node = avg_df['node'].iloc[0]
meas_df_node = avg_df[avg_df['node'] == node]
corr_df_node = corr_df_dict.get(node, meas_df_node.iloc[:0])

fit_df = pd.DataFrame({'node': [node], 'type': ['+'], 'data_type': [None]})
rel_df = pd.DataFrame({
    'node': [node], 'par_code': [None], 'parameter': [node],
    'coefficient': [1.0], 'summation': [1],
    'coeff_par_code': [None], 'coeff_parameter': [None],
})
fit_seed_df = pd.DataFrame({'par_code': [None], 'parameter': [node], 'seed': [0.0]})
tree_df = pd.DataFrame({'node': pd.Series([], dtype=str), 'data_type': pd.Series([], dtype=str)})

t0 = time.perf_counter()
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    out = preprocess(fit_df, rel_df, meas_df_node.copy(), corr_df_node.copy(), fit_seed_df, tree_df, 'AVG', '')
t1 = time.perf_counter()

fit_df2, rel_df2, meas_df2, corr_df2, fit_seed_df2, dep_meas_data, adjust_data = out
parameters = list(fit_seed_df2['parameter_key'].unique())
nodes_list = list(rel_df2['node'].unique())

node_funcs = get_node_funcs(nodes_list, parameters, fit_df2, rel_df2, jit=False)
parameter_funcs = get_parameter_funcs(parameters)
meas_funcs = get_meas_funcs(
    dict(zip(nodes_list, node_funcs)),
    dict(zip(parameters, parameter_funcs)),
    meas_df2,
)
mu = get_mu(meas_funcs)
translate_dep = get_translate_dep(dep_meas_data, parameters, nodes_list, parameter_funcs, node_funcs)
adjust = get_adjust(adjust_data, parameters, nodes_list, parameter_funcs, node_funcs)
t2 = time.perf_counter()

y = jnp.array(meas_df2['value'], dtype=jnp.float64)
error_n = jnp.array(meas_df2['error_n'], dtype=jnp.float64)
error_p = jnp.array(meas_df2['error_p'], dtype=jnp.float64)
corr_mat = get_corr_mat(meas_df2, corr_df2)
corr_mat_inv = jnp.linalg.pinv(corr_mat)
chi2, _, chi2_val, _ = build_chi2(
    y, mu, error_n, error_p, corr_mat_inv,
    lambda x: x,
    translate_dep=translate_dep,
    adjust=adjust,
    use_jit=False,
)
t3 = time.perf_counter()

seed_val = float(meas_df2[meas_df2['node'] == node]['value'].mean())
param_init = np.array([seed_val], dtype=np.float64)
result = scipy_minimize(
    lambda x: chi2_val(jnp.array(x, dtype=jnp.float64)),
    param_init,
    method='BFGS',
)
t4 = time.perf_counter()

hess = jax.hessian(chi2)(jnp.array(result.x, dtype=jnp.float64))
covariance = 2.0 * np.array(jnp.linalg.pinv(hess))
t5 = time.perf_counter()

print(f"node={node}, n_meas={len(meas_df2)}, n_params={len(parameters)}")
print(f"  preprocess:     {1000*(t1-t0):.1f} ms")
print(f"  build_funcs:    {1000*(t2-t1):.1f} ms")
print(f"  build_chi2:     {1000*(t3-t2):.1f} ms")
print(f"  scipy_minimize: {1000*(t4-t3):.1f} ms")
print(f"  jax.hessian:    {1000*(t5-t4):.1f} ms")
print(f"  dep_meas non-None: {sum(d is not None for d in dep_meas_data)}, "
      f"adjust non-None: {sum(d is not None for d in adjust_data)}")
