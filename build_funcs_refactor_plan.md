# build_funcs.py Refactor Plan

This document is intended to be self-contained enough for an implementing agent starting
from a fresh session. Read `src/pdgfits/build_funcs.py`, `src/pdgfits/avg.py`,
`src/pdgfits/fit.py`, `src/pdgfits/func_factory.py`, and `src/pdgfits/build_chi2.py`
before starting.

There is also a shorter earlier sketch in `build_funcs_refactor.txt` at the repo root;
this document supersedes it.

---

## 1. Problem

`avg_test.py` runs ~1000 single-node weighted averages serially.  For each node,
`run_avg` calls `jax.hessian(chi2)(params)` (or `jax.grad`) and these calls take up to
800 ms for nodes with many measurements.

The root cause is inside `get_mu`:

```python
def get_mu(meas_funcs):
    def mu(params):
        return jnp.stack([meas_func(params) for meas_func in meas_funcs])
    return mu
```

This is a Python loop that produces **N individual JAX operations** — one `dot` per
measurement.  When JAX's AD machinery (grad, hessian) traces through `chi2`, it must
re-execute this Python loop, dispatching N separate low-level operations.  For a node
with 30 measurements that is 30 individual dispatches per chi2 evaluation, and for a
Hessian (forward-over-reverse) this is multiplied further.  Replacing the loop with a
single matrix-vector multiply `C @ params` collapses those N ops to 1.

The same Python-loop pattern exists in `get_translate_dep` and `get_adjust`, though
in the avg case these are always all-None and the fix there is a fast-path early return.

---

## 2. Scope

**In scope:** `build_funcs.py` and `avg.py` only.

**Out of scope (do not touch in this refactor):**
- `fit.py` — the full fit uses `use_jit=True`, so JIT amortises the tracing cost over
  many minimizer iterations.  The BR/BRU parameter-mapping paths add risk without
  addressing the reported bottleneck.  Leave for a follow-up.
- `corr_mat.py` / `corr_mat_inv` fast path for the identity case — correct improvement
  but separate from build_funcs; leave as a follow-up.
- Full vectorisation of `translate_dep` and `adjust` beyond the all-None fast path —
  the math (dep_meas linear combinations referencing arbitrary nodes, products of BRs)
  is fiddly and silently corrupts fits if wrong.  Leave for a follow-up.

---

## 3. Data Model — Critical Edges

Read this section carefully; these are the places most likely to cause silent bugs.

### 3.1 How a measurement maps to a model value

`get_meas_funcs` resolves each measurement row to a callable with this precedence:

1. If `meas_df['node']` is in `node_func_dict` (i.e. the node appears in `rel_df`):
   use the corresponding node function.
2. Else if `meas_df['node']` is in `parameter_func_dict` (i.e. the name is a bare
   parameter): use `lambda params: params[j]`.

Both cases must be handled in the vectorised version.  Nuisance parameters (added by
`preprocess`) fall into case 2: their measurement rows have `node = 'nuisance_X'` which
is NOT in `rel_df` but IS in `parameters`.  Their coefficient vector is just the
standard basis vector `e_j` for their parameter index `j`.

### 3.2 Linear-simple detection

A node is **linear-simple** if and only if:
```
equation_type == '+' AND all coeff_parameter_key entries are NULL for this node in rel_df
```

The `coeff_parameter_key` column in `rel_df` is non-null when the coefficient itself
involves another parameter (the `G+`, `R+`, `P` pattern with non-zero `coeff_params` in
`func_factory`).  This is rare but exists; those nodes must fall through to
`func_factory`.

### 3.3 Only summation == 1 matters for '+' nodes

A `+` node has exactly one summation (`max(summation) == 1`).  When building the
coefficient vector, filter:

```python
rel_sub = rel_df[(rel_df['node'] == node) & (rel_df['summation'] == 1)]
```

Other equation types (G+, G*, etc.) have `max(summation) > 1` and use those extra
summation rows.  Including them in a `+`-node coefficient vector would silently corrupt
the result.

### 3.4 Multiple measurements per node

Many measurements in `meas_df` may share the same `node`.  In the coefficient matrix
they all get the **same row** (same coefficient vector).  This is a gather, not a
reduction — each measurement independently predicts the same node value.

### 3.5 Nuisance parameters in `rel_df`?

Nuisance parameters added by `preprocess` are NOT added to `rel_df` or `fit_df`.  They
exist only in `fit_seed_df` (parameters) and `meas_df` (fake measurements).  Do not
look them up in `rel_df`; use the parameter-index path (§3.1, case 2).

---

## 4. Changes to `build_funcs.py`

### 4.1 New function: `get_mu_vectorized`

Add a new function alongside the existing ones.  Do **not** delete or modify
`get_node_funcs`, `get_parameter_funcs`, `get_meas_funcs`, or `get_mu`; they are still
used by `fit.py` and the diagnostics / asym_errors path.

```python
def get_mu_vectorized(parameters, nodes, meas_df, fit_df, rel_df, jit=True):
    """
    Return mu(params) built as a single matrix multiply for linear-simple '+' nodes,
    with a small per-function fallback for non-linear nodes.

    parameters : list of parameter_key strings (order matches params array)
    nodes      : list of node names that appear in rel_df (rel_df['node'].unique())
    meas_df    : preprocessed measurement DataFrame, with 'node' column
    fit_df     : has 'node' and 'type' columns (equation type per node)
    rel_df     : has 'node', 'parameter_key', 'coefficient', 'summation',
                 'coeff_parameter_key' columns
    jit        : passed through to func_factory for the nonlinear fallback callables
    """
```

**Implementation:**

```python
import numpy as np
from pdgfits.func_factory import func_factory

def get_mu_vectorized(parameters, nodes, meas_df, fit_df, rel_df, jit=True):
    n_meas   = len(meas_df)
    n_params = len(parameters)
    param_idx = {p: i for i, p in enumerate(parameters)}
    eq_type_map = dict(zip(fit_df['node'], fit_df['type']))

    # --- Step A: build per-node coefficient vector for linear-simple nodes ---
    # node_coeff[node] = np.array of shape (n_params,), or absent if non-linear.

    node_coeff = {}  # node -> np.ndarray(n_params)

    for node in nodes:
        eq_type = eq_type_map[node]
        if eq_type != '+':
            continue  # non-linear, will use func_factory fallback
        rel_sub = rel_df[(rel_df['node'] == node) & (rel_df['summation'] == 1)]
        has_coeff_params = rel_sub['coeff_parameter_key'].notna().any()
        if has_coeff_params:
            continue  # '+' but non-simple, use func_factory fallback
        coeff_vec = np.zeros(n_params, dtype=np.float64)
        for _, row in rel_sub.iterrows():
            p_key = row['parameter_key']
            if p_key in param_idx:
                coeff_vec[param_idx[p_key]] = row['coefficient']
        node_coeff[node] = coeff_vec

    # Parameters measured directly (nuisance params, and the avg node itself):
    # meas_df rows where meas_node is NOT in nodes but IS in parameters.
    # Coefficient vector = standard basis vector e_j.
    for p in parameters:
        if p not in nodes:  # not a node → direct parameter measurement
            coeff_vec = np.zeros(n_params, dtype=np.float64)
            coeff_vec[param_idx[p]] = 1.0
            node_coeff[p] = coeff_vec
        # (if p IS in nodes, the node path above handles it)

    # --- Step B: classify each measurement row ---

    C = np.zeros((n_meas, n_params), dtype=np.float64)
    nonlinear_idxs  = []  # measurement indices handled by fallback
    nonlinear_funcs = []  # corresponding func_factory callables

    # Cache node-level func_factory functions to avoid building duplicates.
    node_func_cache = {}

    for meas_i, meas_node in enumerate(meas_df['node']):
        if meas_node in node_coeff:
            # Linear-simple (or direct parameter): use matrix row.
            C[meas_i, :] = node_coeff[meas_node]
        else:
            # Non-linear node: build func_factory callable (once per node).
            if meas_node not in node_func_cache:
                # Reuse get_node_funcs logic for this single node.
                node_func_cache[meas_node] = _build_node_func(
                    meas_node, parameters, eq_type_map, rel_df, jit=jit
                )
            nonlinear_idxs.append(meas_i)
            nonlinear_funcs.append(node_func_cache[meas_node])

    C_jnp = jnp.array(C)

    if not nonlinear_idxs:
        # Fast path: all measurements are linear — single matmul.
        def mu(params):
            return C_jnp @ params
    else:
        nl_idxs_arr = jnp.array(nonlinear_idxs, dtype=jnp.int32)

        def mu(params):
            result = C_jnp @ params
            nl_vals = jnp.array([f(params) for f in nonlinear_funcs])
            return result.at[nl_idxs_arr].set(nl_vals)

    return mu
```

Add a private helper `_build_node_func` that replicates the per-node logic from
`get_node_funcs` (build `coefficients` and `coeff_params` arrays for the node, then
call `func_factory`).  This avoids code duplication with `get_node_funcs` and avoids
building functions for ALL nodes when only a small fallback set is needed.

```python
def _build_node_func(node, parameters, eq_type_map, rel_df, jit=True):
    """Build a func_factory callable for a single node (fallback for non-linear nodes)."""
    eq_type = eq_type_map[node]
    rel_sub = rel_df[rel_df['node'] == node]
    n_summations = int(np.max(rel_sub['summation']))
    n_params = len(parameters)
    param_idx = {p: i for i, p in enumerate(parameters)}

    coefficients = []
    coeff_params_list = []
    for s in range(1, n_summations + 1):
        sub = rel_sub[rel_sub['summation'] == s]
        coeff_vec  = np.zeros(n_params, dtype=np.float64)
        cp_vec     = np.zeros(n_params, dtype=np.int32)
        for _, row in sub.iterrows():
            p_key = row['parameter_key']
            if p_key in param_idx:
                j = param_idx[p_key]
                coeff_vec[j] = row['coefficient']
                if pd.notna(row['coeff_parameter_key']):
                    cpk = row['coeff_parameter_key']
                    if cpk not in param_idx and f'nuisance_{cpk}' in param_idx:
                        cpk = f'nuisance_{cpk}'
                    cp_vec[j] = param_idx[cpk] + 1  # +1 offset as in get_node_funcs
        coefficients.append(coeff_vec)
        coeff_params_list.append(cp_vec)

    return func_factory(eq_type, coefficients, coeff_params_list, jit=jit)
```

### 4.2 Fast paths in `get_translate_dep` and `get_adjust`

At the top of each function, before any loop or lambda construction, add an early return:

**`get_translate_dep`:**
```python
def get_translate_dep(dep_meas_data, parameters, nodes, parameter_funcs, node_funcs):
    n_meas = len(dep_meas_data)
    if all(d is None for d in dep_meas_data):
        return lambda params: jnp.zeros(n_meas, dtype=jnp.float64)
    # ... existing code unchanged ...
```

**`get_adjust`:**
```python
def get_adjust(adjust_data, parameters, nodes, parameter_funcs, node_funcs):
    n_meas = len(adjust_data)
    if all(d is None for d in adjust_data):
        return lambda params: jnp.ones(n_meas, dtype=jnp.float64)
    # ... existing code unchanged ...
```

These early returns matter beyond correctness: the current code constructs N lambda
closures even when every entry is None, and that closure-building cost per node adds up
across 1000 averages.

---

## 5. Changes to `avg.py`

Replace the four-call chain:

```python
node_funcs = get_node_funcs(nodes_list, parameters, fit_df, rel_df, jit=False)
parameter_funcs = get_parameter_funcs(parameters)
meas_funcs = get_meas_funcs(
    dict(zip(nodes_list, node_funcs)),
    dict(zip(parameters, parameter_funcs)),
    meas_df,
)
mu = get_mu(meas_funcs)
```

with:

```python
mu = get_mu_vectorized(parameters, nodes_list, meas_df, fit_df, rel_df, jit=False)
```

Also update the import line in `avg.py` to include `get_mu_vectorized`.  Remove the
imports of `get_node_funcs`, `get_parameter_funcs`, `get_meas_funcs`, `get_mu` if they
are no longer referenced anywhere else in `avg.py`.  (`translate_dep` and `adjust` are
still built with the existing helpers; only `mu` changes.)

---

## 6. Test Plan (write tests BEFORE switching callers)

### 6.1 New file: `tests/test_build_funcs.py`

This file must exist and all tests must pass BEFORE `avg.py` is changed.

**Test structure:**

```python
# tests/test_build_funcs.py
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest

from pdgfits.build_funcs import (
    get_node_funcs, get_parameter_funcs, get_meas_funcs, get_mu,
    get_mu_vectorized,
    get_translate_dep, get_adjust,
)
```

**Unit tests using synthetic DataFrames (no DB required):**

Construct minimal `parameters`, `nodes`, `meas_df`, `fit_df`, `rel_df` for each
scenario and assert `mu_new(params) == mu_old(params)`:

1. **Single `+` node, single measurement, one parameter** — the avg case.
2. **Single `+` node, multiple measurements sharing that node** — tests the gather
   (multiple rows same coefficient vector).
3. **Two `+` nodes, two parameters, four measurements** — tests the multi-row matrix.
4. **One nuisance parameter measurement** — `node = 'nuisance_X'`, not in `rel_df`,
   in `parameters`.  Coefficient row must be `e_j`.
5. **Mixed: one `+` node and one `lifetime` node** — verifies the nonlinear fallback
   path alongside the matrix path.
6. **`+` node with non-zero `coeff_parameter_key`** — must fall through to
   `func_factory` (non-simple `+`), not be included in the matrix.
7. **`get_translate_dep` fast path** — pass `dep_meas_data = [None, None, None]`,
   assert the result is `jnp.zeros(3)` and that no lambdas were built (check by timing
   or by inspecting the returned function directly).
8. **`get_adjust` fast path** — same as above for `adjust_data`.

**Integration test using real DB data (mark with `@pytest.mark.db`):**

```python
@pytest.mark.db
def test_mu_vectorized_matches_original_on_avg_nodes():
    from pdgfits.query import avg_queries
    avg_df, corr_df_dict = avg_queries(verbose=False)

    # Test on the first 20 nodes (covers variety without being slow)
    for node in avg_df['node'].unique()[:20]:
        meas_df_node = avg_df[avg_df['node'] == node]
        # ... run preprocess to get meas_df, rel_df, fit_df, parameters, nodes_list ...
        # ... build mu_old via get_node_funcs → get_meas_funcs → get_mu ...
        # ... build mu_new via get_mu_vectorized ...
        params = jnp.array([fit_seed_df[...]['seed'].iloc[0]], dtype=jnp.float64)
        assert jnp.allclose(mu_new(params), mu_old(params), rtol=1e-12, atol=0), \
            f"mu mismatch for node {node} at seed"
        # also check at a perturbed point
        params_perturbed = params * 1.1 + 0.01
        assert jnp.allclose(mu_new(params_perturbed), mu_old(params_perturbed),
                            rtol=1e-12, atol=0), \
            f"mu mismatch for node {node} at perturbed params"
```

---

## 7. Acceptance Criteria

1. `pytest tests/` passes (all existing tests plus the new `test_build_funcs.py` unit
   tests, without the `--db` flag).

2. With `--db`: the integration test above passes for all 20 sampled avg nodes.

3. End-to-end: run `python -m pdgfits.avg_test --node <some_node>` before and after the
   change and confirm `value`, `error_n`, `error_p`, `chi2` match to at least 9
   significant figures.

4. Performance: time `run_avg` on a node with ~30 measurements before and after.
   Expect `jax.grad(chi2)(params)` to be at least 5× faster with the new `mu`.
   (Measure the *second* call to avoid comparing compile time to steady-state time —
   the first call may trigger XLA tracing regardless.)

---

## 8. Follow-up Work (out of scope here)

- **`fit.py` migration:** once the tests are green and `avg.py` is validated, switch
  `fit.py` to also call `get_mu_vectorized`.  Keep `node_funcs` / `parameter_funcs`
  build (still needed for the fit result dict used by `asym_errors.py` and
  `diagnostics.py`).

- **`corr_mat_inv` fast path:** in `avg.py` (or inside `get_corr_mat`), when
  `corr_df` has no rows (no off-diagonal correlations), return `jnp.eye(n_meas)`
  directly instead of calling `jnp.linalg.pinv`.

- **Full `translate_dep` vectorisation:** for the non-None case, `translate_dep[i]` is
  a linear combination of node values plus a constant.  If the referenced nodes are
  themselves linear-simple, the whole thing is `D @ params + d` for a precomputed
  matrix D and vector d.  This is the same matrix-building pattern as `get_mu_vectorized`.

- **Full `adjust` vectorisation:** `adjust[i]` is a product of node values (branching
  fractions); it is not linear.  Vectorisation would require building a batch of node
  evaluations.  Harder; defer.
