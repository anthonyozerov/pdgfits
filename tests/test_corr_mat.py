import pandas as pd
import jax.numpy as jnp
import numpy as np
import pytest
import jax
jax.config.update("jax_enable_x64", True)

from pdgfits.corr_mat import get_corr_mat, correlation_blocks


def make_meas_df(rows):
    return pd.DataFrame(rows, columns=["node", "reference_id", "occurrence"])


def make_corr_df(rows):
    cols = ["node_one", "reference_id_one", "occurrence_one",
            "node_two", "reference_id_two", "occurrence_two", "correlation"]
    return pd.DataFrame(rows, columns=cols)


def test_empty_corr_df_gives_identity():
    meas_df = make_meas_df([("A", "r1", 0), ("B", "r2", 0), ("C", "r3", 0)])
    corr_df = make_corr_df([])
    mat = get_corr_mat(meas_df, corr_df)
    assert jnp.allclose(mat, jnp.eye(3))


def test_one_off_diagonal_entry():
    meas_df = make_meas_df([("nodeA", "ref1", 0), ("nodeB", "ref2", 0), ("nodeC", "ref3", 0)])
    corr_df = make_corr_df([("nodeA", "ref1", 0, "nodeB", "ref2", 0, 0.5)])
    mat = get_corr_mat(meas_df, corr_df)
    assert float(mat[0, 1]) == pytest.approx(0.5)
    assert float(mat[1, 0]) == pytest.approx(0.5)
    assert float(mat[0, 0]) == pytest.approx(1.0)
    assert float(mat[1, 1]) == pytest.approx(1.0)
    assert float(mat[2, 2]) == pytest.approx(1.0)
    assert float(mat[0, 2]) == pytest.approx(0.0)
    assert float(mat[1, 2]) == pytest.approx(0.0)


def test_unknown_node_ignored():
    meas_df = make_meas_df([("A", "r1", 0), ("B", "r2", 0)])
    corr_df = make_corr_df([("A", "r1", 0, "UNKNOWN", "rx", 0, 0.7)])
    mat = get_corr_mat(meas_df, corr_df)
    assert jnp.allclose(mat, jnp.eye(2))


def test_output_dtype_float64():
    meas_df = make_meas_df([("A", "r1", 0)])
    corr_df = make_corr_df([])
    mat = get_corr_mat(meas_df, corr_df)
    assert mat.dtype == jnp.float64


def test_declared_groups_include_zero_links_and_transitive_connections():
    data = make_meas_df([('X', 'r1', 1), ('X', 'r2', 1), ('Y', 'r3', 1),
                         ('Y', 'r4', 1), ('Y', 'r4', 2)])
    declared = make_corr_df([
        ('X', 'r2', 1, 'Y', 'r3', 1, .4),
        ('Y', 'r3', 1, 'Y', 'r4', 1, 0.),
    ])
    matrix, blocks = get_corr_mat(data, declared, return_blocks=True)
    assert blocks == [[1, 2, 3]]
    assert correlation_blocks(matrix) == [[1, 2]]
    assert correlation_blocks(matrix, blocks) == blocks
    assert matrix[2, 3] == 0
    np.testing.assert_array_equal(matrix, get_corr_mat(data, declared))


@pytest.mark.parametrize('blocks', [[[0, 1], [1, 2]], [[0]], [[-1, 1]], [[0, 4]], [[0, 0]]])
def test_invalid_declared_blocks_raise(blocks):
    with pytest.raises(ValueError):
        correlation_blocks(np.eye(3), blocks)


def test_declared_blocks_must_include_every_nonzero_correlation():
    with pytest.raises(ValueError, match='Every nonzero correlation'):
        correlation_blocks(np.array([[1., -.3], [-.3, 1.]]), [])
