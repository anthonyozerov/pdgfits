"""Measurement correlations and the declared groups used by PDG fits."""

import numpy as np


def correlation_blocks(matrix, blocks=None):
    """Return disjoint measurement blocks, excluding isolated measurements.

    With no declaration, use connected nonzero entries of the matrix. Explicit
    blocks may also contain zero correlations, as in the PDG input format.
    """
    if blocks is not None:
        block_ids = np.full(len(matrix), -1, int)
        answer = []
        for number, block in enumerate(blocks):
            indices = np.asarray(block, dtype=int)
            if (indices.ndim != 1 or len(indices) < 2 or len(set(indices)) != len(indices)
                    or np.any(indices < 0) or np.any(indices >= len(matrix))):
                raise ValueError('Correlation blocks need distinct in-range measurement indices')
            if np.any(block_ids[indices] >= 0):
                raise ValueError('Correlation blocks must be disjoint')
            block_ids[indices] = number
            answer.append(sorted(indices.tolist()))
        rows, columns = np.nonzero(matrix - np.diag(np.diag(matrix)))
        if np.any((block_ids[rows] < 0) | (block_ids[rows] != block_ids[columns])):
            raise ValueError('Every nonzero correlation must belong to one explicit block')
        return answer

    remaining, answer = set(range(len(matrix))), []
    while remaining:
        component, pending = [], [min(remaining)]
        while pending:
            i = pending.pop()
            if i not in remaining:
                continue
            remaining.remove(i)
            component.append(i)
            pending.extend(j for j in remaining if matrix[i, j] != 0)
        if len(component) > 1:
            answer.append(sorted(component))
    return answer


def get_corr_mat(meas_df, corr_df, *, return_blocks=False):
    """Build R and optionally preserve explicitly declared correlation links.

    Measurement identity includes the node, reference and occurrence. An
    explicit zero correlation still links a PDG scale group (SDOFIT), unlike
    an absent correlation record. Rows for measurements outside this fit are
    ignored, as they are when building the numerical matrix.
    """
    identities = list(zip(meas_df['node'], meas_df['reference_id'], meas_df['occurrence']))
    matrix = np.eye(len(meas_df))
    links = np.eye(len(meas_df))
    for row in corr_df.itertuples(index=False):
        first = (row.node_one, row.reference_id_one, row.occurrence_one)
        second = (row.node_two, row.reference_id_two, row.occurrence_two)
        if first in identities and second in identities:
            i, j = identities.index(first), identities.index(second)
            matrix[i, j] = matrix[j, i] = row.correlation
            links[i, j] = links[j, i] = 1
    return (matrix, correlation_blocks(links)) if return_blocks else matrix
