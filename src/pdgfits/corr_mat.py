import numpy as np
import jax.numpy as jnp

def get_corr_mat(meas_df, corr_df):
    # build the correlation matrix

    # list of 'ids' for each measurement 
    meas_df_ids = list(zip(meas_df['node'], meas_df['reference_id'], meas_df['occurrence']))

    # form correlation matrix of measurements using the correlation table,
    # starting from an identity matrix
    corr_mat = np.eye(len(meas_df))
    for i, row in corr_df.iterrows():
        # build the ids of the two measurements
        meas_one_id = (row['node_one'], row['reference_id_one'], row['occurrence_one'])
        meas_two_id = (row['node_two'], row['reference_id_two'], row['occurrence_two'])
        
        # If both measurements are in the meas_df, then we add the correlation to the matrix
        if meas_one_id in meas_df_ids and meas_two_id in meas_df_ids:
            meas_one_idx = meas_df_ids.index(meas_one_id)
            meas_two_idx = meas_df_ids.index(meas_two_id)
            corr_mat[meas_one_idx, meas_two_idx] = row['correlation']
            corr_mat[meas_two_idx, meas_one_idx] = row['correlation']
    corr_mat = jnp.array(corr_mat, dtype=jnp.float64)
    return corr_mat