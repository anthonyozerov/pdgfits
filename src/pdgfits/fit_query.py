import os
import warnings
import numpy as np

from dotenv import load_dotenv
import psycopg2
import pandas as pd

from pdgfits.parser import parse_measurement, get_scale

def make_conn():
    load_dotenv()
    conn = psycopg2.connect(host="127.0.0.1", port=5433, dbname=os.getenv("DB_NAME"), user=os.getenv("DB_USER"), password=os.getenv("DB_PW"))
    return conn

def all_fits():
    conn = make_conn()
    QUERY = f"""
    SELECT * FROM fit_control1
    """
    # suppress pandas sqlalchemy warning
    warnings.filterwarnings('ignore', category=UserWarning, 
                       message='.*pandas only supports SQLAlchemy.*')
    fits_df = pd.read_sql_query(QUERY, conn)
    return fits_df

def query_db(fit_label, verbose=True):
    conn = make_conn()

    # suppress pandas sqlalchemy warning
    warnings.filterwarnings('ignore', category=UserWarning, 
                       message='.*pandas only supports SQLAlchemy.*')
    
    QUERY = """
    SELECT algorithm, measurement_type, data_count FROM fit_control1 WHERE label = %s
    """
    if verbose:
        print(QUERY)
    fit_df = pd.read_sql_query(QUERY, conn, params=(fit_label,))
    algorithm = fit_df['algorithm'].iloc[0]
    measurement_type = fit_df['measurement_type'].iloc[0]
    data_count = fit_df['data_count'].iloc[0]
    if verbose:
        print(algorithm)

    # get the relevant nodes from the fit_control2 table
    QUERY = """
    SELECT fit_control2.node, relationship_equation.type, pdgid.data_type
    FROM fit_control2
    LEFT JOIN relationship_equation ON fit_control2.node = relationship_equation.node
    LEFT JOIN pdgid ON fit_control2.node = pdgid.pdgid
    WHERE label = %s
    """
    if verbose:
        print(QUERY)

    fit_df = pd.read_sql_query(QUERY, conn, params=(fit_label,))
    # print the nodes with no equation type:
    # print('Nodes with no equation type:')
    # print(list(fit_df[fit_df['type'].isna()]['node']))
    fit_df['type'] = fit_df['type'].fillna('+') # TODO: IS THIS RIGHT??
    nodes = list(fit_df["node"])

    QUERY = f"""
    SELECT tree.node, tree.data_type
    FROM tree
    WHERE tree.node IN ('{'\', \''.join(nodes)}')
    """
    if verbose:
        print(QUERY)
    tree_df = pd.read_sql_query(QUERY, conn)

    # get the parts of the relationship table containing these nodes
    QUERY = f"""
    SELECT r.node, r.par_code, r.parameter, r.coefficient, r.summation, r.coeff_par_code, r.coeff_parameter
    FROM relationship r
    WHERE r.node IN ('{'\', \''.join(nodes)}')
    AND r.alias_flag IS NULL
    """
    if verbose:
        print(QUERY)

    rel_df = pd.read_sql_query(QUERY, conn)
    rel_df['coefficient'] = rel_df['coefficient'].fillna(1)
    rel_df['summation'] = rel_df['summation'].fillna(1)

    assert rel_df['node'].isin(nodes).all()

    # get the measurements for all of our nodes
    # but only the measurements that are used ('U') and not removed (NULL publication_status)
    QUERY = f"""
    SELECT m.node, m.reference_id, m.occurrence, m.measurement as measurement, ma.measurement as measurement_adjust, u.power_of_ten, u.text, r.source_year, m.systematic_error_clump, m.systematic_error_clump2, im.node as ignore_minus
    FROM measurement m
    LEFT JOIN measurement_adjust ma ON m.node=ma.node AND m.reference_id=ma.reference_id AND m.occurrence=ma.occurrence
    LEFT JOIN units u ON m.node = u.node
    LEFT JOIN reference r ON m.reference_id = r.reference_id
    LEFT JOIN ignore_minus im ON m.node = im.node
    WHERE m.node IN ('{'\', \''.join(nodes)}')
    AND m.place = 'U'
    AND m.publication_status IS NULL
    AND m.fit_flag IS NULL
    AND (m.type IS NULL OR m.type = '{measurement_type}')
    AND m.measurement NOT LIKE '%to%'
    AND u.summary_year IS NULL
    """
    if verbose:
        print(QUERY)

    meas_df = pd.read_sql_query(QUERY, conn)
    if len(meas_df) != data_count:
        print('WARNING: number of measurements found does not match expected number')
    print(f'{len(meas_df)} measurements found for {fit_label}, {data_count} expected')

    # get the correlation coefficients between the measurements
    QUERY = f"""
    SELECT c.node_one, c.reference_id_one, c.occurrence_one, c.node_two, c.reference_id_two, c.occurrence_two, c.correlation
    FROM correlation c
    WHERE c.node_one IN ('{'\', \''.join(nodes)}') AND c.node_two IN ('{'\', \''.join(nodes)}')
    AND c.publication_status IS NULL
    """
    if verbose:
        print(QUERY)
    corr_df = pd.read_sql_query(QUERY, conn)

    # get the fit seeds for the parameters
    QUERY = """
    SELECT par_code, parameter, seed FROM fit_seed WHERE label = %s
    """
    if verbose:
        print(QUERY)
    fit_seed_df = pd.read_sql_query(QUERY, conn, params=(fit_label,))

    return algorithm, measurement_type, fit_df, rel_df, meas_df, corr_df, fit_seed_df, tree_df

def nuisance_corr(nuisance_params, verbose=True):
    nuisance_params = [p.removeprefix('nuisance_') for p in nuisance_params]
    conn = make_conn()
    QUERY = f"""
    SELECT par_code_row, parameter_row, par_code_column, parameter_column, coefficient
    FROM fit_correlation_matrix
    WHERE concat_ws('.', par_code_row, parameter_row) IN ('{'\', \''.join(nuisance_params)}') AND concat_ws('.', par_code_column, parameter_column) IN ('{'\', \''.join(nuisance_params)}')
    """
    if verbose:
        print(QUERY)
    corr_df = pd.read_sql_query(QUERY, conn)
    return corr_df

# function to get the PDG's values for a node
def pdg_value(node):
    node_split = node.split('.')
    if len(node_split) == 1:
        par_code = None
        parameter = node
    else:
        par_code = node_split[0]
        parameter = node_split[1]
    
    conn = make_conn()
    QUERY = """
    SELECT summary
    FROM result_summary
    WHERE ({par_code_cond}) AND parameter = %s
    AND summary_year IS NULL
    AND POSITION('~' IN summary) = 0 -- no tilde in summary
    """
    # suppress pandas sqlalchemy warning
    warnings.filterwarnings('ignore', category=UserWarning, 
                       message='.*pandas only supports SQLAlchemy.*')
    if par_code is None:
        par_code_cond = "par_code IS NULL"
        query_filled = QUERY.format(par_code_cond=par_code_cond)
        summary_df = pd.read_sql_query(query_filled, conn, params=(parameter,))

        QUERY = """
        SELECT text
        FROM units
        WHERE node = %s
        AND summary_year IS NULL
        """
        units_df = pd.read_sql_query(QUERY, conn, params=(parameter,))
        if len(units_df) > 0:
            unit_text = units_df['text'].iloc[0]
        else:
            unit_text = ''
    else:
        par_code_cond = "par_code = %s"
        query_filled = QUERY.format(par_code_cond=par_code_cond)
        summary_df = pd.read_sql_query(query_filled, conn, params=(par_code, parameter))
        unit_text = ''
    return summary_df, unit_text

# function to get the PDG's most precise value for a node
def pdg_most_precise_value(node):
    summary_df, unit_text = pdg_value(node)
    if len(summary_df) == 0:
        return None, None, None
    value, error_n, error_p, _ = zip(*summary_df['summary'].map(parse_measurement))
    error_p = np.array(error_p)
    error_n = np.array(error_n)
    value = np.array(value)
    scale = get_scale(unit_text)
    value *= scale
    error_p *= scale
    error_n *= scale


    error = (error_p + error_n) / 2
    most_precise_idx = np.argmin(error)
    value = value[most_precise_idx]
    error_p = error_p[most_precise_idx]
    error_n = error_n[most_precise_idx]
    return value, error_p, error_n
