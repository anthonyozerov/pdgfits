import os
import pickle
import re
from hashlib import sha1
from pathlib import Path
import warnings
import numpy as np

from dotenv import load_dotenv
import psycopg2
import pandas as pd

from pdgfits.parser import parse_measurement, get_scale

SNAPSHOT_BACKEND = "snapshot"
SNAPSHOT_DIR_ENV = "PDGFITS_SNAPSHOT_DIR"
DATA_BACKEND_ENV = "PDGFITS_DATA_BACKEND"


def _data_backend():
    return os.getenv(DATA_BACKEND_ENV, "db").strip().lower()


def _using_snapshot():
    return _data_backend() == SNAPSHOT_BACKEND


def _snapshot_dir():
    root = os.getenv(SNAPSHOT_DIR_ENV)
    if not root:
        raise RuntimeError(
            f"{SNAPSHOT_DIR_ENV} must be set when {DATA_BACKEND_ENV}=snapshot"
        )
    return Path(root)


def _snapshot_key(value):
    """Return a readable, collision-resistant filename stem for snapshot data."""
    value = str(value)
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_") or "empty"
    digest = sha1(value.encode("utf-8")).hexdigest()[:10]
    return f"{slug[:80]}-{digest}"


def _nuisance_corr_key(nuisance_params):
    stripped = sorted(p.removeprefix("nuisance_") for p in nuisance_params)
    return _snapshot_key("\0".join(stripped))


def _sql_quote_list(values):
    quoted = []
    for value in values:
        escaped = str(value).replace("'", "''")
        quoted.append(f"'{escaped}'")
    return ", ".join(quoted)


def _read_snapshot_pickle(*parts):
    path = _snapshot_dir().joinpath(*parts)
    if not path.exists():
        rel = path.relative_to(_snapshot_dir())
        raise FileNotFoundError(
            f"Snapshot data not found: {rel}. Re-capture the snapshot with this "
            "fit/node included."
        )
    with open(path, "rb") as f:
        return pickle.load(f)


def _write_snapshot_pickle(obj, *parts):
    path = _snapshot_dir().joinpath(*parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def make_conn():
    load_dotenv()
    conn = psycopg2.connect(host="127.0.0.1", port=5433, dbname=os.getenv("DB_NAME"), user=os.getenv("DB_USER"), password=os.getenv("DB_PW"))
    return conn

def all_fits():
    if _using_snapshot():
        return _read_snapshot_pickle("all_fits.pkl")

    conn = make_conn()
    QUERY = f"""
    SELECT * FROM fit_control1
    """
    # suppress pandas sqlalchemy warning
    warnings.filterwarnings('ignore', category=UserWarning,
                       message='.*pandas only supports SQLAlchemy.*')
    fits_df = pd.read_sql_query(QUERY, conn)
    return fits_df

def fit_queries(fit_label, verbose=True):
    if _using_snapshot():
        return _read_snapshot_pickle("fits", f"{_snapshot_key(fit_label)}.pkl")

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
    nodes_sql = _sql_quote_list(nodes)

    QUERY = f"""
    SELECT tree.node, tree.data_type
    FROM tree
    WHERE tree.node IN ({nodes_sql})
    """
    if verbose:
        print(QUERY)
    tree_df = pd.read_sql_query(QUERY, conn)

    # get the parts of the relationship table containing these nodes
    QUERY = f"""
    SELECT r.node, r.par_code, r.parameter, r.coefficient, r.summation, r.coeff_par_code, r.coeff_parameter
    FROM relationship r
    WHERE r.node IN ({nodes_sql})
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
    WHERE m.node IN ({nodes_sql})
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
    WHERE c.node_one IN ({nodes_sql}) AND c.node_two IN ({nodes_sql})
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

def avg_queries(verbose=True):
    if _using_snapshot():
        return _read_snapshot_pickle("avg_queries.pkl")

    conn = make_conn()

    # suppress pandas sqlalchemy warning
    warnings.filterwarnings('ignore', category=UserWarning,
                       message='.*pandas only supports SQLAlchemy.*')

    QUERY = """
    select *
    from (
        select m.node, m.reference_id, m.occurrence, m.measurement as measurement, u.power_of_ten, u.text, r.source_year, m.systematic_error_clump, m.systematic_error_clump2, im.node as ignore_minus, count(*) over (partition by m.node) as node_count
        from measurement m
        left join average_control ac on m.node=ac.node
        LEFT JOIN units u ON m.node = u.node
        LEFT JOIN reference r ON m.reference_id = r.reference_id
        LEFT JOIN ignore_minus im ON m.node = im.node
        where ac.suppress_computation is null and ac.suppress_average is null
        and m.place = 'U'
        and m.publication_status is null
        and m.confidence_level is null
        and measurement not like '*%'
        and measurement not like '%<%'
        and measurement not like '%>%'
        and measurement not like '%seen'
        and measurement not like '~%'
        and measurement like '%+%'
        and ltrim(measurement) not like '@%'
        and u.summary_year is null
    ) filtered
    where node_count > 1
    """
    avg_df = pd.read_sql_query(QUERY, conn)

    nodes = list(avg_df['node'].unique())
    nodes_sql = _sql_quote_list(nodes)

    # get the correlation coefficients between the measurements
    QUERY = f"""
    SELECT c.node_one, c.reference_id_one, c.occurrence_one, c.node_two, c.reference_id_two, c.occurrence_two, c.correlation
    FROM correlation c
    WHERE c.node_one IN ({nodes_sql}) AND c.node_two=c.node_one
    AND c.publication_status IS NULL
    """
    if verbose:
        print(QUERY)
    corr_df = pd.read_sql_query(QUERY, conn)
    if verbose:
        print(f"Correlation df has {len(corr_df)} entries")

    corr_df_dict = {node: corr_df[corr_df['node_one'] == node] for node in nodes}

    return avg_df, corr_df_dict


def nuisance_corr(nuisance_params, verbose=True):
    if _using_snapshot():
        return _read_snapshot_pickle(
            "nuisance_corr", f"{_nuisance_corr_key(nuisance_params)}.pkl"
        )

    nuisance_params = [p.removeprefix('nuisance_') for p in nuisance_params]
    nuisance_params_sql = _sql_quote_list(nuisance_params)
    conn = make_conn()
    QUERY = f"""
    SELECT par_code_row, parameter_row, par_code_column, parameter_column, coefficient
    FROM fit_correlation_matrix
    WHERE concat_ws('.', par_code_row, parameter_row) IN ({nuisance_params_sql}) AND concat_ws('.', par_code_column, parameter_column) IN ({nuisance_params_sql})
    AND type NOT LIKE 'DR'
    """
    if verbose:
        print(QUERY)
    corr_df = pd.read_sql_query(QUERY, conn)
    return corr_df

# function to get the PDG's values for a node
def pdg_value(node):
    if _using_snapshot():
        raise RuntimeError(
            "pdg_value() is not stored in simple snapshots; use "
            "pdg_most_precise_value(), or extend the capture if raw summaries "
            "are needed."
        )

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
    AND type NOT LIKE 'DR'
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
    if _using_snapshot():
        return _read_snapshot_pickle(
            "pdg_most_precise_value", f"{_snapshot_key(node)}.pkl"
        )

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
