import itertools
import numpy as np
import pandas as pd
from itertools import permutations, combinations
from pdgfits.parser import parse_measurement, get_scale, parameter_key, get_dep_meas_data, get_adjust_data
from pdgfits.query import pdg_most_precise_value, nuisance_corr
from collections import defaultdict

# ---------------------------------------------------------------------------
# Domain constants
#
# These node lists encode PDG-specific special cases. They were previously
# inlined inside preprocess(); they are named here so the logic that consumes
# them reads cleanly and so the constants are easy to audit/extend.
# ---------------------------------------------------------------------------

# If any of these nodes is present (or measurement_type == 'UNIV'), the
# universality / particle-antiparticle equality handling kicks in.
UNIVERSALITY_TRIGGER_NODES = ['S024DM', 'S024DTT', 'S022DM']

# Fake measurement nodes that should be removed and replaced by directly
# equating the two underlying parameters (e-mu universality; omega-/antiomega+
# and Xi-/antiXi+ mass and lifetime equality).
UNIVERSALITY_FAKE_NODES = [
    'S013L0U', 'S013LPD', 'S013LQD', 'S013MVD', 'S010L0U',
    'S024DM', 'S024DTT', 'S022DM',
]

# Nodes whose equation type is stored as 'G+' but should be treated as '+'.
EQ_TYPE_FORCE_PLUS = [
    'S013EPH', 'S013EP', 'S023D', 'S085DM', 'S086DM', 'S087DM',
    'S024DM', 'S024DTT',
]

# Equation types whose nodes are partial widths that sum into a total width.
PARTIAL_WIDTH_EQ_TYPES = ['G+', 'G*', 'R+']


# function to parse measurement strings
# first, if needed, for a br_adjust measurement that was scaled by the original authors,
# we unscale it.
# then we parse the rest of the measurement string into a value and errors
def unadjust_measurement(measurement):
    if 'br_adjust' in measurement:
        measurement = measurement.removeprefix('br_adjust:').strip()
        original_measurement = measurement.split(';')[0].strip()
        meas, error_p, error_n, last_err = parse_measurement(original_measurement)

        adjust_measurements = measurement.split(';')[1:]
        # print(adjust_measurements)
        adjustments = []
        for adjust_meas_str in adjust_measurements:
            if 'ADJUST' in adjust_meas_str:
                adjustments.append(adjust_meas_str)
                continue
            rel = adjust_meas_str.split(',')[0].strip()
            # print(rel)
            assert rel in ['/', '*'], f"Unknown relationship: {rel}"
            # print(adjust_meas_str)
            # print(adjust_meas_str.split(',')[1].strip())
            adjust_meas, adjust_error_p, adjust_error_n, _ = parse_measurement(adjust_meas_str.split(',')[1].strip())
            # print(adjust_meas, adjust_error_p, adjust_error_n)

            if rel == '/':
                meas_new = meas * adjust_meas
                error_p_new = meas_new * np.sqrt(max(error_p**2/meas**2 - adjust_error_n**2/adjust_meas**2, 0))
                error_n_new = meas_new * np.sqrt(max(error_n**2/meas**2 - adjust_error_p**2/adjust_meas**2, 0))
            elif rel == '*':
                meas_new = meas / adjust_meas
                error_p_new = meas_new * np.sqrt(max(error_p**2/meas**2 - adjust_error_p**2/adjust_meas**2, 0))
                error_n_new = meas_new * np.sqrt(max(error_n**2/meas**2 - adjust_error_n**2/adjust_meas**2, 0))

            meas = meas_new
            error_p = error_p_new
            error_n = error_n_new
            opposite_rel = '/' if rel == '*' else '*'
            adjust_node = adjust_meas_str.split(',')[2].strip()
            adjustments.append(f'{opposite_rel}, ADJUST, {adjust_node}')
        return (meas, error_p, error_n, None, adjustments)

    else:
        return (*parse_measurement(measurement), None)

# this function takes a rel_df with 6 rows for a systematic_error_clump2 and figures out
# which three nodes are X, Y, and D such that D = X - Y
# written by AI but seems to work
def organize_triplet_clump(rel_df_clump):
    # Each node -> frozenset of (parameter, coefficient) pairs
    nodes = {
        node: frozenset(g.groupby('parameter')['coefficient'].sum().items())
        for node, g in rel_df_clump.groupby('node')
    }

    # Reverse lookup: vector -> node name
    by_vec = {v: n for n, v in nodes.items()}

    def sub(a, b):
        d = dict(a)
        for p, c in b:
            d[p] = d.get(p, 0) - c
        return frozenset((p, c) for p, c in d.items() if c != 0)

    for x, y in permutations(nodes, 2):
        d = by_vec.get(sub(nodes[x], nodes[y]))
        if d and d not in (x, y):
            return {'X': x, 'Y': y, 'D': d}
    raise ValueError("No triplet clump found")


# ---------------------------------------------------------------------------
# preprocess() steps
#
# Each helper below is one self-contained transformation that preprocess()
# applies in sequence. They take and return the dataframes they touch; nothing
# is hidden in shared state. preprocess() at the bottom is the thin orchestrator.
# ---------------------------------------------------------------------------

def _equate_universality_pairs(rel_df, meas_df, fit_seed_df, measurement_type):
    """Remove fake measurements for e-mu universality and omega-/antiomega+ and
    Xi-/antiXi+ mass/lifetime equality, replacing each with a direct relationship
    between the two underlying parameters."""
    if measurement_type == 'UNIV' or any([n in list(meas_df['node']) for n in UNIVERSALITY_TRIGGER_NODES]):
        fake_nodes_in_fit = [node for node in UNIVERSALITY_FAKE_NODES if node in list(rel_df['node'])]
        for node in fake_nodes_in_fit:
            pair = list(rel_df['parameter'][rel_df['node'] == node].unique())
            if pair == ['S022M', 'S022M1']:
                pair = ['S022M1', 'S022M']

            assert len(pair) == 2
            if measurement_type == 'UNIV':
                print(f'for node {node} assuring universality, removing fake meas, and directly equating the two parameters {pair}')
            elif node == 'S024DM':
                print(f'for node {node} equating omega- and antiomega+ masses, directly equating the two parameters {pair}')
            elif node == 'S024DTT':
                print(f'for node {node} equating omega- and antiomega+ lifetimes, directly equating the two parameters {pair}')
            elif node == 'S022DM':
                print(f'for node {node} equating Xi- and antiXi+ masses, directly equating the two parameters {pair}')
            # remove the rows with this node from rel_df
            rel_df = rel_df[rel_df['node'] != node]
            # remove the measurements with this node
            meas_df = meas_df[meas_df['node'] != node]
            if node != 'S022DM':
                # add a row to rel_df for the equal parameter pair
                row = dict(node=pair[0], parameter=pair[1], coefficient=1, summation=1)
                rel_df = pd.concat([rel_df, pd.DataFrame([row])], ignore_index=True)
            # remove the pair[0] node from fit_seed_df
            fit_seed_df = fit_seed_df[fit_seed_df['parameter'] != pair[0]]

    return rel_df, meas_df, fit_seed_df


def _extract_dep_meas(meas_df):
    """Pull dep_meas offsets out of measurement strings, storing them in
    dep_meas_data and replacing each string with its actual value."""
    dep_meas_data = [get_dep_meas_data(m) for m in meas_df['measurement']]
    for i, row in meas_df.iterrows():
        if 'dep_meas' in row['measurement']:
            meas_df.loc[i, 'measurement'] = row['measurement'].split(':')[1].split(',')[0].strip()
    return meas_df, dep_meas_data


def _parse_measurements(meas_df):
    """Parse measurement strings into value/error columns, take absolute values
    for ignore_minus nodes, and extract br_adjust adjustment data."""
    # parse the measurement strings into values and positive and negative errors
    meas_df['value'], meas_df['error_p'], meas_df['error_n'], meas_df['last_err'], adjustments = zip(*meas_df['measurement'].map(unadjust_measurement))
    meas_df['error'] = (meas_df['error_p'] + meas_df['error_n']) / 2
    # where ignore_minus is not na, take the absolute value of the value
    ignore_minus_nodes = list(meas_df['ignore_minus'][meas_df['ignore_minus'].notna()].unique())
    if len(ignore_minus_nodes) > 0:
        print('nodes with ignore_minus:', ignore_minus_nodes)
        print('taking the absolute value of the value for nodes with ignore_minus')
        meas_df['value'] = np.where(meas_df['ignore_minus'].notna(), np.abs(meas_df['value']), meas_df['value'])
    # parse the information about br_adjust
    adjust_data = [get_adjust_data(a) for a in adjustments]
    return meas_df, adjust_data


def _apply_unit_scaling(meas_df):
    """Standardize units (e.g. eV -> MeV) by scaling values and errors.

    Needed for e.g. partial widths that sum up to a total width.
    TODO: this is maybe incomplete.
    """
    meas_df['scale'] = meas_df.apply(lambda row: get_scale(row['text']), axis=1)
    meas_df['value'] *= meas_df['scale']
    meas_df['error_p'] *= meas_df['scale']
    meas_df['error_n'] *= meas_df['scale']
    meas_df['error'] *= meas_df['scale']
    return meas_df


def _force_plus_eq_types(fit_df):
    """Override the equation type of nodes stored as 'G+' that should be '+' (???)."""
    for i, fit_df_row in fit_df.iterrows():
        if fit_df_row['node'] in EQ_TYPE_FORCE_PLUS:
            fit_df.loc[i, 'type'] = '+'
    return fit_df


def _add_lifetime_relations(fit_df, rel_df, fit_seed_df, algorithm):
    """For each lifetime node (data_type 'T'), insert a relationship lifetime=1/width
    and seed the width parameter from the lifetime seed (the width is the fitted
    parameter, not the lifetime)."""
    for i, fit_df_row in fit_df.iterrows():
        if fit_df_row['data_type'] == 'T' and algorithm != 'SPECIALT':
            lifetime_node = fit_df_row['node']
            print('Lifetime node found:', lifetime_node)
            width_node = lifetime_node[:4] + 'W'
            # assert width_node in list(fit_df['node']), f"Width node {width_node} not found in fit_df"
            # print('Width node found:', width_node)
            if lifetime_node in list(rel_df['node']):
                print('Lifetime node found in rel_df:', lifetime_node)
                print('rel_df rows with this node are being deleted')
                print('It is unclear if this is the correct thing to do!')
                rel_df = rel_df[rel_df['node'] != lifetime_node]
            print('Adding new row to rel_df for the lifetime node to make it 1 / the width node')
            row = dict(node=lifetime_node, parameter=width_node, coefficient=1, summation=1, parameter_key=width_node,
                    par_code=None, coeff_par_code=None, coeff_parameter=None, coeff_parameter_key=None)
            rel_df = pd.concat([rel_df, pd.DataFrame([row])], ignore_index=True)

            print('Putting a special equation type in fit_df for this lifetime node')
            # replace the row in fit_df with a row that has type 'lifetime'
            fit_df.loc[i, 'type'] = 'lifetime'
            # place a new row in the fit_seed table for the width node
            width_seed = 1/fit_seed_df[fit_seed_df['parameter'] == lifetime_node]['seed'].iloc[0]
            row = dict(par_code=None, parameter=width_node, seed=width_seed)
            fit_seed_df = pd.concat([fit_seed_df, pd.DataFrame([row])], ignore_index=True)
            # remove the row with the lifetime node from the fit_seed_df
            fit_seed_df = fit_seed_df[fit_seed_df['parameter'] != lifetime_node]
    return fit_df, rel_df, fit_seed_df


def _add_partial_width_relations(fit_df, rel_df, tree_df):
    """Add relationships between partial widths and total widths.

    Each partial-width-to-width relationship is represented as an additional row
    in the relationship table: node = partial width, parameter = width, in a new
    summation (which contains only the one total width parameter).
    """
    # get all nodes corresponding to widths, and check that each particle only has one
    width_nodes = list(tree_df[tree_df['data_type'] == 'G']['node'])
    width_node_particles = list([node[:4] for node in width_nodes])
    assert len(width_node_particles) == len(set(width_node_particles)), f"At least one particle with multiple widths: {width_nodes}"

    for node in rel_df['node'].unique():
        eq_type = fit_df[fit_df['node'] == node]['type'].iloc[0]
        if eq_type in PARTIAL_WIDTH_EQ_TYPES:
            data_type = tree_df[tree_df['node'] == node]['data_type'].iloc[0]
            assert pd.isna(data_type) or data_type == 'E', f"Node {node} has data type {data_type}, expected partial width"
            # print(node)
            # print(width_node_particles)
            assert node[:4] in width_node_particles
            width_node = width_nodes[width_node_particles.index(node[:4])]

            # add a row to rel_df to represent the relationship between the partial width and the width
            max_summation = np.max(rel_df[rel_df['node'] == node]['summation'])
            row = dict(node=node, parameter=width_node, coefficient=1, summation=max_summation+1, parameter_key=width_node,
                    par_code=None, coeff_par_code=None, coeff_parameter=None, coeff_parameter_key=None)
            rel_df = pd.concat([rel_df, pd.DataFrame([row])], ignore_index=True)
    return rel_df


def _build_parameter_keys(rel_df, fit_seed_df):
    """Make keys for parameters/nodes (e.g. 'S013.1', 'S013M') and collect the
    lists of parameters (from fit_seed_df) and nodes (from rel_df)."""
    rel_df['parameter_key'] = rel_df.apply(lambda row: parameter_key(row['par_code'], row['parameter']), axis=1)
    rel_df['coeff_parameter_key'] = rel_df.apply(lambda row: parameter_key(row['coeff_par_code'], row['coeff_parameter']), axis=1)
    fit_seed_df['parameter_key'] = fit_seed_df.apply(lambda row: parameter_key(row['par_code'], row['parameter']), axis=1)

    # parameters are anything in fit_seed_df. nodes are the nodes in rel_df.
    parameters = list(fit_seed_df['parameter_key'].unique())
    nodes = list(rel_df['node'].unique())
    return rel_df, fit_seed_df, parameters, nodes


def _add_nuisance_parameters(rel_df, meas_df, corr_df, fit_seed_df, adjust_data, dep_meas_data, parameters, nodes):
    """Add nuisance parameters for br_adjust / coefficient / dep_meas target
    nodes that are referenced but not themselves in the fit. Each gets a fake
    measurement from the most precise PDG value, plus any correlations between
    nuisance parameters from the fit correlation matrix."""
    # get what the nodes are for br_adjust and coeff_parameters
    # if they are not in the fit, we will add them as nuisance parameters
    adjust_nodes = list(itertools.chain.from_iterable([data[0] for data in adjust_data if data is not None]))
    adjust_nodes = list(np.unique(adjust_nodes))
    if len(adjust_nodes) > 0:
        print('br_adjust nodes:', adjust_nodes)
    adjust_nodes_not_in_fit = [node for node in adjust_nodes if node not in parameters+nodes]
    if len(adjust_nodes_not_in_fit) > 0:
        print('br_adjust nodes not in fit:', adjust_nodes_not_in_fit)
    coeff_nodes = list(np.unique(rel_df['coeff_parameter_key'][rel_df['coeff_parameter_key'].notna()]))
    coeff_nodes_not_in_fit = [node for node in coeff_nodes if node not in parameters+nodes]
    if len(coeff_nodes_not_in_fit) > 0:
        print('coeff nodes not in fit:', coeff_nodes_not_in_fit)

    dep_meas_nodes = list(itertools.chain.from_iterable([data[0] for data in dep_meas_data if data is not None]))
    dep_meas_nodes = list(np.unique(dep_meas_nodes))
    if len(dep_meas_nodes) > 0:
        print('dep_meas nodes:', dep_meas_nodes)
    dep_meas_nodes_not_in_fit = [node for node in dep_meas_nodes if node not in parameters+nodes]
    if len(dep_meas_nodes_not_in_fit) > 0:
        print('dep_meas nodes not in fit:', dep_meas_nodes_not_in_fit)
    nuisance_params = []

    # add nuisance parameters for any nodes that are not in the fit
    # NOTE: sorted() (rather than a bare set) so the order in which nuisance rows
    # are appended is deterministic across runs. The previous bare-set iteration
    # produced a non-reproducible row order in meas_df/fit_seed_df; this is purely
    # a representational fix (results are invariant to measurement ordering).
    for node in sorted(set(adjust_nodes_not_in_fit + coeff_nodes_not_in_fit + dep_meas_nodes_not_in_fit)):
        nuisance_node = f'nuisance_{node}'
        nuisance_params.append(nuisance_node)
        print('adding nuisance parameter:', nuisance_node)

        # create a fake measurement using the most precise PDG value and error
        value, error_p, error_n = pdg_most_precise_value(node)
        meas_dict = {'node': [nuisance_node], 'value': [value], 'error_p': [error_p], 'error_n': [error_n], 'reference_id': [None], 'occurrence': [None]}
        meas_df = pd.concat([meas_df, pd.DataFrame(meas_dict)], ignore_index=True)
        fit_seed_df = pd.concat([fit_seed_df, pd.DataFrame({'parameter_key': [nuisance_node], 'seed': [value]})], ignore_index=True)

        # this fake measurement has no adjustment or dep_meas
        adjust_data.append(None)
        dep_meas_data.append(None)

    # add correlations between nuisance parameters
    if len(nuisance_params) > 0:
        nuisance_corr_df = nuisance_corr(nuisance_params, verbose=False)
        if len(nuisance_corr_df) > 0:
            print('nuisance correlations found, adding to corr_df')
            print(nuisance_corr_df)
            nuisance_corr_df['node_one'] = nuisance_corr_df.apply(lambda row: parameter_key(row['par_code_row'], row['parameter_row']), axis=1)
            nuisance_corr_df['node_two'] = nuisance_corr_df.apply(lambda row: parameter_key(row['par_code_column'], row['parameter_column']), axis=1)
            nuisance_corr_df['correlation'] = nuisance_corr_df['coefficient']/100
            nuisance_corr_df['reference_id_one'] = None
            nuisance_corr_df['occurrence_one'] = None
            nuisance_corr_df['reference_id_two'] = None
            nuisance_corr_df['occurrence_two'] = None
            # prepend 'nuisance_' to node_one and node_two
            nuisance_corr_df['node_one'] = nuisance_corr_df['node_one'].apply(lambda x: 'nuisance_'+x)
            nuisance_corr_df['node_two'] = nuisance_corr_df['node_two'].apply(lambda x: 'nuisance_'+x)
            # drop columns from the nuisance_corr_df to make it have the same cols as corr_df
            nuisance_corr_df = nuisance_corr_df.drop(columns=['par_code_row', 'parameter_row', 'par_code_column', 'parameter_column', 'coefficient'])
            # add to corr_df
            corr_df = pd.concat([corr_df, nuisance_corr_df], ignore_index=True)

    return meas_df, corr_df, fit_seed_df, adjust_data, dep_meas_data


def _add_clump2_correlations(meas_df, rel_df, corr_df):
    """Handle triplets of measurements handled by COMER2 (systematic_error_clump2).

    For each triplet of nodes X, Y, D such that D = X - Y, infer the three
    pairwise correlations from the three reported errors and add them to corr_df.
    """
    clumps = list(meas_df['systematic_error_clump2'][meas_df['systematic_error_clump2'].notna()].unique())
    for clump in clumps:
        print('clump:', clump)
        meas_df_clump = meas_df[meas_df['systematic_error_clump2'] == clump]
        clump_nodes = list(meas_df_clump['node'])
        print('nodes:', clump_nodes)
        if len(meas_df_clump) != 3:
            print(f'WARNING: clump2 "{clump}" has {len(meas_df_clump)} measurements, expected 3. Skipping.')
            continue
        rel_df_clump = rel_df[rel_df['node'].isin(clump_nodes)]
        if len(rel_df_clump) == 2:
            assert len(rel_df_clump['node'].unique()) == 1
            D = rel_df_clump['node'].iloc[0] # D is the node that is X-Y
            assert set(rel_df_clump['coefficient'].unique()) == {1, -1}
            X = rel_df_clump['parameter'].iloc[np.argmax(rel_df_clump['coefficient'])]
            Y = rel_df_clump['parameter'].iloc[np.argmin(rel_df_clump['coefficient'])]
            node_names = {'D': D, 'X': X, 'Y': Y}
        else:
            assert len(rel_df_clump) == 6
            node_names = organize_triplet_clump(rel_df_clump)
            D = node_names['D']
            X = node_names['X']
            Y = node_names['Y']

        print(node_names)

        rows = {k: meas_df_clump[meas_df_clump['node'] == v].iloc[0] for k,v in node_names.items()}

        values = {k: row['value'] for k, row in rows.items()}
        # print('values:', values)
        errors = {}
        ref = {k: row['reference_id'] for k, row in rows.items()}
        occ = {k: row['occurrence'] for k, row in rows.items()}
        for k, row in rows.items():
            error_p = row['error_p']
            error_n = row['error_n']
            if not np.isclose(error_p, error_n):
                print(f'Warning: clump2 "{clump}", node {node_names[k]} has asymmetric errors, symmetrizing.')
            error = (error_p + error_n) / 2
            errors[k] = error
        # print('errors:', errors)
        # print('errors**2:', {k: v**2 for k, v in errors.items()})
        covXY = (errors['X']**2+errors['Y']**2-errors['D']**2)/2
        corrXY = covXY / (errors['X'] * errors['Y'])
        # print('covXY:', covXY)
        covXD = errors['X']**2 - covXY
        corrXD = covXD / (errors['X'] * errors['D'])
        covYD = -errors['Y']**2 + covXY
        corrYD = covYD / (errors['Y'] * errors['D'])
        print('corrXY:', corrXY)
        print('corrXD:', corrXD)
        print('corrYD:', corrYD)
        assert (-1<= corrXY and corrXY <= 1 and -1<= corrXD and corrXD <= 1 and -1<= corrYD and corrYD <= 1)

        # add 3 entries to the correlation matrix for this triplet
        extra_corr = {
            'node_one': [X, X, Y],
            'reference_id_one': [ref['X'], ref['X'], ref['Y']],
            'occurrence_one': [occ['X'], occ['X'], occ['Y']],
            'node_two': [Y, D, D],
            'reference_id_two': [ref['Y'], ref['D'], ref['D']],
            'occurrence_two': [occ['Y'], occ['D'], occ['D']],
            'correlation': [corrXY, corrXD, corrYD]
        }
        corr_df = pd.concat([corr_df, pd.DataFrame(extra_corr)], ignore_index=True)
    return corr_df


def _add_clump_correlations(meas_df, corr_df):
    """Handle systematic_error_clump: add pairwise correlations between
    measurements of one node that share a systematic error (assumed 100%
    correlated)."""
    clumps = list(meas_df['systematic_error_clump'][meas_df['systematic_error_clump'].notna()].unique())
    for clump in clumps:
        print('Handling systematic_error_clump:', clump)

        meas_df_clump = meas_df[meas_df['systematic_error_clump'] == clump]
        # print(meas_df_clump)
        clump_nodes = list(meas_df_clump['node'].unique())
        assert len(clump_nodes) == 1, f"Code assumes that each systematic_error_clump has only one node in the fit"
        node = clump_nodes[0]

        print('Node in this clump:', node)
        print('Measurements in this clump:', len(meas_df_clump))

        syst_err = np.array(meas_df_clump['last_err'])
        total_err = np.array(meas_df_clump['error'])
        ref = np.array(meas_df_clump['reference_id'])
        occ = np.array(meas_df_clump['occurrence'])

        corr_df_vals = corr_df[['node_one', 'node_two', 'reference_id_one', 'reference_id_two', 'occurrence_one', 'occurrence_two']].values.tolist()

        # for every pair of measurements, calculate their correlation, and add it to corr_df
        for (i,j) in combinations(range(len(meas_df_clump)), 2):
            # check that there is no entry in corr_df with these nodes
            if (node, node, ref[i], occ[i], ref[j], occ[j]) in corr_df_vals or (node, node, ref[j], occ[j], ref[i], occ[i]) in corr_df_vals:
                raise ValueError(f"Correlation between systematic error clump measurements {i} and {j} already exists in the correlation table. Not supported.")

            # calculate the correlation between this pair of measurements
            # assumption: the systematic error in both measurements is 100% correlated
            cov_syst_ij = 1*(syst_err[i])*(syst_err[j])
            corr_ij = cov_syst_ij / (total_err[i] * total_err[j])
            print(f'correlation between {node} clump {clump} measurements {i} and {j}: {corr_ij}')
            assert (-1<= corr_ij and corr_ij <= 1)

            # add an entry to the correlation matrix
            extra_corr = {
                'node_one': [node],
                'reference_id_one': [ref[i]],
                'occurrence_one': [occ[i]],
                'node_two': [node],
                'reference_id_two': [ref[j]],
                'occurrence_two': [occ[j]],
                'correlation': [corr_ij]
            }
            corr_df = pd.concat([corr_df, pd.DataFrame(extra_corr)], ignore_index=True)
    return corr_df


def preprocess(fit_df, rel_df, meas_df, corr_df, fit_seed_df, tree_df, algorithm, measurement_type):
    """Turn raw query results into the dataframes the fitter consumes.

    Each step is a named helper (see above); this is just the pipeline.
    """
    rel_df, meas_df, fit_seed_df = _equate_universality_pairs(
        rel_df, meas_df, fit_seed_df, measurement_type)

    meas_df, dep_meas_data = _extract_dep_meas(meas_df)
    meas_df, adjust_data = _parse_measurements(meas_df)
    meas_df = _apply_unit_scaling(meas_df)

    fit_df = _force_plus_eq_types(fit_df)
    fit_df, rel_df, fit_seed_df = _add_lifetime_relations(fit_df, rel_df, fit_seed_df, algorithm)
    rel_df = _add_partial_width_relations(fit_df, rel_df, tree_df)

    rel_df, fit_seed_df, parameters, nodes = _build_parameter_keys(rel_df, fit_seed_df)

    meas_df, corr_df, fit_seed_df, adjust_data, dep_meas_data = _add_nuisance_parameters(
        rel_df, meas_df, corr_df, fit_seed_df, adjust_data, dep_meas_data, parameters, nodes)

    corr_df = _add_clump2_correlations(meas_df, rel_df, corr_df)
    corr_df = _add_clump_correlations(meas_df, corr_df)

    return fit_df, rel_df, meas_df, corr_df, fit_seed_df, dep_meas_data, adjust_data
