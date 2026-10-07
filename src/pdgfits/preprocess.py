"""Translate PDG tables into measurements, relationships and auxiliary inputs."""

from itertools import combinations, permutations

import numpy as np
import pandas as pd

from pdgfits.parser import (
    get_adjust_data, get_dep_meas_data, get_scale, parameter_key, parse_measurement,
)
from pdgfits.query import nuisance_corr, pdg_most_precise_value


UNIVERSALITY_TRIGGER_NODES = ['S024DM', 'S024DTT', 'S022DM']
UNIVERSALITY_FAKE_NODES = [
    'S013L0U', 'S013LPD', 'S013LQD', 'S013MVD', 'S010L0U',
    'S024DM', 'S024DTT', 'S022DM',
]
EQ_TYPE_FORCE_PLUS = [
    'S013EPH', 'S013EP', 'S023D', 'S085DM', 'S086DM', 'S087DM',
    'S024DM', 'S024DTT',
]
PARTIAL_WIDTH_EQ_TYPES = ['G+', 'G*', 'R+']


def unadjust_measurement(measurement):
    """Undo an author's branching-ratio adjustment before parsing the input.

    Remove its quoted variance contribution and record the inverse adjustment,
    which the fit will apply using its own auxiliary parameter.
    """
    if 'br_adjust' not in measurement:
        return (*parse_measurement(measurement), None)

    original, *adjustments = measurement.removeprefix('br_adjust:').strip().split(';')
    value, error_p, error_n, _ = parse_measurement(original.strip())
    remaining = []
    for adjustment in adjustments:
        if 'ADJUST' in adjustment:
            remaining.append(adjustment)
            continue
        operator, quoted, node = [part.strip() for part in adjustment.split(',')[:3]]
        assert operator in ('/', '*'), f'Unknown relationship: {operator}'
        factor, factor_p, factor_n, _ = parse_measurement(quoted)
        if operator == '/':
            adjusted = value * factor
            new_p = adjusted * np.sqrt(max(error_p**2/value**2 - factor_n**2/factor**2, 0))
            new_n = adjusted * np.sqrt(max(error_n**2/value**2 - factor_p**2/factor**2, 0))
        else:
            adjusted = value / factor
            new_p = adjusted * np.sqrt(max(error_p**2/value**2 - factor_p**2/factor**2, 0))
            new_n = adjusted * np.sqrt(max(error_n**2/value**2 - factor_n**2/factor**2, 0))
        value, error_p, error_n = adjusted, new_p, new_n
        inverse = '/' if operator == '*' else '*'
        remaining.append(f'{inverse}, ADJUST, {node}')
    return value, error_p, error_n, None, remaining


def organize_triplet_clump(rel_df_clump):
    """Identify the nodes X, Y, D whose linear relationships satisfy D = X - Y."""
    vectors = {
        node: frozenset(group.groupby('parameter')['coefficient'].sum().items())
        for node, group in rel_df_clump.groupby('node')
    }
    by_vector = {vector: node for node, vector in vectors.items()}
    for x, y in permutations(vectors, 2):
        difference = dict(vectors[x])
        for parameter, coefficient in vectors[y]:
            difference[parameter] = difference.get(parameter, 0) - coefficient
        vector = frozenset((p, c) for p, c in difference.items() if c != 0)
        d = by_vector.get(vector)
        if d and d not in (x, y):
            return {'X': x, 'Y': y, 'D': d}
    raise ValueError('No triplet clump found')


def preprocess(fit_df, rel_df, meas_df, corr_df, fit_seed_df, tree_df, algorithm, measurement_type):
    """Prepare fit tables in order, retaining measurement and parameter ordering.

    Quoted errors are parsed first; relationships and units are then resolved.
    External inputs become nuisance parameters with auxiliary measurements.
    Finally, shared-error groups supply measurement correlations. The returned
    adjustment lists have one entry per measurement, including auxiliary rows.
    """
    # Replace artificial universality measurements by parameter equalities.
    if measurement_type == 'UNIV' or meas_df['node'].isin(UNIVERSALITY_TRIGGER_NODES).any():
        for node in UNIVERSALITY_FAKE_NODES:
            if node not in rel_df['node'].values:
                continue
            pair = list(rel_df.loc[rel_df['node'] == node, 'parameter'].unique())
            if pair == ['S022M', 'S022M1']:
                pair = ['S022M1', 'S022M']
            assert len(pair) == 2
            rel_df = rel_df[rel_df['node'] != node]
            meas_df = meas_df[meas_df['node'] != node]
            if node != 'S022DM':
                row = dict(node=pair[0], parameter=pair[1], coefficient=1, summation=1)
                rel_df = pd.concat([rel_df, pd.DataFrame([row])], ignore_index=True)
            fit_seed_df = fit_seed_df[fit_seed_df['parameter'] != pair[0]]

    # Extract dependent-measurement offsets before parsing the quoted numbers.
    dep_meas_data = [get_dep_meas_data(m) for m in meas_df['measurement']]
    for i, measurement in meas_df['measurement'].items():
        if 'dep_meas' in measurement:
            meas_df.loc[i, 'measurement'] = measurement.split(':')[1].split(',')[0].strip()
    parsed = list(meas_df['measurement'].map(unadjust_measurement))
    meas_df['value'], meas_df['error_p'], meas_df['error_n'], meas_df['last_err'], adjustments = zip(*parsed)
    meas_df['error'] = (meas_df['error_p'] + meas_df['error_n']) / 2
    if meas_df['ignore_minus'].notna().any():
        meas_df['value'] = np.where(meas_df['ignore_minus'].notna(), np.abs(meas_df['value']), meas_df['value'])
    adjust_data = [get_adjust_data(a) for a in adjustments]

    # Preserve the existing unit convention, including the stored last_err.
    meas_df['scale'] = meas_df['text'].map(get_scale)
    for column in ['value', 'error_p', 'error_n', 'error']:
        meas_df[column] *= meas_df['scale']
    fit_df.loc[fit_df['node'].isin(EQ_TYPE_FORCE_PLUS), 'type'] = '+'

    # Lifetimes use reciprocal-width parameters, except in SPECIALT fits.
    for i, row in fit_df.iterrows():
        if row['data_type'] != 'T' or algorithm == 'SPECIALT':
            continue
        lifetime = row['node']
        width = lifetime[:4] + 'W'
        if lifetime in rel_df['node'].values:
            print(f'Replacing the existing relationship for lifetime {lifetime} by 1/{width}.')
            rel_df = rel_df[rel_df['node'] != lifetime]
        relation = dict(node=lifetime, parameter=width, coefficient=1, summation=1,
                        parameter_key=width, par_code=None, coeff_par_code=None,
                        coeff_parameter=None, coeff_parameter_key=None)
        rel_df = pd.concat([rel_df, pd.DataFrame([relation])], ignore_index=True)
        fit_df.loc[i, 'type'] = 'lifetime'
        seed = 1/fit_seed_df.loc[fit_seed_df['parameter'] == lifetime, 'seed'].iloc[0]
        fit_seed_df = pd.concat([
            fit_seed_df, pd.DataFrame([dict(par_code=None, parameter=width, seed=seed)])
        ], ignore_index=True)
        fit_seed_df = fit_seed_df[fit_seed_df['parameter'] != lifetime]

    # Partial-width relationships acquire a total-width factor as a new term.
    width_nodes = list(tree_df.loc[tree_df['data_type'] == 'G', 'node'])
    width_particles = [node[:4] for node in width_nodes]
    assert len(width_particles) == len(set(width_particles)), f'At least one particle with multiple widths: {width_nodes}'
    for node in rel_df['node'].unique():
        equation = fit_df.loc[fit_df['node'] == node, 'type'].iloc[0]
        if equation not in PARTIAL_WIDTH_EQ_TYPES:
            continue
        data_type = tree_df.loc[tree_df['node'] == node, 'data_type'].iloc[0]
        assert pd.isna(data_type) or data_type == 'E', f'Node {node} has data type {data_type}, expected partial width'
        assert node[:4] in width_particles
        width = width_nodes[width_particles.index(node[:4])]
        summation = np.max(rel_df.loc[rel_df['node'] == node, 'summation']) + 1
        relation = dict(node=node, parameter=width, coefficient=1, summation=summation,
                        parameter_key=width, par_code=None, coeff_par_code=None,
                        coeff_parameter=None, coeff_parameter_key=None)
        rel_df = pd.concat([rel_df, pd.DataFrame([relation])], ignore_index=True)

    # Canonical parameter names distinguish branching ratios from other nodes.
    rel_df['parameter_key'] = rel_df.apply(lambda r: parameter_key(r['par_code'], r['parameter']), axis=1)
    rel_df['coeff_parameter_key'] = rel_df.apply(lambda r: parameter_key(r['coeff_par_code'], r['coeff_parameter']), axis=1)
    fit_seed_df['parameter_key'] = fit_seed_df.apply(lambda r: parameter_key(r['par_code'], r['parameter']), axis=1)
    parameters = list(fit_seed_df['parameter_key'].unique())
    nodes = list(rel_df['node'].unique())

    # External adjustment, coefficient and offset inputs become auxiliary rows.
    referenced = {node for data in adjust_data + dep_meas_data if data is not None for node in data[0]}
    referenced.update(rel_df['coeff_parameter_key'].dropna())
    nuisance_params = []
    for node in sorted(referenced - set(parameters + nodes)):
        nuisance = f'nuisance_{node}'
        nuisance_params.append(nuisance)
        value, error_p, error_n = pdg_most_precise_value(node)
        measurement = dict(node=[nuisance], value=[value], error_p=[error_p], error_n=[error_n],
                           reference_id=[None], occurrence=[None])
        meas_df = pd.concat([meas_df, pd.DataFrame(measurement)], ignore_index=True)
        fit_seed_df = pd.concat([
            fit_seed_df, pd.DataFrame({'parameter_key': [nuisance], 'seed': [value]})
        ], ignore_index=True)
        adjust_data.append(None)
        dep_meas_data.append(None)

    if nuisance_params:
        correlations = nuisance_corr(nuisance_params, verbose=False)
        if len(correlations):
            correlations['node_one'] = correlations.apply(
                lambda r: 'nuisance_' + parameter_key(r['par_code_row'], r['parameter_row']), axis=1)
            correlations['node_two'] = correlations.apply(
                lambda r: 'nuisance_' + parameter_key(r['par_code_column'], r['parameter_column']), axis=1)
            correlations['correlation'] = correlations['coefficient'] / 100
            for column in ['reference_id_one', 'occurrence_one', 'reference_id_two', 'occurrence_two']:
                correlations[column] = None
            correlations = correlations.drop(columns=[
                'par_code_row', 'parameter_row', 'par_code_column', 'parameter_column', 'coefficient'])
            corr_df = pd.concat([corr_df, correlations], ignore_index=True)

    # COMER2 triplets: infer covariances from D = X - Y and the three errors.
    for clump in meas_df['systematic_error_clump2'].dropna().unique():
        group = meas_df[meas_df['systematic_error_clump2'] == clump]
        if len(group) != 3:
            print(f'WARNING: clump2 "{clump}" has {len(group)} measurements, expected 3. Skipping.')
            continue
        relations = rel_df[rel_df['node'].isin(group['node'])]
        if len(relations) == 2:
            assert relations['node'].nunique() == 1
            assert set(relations['coefficient'].unique()) == {1, -1}
            names = {
                'D': relations['node'].iloc[0],
                'X': relations['parameter'].iloc[np.argmax(relations['coefficient'])],
                'Y': relations['parameter'].iloc[np.argmin(relations['coefficient'])],
            }
        else:
            assert len(relations) == 6
            names = organize_triplet_clump(relations)
        rows = {key: group[group['node'] == name].iloc[0] for key, name in names.items()}
        errors = {}
        for key, row in rows.items():
            if not np.isclose(row['error_p'], row['error_n']):
                print(f'Warning: clump2 "{clump}", node {names[key]} has asymmetric errors, symmetrizing.')
            errors[key] = (row['error_p'] + row['error_n']) / 2
        covariance_xy = (errors['X']**2 + errors['Y']**2 - errors['D']**2) / 2
        correlations = [
            covariance_xy / (errors['X'] * errors['Y']),
            (errors['X']**2 - covariance_xy) / (errors['X'] * errors['D']),
            (-errors['Y']**2 + covariance_xy) / (errors['Y'] * errors['D']),
        ]
        assert all(-1 <= correlation <= 1 for correlation in correlations)
        pairs = [('X', 'Y'), ('X', 'D'), ('Y', 'D')]
        extra = {'node_one': [names[a] for a, b in pairs],
                 'reference_id_one': [rows[a]['reference_id'] for a, b in pairs],
                 'occurrence_one': [rows[a]['occurrence'] for a, b in pairs],
                 'node_two': [names[b] for a, b in pairs],
                 'reference_id_two': [rows[b]['reference_id'] for a, b in pairs],
                 'occurrence_two': [rows[b]['occurrence'] for a, b in pairs],
                 'correlation': correlations}
        corr_df = pd.concat([corr_df, pd.DataFrame(extra)], ignore_index=True)

    # Other clumps share a fully correlated systematic error on one node.
    for clump in meas_df['systematic_error_clump'].dropna().unique():
        group = meas_df[meas_df['systematic_error_clump'] == clump]
        assert group['node'].nunique() == 1, 'Each systematic_error_clump must contain one node'
        node = group['node'].iloc[0]
        syst_err = np.array(group['last_err'])
        total_err = np.array(group['error'])
        references = np.array(group['reference_id'])
        occurrences = np.array(group['occurrence'])
        identity_columns = ['node_one', 'reference_id_one', 'occurrence_one',
                            'node_two', 'reference_id_two', 'occurrence_two']
        existing = set(corr_df[identity_columns].itertuples(index=False, name=None))
        for i, j in combinations(range(len(group)), 2):
            first = (node, references[i], occurrences[i])
            second = (node, references[j], occurrences[j])
            if first + second in existing or second + first in existing:
                raise ValueError(
                    f'Correlation between systematic error clump measurements {first} and {second} '
                    'already exists in the correlation table. Not supported.'
                )
            correlation = syst_err[i] * syst_err[j] / (total_err[i] * total_err[j])
            assert -1 <= correlation <= 1
            extra = {'node_one': [node], 'reference_id_one': [references[i]],
                     'occurrence_one': [occurrences[i]], 'node_two': [node],
                     'reference_id_two': [references[j]], 'occurrence_two': [occurrences[j]],
                     'correlation': [correlation]}
            corr_df = pd.concat([corr_df, pd.DataFrame(extra)], ignore_index=True)

    return fit_df, rel_df, meas_df, corr_df, fit_seed_df, dep_meas_data, adjust_data
