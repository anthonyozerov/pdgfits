"""Check optimized averages against the original measurement map, offline."""

import argparse
import contextlib
import io
import json
import os
from pathlib import Path

import numpy as np

from pdgfits.avg import run_avg
from pdgfits.query import avg_queries
from pdgfits.scalar_average import scalar_average


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('baseline', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    if os.environ.get('PDGFITS_DATA_BACKEND') != 'snapshot':
        raise RuntimeError('Select the snapshot backend explicitly')
    baseline = {row['id'].split(':', 1)[1]: row for row in
                map(json.loads, args.baseline.read_text().splitlines())}
    data, correlations = avg_queries(verbose=False)
    results = []
    for node in sorted(data.node.unique()):
        with contextlib.redirect_stdout(io.StringIO()):
            fit = run_avg(node, data[data.node == node], correlations[node])
        measurements = fit['meas_df']
        y, en, ep = [measurements[k].to_numpy() for k in ['value', 'error_n', 'error_p']]
        precision = np.linalg.pinv(fit['corr_mat'])
        def original_q(parameters):
            residual = y-np.asarray(fit['mu_adjust'](np.asarray(parameters)))
            sigma = np.clip((2*en*ep-residual*(ep-en))/(en+ep), np.minimum(en, ep), np.maximum(en, ep))
            z = residual/sigma
            return float(z@precision@z)
        def roundoff(parameters, q):
            prediction = np.asarray(fit['mu_adjust'](np.asarray(parameters)))
            dz = np.spacing(np.abs(prediction))/np.minimum(en, ep)
            # Final endpoints must be represented in raw units. A few very
            # precise magnetic moments lose about 1e-6 in Q on this conversion.
            return max(1e-7*max(1, abs(q)), 4*np.sqrt(max(q, 1))*np.linalg.norm(precision, 2)*np.linalg.norm(dz))
        old = baseline[node]
        center = np.asarray(fit['param_values'])
        old_q, new_q = original_q(old['values']), original_q(center)
        assert abs(new_q-fit['chi2_min']) < roundoff(center, new_q), node
        assert abs(old_q-old['chi2']) < roundoff(old['values'], old_q), node
        assert new_q <= old_q+1e-6, (node, old_q, new_q)
        row = {'node': node, 'q_improvement': old_q-new_q,
               'nuisance_parameters': len(center)-1, 'endpoint_residual': 0., 'refinement': None}
        for side in ['lower', 'upper']:
            diagnostic = fit['asym_error_diagnostics']
            point = diagnostic[side+'_profile_point']
            parameters = point.get('fitted_values', [diagnostic[side+'_endpoint']])
            q = original_q(np.array(parameters))
            assert abs(q-diagnostic[side+'_chi2']) < roundoff(parameters, q), node
            assert abs(q-new_q-1) <= .005, (node, side, q-new_q-1)
            row['endpoint_residual'] = max(row['endpoint_residual'], abs(q-new_q-1))
        if len(center) == 1 and np.any(en != ep):
            refined = scalar_average(y, en, ep, precision, grid_size=1025)
            scale = (fit['error_n']+fit['error_p'])/2
            differences = [abs(refined['value']-center[0]),
                           abs(refined['error_n']-fit['error_n']), abs(refined['error_p']-fit['error_p'])]
            row['refinement'] = max(differences)/scale
            assert row['refinement'] < 1e-5, (node, row['refinement'])
        results.append(row)
        if len(results) % 100 == 0:
            print(f'{len(results)} averages independently checked', flush=True)
    args.output.write_text(json.dumps({'averages': len(results), 'checks': results}, indent=2)+'\n')


if __name__ == '__main__':
    main()
