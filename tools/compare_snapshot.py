"""Compare the numerical capture files made by check_snapshot.py."""

import argparse
import json
from pathlib import Path

import numpy as np


def compare(before, after, covariance_tolerance=1e-4):
    def load(path):
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        result = {r['id']: r for r in rows}
        if len(result) != len(rows):
            raise ValueError(f'Duplicate cases in {path}')
        return result

    left, right = load(before), load(after)
    if left.keys() != right.keys():
        raise ValueError(f'Case sets differ: {left.keys() ^ right.keys()}')
    maximum = {}
    failures = []

    def check(case, field, a, b, scale=None, tolerance=1e-7):
        a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
        if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
            failures.append({'case': case, 'field': field, 'error': 'shape or finite-value mismatch'})
            return
        if scale is None:
            scale = np.maximum(np.maximum(abs(a), abs(b)), 1e-100)
        error = float(np.max(abs(a-b)/scale)) if a.size else 0.0
        maximum[field] = max(maximum.get(field, 0), error)
        if error > tolerance:
            failures.append({'case': case, 'field': field, 'relative_error': error})

    for case, old in left.items():
        new = right[case]
        for key in ['parameters', 'nodes']:
            if key in old and old[key] != new[key]:
                raise ValueError(f'{case}: {key} changed')
        check(case, 'chi2', old['chi2'], new['chi2'], scale=max(abs(old['chi2']), 1.0))
        check(case, 'values', old['values'], new['values'])
        if 'node_values' in old:
            check(case, 'node_values', old['node_values'], new['node_values'])
        if 'covariance' in old:
            cov = np.asarray(old['covariance'])
            scale = np.sqrt(np.outer(abs(np.diag(cov)), abs(np.diag(cov))))
            check(case, 'covariance', old['covariance'], new['covariance'],
                  scale=np.maximum(scale, 1e-100), tolerance=covariance_tolerance)
        if 'error_p' in old:
            check(case, 'average_errors', [old['error_p'], old['error_n']], [new['error_p'], new['error_n']])
            for side in ['upper', 'lower']:
                if abs(new[f'{side}_residual']) > 0.005:
                    failures.append({'case': case, 'field': f'{side}_residual', 'error': 'outside tolerance'})
        if 'profiles' in old:
            if old['profiles'].keys() != new['profiles'].keys():
                raise ValueError(f'{case}: profile target set changed')
            for target, profile in old['profiles'].items():
                other = new['profiles'][target]
                check(f'{case}/{target}', 'profile_errors', [profile['error_p'], profile['error_n']],
                      [other['error_p'], other['error_n']])
                for side in ['upper', 'lower']:
                    if abs(other[f'{side}_residual']) > 0.005:
                        failures.append({'case': f'{case}/{target}', 'field': side, 'error': 'outside tolerance'})
    return {'cases': len(left), 'covariance_tolerance': covariance_tolerance,
            'max_scaled_differences': maximum, 'failures': failures}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('before', type=Path)
    parser.add_argument('after', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--covariance-tolerance', type=float, default=1e-4,
                        help='Minuit numerical Hesse tolerance, scaled by marginal variances')
    args = parser.parse_args()
    result = compare(args.before, args.after, args.covariance_tolerance)
    text = json.dumps(result, indent=2) + '\n'
    if args.output:
        args.output.write_text(text)
    print(text)
    raise SystemExit(bool(result['failures']))
