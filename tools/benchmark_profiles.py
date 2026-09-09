"""Time production fits/profiles against an explicitly selected offline snapshot.

Use separate output files and source paths for before/after comparisons. Each
line is flushed immediately so a long campaign retains completed cases.
"""

import argparse
import contextlib
import gc
import json
import os
from pathlib import Path
import signal
import time

import jax
import numpy as np

from pdgfits.asym_errors import calc_asym_errors
from pdgfits.avg import run_avg
from pdgfits.fit import run_fit
from pdgfits.query import all_fits, avg_queries


FOCUS = {
    'G(2000),G(1800)': ['K003DM'],
    'Upsilon(2S)': None,
    'Lam-b-0': ['S040R29'],
    'B0': ['S042B09', 'S042.390', 'S042B95'],
    'B0S-BR': ['S086R04'],
    'eta_c J/psi psi(2S)': ['M026G01', 'M026W', 'nuisance_M071.13'],
    'eta_c(2S)': ['M059.4'],
    'K_3^*(1780)': ['M060.6'],
    'chi_c012 psi(2S)': ['M056.11', 'M056.6', 'M055B1'],
}
AVERAGES = ['M002W', 'M026R08', 'M056R50', 'M057B18', 'M070R84',
            'S032B94', 'S041B46', 'S042CKS', 'S042R2', 'S051R05',
            'S051R06', 'S086R46']


def clean(value):
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, (np.ndarray, jax.Array)):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--kind', choices=['fits', 'averages', 'node-scales', 'pdg-scales'], required=True)
    parser.add_argument('--draws', type=int, default=256)
    parser.add_argument('--scale-profiles', action='store_true', help='Also check every profile target after node scaling')
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--label', action='append')
    parser.add_argument('--timeout', type=int, default=300)
    parser.add_argument('--keep-cache', action='store_true', help='Match a production batch without clearing compiled kernels')
    parser.add_argument('--no-target-retry', action='store_true', help='Do not retry targets individually after a failed group')
    args = parser.parse_args()
    if os.environ.get('PDGFITS_DATA_BACKEND') != 'snapshot':
        raise RuntimeError('Select the snapshot backend explicitly')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(line)['id'] for line in args.output.read_text().splitlines()} if args.output.exists() else set()
    def timed_out(*_):
        raise TimeoutError(f'Case exceeded {args.timeout} seconds')
    signal.signal(signal.SIGALRM, timed_out)
    with args.output.open('a') as output, args.output.with_suffix('.log').open('a', buffering=1) as log:
        def run(case, work):
            if case in done:
                return None
            start = time.perf_counter()
            row = {'id': case}
            signal.alarm(args.timeout)
            try:
                with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                    row.update(work())
                row['status'] = 'ok'
            except Exception as exc:
                row.update(status='error', exception=f'{type(exc).__name__}: {exc}')
                if hasattr(exc, 'history'):
                    row['scale_history'] = exc.history
                if hasattr(exc, 'values'):
                    row['failed_refit'] = {key: getattr(exc, key) for key in
                                           ['values', 'scales', 'fitted_values']}
            finally:
                signal.alarm(0)
            row['seconds'] = time.perf_counter()-start
            output.write(json.dumps(clean(row), allow_nan=False)+'\n')
            output.flush()
            print(f"{case}: {row['status']} {row['seconds']:.2f}s", flush=True)
            return row

        if args.kind in ['fits', 'node-scales', 'pdg-scales']:
            labels = args.label or ([r.label for r in all_fits().itertuples()
                                     if r.algorithm != 'IGNORE' and r.label != 'tauhflav'] if args.all else list(FOCUS))
            for label in labels:
                if args.kind == 'pdg-scales' and all(f'pdg:{label}:{cut}' in done for cut in [False, True]):
                    continue
                case = ('scale:' if args.kind == 'node-scales' else 'fit:')+label
                if case in done:
                    continue
                start = time.perf_counter()
                with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                    fit = run_fit(label, verbose=False)
                if fit is None:
                    raise RuntimeError(f'Fit skipped: {label}')
                central_seconds = time.perf_counter()-start
                if args.kind == 'pdg-scales':
                    from pdgfits.pdg_scaling import pdg_fit_scales
                    for cut in [False, True]:
                        def pdg_scale():
                            comparison = pdg_fit_scales(fit, exclude_weak=cut)
                            result = comparison['refitted_fit']
                            row = {'exclude_weak': cut, 'scales': comparison['node_scales'],
                                   'retained': comparison['retained'], 'values': result['param_values'],
                                   'chi2': result['chi2_min'], 'fit_valid': result['fit_valid']}
                            if args.scale_profiles:
                                row['targets'] = calc_asym_errors(result)
                            return row
                        run(f'pdg:{label}:{cut}', pdg_scale)
                    jax.clear_caches()
                    gc.collect()
                    continue
                if args.kind == 'node-scales':
                    from pdgfits.node_scales import fit_node_scales
                    def scale():
                        result = fit_node_scales(fit, draws=args.draws, seed=81, verbose=True)
                        row = {'scaling': result['node_scaling'], 'values': result['param_values'],
                                'chi2': result['chi2_min'], 'fit_valid': result['fit_valid'],
                                'parameters': result['parameters'], 'covariance': result['covariance'],
                                'unscaled_values': fit['param_values'], 'unscaled_covariance': fit['covariance']}
                        if args.scale_profiles:
                            row['targets'] = calc_asym_errors(result)
                        return row
                    run('scale:'+label, scale)
                    jax.clear_caches()
                    gc.collect()
                    continue
                targets = sorted(set(fit['nodes']+fit['parameters'])) if args.all or FOCUS.get(label) is None else FOCUS[label]
                def profile():
                    values = calc_asym_errors(fit, targets)
                    return {'targets': values, 'chi2': fit['chi2_min'],
                            'values': fit['param_values'], 'covariance': fit['covariance'],
                            'fit_valid': fit['fit_valid'], 'fit_seconds': central_seconds}
                row = run('fit:'+label, profile)
                # A failed group must not hide the other requested targets.
                if row is not None and row['status'] != 'ok' and not args.no_target_retry:
                    for target in targets:
                        run(f'target:{label}:{target}', lambda: {'targets': calc_asym_errors(fit, [target])})
                del fit
                if not args.keep_cache:
                    jax.clear_caches()
                gc.collect()
        else:
            data, correlations = avg_queries(verbose=False)
            nodes = args.label or (sorted(data.node.unique()) if args.all else AVERAGES)
            for node in nodes:
                def average():
                    result = run_avg(node, data[data.node == node], correlations[node])
                    if result is None:
                        raise RuntimeError('Average skipped')
                    return {'values': result['param_values'], 'parameters': result['parameters'],
                            'chi2': result['chi2_min'], 'error_n': result['error_n'], 'error_p': result['error_p'],
                            'diagnostics': result['asym_error_diagnostics']}
                run('average:'+node, average)
                if not args.keep_cache:
                    jax.clear_caches()
                gc.collect()


if __name__ == '__main__':
    main()
