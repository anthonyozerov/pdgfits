"""Run averages and export their numerical results, including a single node."""

import argparse
from pathlib import Path

import pandas as pd

from pdgfits.query import avg_queries
from pdgfits.avg import run_avg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--node', help='Run a single node')
    parser.add_argument('--start_from', help='Resume at a selected node')
    parser.add_argument('--output', type=Path, default=Path('avg_results.csv'))
    parser.add_argument('--test-parse', action='store_true')
    parser.add_argument('--interesting', action='store_true', help='Select nodes from avg-node-ex.yaml')
    parser.add_argument('--contour', action='store_true', help='Save Minuit contour plots')
    parser.add_argument('--corr_floor', type=float, default=0.0)
    args = parser.parse_args()
    data, correlations = avg_queries(verbose=False)
    nodes = list(data.node.unique())
    if args.interesting:
        import yaml
        config = yaml.safe_load(Path(__file__).with_name('avg-node-ex.yaml').read_text())
        nodes = [node for node in config['interesting'] if node in nodes]
    elif args.node:
        if args.node not in nodes:
            parser.error(f'Node {args.node} not found')
        nodes = [args.node]
    if args.start_from:
        if args.start_from not in nodes:
            parser.error(f'Start node {args.start_from} not in selection')
        nodes = nodes[nodes.index(args.start_from):]

    rows = []
    for node in nodes:
        result = run_avg(node, data[data.node == node], correlations[node],
                         skip_avg=args.test_parse, contours=args.contour,
                         contours_dir='avg-results/contours', corr_floor=args.corr_floor)
        if result is None:
            continue
        rows.append({
            'node': node, 'value': float(result['param_values'][result['parameters'].index(node)]),
            'error_n': result['error_n'], 'error_p': result['error_p'], 'chi2': result['chi2_min'],
            'ndof': result['ndof'], 'n_meas': result['n_meas'],
            'n_primary_meas': result['n_primary_meas'], 'n_params': len(result['parameters']),
        })
    if rows:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(args.output, index=False)
        print(f'Saved {len(rows)} averages to {args.output}')


if __name__ == '__main__':
    main()
