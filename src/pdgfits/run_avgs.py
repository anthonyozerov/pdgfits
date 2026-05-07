import jax
jax.config.update("jax_enable_x64", True)

import argparse
import os
import numpy as np
import pandas as pd
import yaml

from pdgfits.query import avg_queries
from pdgfits.avg import run_avg


def main():
    parser = argparse.ArgumentParser(description='Run PDG weighted averages for all nodes.')
    parser.add_argument('--node', type=str, default=None, help='Run a single node average')
    parser.add_argument('--start_from', type=str, default=None, help='Resume from a given node')
    parser.add_argument('--output', type=str, default='avg_results.csv', help='Output CSV path')
    parser.add_argument('--test-parse', action='store_true', help='Test measurement parsing and exit')
    parser.add_argument('--interesting', action='store_true', help='Run interesting nodes from avg-node-ex.yaml with contour plots saved to avg-results/contours')
    args = parser.parse_args()
    test_parse = args.test_parse

    print('Querying all averages...')
    avg_df, corr_df_dict = avg_queries(verbose=False)
    nodes = list(avg_df['node'].unique())
    print(f'Found {len(nodes)} nodes to average')

    contours = False
    contours_dir = None

    if args.interesting:
        yaml_path = os.path.join(os.path.dirname(__file__), 'avg-node-ex.yaml')
        with open(yaml_path) as f:
            node_ex = yaml.safe_load(f)
        interesting_nodes = node_ex.get('interesting', [])
        nodes = [n for n in interesting_nodes if n in nodes]
        missing = [n for n in interesting_nodes if n not in avg_df['node'].unique()]
        if missing:
            print(f'Warning: interesting nodes not found in data: {missing}')
        contours = True
        contours_dir = 'avg-results/contours'
    elif args.node is not None:
        nodes = [n for n in nodes if n == args.node]
        contours = True
        if not nodes:
            print(f'Node {args.node} not found in average data.')
            return

    start_idx = 0
    if args.start_from is not None:
        if args.start_from in nodes:
            start_idx = nodes.index(args.start_from)
        else:
            print(f'Warning: start_from node {args.start_from} not found, starting from beginning.')
    nodes = nodes[start_idx:]

    results = []
    for i, node in enumerate(nodes):
        meas_df_node = avg_df[avg_df['node'] == node]
        corr_df_node = corr_df_dict[node]

        result = run_avg(node, meas_df_node, corr_df_node, skip_avg=test_parse, contours=contours, contours_dir=contours_dir)
        if result is not None:
            print(result['error_n'], result['error_p'])
            results.append(result)

        if (i + 1) % 100 == 0:
            print(f'[{i+1}/{len(nodes)}] done so far...')
    if test_parse:
        exit()


    rows = []
    for r in results:
        node = r['node']
        val = float(r['param_values'][0])
        n_params = len(r['parameters'])
        rows.append({
            'node': node,
            'value': val,
            'error_n': r['error_n'],
            'error_p': r['error_p'],
            'chi2': r['chi2_min'],
            'ndof': r['n_meas'] - n_params,
            'n_meas': r['n_meas'],
            'n_params': n_params,
        })

    if rows and len(rows) > 1:
        results_df = pd.DataFrame(rows)
        results_df.to_csv(args.output, index=False)
        print(f'\nResults saved to {args.output}')

        print(f'\nchi2 / ndof distribution (ndof > 0):')
        valid = results_df[results_df['ndof'] > 0].copy()
        valid['chi2_per_dof'] = valid['chi2'] / valid['ndof']
        print(valid['chi2_per_dof'].describe())

        high_chi2 = valid[valid['chi2_per_dof'] > 5].sort_values('chi2_per_dof', ascending=False)
        if len(high_chi2) > 0:
            print(f'\nNodes with chi2/ndof > 5 ({len(high_chi2)} total):')
            print(high_chi2[['node', 'value', 'error_n', 'error_p', 'chi2', 'ndof', 'chi2_per_dof']].head(20).to_string(index=False))


if __name__ == '__main__':
    main()
