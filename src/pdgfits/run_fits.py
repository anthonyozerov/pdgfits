import argparse
from decimal import Decimal

import jax
jax.config.update("jax_enable_x64", True)

from pdgfits.fit_query import all_fits
from pdgfits.fit import run_fit
from pdgfits.diagnostics import compare_to_pdg, meas_diagnostics, meas_sensitivity
from pdgfits.asym_errors import calc_asym_errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start_from', type=str, default=None)
    parser.add_argument('--fit_type', type=str, default=None)
    parser.add_argument('--fit_label', type=str, default=None)
    parser.add_argument('--measurement_type', type=str, default=None)
    parser.add_argument('--compare_to_pdg', action='store_true', default=False)
    parser.add_argument('--calc_asym_errors', action='store_true', default=False)
    parser.add_argument('--meas_diagnostics', action='store_true', default=False)
    parser.add_argument('--meas_sensitivity', action='store_true', default=False)
    parser.add_argument('--optimizer', type=str, default='minuit', choices=['minuit', 'scipy'])
    parser.add_argument('--fit_space', type=str, default='unconstrained', choices=['unconstrained', 'constrained'])
    args = parser.parse_args()

    fits_df = all_fits()
    fits_df_sub = fits_df[fits_df['algorithm'] != 'IGNORE']

    if args.fit_type is not None:
        fits_df_sub = fits_df_sub[fits_df_sub['algorithm'] == args.fit_type]
    if args.fit_label is not None:
        fits_df_sub = fits_df_sub[fits_df_sub['label'] == args.fit_label]
    if args.measurement_type is not None:
        fits_df_sub = fits_df_sub[fits_df_sub['measurement_type'] == args.measurement_type]

    labels = list(fits_df_sub['label'])
    chi2s_pdg = list(fits_df_sub['chi_square'])

    start_from_idx = 0
    if args.start_from is not None:
        start_from_idx = labels.index(args.start_from)

    for i in range(start_from_idx, len(labels)):
        label = labels[i]
        chi2_pdg = chi2s_pdg[i]
        print('=' * 100)
        print(f'{label}')

        if label == 'tauhflav':
            print('skipping tauhflav')
            continue

        fit = run_fit(label, verbose=True, optimizer=args.optimizer, fit_space=args.fit_space)
        if fit is None:
            continue

        print(f'chi2 obtained: {Decimal(fit["chi2_min"]):.2E}')
        print(f'chi2 obtained by PDG: {Decimal(chi2_pdg):.2E}')

        if args.compare_to_pdg:
            compare_to_pdg(fit)

        if args.calc_asym_errors:
            calc_asym_errors(fit)

        if args.meas_diagnostics:
            meas_diagnostics(fit)

        if args.meas_sensitivity:
            meas_sensitivity(fit)


if __name__ == '__main__':
    main()
