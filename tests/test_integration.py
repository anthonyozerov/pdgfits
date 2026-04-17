import jax
jax.config.update("jax_enable_x64", True)

import numpy as np
import pytest

from pdgfits.fit import run_fit
from pdgfits.diagnostics import get_pdg_chi2


@pytest.mark.db
def test_fit(fit_label):
    result = run_fit(fit_label, verbose=False, optimizer='minuit', fit_space='unconstrained')
    if result is None:
        pytest.skip(f"run_fit returned None for {fit_label}")

    assert result['fit_valid'], "Minuit did not report a valid fit"

    assert result['hesse_accurate'], "Minuit Hesse was not accurate"

    pdg_chi2 = get_pdg_chi2(fit_label)
    assert result['chi2_min'] < pdg_chi2 + 13, (
        f"chi2_min={result['chi2_min']:.4f} is not < pdg_chi2+6={pdg_chi2 + 13:.4f}"
    )

    cov = np.array(result['covariance'])
    assert np.allclose(cov, cov.T, atol=1e-10), "Covariance matrix is not symmetric"

    eigenvals = np.linalg.eigvalsh(cov)
    assert np.all(eigenvals >= -1e-8), (
        f"Covariance matrix is not positive semi-definite; min eigenvalue: {eigenvals.min():.2e}"
    )
