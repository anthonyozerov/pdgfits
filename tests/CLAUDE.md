# CLAUDE.md — tests/

## Overview

pytest-based test suite. Unit tests require no DB; integration tests require an active SSH tunnel to `127.0.0.1:5433` and a `.env` file with credentials.

## Running tests

```bash
pytest tests/                   # unit tests only
pytest tests/ -v -m db --db    # include DB integration tests (verbose)
pytest tests/ -v -s -m db --db # also show captured stdout from run_fit
pytest tests/ -m db --db -n 4  # parallel (requires pytest-xdist)
```

## Files

- **`conftest.py`** — adds the `--db` CLI flag; registers the `db` marker; skips `db`-marked tests when `--db` is absent; uses `pytest_generate_tests` to lazily parametrize `fit_label` from `query.all_fits()` only when `--db` is active (filters out `algorithm='IGNORE'` and the `tauhflav` label).
- **`test_integration.py`** — single `@pytest.mark.db` test parametrized over all fit labels. For each label it calls `run_fit(label, optimizer='minuit', fit_space='unconstrained')` and asserts:
  1. `result['fit_valid']` — Minuit reports a valid fit
  2. `result['hesse_accurate']` — Hesse covariance succeeded
  3. `result['chi2_min'] < pdg_chi2 + 13` — chi2 not much worse than PDG value
  4. Covariance matrix is symmetric (`np.allclose(cov, cov.T, atol=1e-10)`)
  5. Covariance matrix is positive semi-definite (all eigenvalues ≥ −1e-8)
  - Fits returning `None` from `run_fit` are skipped via `pytest.skip`.
- **`test_preprocess.py`** — characterization (golden) tests for `preprocess()`. Run with **no DB**: each fixture carries the inputs, the recorded results of the two DB calls `preprocess` makes internally (`pdg_most_precise_value`, `nuisance_corr`, replayed via `monkeypatch`), and the expected outputs. Compares the five output dataframes order-insensitively (row order is not a contract — see `src/pdgfits/CLAUDE.md`) and the `dep_meas_data`/`adjust_data` lists positionally. Skips if no fixtures are present; a synthetic `test_ignore_minus_takes_absolute_value` always runs (the `ignore_minus` branch is exercised by no live fit). These golden tests are the precise safety net for refactoring `preprocess`; the durable parity check against PDG is `test_integration.py`.
- **`fixtures/capture_preprocess_fixtures.py`** — one-time capture script (run against a live DB tunnel): `python tests/fixtures/capture_preprocess_fixtures.py`. Snapshots every non-IGNORE fit's `preprocess` inputs (copied *before* the call, since `preprocess` mutates them), the recorded internal DB calls, and outputs to `tests/fixtures/preprocess/*.pkl`, and prints a path-coverage summary. **Fixtures are gitignored** (the DB changes over time — regenerate rather than commit). Committed artifacts are the script and `test_preprocess.py`.
