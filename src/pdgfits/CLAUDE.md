# Package guide

The current workflow and statistical interpretation are in the root README.md; repository development rules are in AGENTS.md.

The fit pipeline is `query -> preprocess -> prediction/correlation/objective builders -> parameter mapping -> minimization -> profile endpoints`. Averages use the same data preprocessing and objective, an identity parameter map, and direct-coordinate nuisance profiling.

`run_fit` and `run_avg` return plain dictionaries containing parameter names/values, prediction functions, the objective and its open-data variant, input dataframes and interval information. Coordinate maps preserve the distinction between physical parameters and optimizer coordinates. Fit covariances are in optimizer coordinates.

`get_node_funcs` and the nonlinear measurement predictor use the same relationship compiler. Arrays follow their explicit parameter/node lists; dataframe row order is not a cross-run contract, except that per-measurement adjustment lists correspond positionally to the processed measurement rows.

`profiles.py` contains fixed-target optimization and `ProfilePoint`; `asym_errors.py` contains root/endpoint records and scalar searches. Compatibility imports remain available from `asym_errors`. Prefer returned `ProfileRoot` diagnostics over the legacy `binary_search_error.last_diagnostics` attributes in new code.

Unused eliminated-parameter maps and the unvalidated scale-fitting objective were removed from the active package during the September simplification. The original implementations are preserved in squash commit `bf3d97c`. Use `birge.block_birge` only as an explicitly requested experimental diagnostic.
