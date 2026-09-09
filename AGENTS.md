# Working on pdgfits

Read README.md for current behavior and assumptions. `SCIENTIFIC_CONTEXT.md`, if present locally, is historical context and contains outdated dataset sizes, execution instructions and statistical claims.

- Use the offline snapshot backend for ordinary development. Explicitly set `PDGFITS_DATA_BACKEND=snapshot`, `PDGFITS_SNAPSHOT_DIR`, and `PYTHONPATH` when comparing checkouts. Do not default to a live database call.
- Run `python -m pytest` (test collection is limited to `tests/`). Numerical changes need the snapshot regression capture/comparison in `tools/`, including difficult profile cases and averages with auxiliary inputs.
- Preserve the measurement/relationship model when refactoring. Keep corrections to statistical behavior explicit and documented with evidence.
- `profiles.py` minimizes at a fixed target; `asym_errors.py` searches and checks endpoints. Feasibility alone is insufficient. Do not silently accept failed profiles or endpoints outside the requested objective residual tolerance.
- Keep projected starts, exact-Hessian fallback, KKT/descent checks and bracket contraction unless a hard-case comparison supports a change. The June ablations are in `notes/asym-profile-simplification.md` and `notes/asym-fit-simplified-broad-sweep.md`.
- Main results are plain dictionaries. Prefer small direct functions to new frameworks or wrapper layers. Keep experimental scripts and generated results outside the installed package.
- Do not infer confidence-interval coverage from numerical convergence. The asymmetric objective is not automatically a complete repeated-sampling model; block Birge diagnostics are not validated general interval rescalings.
- Preserve local ignored snapshots/fixtures and the nested transfer source. Do not commit credentials or bulky stdout logs.
- Keep documentation synchronized with commands, result fields and intentional numerical changes.
