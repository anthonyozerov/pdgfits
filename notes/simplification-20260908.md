# Fitter simplification, 8 September 2026

The baseline is squash import `bf3d97c`, incorporating the independent remote
history through `3691d0e`. The nested transfer source is untouched. No live
database or PDG tunnel was used in this refactor.

## Structure

- `profiles.py` owns fixed-target minimization. `asym_errors.py` owns bracketing,
  roots and endpoint diagnostics. Both fits and nuisance-bearing averages use
  these routines; the old `binary_search_error` wrapper remains available.
- Consolidated SciPy candidate selection and repeated solver attempts, Minuit's
  fixed retry sequence, and relationship compilation. Kept the solver ordering,
  coordinate preconditioning and difficult-case safeguards supported by the June
  experiments.
- Removed unused eliminated-parameter mapping and alternative chi-square code.
  Moved the import-time timing script to `tools/legacy_timing.py`.
- Simplified the average CLI, added its installed entry point, and updated package
  documentation and pytest collection scope.

## Intentional corrections

- A finite but unsuccessful profile evaluation no longer certifies an endpoint.
  The requested objective residual tolerance is enforced literally.
- An unconverged feasible nuisance starting point is no longer automatically a
  successful profile minimum. Nonfinite descent checks now fail certification.
- A single average is saved to CSV, using its named primary parameter.
- Average `n_meas` includes auxiliary constraints, with a separate
  `n_primary_meas` and nominal `ndof = n_meas - n_params`.
- Experimental block-Birge rescaling no longer runs or prints rescaled intervals
  automatically. It remains an explicitly requested diagnostic, with its limited
  statistical justification documented.

## Validation

The offline suite passes **164 tests, 2 skipped**. The new tests exercise failed
finite profiles, a requested tolerance that the old code silently relaxed, and
unconverged nuisance starts. A real `M026R08` CLI run saves one row with four
measurements (two primary plus two auxiliary), three parameters and one nominal
degree of freedom.

A separate-process before/after capture covers **369 cases**: all 81 fit central
solutions and covariances, difficult profile targets, four SciPy cases, and 284
averages including every nuisance-bearing average. The final comparison is in
`refactor-validation/final-comparison.json`; source capture hashes, exact target
count and dependency versions are in `refactor-validation/provenance.json`.

Maximum scaled differences are 3.7e-15 in objective, 8.6e-12 in parameters,
3.6e-8 in profile errors, and zero in average interval errors. All checked
endpoints satisfy the 0.005 objective residual requirement.

Minuit's numerical Hesse covariance differs by up to **4.86e-5** when scaled by
the corresponding marginal standard deviations. The initial 1e-6 comparison
threshold therefore failed; that report is preserved as `comparison.json`.
At an identical parameter point for the largest case, `f_1(1285)`, the objective
is identical and automatic-differentiation Hessians agree to 4.4e-17 relative
maximum norm. A second case, `phi(1020)`, agrees to 1.4e-16. This identifies
roundoff sensitivity of the numerical covariance estimate, rather than a changed
objective. The comparison tool exposes its covariance tolerance and defaults to
1e-4; central values and profile errors retain the stricter 1e-7 tolerance.

This validation establishes numerical agreement on the snapshot, not frequentist
coverage of asymmetric Delta-Q intervals or proof of global optimality.
