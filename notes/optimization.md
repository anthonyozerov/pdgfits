# Fitting and interval optimization, September 2026

The changes preserve the measurement functions, correlations and asymmetric-error
objective. They improve how the same problem is represented and solved.

* Profile parameters use uncertainty units. Shared compiled derivatives avoid
  recompilation for each target; the endpoint search uses the approximately
  quadratic shape of Q to take bracketed square-root secant steps. If successive
  steps stop contracting, it bisects the bracket. This prevents steep endpoints
  from exhausting the search while preserving the fast quadratic case.
* Branching-fraction fits use an affine physical chart for profiling and repeated
  fits. Exact zero, one and sum constraints are accessible. An interval reaching
  a physical boundary is returned with an explicit boundary flag.
* Direct averages use a scalar search over the error-rule pieces, including their
  knots. Auxiliary-parameter averages share product-model kernels and minimize
  the residual vector. Independent measurements avoid dense whitening products.
* Repeated simulated fits share compiled kernels and run in batches. Failed
  stationarity checks fall back to the scalar solver. Local conditioning and
  explicit one-sided derivative checks handle asymmetric knots; small optimizer
  steps alone are not accepted as convergence.
* General node scales retain the best completed score update when switching
  solvers. Numerical derivatives use a controlled step in log scale, including
  near the scale floor; relative steps there can be smaller than inner-fit
  numerical error. Their score residual and Monte Carlo error are separate.

The profile safeguards remain: feasibility and stationarity checks, projected
starts, curvature and descent checks, and direct verification of returned
endpoints in the original objective. These checks do not prove global optimality
or frequentist coverage.

## Measured performance

Against commit `51e2184`, 45 matched, successfully completed fit groups, each with
all its profile targets, take **1,214.08 seconds before and 68.65 seconds after**,
including the central fits: **17.69 times faster**. Failed or timed-out baseline
groups are excluded from the speed ratio. All 81 supported snapshot fits and all
1,395 targets complete with the new solver.

All 2,647 averages complete. Recorded complete runs take 50–72 seconds, versus
197.94 seconds before. This is a smaller gain; there is no uniform tenfold speed
claim for every call. Timings use one BLAS thread, the same offline snapshot and
this machine's PDG Python environment. Concurrent validation jobs affect elapsed
time. The fit/profile timings exclude the benchmark's intervening cache clearing
and output serialization. General simulation-based node scaling is new work and has no corresponding
baseline speed ratio.

For the large χc/ψ simulation batch, handing the remaining difficult cases to
the scalar solver after eight vector steps, instead of 32, takes 2.22 rather
than 4.58 seconds in a matched check. All 256 objective values agree within
7×10⁻¹⁰, and the same 14 cases require the safeguarded solver.

Independent checks reconstruct the original average objective at the old and new
minima and both new endpoints. All 2,647 pass; 824 asymmetric scalar cases also
pass a finer search. The maximum endpoint Q residual is 0.004988, within the
requested 0.005. Some shallow minima move slightly as they are solved more
accurately; agreement with an old approximate answer is not the acceptance rule.
The final 369-case snapshot comparison leaves all central values, objective
minima, node predictions and covariances identical. Some average endpoints move
within the requested objective tolerance (largest error-width change 0.44%);
their independent original-objective checks pass. Fit profile widths agree to
6.4×10⁻¹² relative precision in this capture.

## Complete validation

The [compact validation report](optimizer-validation.json) records all cases,
endpoint residuals, scale residuals and the matched timing workload:

| Calculation | Completed fits or averages | Checked profile targets |
|---|---:|---:|
| Ordinary fits | 81 | 1,395 |
| General node scales | 81 | 1,395 |
| Current PDG scale pass, both exclusion choices | 162 | 2,790 |
| Averages, independently checked | 2,647 | primary intervals checked separately |

General scales use 256 draws, seed 81, numerical tolerance 0.001 and no Monte
Carlo tolerance allowance. The largest coupled eta_c/J/psi/psi(2S) case passes
at residual 0.000993 after 1,303 scale evaluations, including all 71 profile
targets. Its completed run takes 31.4 minutes; the median general-scale case
takes about five seconds. The earlier 30-minute benchmark timeout is retained
in the report with its successful retry. The final endpoint safeguard also resolves
an eta_c(2S) profile search that stalled at a steep bracket endpoint. All scaled
profiles are checked again at the unchanged, validated scales; the separate
profile-check timings and original scale-run sources are retained in the report.
The fitting test suite passes 199
tests, with two optional database tests skipped.

## Reproduce

Explicitly select `PDGFITS_DATA_BACKEND=snapshot`, `PDGFITS_SNAPSHOT_DIR`, and the
checkout's `PYTHONPATH=src`. Use separate output files for each implementation.

```bash
python tools/benchmark_profiles.py fits.jsonl --kind fits --all
python tools/benchmark_profiles.py averages.jsonl --kind averages --all --keep-cache
python tools/verify_averages.py before-averages.jsonl checked-averages.json
python tools/benchmark_profiles.py scales.jsonl --kind node-scales --all --scale-profiles --timeout 7200
python tools/benchmark_profiles.py pdg-scales.jsonl --kind pdg-scales --all --scale-profiles --timeout 600
python -m pytest -q
```

The node-scale benchmark fixes 256 simulation draws and seed 81. The PDG benchmark
runs both exclusion choices. Each completed case is written immediately; a failed
case remains visible and requires a new output file when retried. `tauhflav` and
entries marked `IGNORE` are outside the existing supported fit sweep.
