# Fitting and interval optimization, September 2026

The current revision replaces simulation-based general node scales with a single
local-geometry calculation and simplifies the refitting code. Measurement
functions, the asymmetric-error objective and the ordinary profile solver are
preserved.

## What was removed

* The normalized-kernel samplers, simulated score expectations and outer scale
  root/least-squares solvers. Scale estimation now uses the residual Jacobian
  and a small matrix projection; see [the method](general-node-scales.md).
* The custom batched damped-Newton optimizer for simulated fits.
* Repeated reconditioning/polishing cycles, up to eight derivative-free recovery
  rounds, and special callback exceptions in the mean refitter.

`node_scales.py` is 131 lines rather than 419; `refit.py` is 251 rather than 408.
Including the removed command-line options, production Python has 448 fewer
lines. The retired Monte Carlo options and result fields are removed explicitly.
The Gaussian linear REML reference is still a separate optional comparator.

## What remains and why

The mean refitter uses SciPy least squares with Jacobian scaling, or SLSQP when
physical constraints are present. If needed, it minimizes once along a nearby
asymmetric-error knot, then uses at most one derivative-free fallback. Feasibility,
one-sided stationarity, local descent and covariance checks remain.

The corner polish has a concrete regression behind it: coordinate searches can
find no decrease even when a direction *along* the corner decreases Q. Dropping
that polish shifted three auxiliary-input averages by less than 0.001 reported
standard errors, but raised Q by about 2–4×10⁻⁶. A short constrained polish
restores their previous minima within 5×10⁻¹². A two-parameter analytic example
checks this mechanism directly. Coordinate-wise stopping alone is not accepted
at an asymmetric corner.

The ordinary fixed-target profile solver is unchanged. It retains its projected
starts, exact-Hessian fallback, KKT/descent checks, physical bounds and contracting
endpoint bracket. Those safeguards were independently justified by difficult
profile cases before the simulation-scale work. Shared derivatives, uncertainty
units and direct scalar-average searches are also retained; they are simpler
than adding more optimizer layers.

## Validation and performance

The [current validation report](optimizer-validation.json) records the final
local-scale, PDG-comparator, average and snapshot checks. The 369-case comparison
against `e605082` includes all ordinary central fits, selected difficult profiles
and averages with auxiliary inputs. Direct average checks reconstruct the
original measurement objective at minima and endpoints. Passing numerical
checks does not establish global optimality or frequentist coverage.

The current sweep passes 81 locally scaled fits / 1,395 targets, 162 PDG scale
variants / 2,790 targets, and all 2,647 averages (including 824 scalar refinements).
The test suite passes 197 tests, with two optional live-database tests skipped.
All ordinary snapshot outputs in the 369-case comparison are unchanged. One
auxiliary average, S041B41, exceeds the comparison's strict 1e-7 relative error
tolerance by a factor of 2.12: its interval errors change by 2.12e-7 relatively
after recovering the same minimum to 2.1e-13 in Q. Its independently checked
endpoint Q residuals change by less than 7e-8 and remain within 0.005. This
reviewed numerical-equivalence warning is retained in the report.

The earlier **17.69×** result was measured at `e605082` against `51e2184` on
45 matched successful ordinary fit/profile groups: 1,214.08 versus 68.65 seconds.
The ordinary fit/profile source is unchanged in this simplification. This is a
historical workload measurement, not a new claim of uniform speedup for all APIs.
The former 31-minute scale calculation was a different statistical procedure;
its removal is not described as accelerating an equivalent Monte Carlo estimate.
Current local-scale timings, including their checked profiles, are in the report.

The full older validation report and simulation implementation are available in
Git at `e605082`; they do not describe the current node-scale method. The local
raw development runs are under `pdgstudy/.local/geometry-audit/`.
