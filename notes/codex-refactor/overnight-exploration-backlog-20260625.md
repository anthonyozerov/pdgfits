# Overnight exploration backlog for pdgfits asymmetric-error / fitting rethink

Purpose: give the autonomous supervisor a concrete menu of high-value directions after the current MINOS comparator work. This should prevent the loop from drifting back into micro-optimizations or interface cleanup.

Priority order remains:

1. scientific/statistical correctness;
2. simple/minimal robust method;
3. reproducible quantitative evidence;
4. speed/performance;
5. clever code.

Do one bounded Codex round at a time, with a durable report and quantitative artifacts. Negative results are useful if they rule out an idea on hard cases.

## Priority 1: Borrow the useful parts of MINOS, not MINOS wholesale

Read first:

- `notes/codex-refactor/minuit-minos-method-note-20260625.md`
- `notes/codex-refactor/asym-refactor-round07-minos-comparator-20260625.md`

Promising follow-ups:

### 1A. MINOS-style covariance-predicted conditional starts

MINOS uses the error matrix to predict the sub-minimum of nuisance/correlated parameters when the profiled coordinate is displaced. pdgfits currently relies on continuation/MLE projection plus SLSQP/trust-constr/KKT/descent checks.

Experiment:

- For direct-coordinate and arbitrary-target hard endpoints, construct a local linear/quadratic predictor for the free fitted coordinates at fixed target value using the covariance/Hessian and target gradient.
- Feed that predicted point as an additional start to the existing constrained-profile solver.
- Compare against current starts on hard rows: profile calls, function evals, acceptance method, endpoint residuals, and whether any lower fixed-target chi2 is found.
- Keep as optional/diagnostic unless it clearly reduces calls or improves certificates without changing endpoints.

Success criteria:

- Same endpoints/residuals as current production, but fewer profile calls or cleaner KKT/certificate paths on hard targets; or a documented rejection if predictor is unstable/misleading.

### 1B. HESSE-vs-MINOS / parabolic-vs-profile nonlinearity diagnostic

Round 7 showed HESSE and MINOS agree exactly on nearly quadratic direct-coordinate fits and differ modestly on nonlinear/nuisance cases.

Experiment:

- Add a notes-only diagnostic that records parabolic/HESSE-predicted endpoint, profile endpoint, and their relative disagreement on a stratified target set.
- Use disagreement to rank cases for expensive MINOS-style/profile-curve diagnostics.
- This is evidence tooling, not a replacement for profile endpoints.

Success criteria:

- A ranked list of targets where nonlinearity/asymmetry is biggest, with examples explaining why.

### 1C. Structured crossing statuses like `MnCross`

MINOS reports valid side, new minimum, parameter limit, call limit, invalid crossing. pdgfits now has `ProfileRoot`; use it to make statuses more systematic.

Experiment:

- Make diagnostic harnesses consume `binary_search_error.last_root` directly.
- Add explicit notes-only summaries of endpoint status categories across hard sweeps.
- Do not do more wrapper cleanup unless it enables a method comparison.

## Priority 2: Failure search before new machinery

If current diagnostics keep finding no discrepancy, search harder for real failure modes.

### 2A. Stratified broad failure search

Run a resumable notes-only sweep over endpoints stratified by:

- `+descent-check` vs KKT/projected-gradient acceptance;
- endpoint residual near tolerance boundary;
- high projected/KKT gradient;
- chart saturation / tiny BRU parameters;
- HESSE-vs-profile disagreement;
- profile-curve nonmonotonicity risk.

For each selected endpoint, run cheap diagnostics first, then escalate to multistart/profile-curve/MINOS-style checks only for suspicious rows.

Success criteria:

- Either a concrete failure/lower fixed-target chi2 is found, or a prioritized risk table is produced with no false claims.

### 2B. Disconnected feasible-component / globality stress test

The current certificates are local. Round 3 multistart was local covariance-scale. Try more global starts where feasible:

- Latin-hypercube or Sobol starts in fitted coordinates for low-dimensional hard fits;
- boundary-near starts for BR/BRU targets;
- physical-chart starts only where conditioning is not horrible;
- timebox aggressively.

Success criteria:

- Find a lower constrained chi2 or document that broader starts return to the same profile on selected hard cases.

## Priority 3: Chart / parameterization rethinks for BR/BRU

Round 6 showed naive direct physical coordinates can match production on some BRU cases but are awful for `B0::S042B95` due to tiny parameters and arctan inverse saturation.

### 3A. Alternative bounded charts

Compare current arctan/sigmoid-like fitted map against alternatives for notes-only hard cases:

- logit chart for `(0,1)` BRU parameters;
- softplus/log chart for tiny positive parameters when upper bound is irrelevant;
- simplex-aware additive-log-ratio / centered-log-ratio charts for BR groups, if applicable.

Do not change production maps unless evidence is strong. The first goal is condition-number / stationarity / profile-solve comparison.

Success criteria:

- A hard target where an alternative chart gives equal endpoint chi2 with cleaner stationarity/certificates, or a clear rejection.

### 3B. Reduced-coordinate physical profile for specific low-dimensional cases

For a small fit where the target constraint can be solved analytically or by eliminating one coordinate, compare reduced-coordinate profiling against generic equality-constrained profiling.

Success criteria:

- Same endpoint with simpler local optimization/certification, or evidence that reduction is too target-specific.

## Priority 4: Venzon-Moolgavkar / robust-VM endpoint solver spike

Venzon-Moolgavkar solves endpoint equations directly: profile level plus nuisance stationarity. Robust VM adds trust-region safeguards because plain Newton-like steps can fail.

Experiment:

- Pick a low-dimensional, smooth, direct-coordinate case first.
- Prototype notes-only endpoint equation/trust-region solver using JAX gradients/Hessians.
- Compare to bisection+profile and iminuit/MINOS.
- Do not replace bisection unless it is as robust and simpler on hard cases.

Success criteria:

- Fewer profile solves with matching endpoints and clear failure/status behavior; otherwise reject.

## Priority 5: Statistical-modeling diagnostics, not just numerical solvers

Even if numerical endpoints are right, Wilks/coverage assumptions can fail near boundaries, constraints, or non-Gaussian/asymmetric PDG inputs.

Possible notes-only directions:

- Identify endpoints near physical boundaries where Wilks one-sigma interpretation is questionable.
- Compare profile intervals to parabolic/HESSE and bootstrap/simulation for a tiny synthetic or snapshot-derived toy, if feasible.
- Flag cases where asymmetric interpolation or boundary regularity, not optimizer behavior, is the dominant uncertainty.

Success criteria:

- A scientifically honest risk note: which intervals are numerically stable but statistically delicate.

## Priority 6: Performance only after method questions

Only pursue speed after correctness/method experiments plateau.

Good performance targets:

- MINOS-style starts reducing profile calls;
- caching profile points/root traces cleanly through `ProfileRoot`;
- broad-sweep resumability and triage to avoid wasting JAX time.

Bad performance targets:

- shaving tiny call counts without improving robustness/evidence;
- unbracketed root finders that reduce calls but weaken failure behavior;
- more solver fallbacks without failure cases.

## Suggested supervisor policy

At each 20-minute tick after a Codex round completes:

1. If the last round found a real failure, steer to a principled fix/design.
2. Else if MINOS borrowing has not yet been tried beyond the comparator, prioritize Priority 1A or 1B.
3. Else rotate through Priorities 2–4, one bounded experiment per round.
4. If three consecutive rounds are purely negative and no credible next experiment remains, stop launching new rounds and leave a clear synthesis report.

Every next prompt should ask Codex to state explicitly whether the round produced:

- a production method change;
- a notes-only comparator/diagnostic;
- a concrete failure found;
- an idea rejected and why;
- recommended next round.
