# Asym Refactor Round 04 Architecture Rethink

Date: 2026-06-24 23:15 PDT

Branch: `codex/asym-refactor-simplify-20260624-221852`

Production code changed: no.

Commit hash: final commit hash is reported in the chat final response. It is
not embedded here because embedding the self-referential hash would change the
commit.

## Scope

This round deliberately did not start from a production edit. I reread the
scientific context, current-method notes, and rounds 1-3, then mapped the
current fit and average profile code. I added one notes-only diagnostic spike:
a profile-curve monotonicity scan for three hard fit-side targets.

Relevant files read:

- `SCIENTIFIC_CONTEXT.md`
- `AGENTS.md`
- `src/pdgfits/CLAUDE.md`
- `notes/asymmetric-error-algorithm-evidence.md`
- `notes/asym-profile-simplification.md`
- `notes/asym-fit-simplified-broad-sweep.md`
- `notes/endpoint-search/asym-endpoint-search-exploration.md`
- `notes/codex-refactor/asym-refactor-round01-20260624-2228.md`
- `notes/codex-refactor/asym-refactor-round02-supervised-20260624-2241.md`
- `notes/codex-refactor/asym-refactor-round03-supervised-20260624-2307.md`
- `notes/codex-refactor/supervisor-state.md`
- `src/pdgfits/asym_errors.py`
- `src/pdgfits/avg.py`
- `src/pdgfits/fit.py`
- `src/pdgfits/param_maps.py`

## Current Architecture Critique

The current asymmetric-error method is scientifically pointed in the right
direction. It returns profile-likelihood/Wilks endpoints, not optimizer
success flags:

```text
profile_chi2(endpoint) = chi2_min + 1
```

with profile minimization over all other fitted/nuisance coordinates.

The fit-side implementation is:

1. Build a scalar target function over physical parameters.
2. Minimize chi2 in fitted-coordinate space subject to
   `target_func(fitted_params_to_params(fp)) = fixed_value`.
3. Project starts onto the scalar constraint.
4. Try SLSQP, then exact-Hessian `trust-constr`.
5. KKT-polish exact-Hessian candidates.
6. Accept only finite, scaled-feasible points with projected/KKT stationarity
   or the existing no-meaningful-feasible-descent certificate.
7. Use bracketed bisection to locate both endpoints and verify endpoint
   residuals.

The average-side implementation is different and should stay different:

1. The primary reported value is a direct coordinate.
2. For each fixed primary value, minimize only nuisance coordinates.
3. Use BFGS with gradient, then Nelder-Mead only when needed.
4. Use the same bracketed endpoint search and endpoint residual verification.

The split is mathematically justified. Averages do not need a generic nonlinear
equality constraint. Forcing them through the fit-side constrained optimizer
would add fragility without a scientific gain.

The architecture problems are mostly at the abstraction and evidence layers:

- `binary_search_error()` is shared, but the object it calls is only an
  untyped closure. The endpoint search cannot ask whether a profile point is
  certified by KKT, descent check, direct nuisance minimization, or a future
  reduced chart except through side effects.
- Fit-side profiling is a monolithic closure mixing start projection, solver
  selection, KKT polishing, descent probing, diagnostics, and continuation
  state. This makes it harder to compare alternative mathematical
  formulations fairly.
- Average and fit profile diagnostics have similar semantics but are not
  represented by a shared problem/result interface.
- Bisection assumes the endpoint side has a usable monotone crossing. That is
  appropriate as a conservative root finder, but disconnected or nonmonotone
  profile curves are only caught indirectly by failed brackets or endpoint
  residuals.
- BR/BRU and softmax/arctan parameter maps solve the physical-domain problem
  through fitted-coordinate charts. That is useful for central fits, but it can
  make profile stationarity norms hard to interpret near chart saturation or
  simplex edges.
- The current local certificates are meaningful local numerical certificates,
  not global constrained-profile proofs. Round 3 did not find lower fixed-target
  chi2 values from multistart probes on five hard endpoints, but the globality
  risk remains conceptual.

Decision-level view: the current algorithm should not be replaced by another
solver stack. Its weak point is not "missing one more fallback"; it is the lack
of a stricter profile-problem interface and systematic profile-curve/globality
diagnostics around the existing solver.

## Candidate Architecture Comparison

Scores use `High`, `Medium`, and `Low` in the favorable direction except
`Implementation risk`, where `Low` is better.

| Candidate | Scientific validity | Simplicity | Robustness / certification | Expected speed | Implementation risk | Evidence needed | Compatibility |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Shared `ProfileProblem` abstraction with specialized fit and average implementations | High | Medium | High, if profile point certificates become explicit | Neutral | Medium | Same hard-set outputs and diagnostics must match current behavior exactly | High. It preserves direct averages and constrained fits |
| Direct fixed-coordinate profile for averages only | High | High | High for averages because the target is a coordinate | Good | Low | Existing full average sweep already supports it; rerun only after edits | Already current behavior |
| Chart-aware / reduced-coordinate profiles for BR/BRU/simplex targets | High if charts are valid and boundary-aware | Medium to Low | Potentially High, but only after careful chart and boundary diagnostics | Unknown | High | Compare against current fixed-target chi2 on BR/BRU hard rows, especially descent-check endpoints and boundary-near targets | Medium. Must not change parameter-map semantics or interpolation |
| Lagrangian/KKT-root continuation solver | High locally | Low to Medium | Potentially stronger local certificate than optimizer success plus ad hoc checks | Unknown | High | Needs hard-case comparison against SLSQP/trust-constr and failure behavior on nonsmooth/clipped-error points | Medium. Must handle nonsmooth interpolation clips and rank issues |
| Tangent-space local minimization/certification layer | High as a diagnostic | Medium | Medium to High for local checks; not global | Neutral to slow | Medium | Run on accepted `+descent-check` and high-gradient endpoints; compare with round 3 multistart | High as diagnostics, Medium as default acceptance |
| Profile-curve monotonicity / residual scans | High as evidence tooling | High | Medium. It can reveal nonmonotone or disconnected behavior but cannot prove absence | Slow if broad, bounded if targeted | Low | Run on hard endpoints and later on all descent-check/KKT rows with resumable output | High. Notes-only initially |
| Minuit/MINOS-style profiling as comparator | Medium to High conceptually | Medium | Medium. Useful independent HEP comparator, not automatically better for custom constraints/maps | Unknown | Medium to High | Curated low-dimensional and average cases first; compare endpoints and residuals | Medium. Good comparator, poor default without evidence |
| Replace bisection with secant/Newton/Brent endpoint root | Scientific validity High only if bracketed and verified | Medium | Medium. Root method does not fix bad profile solves | Potential speed win | Medium | Hard-case benchmark must show fewer profile calls with unchanged residual/failure behavior | Medium. Previous endpoint work did not justify it |
| Put endpoint search behind a stricter `ProfileRoot` interface while retaining bisection | High | High | High. Separates root policy from profile solve quality | Neutral | Low to Medium | Unit tests plus focused hard set should show identical endpoint values | High. Best near-term refactor candidate |

## Architecture Recommendations

### 1. Keep the Mathematical Invariant Untouched

Do not change the endpoint rule, residual verification, or constrained-profile
acceptance thresholds without a failure case. The current staged evidence is
stronger than the evidence for any proposed replacement.

### 2. Separate Profile Evaluation From Endpoint Root Search

The endpoint search should operate on an explicit object, not a bare callable:

```python
class ProfileProblem:
    target_name: str
    target_value: float
    chi2_min: float
    lower_initial: float
    upper_initial: float

    def evaluate(self, value: float) -> ProfilePoint:
        ...
```

Then `ProfileRoot` can own only:

- bracket contraction/expansion;
- bisection;
- endpoint residual verification;
- root-search counters;
- optional monotonicity trace hooks.

This is not a new algorithm. It is an architectural constraint that keeps
future solvers honest: every profile implementation has to return the same
kind of `ProfilePoint` with explicit feasibility and certificate fields.

### 3. Preserve Specialized Profile Implementations

A shared abstraction should not mean one shared optimizer.

- `FitConstrainedProfileProblem`: current scalar equality-constrained fitted
  coordinate profile.
- `AverageFixedCoordinateProfileProblem`: current direct primary coordinate
  profile over nuisance coordinates.
- Future `ReducedChartProfileProblem`: optional notes-only BR/BRU/simplex
  experiment.

This keeps the average path simple and avoids degrading the most validated
part of the system.

### 4. Treat Chart-Aware BR/BRU Work As A Research Spike

BR/BRU chart-aware methods are the most mathematically interesting alternative,
but also the riskiest. They may improve stationarity interpretation near
arctan/softmax saturation and simplex edges. They may also multiply code paths
and create subtle boundary bugs.

A useful next experiment would not replace anything. It would pick one
low-dimensional BR/BRU hard target and compare fixed-target profile chi2 from:

- current fitted-coordinate equality profile;
- direct physical-parameter reduced chart, where feasible;
- local tangent-space KKT residuals in the physical chart.

Ship none of it until it finds a real lower constrained chi2 or a simpler
certificate on hard cases.

### 5. Use Profile-Curve Diagnostics To Stress Bisection Assumptions

Conservative bracketed bisection should remain the endpoint finder. The better
architectural move is to add diagnostics that can tell us whether the profile
curve looks well behaved on hard rows:

- monotone delta-chi2 away from the MLE on each side;
- endpoint residual at fraction 1.0;
- beyond-endpoint delta-chi2 at, for example, 1.2x endpoint distance;
- method/certificate changes along the curve;
- failures or jumps that suggest disconnected feasible components.

This round added a notes-only version of that diagnostic.

## Notes-Only Spike: Profile-Curve Scan

Added:

- `notes/codex-refactor/run_asym_profile_curve_scan.py`
- `notes/codex-refactor/round04_profile_curve_summary.csv`
- `notes/codex-refactor/round04_profile_curve_summary.jsonl`
- `notes/codex-refactor/round04_profile_curve_points.csv`
- `notes/codex-refactor/round04_profile_curve_points.jsonl`

The script samples fixed-target profile chi2 values at fractions:

```text
0.0, 0.25, 0.5, 0.75, 1.0, 1.2
```

where fraction 0.0 is the MLE target, fraction 1.0 is the previously verified
endpoint from `notes/asym_fit_sweep_results_simplified.csv`, and fraction 1.2
is a modest beyond-endpoint check on the same side.

Targets:

- `B0::S042B95`
- `B0S-BR::S086.37`
- `eta_c J/psi psi(2S)::M026W`

These were chosen because they include hard labels, BR/BRU-like structure, and
descent-check/KKT-descent endpoints from prior rounds.

### Commands

Compile:

```bash
env PYTHONPATH=/root/pdgfits-private/src \
  PDGFITS_DATA_BACKEND=snapshot \
  PDGFITS_SNAPSHOT_DIR=/root/pdgfits-private/data/pdg-snapshot \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  /root/micromamba/micromamba run -r /root/micromamba-root -n pdg \
  python -m py_compile notes/codex-refactor/run_asym_profile_curve_scan.py
```

Run:

```bash
prlimit --as=7800000000 --rss=7800000000 -- \
  env PYTHONPATH=/root/pdgfits-private/src \
  PDGFITS_DATA_BACKEND=snapshot \
  PDGFITS_SNAPSHOT_DIR=/root/pdgfits-private/data/pdg-snapshot \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  /root/micromamba/micromamba run -r /root/micromamba-root -n pdg \
  python notes/codex-refactor/run_asym_profile_curve_scan.py \
    --target 'B0::S042B95' \
    --target 'B0S-BR::S086.37' \
    --target 'eta_c J/psi psi(2S)::M026W' \
    --fractions 0.0,0.25,0.5,0.75,1.0,1.2 \
    --summary-csv notes/codex-refactor/round04_profile_curve_summary.csv \
    --summary-jsonl notes/codex-refactor/round04_profile_curve_summary.jsonl \
    --points-csv notes/codex-refactor/round04_profile_curve_points.csv \
    --points-jsonl notes/codex-refactor/round04_profile_curve_points.jsonl \
    --stdout-log notes/logs/round04_profile_curve_scan_stdout.log
```

Result:

```text
wrote 6 summary rows and 36 point rows
```

The stdout log is small but left under `notes/logs/` uncommitted by policy.

### Results

| Label | Target | Side | Points ok | Endpoint residual | Beyond delta chi2 | Nonmonotone steps |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `B0` | `S042B95` | lower | 6/6 | 0.001156 | 1.442677 | 0 |
| `B0` | `S042B95` | upper | 6/6 | -0.004138 | 1.434581 | 0 |
| `B0S-BR` | `S086.37` | lower | 6/6 | -0.000857 | 1.449316 | 0 |
| `B0S-BR` | `S086.37` | upper | 6/6 | 0.001152 | 1.431730 | 0 |
| `eta_c J/psi psi(2S)` | `M026W` | lower | 6/6 | 0.000707 | 1.441063 | 0 |
| `eta_c J/psi psi(2S)` | `M026W` | upper | 6/6 | -0.000935 | 1.438735 | 0 |

The sampled curves were monotone away from the MLE on all six sides. Fraction
1.0 matched the existing endpoint residual tolerance. Fraction 1.2 remained
above the Wilks target on every side. This does not prove global monotonicity or
connectedness, but it is a useful negative result: no obvious bisection-side
pathology appeared on this small hard set.

Representative sampled deltas:

```text
B0 / S042B95 lower:
0.0 -> 0.000000
0.25 -> 0.062488
0.50 -> 0.250052
0.75 -> 0.562767
1.00 -> 1.001156
1.20 -> 1.442677

B0S-BR / S086.37 upper:
0.0 -> -0.000003
0.25 -> 0.064038
0.50 -> 0.254512
0.75 -> 0.567984
1.00 -> 1.001152
1.20 -> 1.431730

eta_c J/psi psi(2S) / M026W lower:
0.0 -> -0.000028
0.25 -> 0.062550
0.50 -> 0.250178
0.75 -> 0.562887
1.00 -> 1.000707
1.20 -> 1.441063
```

Tiny negative deltas at fraction 0.0 are numerical noise from resolving the
MLE fixed-target profile, not a scientific issue.

## What I Would Not Change

- Do not change `build_chi2.py` or the asymmetric interpolation model. This
  round found no scientific reason to alter the likelihood surface.
- Do not weaken endpoint residual verification. It is the central correctness
  invariant.
- Do not weaken constrained-profile acceptance checks. Feasible target
  constraints and optimizer success remain insufficient.
- Do not push averages through generic equality constraints. The current direct
  fixed-coordinate average profile is simpler and better matched to the
  problem.
- Do not replace bracketed bisection with secant/Newton logic. Endpoint root
  speed is not the limiting scientific issue, and bisection plus verification
  is easy to audit.
- Do not add another default fit-side fallback without a hard-case failure that
  it fixes. Round 3 and this round found no such failure.
- Do not treat Minuit/MINOS as a drop-in replacement. It is valuable as a HEP
  conceptual comparator or independent cross-check, but the current code has
  custom parameter maps, nuisance structure, and endpoint verification
  requirements that still need explicit checks.

## Recommended Next Codex Round

The next round should choose one of these two concrete tracks.

Preferred track: architecture prototype without behavior change.

1. Add a notes-only or guarded production `ProfileProblem`/`ProfileRoot` design
   prototype.
2. Wrap the current fit constrained profile and average fixed-coordinate
   profile behind that interface.
3. Reproduce a focused hard set and one average set, proving endpoint values,
   residuals, and profile-method diagnostics are unchanged.
4. Do not change default behavior unless the refactor is mechanically exact and
   tests/hard-set output match.

Alternative track: broaden the curve diagnostic.

1. Make `run_asym_profile_curve_scan.py` resumable and able to select all rows
   from `notes/asym_fit_sweep_results_simplified.csv` with endpoint methods
   containing `descent-check` or `KKT`.
2. Run it on a bounded high-risk subset first, for example all `B0`,
   `B0S-BR`, `eta_c J/psi psi(2S)`, and `Lam-b-0` descent-check rows.
3. Flag nonmonotone steps, endpoint residual drift, beyond-endpoint failures,
   and method discontinuities.
4. Only if a pathology appears, design a solver or endpoint-search change
   targeted at that failure.

I would not make chart-aware BR/BRU profiling the immediate next default-code
round. It is promising, but should first be a notes-only mathematical spike on
one low-dimensional hard target with direct comparison to the current fixed
target profile chi2.

## Bottom Line

The current method is not merely an optimizer stack; it is a conservative
profile-likelihood engine with endpoint residual verification and local
constrained-profile certificates. Its strongest remaining weakness is not a
missing fallback, but the lack of explicit architecture separating:

- profile problem definition;
- profile point certification;
- endpoint root finding;
- curve/globality diagnostics.

This round's small profile-curve scan found no nonmonotone behavior on three
hard targets. That supports leaving production code unchanged and investing the
next round in interface clarity and broader diagnostics rather than clever new
solvers.
