# General node scales

`fit_node_scales(fit)` extends the separate-scale calculation to nonlinear
measurands, asymmetric errors, physical bounds and known correlations. It uses
the existing measurement map and objective. It is an explicit working model and
an opt-in estimating procedure, not a claim that PDG errors follow this law or
that its final confidence intervals are calibrated.

## Model and update

Let Q(θ,s;y) be the existing interpolated-error objective with both quoted widths
of row i multiplied by its node scale s_g ≥ 1. The sampling density is proportional
to exp(−Q/2) in the original measurement units. For nonsingular inputs, changing
the scales multiplies its normalizer by the product of the row scales. Asymmetric
quoted intervals alone would not have determined this sampling density.

At each scale vector, fit θ, simulate measurements from the fitted model, and
refit θ in every simulation. For each node calculate

    A_g = −½ ∂Q / ∂log(s_g),
    adjusted score = observed A_g − mean refitted simulated A_g.

The normalizer's derivative cancels in this subtraction. Inflation balances the
observed and expected contributions; a node at the scale floor may have a negative
adjusted score. Contributions can be negative with cross-node correlations, so
taking an unconditional square root of observed/expected contributions is not
a valid update. That ratio is used only for eligible positive initial seeds.

For a linear Gaussian mean model, the expected contribution is
½ tr(P ∂V/∂log(s_g)), where
P = V⁻¹ − V⁻¹X(XᵀV⁻¹X)⁻¹XᵀV⁻¹. The resulting equations are the Gaussian REML
equations. A single ordinary average recovers the clipped Birge ratio. This is
checked numerically, including the Monte Carlo uncertainty in the expectation.

Centering a profile score is an established adjustment idea; see
[McCullagh and Tibshirani (1990)](https://academic.oup.com/jrsssb/article/52/2/325/7027900).
Their work also considers an information adjustment. This implementation uses
score centering to estimate scales; it does not construct their full adjusted
likelihood or borrow an interval-calibration guarantee from it.

The outer solver first tries a short joint score update, then a bounded joint
least-squares solve, with scalar bracketing retained for corners. Every update
refits the full mean model. Common random draws keep
the calculation reproducible. Finite simulated profiles can jump when their
active minimum changes. The default stopping budget is the strict numerical tolerance
(`mc_tolerance=0`). An explicit nonzero `mc_tolerance` can allow a fraction of
one simulation standard error; both raw and excess residuals are returned.
Raising `draws` reduces the statistical simulation error.

The stopping residual is in log-scale units. With n_g measurements in node g,
it is the largest absolute component of

    max(log(s_g) + asinh((observed A_g − expected A_g)/(2 n_g)), 0) − log(s_g).

The default tolerance is 0.001. This includes the one-sided floor condition;
`raw_score_residual` uses this normalization without a Monte Carlo allowance.

## Asymmetry, bounds and exact dependence

For a full-rank correlation block, write z=r/σ(r). Inside the interpolation
segment, r=a z/(1+b z), with a=2e₋e₊/(e₋+e₊) and b=(e₊−e₋)/(e₋+e₊).
Its Jacobian is a/(1+b z)². Reweighting correlated normal draws by this Jacobian
gives exact rejection sampling from the normalized Q kernel. Symmetric errors
recover correlated Gaussian sampling exactly.

At an asymmetric-error knot, an arbitrary autodifferentiation branch need not
give the derivative of the profiled objective. The calculation uses a convex
combination of the one-sided derivatives consistent with mean-fit stationarity,
including active physical-boundary multipliers. The single-kink result is checked
against independently refitted finite differences.

Exactly dependent reported summaries do not define a density on the full ambient
space. They are sampled on their fixed physical measurement plane. Singular
symmetric blocks use an exact Gaussian draw on that plane. Singular asymmetric
blocks use Student proposals and importance-weighted expectations, with effective
sample size and Monte Carlo error reported. The raw relation is checked against
the measurements and model. The two actual dependent mass-fit systems are retained.
Here the correlation matrix specifies the Q kernel; independently scaled entries
do not by themselves define a new full-rank covariance or a new exact relation.

## What is and is not established

The implementation explicitly checks mean convergence, physical feasibility,
profile endpoints, sampler moments, the Gaussian limit and nonlinear/asymmetric
examples. Simulations are batched; solutions that do not meet the fast path's
stationarity test go through the scalar safeguarded optimizer.

Scale estimates can be weakly identified, and general score equations can have
multiple solutions. The algorithm uses a fixed initialization and safeguarded
updates; it is not a uniqueness or global-optimality proof. A floor value is not evidence that a
node's uncertainty is known to be correct. The returned covariance and profile
intervals condition on estimated scales, and therefore omit scale-estimation
uncertainty. Neither score balancing nor successful optimization establishes
nominal coverage. The separate Gaussian calibration experiments show why that
distinction matters. At active bounds or nonsmooth minima the returned observed
curvature is not marked as an accurate Hesse approximation; use the checked
profile interval to describe the objective’s shape.

`pdg_fit_scales` supplies a comparison using the PDG scaling pass on the same
Python mean-fit objective: one initial precision selection, a refit, separate
pull scales, a covariance update, and another refit. Exactly dependent blocks
receive a common block factor; ordinary correlation blocks use the Fortran
rank-one update. Original and refitted centers remain separate. This isolates
the scale prescription rather than claiming an exact port of all legacy
asymmetric-error propagation. Its linear limit agrees with the direct linear
reference, with and without exclusions and with correlated inputs.
