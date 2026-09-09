# General node scales from local fit geometry

`fit_node_scales(fit)` now calculates scales **once at the unscaled fit**, then
refits the original objective with those scales. It uses no random draws and no
outer scale solver. It replaces the earlier simulation-and-refit estimator.
That former implementation remains in Git history at `e605082`.

## The calculation

Let z be the normalized measurement residuals and R their supplied correlation
matrix. Linearize z with respect to the free fit coordinates, giving Jacobian A.
Write R = L Lᵀ and let U be an orthonormal basis for the columns of L⁺ A. In the
local Gaussian tangent model, fitting removes U's directions, so

    C_res = L (I − U Uᵀ) Lᵀ.

For a group g containing measurements of one node, calculate

    Q_g = z_gᵀ R_gg⁺ z_g,
    E_g = tr(R_gg⁺ C_res,gg),
    S_g = max(1, sqrt(Q_g/E_g)).

These are a node's **marginal** quadratic disagreement and local expected
disagreement. Both are nonnegative. Known cross-node correlations enter the
full-fit projection. When nodes are correlated, their marginal Q_g values need
not sum to the total fit Q. We do not split signed cross terms between nodes or
assign arbitrarily rotated whitening coordinates to physical nodes.

For independent symmetric measurements, E_g is the sum of their residual
leverages, Σ(1−h_i). Over all independent nodes it sums to n−rank(A).
One ordinary average gives Q/(n−1) exactly. A single correlated node similarly
gives the generalized Birge ratio using its full covariance. For a Gaussian
linear fit these expectation statements are exact under the supplied covariance.
For a general fit they describe the local reference model, not an exact
repeated-sampling expectation for the full nonlinear estimator.

The scale is computed at the original errors; it is not recomputed after the
scaled fit. In a joint fit, inconsistency in one node can move the fitted mean
and thereby inflate another node's diagnostic. This single pass does not solve
that variance-attribution problem or claim to estimate each unknown true
variance without bias. The separate Gaussian REML reference remains in
`birge.fit_linear_scales`; it is not invoked by the general method.

## Asymmetry, bounds and dependence

The Jacobian differentiates the **normalized residual**, including the slope of
the existing asymmetric-error interpolation. Thus its geometry follows the same
objective being fitted. At an interpolation knot the derivative is not unique;
the calculation uses the average of the two one-sided slopes and reports the
knot rows. This is an explicit local approximation, not a hidden sampling law.

The original fit is polished in physical coordinates before calculating the
geometry. Exact equalities are removed by that coordinate chart. At an active
inequality boundary the projection uses directions tangent to the current
face and reports the active constraints. Conditioning on that face is not the
unconditional Gaussian boundary-mixture distribution.

Exactly dependent correlated summaries have fewer supported measurement
directions. Their null directions are removed from the geometry; linked nodes
share one scale so scaling does not break their exact measurement relation.
Ordinary nonsingular correlations permit different node scales, with each
correlation coefficient retained. A group with zero residual information is
reported as `estimable=False` and left at scale one; this does not establish
that its quoted errors are correct.

Both quoted error widths are multiplied by S. The actual scaled mean and
profile fits still use the full nonlinear/asymmetric objective. Their intervals
condition on the estimated scales and omit scale-estimation uncertainty.
Numerical convergence does not establish confidence-interval coverage.

## Current PDG comparison and exclusions

`pdg_fit_scales` is the separate comparator for the PDG prescription: choose weak
inputs once, refit, compute separate pull scales, update the covariance, and
refit. It returns the original and refitted centers separately. It applies that
prescription to the same Python objective; it is not a complete port of the
Fortran asymmetric-error propagation.

For independent measurements the PDG rule averages individual squared pulls,
while the local rule divides a node's total Q by its total expected Q. They
agree when residual leverage is equal within a node, but need not agree
otherwise. Neither equivalence nor superiority is assumed.

The general local method makes no extra precision cut. The exclusion rule's
intended case must nevertheless be tested: well-calibrated weak measurements
can dilute the inferred scale when precise measurements underestimate their
errors. Removing them can appropriately **increase** the scale. This is a
reason to compare both rules, not to conclude that no exclusions is always
preferable. The companion `pdgstudy` experiments include this case explicitly.

## Use

```python
from pdgfits.fit import run_fit
from pdgfits.node_scales import fit_node_scales
from pdgfits.asym_errors import calc_asym_errors

scaled = fit_node_scales(run_fit(label))
print(scaled['node_scaling']['groups'])
intervals = calc_asym_errors(scaled)
```

`node_scaling` records the method, node scales, groups, observed and expected
marginal Q, effective ranks, asymmetric/knot rows, active bounds and original
fit values. `input_scales` holds the factor for each measurement. The former
`draws`, `seed`, score tolerances and simulation-history fields have been removed
rather than silently ignored.
