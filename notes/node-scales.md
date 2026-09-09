# Node scales: experimental linear-Gaussian reference

Richie Bonventre's *PDG Fits, Averages, and Scale Factors* (21–22 November
2024) is available locally at
`../binomial/src/pdg/PDGFits_collab2024.pdf`. Slides 13–20 distinguish the old
weak-input and pull prescriptions from the proposed observed/expected χ² ratio.
This implementation is a reference model for that discussion, not a reproduction
of every PDG convention or a claim that the proposal was subsequently adopted.

`birge.linear_birge(y, X, V, nodes)` computes exact residual expectations for a
linear unconstrained mean and fixed positive-definite Gaussian covariance. It
keeps correlated components together for nonnegative diagnostic contributions.
The returned response matrix shows how unknown block variances mix into residual
contributions after fitting. Dividing once by a nominal expected contribution
does not generally estimate each block's own scale when the true scales differ.

`birge.fit_linear_scales(y, X, V, nodes)` uses a more direct model: one scale
s≥1 per node, with measurement covariance D_s V D_s. This preserves correlations,
including correlations between differently scaled nodes. At each scale update,
solve the ordinary GLS fit. Optimize the Gaussian restricted likelihood

    [log det V_s + log det (X' V_s^-1 X) + r' V_s^-1 r] / 2.

For a single ordinary average this gives the usual clipped Birge scale
sqrt(max(1, Q/(N−1))). The determinant adjustment accounts for fitting the mean.
For multiple nodes, the derivative with respect to log(s_g²) is
(expected contribution − observed contribution)/2. These contributions can be
negative with cross-node correlations; the implementation optimizes the full
criterion rather than taking a square root of a potentially negative ratio.
The fixed full-rank mean basis makes the determinant well defined even when X
has redundant columns. No automatic weak-input exclusion is performed.

Returned measurement covariance and fitted parameters are conditional on the
estimated scales. They do not include uncertainty in the scales. A saturated
node may supply no information about its scale; an optimum at the floor is not
proof that its quoted uncertainty is correct. A successful bounded optimization
is a stationarity check, not a global-optimum theorem.

These scale routines remain opt-in. Do not substitute an observed Hessian or
fitted asymmetric covariance into the exact expectation argument. The separate
[general implementation](general-node-scales.md) makes an explicit sampling
choice for nonlinear means, active constraints and asymmetric errors. Interval
calibration remains a further statistical question.

Checks cover ordinary Birge recovery, independent nodes with different scales,
correlated inputs without correlation reversal, expected residual mixing,
physical-unit and row-order invariance, nonidentifiable residual scales, and an
independent derivative-free REML optimization with cross-node correlations.
`pdgstudy/tools/experiment_node_scales.py` adds 6,000 Gaussian toy fits. It shows
that node-specific scale estimation can help localize inconsistency, but ordinary
±1 conditional-SE intervals still undercover substantially when there are only
three inputs per node and a genuinely common inflation. It is not ready to be
advertised as an interval-calibration solution.

The broader statistical and exclusion findings, with reproducible figures, are
in `pdgstudy/notes/birge-and-exclusions.md`. General REML background:
Bates et al. (2015), [Fitting Linear Mixed-Effects Models Using lme4](https://www.jstatsoft.org/article/view/v067i01).
The node-scale covariance model and derivative calculation here are explicit
specializations, not a claim that lme4 implements this particular API.

Validation on this addition: 170 offline tests passed (two skipped), and all
369 saved snapshot cases exactly matched the pre-addition results, including
fit values, covariances and profile endpoints. The two real mass-fit examples
also have explicitly checked linear measurement maps and symmetric inputs.

## Comparison with the PDG prescription

The [2026 introduction, §5.2](https://pdg.lbl.gov/2026/reviews/rpp2026-rev-rpp-intro.pdf)
still documents the separate-pull fit formula, the average precision cutoff, and
original-center reporting. `pdg_scaling.py` implements these baselines, using the
local `savg.f`, `sscafac.f` and `sbrfit.f` to fix implementation details.
`MAXTRY=1`, the fit scaling threshold is 1.001, and the pull guard is 1e-5.
Correlated blocks use the legacy rank-one covariance adjustment. Singular
measurement-dependency matrices and nonlinear asymmetric fit-error propagation
are not implemented by this linear comparator.

The new average agrees with the original Fortran numerical routines on all
2,449 modern quantities: largest mean difference 6.9e-13 quoted-error units;
relative scaled-error/scale differences below 5.8e-8 (the Fortran scale uses
single precision). The original numerical routines were compiled unchanged
with a minimal input COMMON-block harness.

Across seven eligible real mass fits, the PDG and REML conditional errors are
usually close: median ratio 1.00, largest difference about 17% with all inputs.
The evidence does not support replacing the existing separate-pull formula
wholesale. Correlation modification and original-center reporting are distinct
issues, and the simulations show regimes in which they matter substantially.
Excluding weak inputs can help or hurt under explicitly different sampling models;
no exclusions is retained as the simpler primary alternative for further study.
