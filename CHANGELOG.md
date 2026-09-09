# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `gamma` now also accepts factory strings like `"max_rel_risk(0.1)"` for the
  size-related estimators, resolving per observation like the callable form
  (built-in factories: `max_rel_risk`/`max_abs_risk`).
- The canonical coordinates are now guaranteed to be ordered by _decreasing_
  variance `d` (coordinate 0 has the largest variance), matching the risk-curve
  axis convention; the canonical prior diagonal `pi` follows the same order and
  is non-increasing within each block of (numerically-)equal `d`.
- New `prior_cov` argument to the prior-based shrinkage estimators (`bayes`,
  `robust_bayes`, `tan_bayes`, `tan`, `minimax_bayes`) and `shrink`,
  `estimate_risk` / `estimate_risk_curve`: an explicit prior covariance matrix
  `Theta` in the original coordinates, making the Gaussian prior
  `theta ~ N(0, gamma * Theta)` (defaults to `Q^{-1}`, i.e. the previous
  homoscedastic canonical prior). The canonicalization rotates the canonical
  frame so that `Theta` is diagonal in canonical coordinates (with the rotation
  free when `cov` is proportional to `Q^{-1}`), and `gamma` continues to scale
  the prior per coordinate. `berger`, which involves no prior, rejects
  `prior_cov` with a `TypeError`.
- `tan`'s infinite-`gamma` limit (`A†_inf`) now respects an explicit prior shape
  whenever one is given, ranking coordinates by `d_j^2 / pi_j` instead of the
  flat `d_j^2`; with the default homoscedastic prior both coincide.
- The `prior_cov` diagonalization now recognises `cov` proportional to `Q^{-1}`
  at numerical roundoff (relative `sqrt(eps)`), so any positive-definite prior
  is accepted with `Q = inv(cov)` even for ill-conditioned covariances.
- New shrinkage estimator `robust_bayes` (Berger's 1982 `delta^RB`, Tan2015
  Equation 7) with a homoscedastic prior `Gamma = gamma I`, a non-minimax
  robust-to-prior-misspecification estimator, registered in `shrink` and
  exported from `nustattools.stats.shrinkage`.
- New shrinkage estimator `tan_bayes` (delta_{A,c} from Tan2015, Section 3, with
  the Bayes-rule shrinkage direction A = D(D + gamma I)^{-1}), registered in
  `shrink` and exported from `nustattools.stats.shrinkage`.
- New shrinkage estimator `bayes` (Bayes rule under the homoscedastic prior
  Gamma = gamma I in canonical coordinates), a non-minimax estimator registered
  in `shrink` and exported from `nustattools.stats.shrinkage`.
- New shrinkage estimator `minimax_bayes` (Berger's 1982 `delta^MB`, Tan2015
  Equation 8) with a homoscedastic prior `Gamma = gamma I`, registered in
  `shrink` and exported from `nustattools.stats.shrinkage`.

### Removed

- Dropped support for Python 3.8 and 3.9. The minimum required Python version is
  now 3.10.

### Changed

- The `gamma` parameter of the `bayes`, `robust_bayes` and `tan_bayes`
  estimators accepts a numeric `float`, a one-dimensional `numpy.ndarray` with
  one prior scale per observation (its shape must match the leading batch
  dimensions of `x`), or the string `"empirical"` to infer the per-observation
  scale from the data as `||y||^2 / p` where `y` is the centered data in
  canonical coordinates and `p` is the effective dimension. The per-observation
  scale — whether supplied explicitly or inferred via `"empirical"` — is passed
  to the estimators as a broadcast array, so the whole batch is solved
  vectorized with no per-observation Python loop. The `tan` and `minimax_bayes`
  estimators' `gamma` parameter is now strictly a non-negative `float` (no
  string, no per-observation scale): their gamma-dependent coordinate ranking or
  segmentation does not (yet) admit a per-observation scale.
- The `gamma="empirical"` prior scale is now dispatched through an internal
  registry of named presets (only `"empirical"` is currently registered), which
  lays the groundwork for future presets and callable prior scales.
- The `gamma` parameter of the `bayes`, `robust_bayes` and `tan_bayes`
  estimators may also be a callable `f(d, pi, y)`, where `d` are the canonical
  coordinate variances, `pi` the canonical diagonal of the prior covariance (all
  ones without `prior_cov`) and `y` the centered canonical data; it must return
  a real-valued, non-negative array of prior scales, either a singular scalar
  (one scale shared by every observation, e.g. inferred only from `d` / `pi`) or
  an array whose shape matches the batch dimensions of `y`. The `tan` and
  `minimax_bayes` estimators remain strictly `float` only.
- The `"empirical"` preset computes the prior-aware MLE-like scale
  `||y / sqrt(pi)||^2 / p` (so the default homoscedastic prior recovers the
  classic `||y||^2 / p`); the callable now receives `pi` for custom scale
  formulas.
- The canonical-prior diagonal threaded to the canonical estimators is named
  `pi` (matching `d`), not `pi_diag`.
- `estimate_risk` and `estimate_risk_curve` are faster: the quadratic loss is
  evaluated without the `(p, p)` einsum, and `estimate_risk_curve` now draws the
  Monte Carlo noise once as `N(0, cov)` and translates it to each sweep point
  (common random numbers) instead of re-drawing per point. Results are unchanged
  for seeded calls.
- Only the `shrink` front-end is now exported from the `nustattools.stats`
  package; the other shrinkage estimators (e.g. `berger` and `tan`) and helpers
  are available through the `nustattools.stats.shrinkage` submodule.
- `nustattools.stats.shrinkage` is now a subpackage: the implementation is split
  across private `_core`, `_estimators` and `_risk` modules (which keeps
  individual files well under 2000 lines), with the public API re-exported from
  `nustattools.stats.shrinkage` unchanged.
- The API docs now render each module on its own page; in particular
  `nustattools.stats.shrinkage` is documented on a separate page below
  `nustattools.stats` (via `sphinx-apidoc --separate`).
- The affine-subspace interface of the shrinkage estimators changed: the
  idempotent `projection` matrix argument was replaced by `dirs`, a matrix whose
  columns span the subspace (see the `Added` entry).
- Berger's estimator renames its shrinkage-magnitude parameter `c` to
  `strength`, now expressed as a fraction of the optimal minimax value (0 =
  identity, 1 = optimal, 2 = boundary of the minimax class).
- Clarified the documentation of the `tan` estimator's `gamma` parameter:
  `gamma=0` corresponds to a prior proportional to the covariance and
  `gamma=inf` to a flat prior uniform in the canonical coordinate space (per Tan
  2015, Section 3.3), rather than literally "no prior".
- The `stats.shrinkage.tan` estimator's `gamma` parameter now accepts any
  non-negative value, not just `0` and `inf`; finite positive `gamma`
  interpolates between the two extreme priors.
- Rewrote the shrinkage docstrings to be self-contained: they now define the
  canonical-form change of coordinates, the canonical coordinate variances
  `d_j`, and the shrinkage-direction matrix `A` (and the `A†_0`/`A†_∞` limits),
  so they are readable without the reference papers.

### Fixed

- Integer "axis j" directions in `estimate_risk_curve` are now resolved using
  the shared frame's prior ordering: within blocks of (numerically-)equal
  canonical variance `d` the frame orders coordinates by decreasing prior
  diagonal `pi`, so axis `j` picks the estimator's canonical coordinate `j`.
  Previously the axis tie-break ignored the prior (an `argsort` on the all-equal
  `d`), so e.g. with `cov = Q^{-1}` axes pointing at equal prior variances (such
  as `-1` and `-2` with two equal trailing `pi` entries) produced different risk
  curves.
- `estimate_risk_curve` now maps true means back to the original space with the
  same (prior-rotated) canonical frame the estimators canonicalize the data in,
  so integer "axis j" and raw-vector directions sweep the estimator's canonical
  coordinate `j` (whose prior diagonal `pi` is non-increasing within each block
  of equal `d`) instead of the unrotated-frame coordinate. The shared
  canonical-frame construction is factored into a private `_canonical_frame`
  helper used by both the estimators and the risk curve.
- `tan_bayes`'s infinite-`gamma` limit no longer degenerates to the identity:
  since `delta_{A,c}` is invariant under a scalar rescaling of the Bayes-rule
  direction `A`, the `gamma = inf` limit uses the direction `a_j = d_j / pi_j`
  (shrinkage proportional to variance, like `tan`'s `A†_inf`), reducing to the
  identity only when the corresponding `c* = c*(D, diag(d/pi))` is non-positive.
- The documentation of `estimate_risk_curve` now states that the loss matrix `Q`
  must be positive definite; unlike the estimators and `estimate_risk`, the risk
  curve does not accept a positive semi-definite `Q`.

### Added

- The shrinkage estimators (`shrink`, `berger`, `tan`) and `estimate_risk` now
  accept a positive semi-definite loss matrix `Q`: the null space of `Q` (where
  the loss is zero) is treated as an additional set of no-shrink directions, so
  its covariance-metric projection is kept at the data value while shrinkage
  acts in the covariance-metric complement.
- Support for Python 3.14.
- Documented the `uv`-based development workflow (`uv sync`, `uv lock`) in the
  README and `AGENTS.md`.
- CI now checks that `uv.lock` is up to date and uses `uv` to install `nox`.
- Paper citation in docstrings of `FMaxStatistic`, `QMaxStatistic`,
  `OptimalFMaxStatistic`, and `Cee2`.
- Citation of "Plotting correlated data" (Koch 2026, arXiv:2601.20805) in
  docstrings of `corlines`, `wedgeplot`, and `pcplot`.
- Citation of the same paper and the original Hinton diagram reference (Hinton &
  Shallice 1991) in the docstring of `hinton`.
- Shrinkage estimators for a multivariate normal mean in `stats.shrinkage`,
  including Berger's minimax estimator and a canonical-form front-end that
  handles general covariance and loss matrices.
- `stats.shrinkage.tan`, Tan's improved minimax shrinkage estimator (Tan 2015),
  which segments coordinates by Bayes importance and improves on Berger's
  estimator when the truth concentrates in low-variance coordinates. Supports
  the `gamma=0` (no prior) and `gamma=inf` (flat prior) special cases.
- The shrinkage estimators accept `offset` and `dirs` arguments: an arbitrary
  affine subspace `offset + span(dirs)` is specified by its spanning vectors,
  and the projection onto it is built in the covariance (precision) metric (per
  Tan 2016, Section 3.3) so the fitted and residual components are uncorrelated.
  The residual is then shrunk recursively, conserving the full squared-error
  loss exactly.
- Added `stats.estimate_risk`, a Monte-Carlo helper to estimate the risk (and
  its standard error) of a shrinkage estimator on shared samples, making
  performance comparisons between estimators easier.
- Added `stats.estimate_risk_curve`, which sweeps the estimated risk of several
  shrinkage estimators against the true mean along configurable directions
  (named, canonical axis, or a raw vector) at increasing distances, returning
  records ready for a data frame and a seaborn plot. Each record includes the
  Mahalanobis distance of the true mean, and integer directions accept negative
  indices (e.g. -1 is the smallest variance).

### Fixed

- `stats.shrinkage.tan` no longer mis-applies the shrinkage weights to the wrong
  coordinates when the covariance is not already sorted: the estimator now
  reorders the data to match the Bayes-importance ordering before applying the
  paper's coordinate-wise shrinkage (Tan, 2015).

## [0.8.0]

### Changed

- Changed how the `pcplot` works. Now plots multiple principal components.

## [0.7.0]

### Added

- `wedgeplot`, `corlines` and `pcplot` plots.

## [0.6.0]

### Added

- `plotting` module with `hinton` plot function.

## [0.5.0]

### Added

- Argument to calculate null space of model projection for Goodness of Fit tests
  with the covariance derating method.

## [0.4.1]

### Fixed

- Readthedocs.io checkout process.

## [0.4.0]

### Changed

- Renamed `derate_covariance` parameter `accuracy` to `precision`.
- Use Generalised Chi Squared distribution to calculate critical values.

### Added

- Argument to change whitening transform in `derate_covariance`.

## [0.3.1]

### Fixed

- Fixed numerical precision issues in covariance derating.

## [0.3.0]

### Changed

- Moved `robust` module into `stats`.

### Added

- Multiple TestStatistics and RVTestStatistic
- Covariance derating now works with known 0 off-diagonal blocks.

## [0.2.1]

### Fixed

- Cee and Cee2

## [0.2.0]

### Added

- Distributions used for robust test statistics.

## [0.1.0]

### Added

- Covariance derating for unknown correlations.
