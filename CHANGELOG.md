# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- New shrinkage estimator `tan_bayes` (delta_{A,c} from Tan2015, Section 3, with
  the Bayes-rule shrinkage direction A = D(D + gamma I)^{-1}), registered in
  `shrink` and exported from `nustattools.stats.shrinkage`.
- New shrinkage estimator `bayes` (Bayes rule under the homoscedastic prior
  Gamma = gamma I in canonical coordinates), a non-minimax estimator registered
  in `shrink` and exported from `nustattools.stats.shrinkage`.
- New shrinkage estimator `berger_mb` (Berger's 1982 `delta^MB`, Tan2015
  Equation 8) with a homoscedastic prior `Gamma = gamma I`, registered in
  `shrink` and exported from `nustattools.stats.shrinkage`.

### Removed

- Dropped support for Python 3.8 and 3.9. The minimum required Python version is
  now 3.10.

### Changed

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
