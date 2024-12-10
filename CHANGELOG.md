# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
