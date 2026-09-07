"""Shrinkage estimators for a multivariate normal mean.

This subpackage implements shrinkage estimators for the problem of estimating
the mean ``theta`` of :math:`X \\sim N(\\theta, \\Sigma)` under the loss
:math:`(\\delta - \\theta)^\\mathrm{T} Q (\\delta - \\theta)`.

Following [Tan2015]_, the general problem can always be transformed into the
canonical form where ``Sigma`` is diagonal, ``Q`` is the identity matrix and
the loss reduces to the sum of squared errors.  The public estimators accept a
general covariance matrix ``cov`` and loss matrix ``Q``, canonicalize the
problem internally, and transform the result back.  This keeps the
per-estimator implementations simple: they only ever need to shrink a vector
towards zero with independent coordinates of varying variance.

The estimators may shrink towards an arbitrary affine subspace
``offset + span(dirs)``, where ``dirs`` is a matrix whose columns span the
affine direction.  Following [Tan2016]_, Section 3.3, the projection onto this
subspace is built in the *covariance* (precision) metric: for a matrix
``V = dirs_star`` in canonical coordinates, the projector is
``P = V (V^T D^{-1} V)^{-1} V^T D^{-1}``.  This makes the fitted component
``P y`` and the residual ``(I - P) y`` statistically uncorrelated, so their
risks add and each can be improved independently.  The component in the
subspace is kept and the residual is shrunk towards zero; because the residual
basis ``l2`` is taken orthonormal (eigenvectors of the symmetric residual
covariance), the change of coordinates is an isometry of the squared-error
loss, so the reduced shrinkage conserves the full loss exactly.

The *prior* of the Bayes-rule estimators — :func:`bayes`, :func:`robust_bayes`
and :func:`tan_bayes` — and of the gamma-based minimax estimators
(:func:`tan` and :func:`minimax_bayes`) is a Gaussian
:math:`\\theta \\sim N(0, \\gamma \\Theta)`.  It has a *shape* ``Theta`` and a
*scale* ``gamma``:

- ``Theta`` — the prior covariance, of shape ``(p, p)`` in the same
  coordinates as ``x`` — is either left at its default ``Q^{-1}`` (the
  current homoscedastic prior in canonical coordinates) or set explicitly via
  the ``prior_cov`` argument of the estimators.  When given, the
  canonicalization *rotates the canonical frame* — always possible when the
  canonical coordinate variances coincide, in particular when ``cov`` is
  proportional to ``Q^{-1}`` — so that ``Theta`` is diagonal in the canonical
  coordinates, and passes that diagonal ``diag(pi)`` to the estimator.  The
  prior then acts per coordinate: the Bayes-rule direction/weight
  ``d_j / (d_j + gamma * pi_j)`` and the posterior mean
  ``gamma * pi_j / (d_j + gamma * pi_j)`` replace their homoscedastic
  counterparts.  ``Theta`` must be symmetric positive definite and must be
  diagonalizable in the canonical coordinates (its canonical off-diagonals
  must vanish up to numerical precision); for ``cov`` proportional to
  ``Q^{-1}`` this is automatic since the canonical variances then coincide and
  the rotation is unconstrained, so any positive-definite prior works there.
  The proportionality is detected with a ``sqrt(eps)``-relative, roundoff-aware
  check (:func:`_cov_proportional_to_qinv`), so it covers numerically-inverted
  matrices such as ``Q = inv(cov)`` rather than requiring exact
  proportionality.
- ``gamma`` is the prior *scale* — a non-negative number scaling the whole
  prior.  It can be given in any of four forms:

- a non-negative ``float`` giving a single prior scale shared by every
  coordinate and observation;
- the string ``"empirical"`` to infer the scale per observation from the data
  as ``gamma = ||y / sqrt(pi)||^2 / p_eff``, where ``y`` is the centered data in
  canonical coordinates, ``pi`` is the canonical diagonal of the prior
  covariance (all ones without ``prior_cov``) and ``p_eff`` is the effective
  dimension (one scale per observation, so a batched ``x`` — e.g. the draws
  of a risk sweep — yields one gamma per draw);
- a one-dimensional ``numpy.ndarray`` giving an explicit prior scale per
  observation; its shape must match the leading batch dimensions of ``x``
  (``()`` for a single vector);
- a callable ``f(d, pi, y)`` that computes the prior scales from the canonical
  coordinate variances ``d`` (shape ``(p,)``), the canonical diagonal ``pi`` of
  the prior covariance (shape ``(p,)``; all ones without ``prior_cov``), and
  the centered canonical data ``y``.  It may return either a singular scalar
  (a single scale shared by every observation, e.g. when the scale is inferred
  only from ``d`` / ``pi`` and not the data) or a real-valued, non-negative
  array whose shape equals ``y.shape[:-1]`` (one prior scale per
  observation).  Any error from the callable is reported as a
  :class:`TypeError`, and a shape mismatch or negative return as a
  :class:`ValueError`.

Every form resolves to one prior scale per observation, which is passed to the
estimator as an array broadcast against the batch dims of ``x``; the supported
estimators vectorize, so the whole batch is solved with no per-observation
Python loop.  The :func:`tan` and :func:`minimax_bayes` estimators, whose
gamma-dependent coordinate ranking or segmentation does not (yet) admit a
per-observation scale, accept only a non-negative ``float``; their ranking
may still depend on the per-coordinate prior shape ``diag(pi)``, which is
observation-independent.

The loss matrix ``Q`` may be positive *semi*-definite.  Its null space carries
no loss, so it is treated as an *additional set of no-shrink directions*,
exactly like the columns of ``dirs``: the null space of ``Q`` is added to the
no-shrink subspace ``span(dirs)`` and the estimator acts only on the
covariance-metric complement ``span(dirs) + null(Q)``, where the restricted
loss is positive definite.  Concretely, the covariance-metric projection of the
estimate onto ``span(dirs) + null(Q)`` equals that of the observed data (kept
at its data value), and shrinkage is applied in the complement.  Because only
the spanned space matters, whether a given direction comes from ``dirs`` or
from ``null(Q)`` is irrelevant; specifying directions already lying in
``null(Q)`` via ``dirs`` has no additional effect.  The risk on the full
problem is unaffected by how the estimate is set in ``null(Q)``: the loss is
blind to that component, so any choice there carries the same risk.

The implementation is split across three private modules,
``nustattools.stats.shrinkage._core`` (shared validation, canonicalization,
the affine-subspace projector and the ``_estimate`` front-end),
``nustattools.stats.shrinkage._estimators`` (the concrete estimators
``berger``, ``tan``, ``minimax_bayes``, ``tan_bayes``, ``robust_bayes`` and
``bayes``, the ``shrink`` front-end and the ``_METHODS`` registry)
and ``nustattools.stats.shrinkage._risk`` (the Monte-Carlo risk-estimation
helpers ``estimate_risk`` and ``estimate_risk_curve``).

References
----------

.. [Tan2015] Z. Tan, "Improved minimax estimation of a multivariate normal mean
    under heteroscedasticity," Bernoulli 21(1), 574-603 (2015),
    https://arxiv.org/abs/1505.07607

.. [Tan2016] Z. Tan, "Steinized empirical Bayes estimation of heteroscedastic
    data," Statistica Sinica 26(3), 1219-1248 (2016),
    https://doi.org/10.5705/ss.202014.0069

"""

from __future__ import annotations

from ._core import (
    # The underscore-prefixed helpers are re-exported for internal use only
    # (the test-suite exercises them directly).  They are NOT public API:
    # names, signatures and behaviour may change without notice, so do not
    # rely on them outside this package's tests.
    _canonicalize,  # noqa: F401
    _canonicalize_prior,  # noqa: F401
    _dirs_projection,  # noqa: F401
    _estimate,  # noqa: F401
    _estimate_split,  # noqa: F401
    _group_degenerate,  # noqa: F401
    _merge_dirs,  # noqa: F401
    _subspace_reduce,  # noqa: F401
)
from ._estimators import (
    bayes,
    berger,
    minimax_bayes,
    robust_bayes,
    shrink,
    tan,
    tan_bayes,
)
from ._risk import estimate_risk, estimate_risk_curve

__all__ = [
    "bayes",
    "berger",
    "estimate_risk",
    "estimate_risk_curve",
    "minimax_bayes",
    "robust_bayes",
    "shrink",
    "tan",
    "tan_bayes",
]
