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
``berger``, ``tan`` and ``berger_mb``, the ``shrink`` front-end and the
``_METHODS`` registry)
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
    _dirs_projection,  # noqa: F401
    _estimate,  # noqa: F401
    _estimate_split,  # noqa: F401
    _merge_dirs,  # noqa: F401
    _subspace_reduce,  # noqa: F401
)
from ._estimators import berger, berger_mb, shrink, tan
from ._risk import estimate_risk, estimate_risk_curve

__all__ = [
    "berger",
    "berger_mb",
    "estimate_risk",
    "estimate_risk_curve",
    "shrink",
    "tan",
]
