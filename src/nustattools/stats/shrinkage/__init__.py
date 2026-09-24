"""Shrinkage estimators for a multivariate normal mean.

This module implements shrinkage estimators for the problem of estimating
the mean :math:`\\theta` of :math:`\\vec x \\sim N(\\vec\\theta, \\Sigma)`
under the quadratic loss :math:`(\\vec\\delta - \\vec\\theta)^\\mathrm{T} Q
(\\vec\\delta - \\vec\\theta)`, where :math:`\\vec\\delta` is the estimatate.
The aim of shrinkage estimators is to reduce the expectation value of the loss,
the risk, for some or all possible values of :math:`\\vec\\theta`, compared to
the risk of the Maximum Likelihood Estimator (MLE) :math:`\\vec\\delta = \\vec
x`.

Following [Tan2015]_, the general problem can always be transformed into the
canonical form :math:`\\vec y = T \\vec x`, with a suitable transformation
matrix :math:`T`. In the canonical form, :math:`\\Sigma` is a diagonal matrix
:math:`D`, :math:`Q` is the identity matrix and the loss reduces to the sum of
squared errors. The public estimators accept a general covariance matrix
``cov`` and loss matrix :math:`Q`, canonicalize the problem internally, and
transform the result back. The canonical coordinates are ordered by
*decreasing* variance :math:`d_i` (coordinate :math:`i = 0` has the largest
variance).

The estimators may shrink towards an arbitrary affine subspace :math:`\\vec o +
\\operatorname{span}(\\mathcal D)`, where :math:`\\vec o` is an ``offset`` and
:math:`\\mathcal D` is a set of direction vectors that span the affine subspace
(providede as a matrix ``dirs`` with the vectors making up the columns).
Following [Tan2016]_, Section 3.3, the projection onto this subspace is built
in the data covariance (precision) metric: Let :math:`V` be the matrix of
direction vectors in the canonical coordinates. Then the projector is :math:`P
= V (V^T D^{-1} V)^{-1} V^T D^{-1}`.  This makes the fitted component :math:`P
y` and the residual :math:`(I - P) y` statistically uncorrelated, so their
risks add and each can be improved independently. The component in the affine
subspace is kept and the residual is shrunk towards zero.

The prior used in some estimators -- e.g. :func:`bayes`, :func:`robust_bayes`,
:func:`tan_bayes`, and :func:`minimax_bayes` -- is a Gaussian
:math:`\\vec\\theta \\sim N(\\vec 0, \\gamma \\Gamma)`.  It has a shape
:math:`\\Gamma` (provided as ``prior_cov`` and a scale :math:`\\gamma`. The
prior must be diagonal in the canonical space. Its elements :math:`\\pi_i`
follow the same large-to-small ordering as the data covariance. They are
non-increasing within each block of (numerically-)equal :math:`d_i`.

There are three cases that ensure that the prior is diagonal in the canonical
coordinates:

- :math:`\\Gamma \\propto Q^{-1}` -- In this case, :math:`Q = I` in the
  canonical space, so :math:`\\Gamma \\propto I`. The default assumption when
  no explicit ``prior_cov`` is specified is :math:`\\Gamma = Q^{-1}`

- :math:`\\Gamma \\propto \\Sigma` -- In this case, :math:`\\Sigma = D` in the
  canonical space, so :math:`\\Gamma \\propto D`.

- :math:`\\Sigma \\propto Q^{-1}` -- In this case, :math:`\\Sigma \\propto I`
  in the canonical space, so any :math:`\\Gamma` can be made diagonal by a free
  rotation.

The scaling factor :math:`\\gamma` can be specified in five ways:

- As a non-negative ``float`` giving a single prior scale shared by every
  coordinate and observation.

- The string ``"empirical"`` to infer the scale per observation from the data
  as :math:`\\gamma = \\sum_i y_i^2 / \\pi_i / p_{\\mathrm{eff}}`, where
  :math:`p_{\\mathrm{eff}}` is the effective number parameters. The latter can
  be lower than the dimension of the data :math:`\\vec x`, when shrinking
  towards a subspace.

- A one-dimensional :class:`numpy.ndarray` giving an explicit prior scale per
  observation; its shape must match the leading batch dimensions of
  ``x`` (``()`` for a single vector).

- A string ``"max_rel_risk({alpha}, {precision})"`` or ``"max_rel_risk({beta},
  {precision})"``. The arguments must be given as numeric literals (e.g.
  ``"max_abs_risk(1, 1e-3)"``).

  These empirical methods chose :math:`\\gamma` such that the application of
  Bayes' rule (e.g. with :func:`bayes`) will lead to a shift of the data by a
  vector with a length of at most ``alpha`` in the canonical space. If the
  data is far away from the shrinkage target, where the reduction of variance
  is negligible, this bias is equal to the increase of risk. Hence the name
  ``max_*_risk``.
  In the case of ``max_rel_risk``, the risk increase ``alpha`` is calculated as
  a fraction ``beta`` of the MLE risk.

  The value of :math:`gamma` that leads to a data shift of the requested length
  needs to be approximated numerically. The iterative process is stopped as
  soon as the actual length is within ``alpha +/- precision``. If no
  ``precision`` is specified, the first order approximation

  .. math::

    \\gamma =  \\sqrt{ \\frac{\\sum_i \\left(\\frac{d_i y_i}{\\pi_i}\\right)^2}
                             {\\alpha \\sum_j d_j}}

  is used. This approximation overestimates :math:`\\gamma` for data points
  close to the shrinkage target, leading to weaker shrinkgge. It is always
  guaranteed that :math:`\\gamma \\ge 0`, both in the first order approximation
  and the numerical evaluation. In the latter case, a data point within a
  distance of ``alpha`` from the shrinkage target will be pulled exactly to the
  shrinkage target.

- A callable ``f(d, pi, y)`` that computes the prior scales from the canonical
  coordinate data variances :math:`d_i`, and prior variances :math:`\\pi_i`
  (both provided as a vector with shape ``(p_eff,)``; and the canonical data
  :math:`\\vec y`. It may return either a singular scalar (a single scale
  shared by every observation, e.g. when the scale is inferred only from
  :math:`d_i`  and :math:`\\pi_i` and not the data) or a real-valued,
  non-negative array whose shape equals ``y.shape[:-1]`` (one prior scale per
  observation).  Any error from the callable is reported as a
  :class:`TypeError`, and a shape mismatch or negative return as a
  :class:`ValueError`.

Some estimators do not support per-observation empirical :math:`\\gamma` and
only accept a single non-negative ``float``.

The loss matrix :math:`Q` may be positive *semi*-definite.  Its null space
carries no loss, so it is treated as an additional set of no-shrink directions,
exactly like the columns of ``dirs``. The null space of :math:`Q` is added to
the no-shrink subspace :math:`\\operatorname{span}(\\mathcal D)` and the
estimator acts only on the covariance-metric complement of
:math:`\\operatorname{span}(\\mathcal D) + \\operatorname{null}(Q)`, where the
restricted loss is positive definite.

Concretely, the covariance-metric projection of the estimate onto
:math:`\\operatorname{span}(\\mathcal D) + \\operatorname{null}(Q)` equals that
of the observed data (kept at its data value), and shrinkage is applied in the
complement.  Because only the spanned space matters, whether a given direction
comes from ``dirs`` or from :math:`\\operatorname{null}(Q)` is irrelevant.
Specifying directions already lying in :math:`\\operatorname{null}(Q)` via
``dirs`` has no additional effect.  The risk on the full problem is unaffected
by how the estimate is set in :math:`\\operatorname{null}(Q)`: the loss is
blind to that component, so any choice there carries the same risk.

References
----------

.. [Tan2015] Z. Tan, "Improved minimax estimation of a multivariate normal mean
    under heteroscedasticity," Bernoulli 21(1), 574-603 (2015),
    https://arxiv.org/abs/1505.07607

.. [Tan2016] Z. Tan, "Steinized empirical Bayes estimation of heteroscedastic
    data," Statistica Sinica 26(3), 1219-1248 (2016),
    https://doi.org/10.5705/ss.202014.0069

.. [Cohen1966] A. Cohen, "All Admissible Linear Estimates of the Mean Vector,"
    The Annals of Mathematical Statistics 37(2), 458-463 (1966),
    https://doi.org/10.1214/aoms/1177699528

.. [Eldar2006] Y. C. Eldar, "Comparing Between Estimation Approaches:
    Admissible and Dominating Linear Estimators," IEEE Transactions on Signal
    Processing 54(5), 1689-1703 (2006),
    https://doi.org/10.1109/tsp.2006.870559

"""

from __future__ import annotations

from ._bayes import bayes, robust_bayes
from ._coordinate import minimax_bayes, tan, tan_bayes
from ._dispatch import shrink
from ._linear import matmul
from ._minimax import berger
from ._risk import estimate_risk, estimate_risk_curve

__all__ = [
    "bayes",
    "berger",
    "estimate_risk",
    "estimate_risk_curve",
    "matmul",
    "minimax_bayes",
    "robust_bayes",
    "shrink",
    "tan",
    "tan_bayes",
]
