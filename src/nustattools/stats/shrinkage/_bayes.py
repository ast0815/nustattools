"""Bayes-rule shrinkage estimators for a multivariate normal mean.

This private module implements the non-minimax Bayes-rule estimators
(:func:`bayes` and :func:`robust_bayes`) and their canonical-form
implementations (:func:`_bayes_canonical` and
:func:`_robust_bayes_canonical`).  Each estimator only ever needs to shrink a
vector towards zero with independent coordinates of varying variance (the
canonical form); the shared machinery in
:mod:`nustattools.stats.shrinkage._core` validates and canonicalizes the
general problem.

"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._core import (
    GammaCallable,
    _check_gamma_nonnegative,
    _estimate,
    _prior_diagonal,
)


def _robust_bayes_canonical(
    x: NDArray[Any],
    d: NDArray[Any],
    *,
    strength: float,
    gamma: float | NDArray[Any],
    pi: NDArray[Any] | None = None,
) -> NDArray[Any]:
    """The robust generalised Bayes estimator :math:`\\delta^{\\mathrm{RB}}` in
    canonical form.

    Implements [Tan2015]_, Equation (7) (Berger, 1982) for the canonical
    problem where the covariance is the diagonal matrix
    :math:`D = \\operatorname{diag}(d)` and
    the loss is the identity, under the homoscedastic prior
    :math:`\\theta \\sim N(0, \\gamma I)`.

    ``x`` has shape ``(..., p)`` with coordinate variances ``d`` of shape
    ``(p,)``.  ``strength`` scales the shrinkage constant :math:`(k-2)_+`
    (with :math:`k = \\mathrm{len}(d)`): ``strength = 1`` recovers Tan's
    version and ``strength = 2``
    Berger's original :math:`2(k-2)_+`.  ``gamma`` is the (finite,
    non-negative)
    prior scale, either a scalar (shared across observations) or an array
    matching the batch dims of ``x`` (one prior scale per observation).
    ``pi`` (default all ones) is the canonical diagonal of an explicit prior
    covariance, making the effective per-coordinate prior variance
    :math:`\\gamma \\pi_j`.  The coordinates are in canonical order: ``d`` is
    non-increasing and ``pi`` is its aligned prior diagonal, non-increasing
    within each block of (numerically-)equal ``d``.

    The estimator is *not* minimax: it is robust to misspecification of the
    prior but may have greater risk than the identity estimator.  Unlike
    :func:`minimax_bayes` (which uses the same Bayes-rule weight
    :math:`w_j = d_j/(d_j + \\gamma)` but with a coordinate-wise minimax
    magnitude),
    here the shrinkage magnitude is the scalar
    :math:`m = \\min\\{1, \\mathrm{strength}\\,(k-2)_+/S\\}`
    with :math:`S = \\sum_j x_j^2/(d_j + \\gamma)`, applied uniformly:

    .. math::

        \\delta_j = (1 - m \\, w_j) \\, x_j

    Since :math:`m \\le 1` and :math:`w_j \\le 1`, the factor
    :math:`1 - m w_j` is always
    non-negative, so no positive-part truncation is needed.

    """

    if len(d) < 3:
        return x
    p_eff = len(d)
    c_k = strength * (p_eff - 2)
    if c_k <= 0:
        return x

    pi = _prior_diagonal(d, pi)
    g = np.asarray(gamma, dtype=float)[..., None]
    d_plus_g = d + g * pi
    weight = d / d_plus_g
    s_val = np.sum(x**2 / d_plus_g, axis=-1)
    # When s_val = 0 (e.g. x = 0) the ratio is infinite so m = 1; suppress the
    # divide warning since the min() below maps it correctly.
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = c_k / s_val
    m_k = np.minimum(1.0, ratio)
    factor = 1.0 - m_k[..., None] * weight
    return cast(NDArray[Any], factor * x)


def robust_bayes(
    x: ArrayLike,
    cov: ArrayLike | None = None,
    *,
    Q: ArrayLike | None = None,
    strength: float = 1.0,
    gamma: float | str | GammaCallable | NDArray[Any] = 1.0,
    offset: ArrayLike | None = None,
    dirs: ArrayLike | None = None,
    prior_cov: ArrayLike | None = None,
) -> NDArray[Any]:
    """The robust generalised Bayes estimator :math:`\\delta^{\\mathrm{RB}}`.

    This is Berger's (1982) robust generalised Bayes estimator reviewed in
    [Tan2015]_, Section 2, Equation (7), for a multivariate normal mean under
    the homoscedastic prior :math:`\\theta \\sim N(0, \\gamma I)`.  It shrinks
    the coordinates in the direction of the Bayes rule but caps the shrinkage
    magnitude so the shrinkage ratio never exceeds one:

    .. math::

        \\delta = \\left(1 - \\min\\left(1,
            \\frac{\\text{strength}\\, (k-2)_+}{S}\\right) w\\right) \\odot x

    where :math:`w_j = d_j/(d_j + \\gamma)` is the Bayes-rule weight and
    :math:`S = \\sum_j x_j^2/(d_j + \\gamma)` in canonical coordinates.  The
    estimator is *non-minimax* (unlike :func:`minimax_bayes`): it is expected to
    provide significant risk reduction over the identity when the prior is
    well-specified but is robust to misspecification.

    Parameters
    ----------
    x : array_like
        Observed data.  A single vector of shape ``(p,)`` or a stack of
        observations of shape ``(..., p)``.  The estimator is applied to each
        observation over the last axis.
    cov : array_like, default=None
        The known covariance matrix of ``x``, of shape ``(p, p)``.  Must be
        symmetric and positive definite.  Defaults to the identity matrix.
    Q : array_like, default=None
        The known loss matrix, of shape ``(p, p)``.  May be positive
        semi-definite; see the :mod:`nustattools.stats.shrinkage` module
        docstring for how the loss-free null space is handled.  Defaults to the
        identity, i.e. squared-error loss.
    strength : float, default=1.0
        Shrinkage strength as a fraction of the critical value :math:`(k-2)_+`.
        ``strength = 0`` gives the identity estimator, ``strength = 1`` is
        Tan's version (with the constant :math:`(k-2)_+`) and ``strength = 2``
        Berger's original version (with :math:`2(k-2)_+`).  Values outside
        :math:`[0, 2]` are
        accepted but push the estimator further from its recommended operating
        range.
    gamma : float, str, callable, or numpy.ndarray, default=1.0
        Non-negative prior scale; see the :mod:`nustattools.stats.shrinkage`
        module docstring for the accepted forms (scalar, ``"empirical"``,
        per-observation array, factory string like ``"max_rel_risk(0.1)"``,
        or callable) and the per-observation shape
        contract.  :math:`\\gamma = 0` corresponds to the spherically
        symmetric limiting form
        :math:`\\{1 - \\mathrm{strength}\\,(k-2)_+/(X^T D^{-1} X)\\}_+ x`
        while larger ``gamma`` shrinks coordinates more strongly in the
        direction of the Bayes rule.  Because the Bayes weight
        :math:`d_j/(d_j + \\gamma)` vanishes and :math:`S` vanishes as
        :math:`\\gamma \\to \\infty`,
        the estimator reduces to the identity there, so no infinite-gamma
        parameter is supported.
    offset : array_like, default=None
        A point of shape ``(p,)`` towards which to shrink.  Defaults to zero,
        i.e. shrinking towards the origin.
    dirs : array_like, default=None
        A matrix of shape ``(p, k)`` whose columns span the affine direction
        of shrinkage.  If given, the estimate shrinks towards the affine
        subspace :math:`\\mathrm{offset} + \\operatorname{span}(\\mathrm{dirs})`:
        the component in the subspace is kept and the residual
        :math:`(I - P)(x - \\mathrm{offset})` (with :math:`P` the
        covariance-metric projector) is shrunk towards zero in the complement.
        If ``None``, the estimate shrinks towards the single point ``offset``.
        When ``Q`` is singular, the null space of ``Q`` is added to the
        no-shrink subspace; see the :mod:`nustattools.stats.shrinkage` module
        docstring for the details.
    prior_cov : array_like, default=None
        The prior covariance matrix, of shape ``(p, p)``, in the same
        coordinates as ``x`` — the fixed covariance ``Theta`` of a Gaussian
        prior :math:`\\theta \\sim N(0, \\gamma \\Theta)`.  Defaults to
        :math:`Q^{-1}` (the current homoscedastic prior in canonical
        coordinates).
        When given, the canonicalization rotates the canonical frame (where the
        canonical variances allow; always, when ``cov`` is proportional to
        :math:`Q^{-1}`) so that ``Theta`` is diagonal in the canonical
        coordinates,
        and that diagonal ``diag(pi)`` replaces the implicit identity of the
        homoscedastic prior: ``gamma`` still scales the prior, now per
        coordinate, so the Bayes weight becomes
        :math:`d_j/(d_j + \\gamma \\pi_j)`.  ``Theta`` must be symmetric
        positive definite and must be diagonalizable in the canonical
        coordinates
        (automatic for ``cov`` proportional to :math:`Q^{-1}`); see the
        :mod:`nustattools.stats.shrinkage` module docstring.

    Returns
    -------
    delta : numpy.ndarray
        The shrinkage estimate of the mean, with the same shape as ``x``.

    Notes
    -----
    The estimator first transforms the problem to *canonical form* (diagonal
    covariance, identity loss), which is lossless, and applies the direction
    there.  In canonical coordinates the estimator is componentwise
    :math:`\\delta_j = (1 - m w_j) x_j` where :math:`w_j = d_j/(d_j +
    \\gamma)` is the
    Bayes-rule weight and
    :math:`m = \\min\\{1, \\mathrm{strength}\\,(k-2)_+/S\\}` is a scalar
    shrinkage ratio with :math:`S = \\sum_j x_j^2/(d_j + \\gamma)`.  Because
    :math:`m` is capped at one and :math:`w_j` never exceeds one, the
    per-coordinate factor is
    always non-negative, so the estimator needs no positive-part truncation
    (unlike :func:`berger` and :func:`tan`).

    The estimator is generally *not* minimax: its risk can exceed the minimax
    risk ``trace(Q @ cov)`` when the true mean is far from the prior mean.
    However, it is robust to misspecification of the prior and can have
    substantially lower risk than any minimax estimator when the prior is
    well-specified.

    Examples
    --------

    >>> import numpy as np
    >>> import nustattools.stats.shrinkage as sh
    >>> rng = np.random.default_rng(0)
    >>> x = rng.normal(size=5)
    >>> sh.robust_bayes(x).shape
    (5,)

    """

    _check_gamma_nonnegative(gamma)

    return _estimate(
        x=x,
        cov=cov,
        q=Q,
        canonical_estimator=_robust_bayes_canonical,
        strength=strength,
        gamma=gamma,
        offset=offset,
        dirs=dirs,
        prior_cov=prior_cov,
    )


def _bayes_canonical(
    x: NDArray[Any],
    d: NDArray[Any],
    *,
    gamma: float | NDArray[Any],
    pi: NDArray[Any] | None = None,
) -> NDArray[Any]:
    """Bayes rule in canonical form under the prior
    :math:`\\Gamma = \\operatorname{diag}(\\gamma \\pi)`.

    The canonical problem has diagonal covariance
    :math:`D = \\operatorname{diag}(d)` and identity
    loss.  Under the prior :math:`\\theta^* \\sim N\\bigl(0,
    \\operatorname{diag}(\\gamma \\pi)\\bigr)` (with ``pi``,
    the canonical diagonal of an explicit prior covariance, all ones by
    default) the posterior mean (Bayes rule) is:

    .. math:: \\delta_j = \\frac{\\gamma \\pi_j}{d_j + \\gamma \\pi_j} \\, x_j^*.

    ``x`` has shape ``(..., p)`` with coordinate variances ``d`` of shape
    ``(p,)``.  The coordinates are in canonical order: ``d`` is non-increasing
    and ``pi`` is its aligned prior diagonal, non-increasing within each block
    of (numerically-)equal ``d``.  :math:`\\gamma \\ge 0` is the prior scale,
    either a scalar (shared across
    observations) or an array matching the batch dims of ``x`` (one prior scale
    per observation):

    - :math:`\\gamma = 0`: degenerate prior (point mass at zero); the estimate
      is zero.
    - :math:`\\gamma = \\infty`: flat prior; the estimate is the identity
      :math:`\\delta = x`.
    - :math:`0 < \\gamma < \\infty`: proper prior; coordinates with larger
      variance :math:`d_j` are shrunk less (the shrinkage factor
      :math:`\\gamma \\pi_j/(d_j + \\gamma \\pi_j)` decreases with
      :math:`d_j`), and a larger prior variance :math:`\\pi_j` shrinks
      coordinate :math:`j` less.

    """

    pi = _prior_diagonal(d, pi)
    g = np.asarray(gamma, dtype=float)[..., None]
    with np.errstate(divide="ignore", invalid="ignore"):
        tau = np.where(pi == 0.0, 0.0, g * pi)
        factor = np.where(np.isfinite(tau), tau / (d + tau), 1.0)
    return cast(NDArray[Any], factor * x)


def bayes(
    x: ArrayLike,
    cov: ArrayLike | None = None,
    *,
    Q: ArrayLike | None = None,
    gamma: float | str | GammaCallable | NDArray[Any] = 1.0,
    offset: ArrayLike | None = None,
    dirs: ArrayLike | None = None,
    prior_cov: ArrayLike | None = None,
) -> NDArray[Any]:
    """Bayes rule shrinkage estimator for a multivariate normal mean.

    Applies the posterior-mean (Bayes rule) estimator under the homoscedastic
    prior :math:`\\theta \\sim N(0, \\gamma I)` in the canonical coordinate
    space (where the covariance is diagonal and the loss is the identity).  This
    is a *non-minimax* estimator: it does not dominate the identity estimator
    uniformly over the parameter space, but can have substantially lower risk
    when the true mean is close to the prior mean.

    Parameters
    ----------
    x : array_like
        Observed data.  A single vector of shape ``(p,)`` or a stack of
        observations of shape ``(..., p)``.  The estimator is applied to each
        observation over the last axis.
    cov : array_like, default=None
        The known covariance matrix of ``x``, of shape ``(p, p)``.  Must be
        symmetric and positive definite.  Defaults to the identity matrix.
    Q : array_like, default=None
        The known loss matrix, of shape ``(p, p)``.  May be positive
        semi-definite; see the :mod:`nustattools.stats.shrinkage` module
        docstring for how the loss-free null space is handled.  Defaults to the
        identity, i.e. squared-error loss.
    gamma : float, str, callable, or numpy.ndarray, default=1.0
        Non-negative prior scale; see the :mod:`nustattools.stats.shrinkage`
        module docstring for the accepted forms (scalar, ``"empirical"``,
        per-observation array, factory string like ``"max_rel_risk(0.1)"``,
        or callable) and the per-observation shape
        contract.  ``gamma = 0`` gives the degenerate estimate
        ``delta = 0``; ``gamma = inf`` gives the identity estimate
        ``delta = x``; intermediate values interpolate between the two,
        scaling the prior shape (``prior_cov``, or the default identity).
    offset : array_like, default=None
        A point of shape ``(p,)`` towards which to shrink.  Defaults to zero,
        i.e. shrinking towards the origin.
    dirs : array_like, default=None
        A matrix of shape ``(p, k)`` whose columns span the affine direction
        of shrinkage.  If given, the estimate shrinks towards the affine
        subspace :math:`\\mathrm{offset} + \\operatorname{span}(\\mathrm{dirs})`:
        the component in the subspace is kept and the residual
        :math:`(I - P)(x - \\mathrm{offset})` (with :math:`P` the
        covariance-metric projector) is shrunk towards zero in the complement.
        If ``None``, the estimate shrinks towards the single point ``offset``.
        When ``Q`` is singular, the null space of ``Q`` is added to the
        no-shrink subspace; see the :mod:`nustattools.stats.shrinkage` module
        docstring for the details.
    prior_cov : array_like, default=None
        The prior covariance matrix, of shape ``(p, p)``, in the same
        coordinates as ``x`` — the fixed covariance ``Theta`` of a Gaussian
        prior :math:`\\theta \\sim N(0, \\gamma \\Theta)`.  Defaults to
        :math:`Q^{-1}` (the current homoscedastic prior in canonical
        coordinates).
        When given, the canonicalization rotates the canonical frame (where the
        canonical variances allow; always, when ``cov`` is proportional to
        :math:`Q^{-1}`) so that ``Theta`` is diagonal in the canonical
        coordinates,
        and that diagonal :math:`\\operatorname{diag}(\\pi)` replaces the
        implicit identity of the
        homoscedastic prior: ``gamma`` still scales the prior, now per
        coordinate, so the Bayes weight becomes
        :math:`d_j/(d_j + \\gamma \\pi_j)`.  ``Theta`` must be symmetric
        positive definite and must be diagonalizable in the canonical
        coordinates
        (automatic for ``cov`` proportional to :math:`Q^{-1}`); see the
        :mod:`nustattools.stats.shrinkage` module docstring.

    Returns
    -------
    delta : numpy.ndarray
        The shrinkage estimate of the mean, with the same shape as ``x``.

    Notes
    -----
    The estimator first transforms the problem to *canonical form* (diagonal
    covariance, identity loss), which is lossless, and applies the Bayes rule
    there.  In canonical coordinates the posterior mean under
    :math:`\\theta^* \\sim N(0, \\gamma \\operatorname{diag}(\\pi))` is

    .. math:: \\delta_j = \\frac{\\gamma \\pi_j}{d_j + \\gamma \\pi_j} \\, x_j^*,

    where :math:`d_j` is the variance of the :math:`j`-th canonical coordinate
    and :math:`\\pi_j` the :math:`j`-th diagonal of the explicit prior (all
    ones without
    ``prior_cov``).  Coordinates with larger variance are shrunk less, which is
    the opposite of Berger's minimax estimator (which shrinks inversely
    proportional to variance).

    The Bayes rule is generally *not* minimax: its risk exceeds the minimax
    risk :math:`\\operatorname{tr}(Q \\, \\mathrm{cov})` when the true mean is
    far from the prior mean.
    However, when the prior is well-specified (the true mean is near zero), the
    Bayes rule can have substantially lower risk than any minimax estimator.

    Examples
    --------

    >>> import numpy as np
    >>> import nustattools.stats.shrinkage as sh
    >>> rng = np.random.default_rng(0)
    >>> x = rng.normal(size=5)
    >>> sh.bayes(x).shape
    (5,)

    """

    _check_gamma_nonnegative(gamma)

    return _estimate(
        x=x,
        cov=cov,
        q=Q,
        canonical_estimator=_bayes_canonical,
        gamma=gamma,
        offset=offset,
        dirs=dirs,
        prior_cov=prior_cov,
    )
