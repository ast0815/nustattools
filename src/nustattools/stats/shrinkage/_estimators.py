"""Individual shrinkage estimators and the ``shrink`` front-end.

This private module defines the concrete minimization estimators
(:func:`berger`, :func:`tan`, :func:`berger_mb` and :func:`tan_bayes`), the
non-minimax Bayes rule estimator (:func:`bayes`), their canonical-form
implementations (:func:`_berger_canonical`, :func:`_tan_canonical`,
:func:`_mb_canonical`, :func:`_tan_bayes_canonical` and
:func:`_bayes_canonical`), the :func:`shrink` front-end that dispatches to a
named estimator, and the ``_METHODS`` registry used by :func:`shrink` and the
risk-estimation helpers.

Each estimator only ever needs to shrink a vector towards zero with independent
coordinates of varying variance (the canonical form); the shared machinery in
:mod:`nustattools.stats.shrinkage._core` validates and canonicalizes the general
problem.

"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._core import _estimate


def _berger_canonical(
    x: NDArray[Any], d: NDArray[Any], positive: bool, strength: float = 1.0
) -> NDArray[Any]:
    """Berger's minimax estimator in canonical form.

    Implements [Tan2015]_, Equations (6), for the canonical problem where the
    covariance is the diagonal matrix ``D = diag(d)`` and the loss is the
    identity: with :math:`S = x^T D^{-2} x`,

    .. math:: \\delta_j = \\left(1 - \\frac{c}{d_j S}\\right)_+ x_j.

    ``x`` has shape ``(..., p)`` with coordinate variances ``d`` of shape
    ``(p,)``.  ``strength`` controls the shrinkage magnitude as a fraction of
    the optimal value ``c = p_eff - 2`` (where ``p_eff = len(d)``); the
    estimator is minimax for ``0 <= strength <= 2``.

    """

    p_eff = len(d)
    c = strength * (p_eff - 2)
    if c <= 0:
        return x
    dinv = 1.0 / d
    s = np.sum(x**2 * dinv**2, axis=-1)
    factor = 1.0 - c * dinv / s[..., None]
    if positive:
        factor = np.maximum(factor, 0.0)
    return cast(NDArray[Any], factor * x)


def berger(
    x: ArrayLike,
    cov: ArrayLike | None = None,
    *,
    Q: ArrayLike | None = None,
    positive: bool = True,
    strength: float = 1.0,
    offset: ArrayLike | None = None,
    dirs: ArrayLike | None = None,
) -> NDArray[Any]:
    """Berger's minimax shrinkage estimator for a multivariate normal mean.

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
    positive : bool, default=True
        Use the positive-part estimator, which dominates the plain one.
    strength : float, default=1.0
        Shrinkage strength as a fraction of the optimal value ``c* = p_eff -
        2`` (where ``p_eff`` is the effective dimension).  ``strength = 0``
        gives the identity estimator, ``strength = 1`` the optimal minimax
        estimator, and ``strength = 2`` the boundary of the minimax class.
        Must be in ``[0, 2]``.
    offset : array_like, default=None
        A point of shape ``(p,)`` towards which to shrink.  Defaults to zero,
        i.e. shrinking towards the origin.
    dirs : array_like, default=None
        A matrix of shape ``(p, k)`` whose columns span the affine direction
        of shrinkage.  If given, the estimate shrinks towards the affine
        subspace ``offset + span(dirs)``: the component in the subspace is kept
        and the residual ``(I - P) (x - offset)`` (with ``P`` the
        covariance-metric projector) is shrunk towards zero in the complement.
        If ``None``, the estimate shrinks towards the single point ``offset``.
        When ``Q`` is singular, ``null(Q)`` is added to the no-shrink subspace;
        see the :mod:`nustattools.stats.shrinkage` module docstring for the
        details.

    Returns
    -------
    delta : numpy.ndarray
        The shrinkage estimate of the mean, with the same shape as ``x``.

    Notes
    -----
    The estimator first transforms the problem to *canonical form* (diagonal
    covariance, identity loss), which is lossless, and applies this direction
    there.  Because it shrinks inversely proportional to variance, coordinates
    with small variances are shrunk more strongly.  See [Tan2015]_, Section 2.

    Examples
    --------

    >>> import numpy as np
    >>> import nustattools.stats.shrinkage as sh
    >>> rng = np.random.default_rng(0)
    >>> x = rng.normal(size=5)
    >>> sh.berger(x).shape
    (5,)

    """

    if strength < 0 or strength > 2:
        msg = "strength must be in [0, 2]."
        raise ValueError(msg)

    return _estimate(
        x,
        cov,
        Q,
        _berger_canonical,
        strength=strength,
        positive=positive,
        offset=offset,
        dirs=dirs,
    )


def _tan_canonical(
    x: NDArray[Any],
    d: NDArray[Any],
    *,
    positive: bool,
    strength: float,
    gamma: float,
) -> NDArray[Any]:
    """Tan's improved minimax estimator in canonical form.

    Implements [Tan2015]_, Theorem 2.  The estimator automatically segments
    coordinates into two groups based on Bayes "importance":

    - **High-importance** coordinates are shrunk inversely proportional to
      their variance (Berger direction).
    - **Low-importance** coordinates are shrunk in the direction of the Bayes
      rule.

    ``x`` has shape ``(..., p)`` with coordinate variances ``d`` of shape
    ``(p,)``.  ``strength`` scales the shrinkage constant ``c*``: the
    estimator is minimax for ``0 <= strength <= 2``.

    The estimator belongs to the minimax class ``(I - lambda A) x``, where
    ``A = diag(a_1, ..., a_p)`` is a nonnegative diagonal *shrinkage-direction*
    matrix chosen, independently of the data, to approximately minimize the
    Bayes risk (see [Tan2015]_, Theorem 2); ``A†`` denotes this optimal choice,
    and ``A†_0`` / ``A†_∞`` its two extreme limits below.

    ``gamma`` selects the prior :math:`\\theta \\sim N(0, \\gamma I)` in the
    Bayes-risk criterion of [Tan2015]_, Section 3.3, i.e. a homoscedastic prior
    in the *canonical* coordinate space with scale ``gamma``.  The prior enters
    through the Bayes importance :math:`d_j^* = d_j^2/(d_j + \\gamma)`.  Any
    non-negative ``gamma`` is accepted:

    - ``gamma = 0`` (``A†_0``): the limit of a prior proportional to the
      covariance, i.e. :math:`\\Gamma \\propto \\operatorname{diag}(d)_j`
      in canonical form (so :math:`d_j^* \\propto d_j`).  Low-importance
      coordinates receive equal shrinkage weight.
    - ``0 < gamma < inf``: a homoscedastic prior of scale ``gamma`` in the
      canonical coordinates, ranking coordinates by :math:`d_j^2/(d_j + \\gamma)`.
      Low-importance coordinates are shrunk in the Bayes-rule direction
      :math:`d_j/(d_j + \\gamma)`.
    - ``gamma = inf`` (``A†_∞``): the flat homoscedastic prior
      :math:`\\Gamma \\propto I`, uniform in the *canonical* coordinate space,
      not the original variable space (so :math:`d_j^* \\propto d_j^2`).
      Low-importance coordinates are shrunk proportional to their variance.

    As ``gamma`` increases the relative importance ordering of the coordinates
    ranges from ``d_j`` (``gamma = 0``) through :math:`d_j^2/(d_j + \\gamma)`
    to ``d_j^2`` (``gamma = inf``).

    """

    p_eff = len(d)
    if p_eff < 3:
        return x

    # Bayes importance d* = d^2/(d+gamma), weight (d+gamma)/d^2 and the
    # low-importance Bayes-rule direction a = d/(d+gamma) (see Corollary 3).
    if gamma == 0.0:
        d_star = d
        weight = 1.0 / d
        low_a = np.ones(p_eff)
    elif not np.isfinite(gamma):
        d_star = d**2
        weight = 1.0 / d**2
        low_a = d.copy()
    else:
        d_plus_g = d + gamma
        d_star = d**2 / d_plus_g
        weight = d_plus_g / d**2
        low_a = d / d_plus_g

    # Sort by decreasing Bayes importance.
    order = np.argsort(d_star)[::-1]
    d_sorted = d[order]
    d_star_sorted = d_star[order]
    weight_sorted = weight[order]

    # Find segmentation index nu: smallest k (3 <= k <= p-1) such that
    # (k-2) / sum_{j<=k} weight[j] > d*_{k+1}.  If none, nu = p_eff.
    cum_weight = np.cumsum(weight_sorted)
    nu = p_eff
    for k in range(3, p_eff):
        if (k - 2) / cum_weight[k - 1] > d_star_sorted[k]:
            nu = k
            break

    # Compute optimal A† (diagonal elements).  nu >= 3 so S > 0 always.
    S = cum_weight[nu - 1]
    a_star = np.empty(p_eff)
    a_star[:nu] = (nu - 2) / (S * d_sorted[:nu])
    a_star[nu:] = low_a[order[nu:]]

    # c*(D, A†) = M_nu = (nu-2)^2 / S + sum_{j>nu} d_j / (d_j + gamma_j) * d*_j.
    # Low-importance term = sum d*_j (since a†_j * d*_j = d_j/(d_j+gamma_j) *
    # d_j^2/(d_j+gamma_j), which is not 1 in general; the c* term is M_nu with
    # the low-importance part sum_{j>nu} d_j^2/(d_j+gamma_j) = sum_{j>nu} d*_j
    # for a diagonal A, per Corollary 3).  For gamma=inf (a rescaled limit),
    # d*_j = d_j^2 so the term is sum d_j^2.
    c_star_val = (nu - 2) ** 2 / S
    if nu < p_eff:
        if not np.isfinite(gamma):
            c_star_val += np.sum(d_sorted[nu:] ** 2)
        else:
            c_star_val += np.sum(d_star_sorted[nu:])

    # Apply estimator: delta_j = (1 - strength * c* * a*_j / (a*^2 . x^2))_+ * x_j.
    # a_star is indexed by descending-importance sorted position, so x must be
    # reordered to match before applying and then mapped back.
    x_sorted = x[..., order]
    s_val = np.sum(a_star**2 * x_sorted**2, axis=-1)
    c_actual = strength * c_star_val
    factor = 1.0 - c_actual * a_star / s_val[..., None]
    if positive:
        factor = np.maximum(factor, 0.0)
    delta_sorted = cast(NDArray[Any], factor * x_sorted)
    delta = np.empty_like(delta_sorted)
    delta[..., order] = delta_sorted
    return delta


def tan(
    x: ArrayLike,
    cov: ArrayLike | None = None,
    *,
    Q: ArrayLike | None = None,
    positive: bool = True,
    strength: float = 1.0,
    gamma: float = 0.0,
    offset: ArrayLike | None = None,
    dirs: ArrayLike | None = None,
) -> NDArray[Any]:
    """Tan's improved minimax shrinkage estimator for a multivariate normal mean.

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
    positive : bool, default=True
        Use the positive-part estimator, which dominates the plain one.
    strength : float, default=1.0
        Shrinkage strength as a fraction of the optimal value ``c*``.  Must
        be in ``[0, 2]``.  ``strength = 0`` gives the identity estimator,
        ``strength = 1`` the optimal minimax estimator, and ``strength = 2``
        the boundary of the minimax class.
    gamma : float, default=0.0
        Non-negative prior scale controlling the Bayes importance segmentation
        [Tan2015]_.  Must be ``>= 0``.  The estimator first transforms the
        problem to *canonical form*, in which the covariance is the diagonal
        matrix ``D = diag(d)`` and the loss is the identity, so ``d_j`` below
        is the variance of the ``j``-th canonical coordinate (the transformed
        problem has the same risk, so the transform is always lossless).

        The prior is homoscedastic in the canonical space, :math:`\\Gamma
        \\propto \\gamma I`, entering through the Bayes importance
        :math:`d_j^* = d_j^2/(d_j + \\gamma)`:

        - ``gamma = 0`` (``A†_0``): coordinates are ranked by their variance
          ``d_j``.
        - ``gamma = inf`` (``A†_∞``): coordinates are ranked by ``d_j²``.
        - intermediate ``gamma``: coordinates are ranked by
          ``d_j² / (d_j + gamma)``, so the importance ordering ranges
          continuously between ``d_j`` and ``d_j²`` as ``gamma`` grows.

    offset : array_like, default=None
        A point of shape ``(p,)`` towards which to shrink.  Defaults to zero,
        i.e. shrinking towards the origin.
    dirs : array_like, default=None
        A matrix of shape ``(p, k)`` whose columns span the affine direction
        of shrinkage.  If given, the estimate shrinks towards the affine
        subspace ``offset + span(dirs)``: the component in the subspace is kept
        and the residual ``(I - P) (x - offset)`` (with ``P`` the
        covariance-metric projector) is shrunk towards zero in the complement.
        If ``None``, the estimate shrinks towards the single point ``offset``.
        When ``Q`` is singular, ``null(Q)`` is added to the no-shrink subspace;
        see the :mod:`nustattools.stats.shrinkage` module docstring for the
        details.

    Returns
    -------
    delta : numpy.ndarray
        The shrinkage estimate of the mean, with the same shape as ``x``.

    Notes
    -----
    The estimator first transforms the problem to *canonical form* (diagonal
    covariance, identity loss), which is lossless.  It then automatically
    segments the coordinates into two groups based on Bayes "importance"
    [Tan2015]_.  High-importance coordinates are shrunk inversely proportional
    to their variance (like Berger's estimator), while low-importance
    coordinates are shrunk in the direction of the Bayes rule (shrinkage
    proportional to variance).  This yields both minimaxity and effective risk
    reduction, whereas Berger's estimator shrinks low-variance coordinates too
    aggressively and the Bayes rule is generally non-minimax.

    Examples
    --------

    >>> import numpy as np
    >>> import nustattools.stats.shrinkage as sh
    >>> rng = np.random.default_rng(0)
    >>> x = rng.normal(size=5)
    >>> sh.tan(x).shape
    (5,)

    """

    if strength < 0 or strength > 2:
        msg = "strength must be in [0, 2]."
        raise ValueError(msg)
    if gamma < 0:
        msg = "gamma must be non-negative."
        raise ValueError(msg)

    return _estimate(
        x,
        cov,
        Q,
        _tan_canonical,
        strength=strength,
        gamma=gamma,
        positive=positive,
        offset=offset,
        dirs=dirs,
    )


def _mb_canonical(
    x: NDArray[Any],
    d: NDArray[Any],
    *,
    positive: bool,
    strength: float,
    gamma: float,
) -> NDArray[Any]:
    """Berger's improved minimax estimator ``delta^MB`` in canonical form.

    Implements [Tan2015]_, Equation (8) (Berger's (1982) estimator, reviewed in
    Tan2015, Section 2) for the canonical problem where the covariance is the
    diagonal matrix ``D = diag(d)`` and the loss is the identity, under the
    homoscedastic prior :math:`\\theta \\sim N(0, \\gamma I)`.

    ``x`` has shape ``(..., p)`` with coordinate variances ``d`` of shape
    ``(p,)``.  ``strength`` scales the shrinkage constant ``(k - 2)_+``:
    ``strength = 1`` recovers Tan's version and ``strength = 2`` Berger's
    original ``2(k - 2)_+``; minimaxity holds for ``0 <= strength <= 2``.
    ``gamma`` is the (finite, non-negative) prior scale.

    """

    p_eff = len(d)
    if p_eff < 3:
        return x

    # Bayes importance d* = d^2/(d+gamma), Bayes-rule weight w = d/(d+gamma)
    # and the cumulative shrinkage statistic S_k = sum_{l<=k} x_l^2/(d_l+gamma).
    # For gamma >= 0 these are all well-defined, with gamma=0 the Bhattacharya
    # limit (d* = d, w = 1).  As gamma -> inf, w -> 0 and the estimator reduces
    # to the identity (delta = X), so no separate limit is needed.
    d_plus_g = d + gamma
    d_star = d**2 / d_plus_g
    weight = d / d_plus_g

    # Sort by decreasing Bayes importance and reorder x to match.
    order = np.argsort(d_star)[::-1]
    d_star_sorted = d_star[order]
    weight_sorted = weight[order]
    x_sorted = x[..., order]

    # S_k = sum_{l<=k} x_l^2 / (d_l + gamma), an increasing cumulative sum.
    s_cum = np.cumsum(x_sorted**2 / d_plus_g[order], axis=-1)

    # m_k = min{1, strength*(k-2)_+ / S_k}; recall k is 1-based in the paper, so
    # at 0-based index i the term is strength * max(0, i-1).  m_0 = m_1 = 0, so
    # the k=1,2 coordinates contribute nothing, as expected from (k-2)_+.
    c_k = strength * np.maximum(np.arange(p_eff) - 1, 0.0)
    m_k = np.minimum(1.0, c_k / s_cum)

    # t_k = (d*_k - d*_{k+1}) * m_k, with d*_{p+1} = 0.  The bracket in Eq. (8)
    # is B_j = (1/d*_j) * sum_{k>=j} t_k, a reverse cumulative sum.
    d_next = np.concatenate((d_star_sorted[1:], np.zeros(1)))
    t_k = (d_star_sorted - d_next) * m_k
    bracket = np.cumsum(t_k[..., ::-1], axis=-1)[..., ::-1] / d_star_sorted

    # delta_j = x_j * (1 - w_j * B_j), with the optional positive part.
    factor = 1.0 - weight_sorted * bracket
    if positive:
        factor = np.maximum(factor, 0.0)
    delta_sorted = cast(NDArray[Any], factor * x_sorted)
    delta = np.empty_like(delta_sorted)
    delta[..., order] = delta_sorted
    return delta


def berger_mb(
    x: ArrayLike,
    cov: ArrayLike | None = None,
    *,
    Q: ArrayLike | None = None,
    positive: bool = True,
    strength: float = 1.0,
    gamma: float = 0.0,
    offset: ArrayLike | None = None,
    dirs: ArrayLike | None = None,
) -> NDArray[Any]:
    """Berger's improved minimax shrinkage estimator ``delta^MB``.

    This is Berger's (1982) estimator reviewed in [Tan2015]_, Section 2,
    Equation (8), for a multivariate normal mean under a homoscedastic prior
    :math:`\\theta \\sim N(0, \\gamma I)`.  It combines the Bayes rule (shrinkage
    proportional to variance) with a minimax shrinkage magnitude that keeps the
    estimator minimax over the whole parameter space.

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
    positive : bool, default=True
        Use the positive-part estimator, which dominates the plain one.
    strength : float, default=1.0
        Shrinkage strength as a fraction of the critical value.  ``strength = 0``
        gives the identity estimator, ``strength = 1`` is Tan's version (with
        the constant ``(k - 2)_+``) and ``strength = 2`` Berger's original
        version (with ``2(k - 2)_+``).  Must be in ``[0, 2]``.
    gamma : float, default=0.0
        Non-negative prior scale in the homoscedastic prior
        :math:`\\theta \\sim N(0, \\gamma I)` (in the canonical coordinates).
        Must be ``>= 0``.  ``gamma = 0`` corresponds to the limiting
        Bhattacharya estimator; larger ``gamma`` shrinks coordinates more
        strongly in the direction of the Bayes rule.  Because the Bayes weight
        ``d_j/(d_j + gamma)`` vanishes as ``gamma -> inf``, the estimator reduces
        to the identity there, so no infinite-gamma parameter is supported.
    offset : array_like, default=None
        A point of shape ``(p,)`` towards which to shrink.  Defaults to zero,
        i.e. shrinking towards the origin.
    dirs : array_like, default=None
        A matrix of shape ``(p, k)`` whose columns span the affine direction
        of shrinkage.  If given, the estimate shrinks towards the affine
        subspace ``offset + span(dirs)``: the component in the subspace is kept
        and the residual ``(I - P) (x - offset)`` (with ``P`` the
        covariance-metric projector) is shrunk towards zero in the complement.
        If ``None``, the estimate shrinks towards the single point ``offset``.
        When ``Q`` is singular, ``null(Q)`` is added to the no-shrink subspace;
        see the :mod:`nustattools.stats.shrinkage` module docstring for the
        details.

    Returns
    -------
    delta : numpy.ndarray
        The shrinkage estimate of the mean, with the same shape as ``x``.

    Notes
    -----
    The estimator first transforms the problem to *canonical form* (diagonal
    covariance, identity loss), which is lossless, and applies the direction
    there.  Unlike :func:`tan`, which approximates the minimax optimal
    shrinkage direction, this estimator uses Berger's explicit minimax
    magnitude in the direction of the Bayes rule [Tan2015]_, Section 2.

    Examples
    --------

    >>> import numpy as np
    >>> import nustattools.stats.shrinkage as sh
    >>> rng = np.random.default_rng(0)
    >>> x = rng.normal(size=5)
    >>> sh.berger_mb(x).shape
    (5,)

    """

    if strength < 0 or strength > 2:
        msg = "strength must be in [0, 2]."
        raise ValueError(msg)
    if gamma < 0:
        msg = "gamma must be non-negative."
        raise ValueError(msg)

    return _estimate(
        x,
        cov,
        Q,
        _mb_canonical,
        strength=strength,
        gamma=gamma,
        positive=positive,
        offset=offset,
        dirs=dirs,
    )


def _tan_bayes_canonical(
    x: NDArray[Any],
    d: NDArray[Any],
    *,
    positive: bool,
    strength: float,
    gamma: float,
) -> NDArray[Any]:
    """``delta_{A,c}`` in canonical form with the Bayes-rule shrinkage direction.

    Implements [Tan2015]_, Section 3, Equation (9): the estimator
    ``delta_{A,c} = (I - c A / (x^T A^2 x)) x`` where ``A = diag(a)`` is the
    Bayes-rule shrinkage direction with ``a_j = d_j / (d_j + gamma)``.

    ``x`` has shape ``(..., p)`` with coordinate variances ``d`` of shape
    ``(p,)``.  ``strength`` scales the minimax constant
    ``c*(D, A) = tr(DA) - 2 lambda_max(DA)``: the estimator is minimax for
    ``0 <= strength <= 2``.

    The Bayes-rule direction ``a_j = d_j/(d_j + gamma)`` is proportional to
    variance: high-variance coordinates are shrunk more (like Berger's
    estimator), while low-variance coordinates are shrunk less.  As ``gamma``
    varies:

    - ``gamma = 0``: ``a_j = 1`` (A = I), reducing to the Berger direction with
      ``c* = tr(D) - 2 max(d)``.
    - ``gamma = inf``: ``a_j = 0`` (A = 0), no shrinkage (identity estimator).

    """

    p_eff = len(d)
    if p_eff < 3:
        return x

    a = d / (d + gamma)
    da = d * a
    c_star_val = float(np.sum(da) - 2.0 * np.max(da))
    if c_star_val <= 0.0:
        return x

    s_val = np.sum(a**2 * x**2, axis=-1)
    c_actual = strength * c_star_val
    factor = 1.0 - c_actual * a / s_val[..., None]
    if positive:
        factor = np.maximum(factor, 0.0)
    return cast(NDArray[Any], factor * x)


def tan_bayes(
    x: ArrayLike,
    cov: ArrayLike | None = None,
    *,
    Q: ArrayLike | None = None,
    positive: bool = True,
    strength: float = 1.0,
    gamma: float = 1.0,
    offset: ArrayLike | None = None,
    dirs: ArrayLike | None = None,
) -> NDArray[Any]:
    """Shrinkage estimator ``delta_{A,c}`` with the Bayes-rule direction.

    Applies [Tan2015]_, Section 3, Equation (9) — the class of minimax
    estimators ``delta_{A,c} = (I - c A / (x^T A^T Q A x)) x`` — with the
    shrinkage-direction matrix ``A`` fixed to the Bayes rule under the
    homoscedastic prior :math:`\\theta \\sim N(0, \\gamma I)` in canonical
    coordinates.  In canonical form (diagonal covariance ``D = diag(d)``,
    identity loss), ``A = diag(a)`` with ``a_j = d_j / (d_j + gamma)``.

    Unlike :func:`tan` (which optimises ``A`` by approximately minimising the
    Bayes risk among all minimax estimators), this estimator uses the
    Bayes-rule direction directly.  The shrinkage magnitude is controlled by
    ``strength * c*(D, A)`` where ``c*(D, A) = tr(DA) - 2 lambda_max(DA)``.
    The estimator is minimax for ``0 <= strength <= 2``.

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
    positive : bool, default=True
        Use the positive-part estimator, which dominates the plain one.
    strength : float, default=1.0
        Shrinkage strength as a fraction of the minimax constant
        ``c*(D, A) = tr(DA) - 2 lambda_max(DA)``.  ``strength = 0`` gives the
        identity estimator, ``strength = 1`` the optimal minimax value, and
        ``strength = 2`` the boundary of the minimax class.  Values outside
        ``[0, 2]`` are accepted but the estimator is no longer guaranteed
        minimax.
    gamma : float, default=1.0
        Non-negative prior scale in the homoscedastic prior
        :math:`\\theta \\sim N(0, \\gamma I)` (in the canonical coordinates).
        Must be ``>= 0``.  Controls the Bayes-rule shrinkage direction
        ``a_j = d_j / (d_j + gamma)``:

        - ``gamma = 0``: ``a_j = 1`` (A = I), the Berger direction with
          ``c* = tr(D) - 2 max(d)``.
        - ``gamma = inf``: ``a_j = 0`` (A = 0), no shrinkage (identity).
        - intermediate ``gamma``: coordinates with larger variance ``d_j`` are
          shrunk more (proportional to ``d_j/(d_j + gamma)``).

    offset : array_like, default=None
        A point of shape ``(p,)`` towards which to shrink.  Defaults to zero,
        i.e. shrinking towards the origin.
    dirs : array_like, default=None
        A matrix of shape ``(p, k)`` whose columns span the affine direction
        of shrinkage.  If given, the estimate shrinks towards the affine
        subspace ``offset + span(dirs)``: the component in the subspace is kept
        and the residual ``(I - P) (x - offset)`` (with ``P`` the
        covariance-metric projector) is shrunk towards zero in the complement.
        If ``None``, the estimate shrinks towards the single point ``offset``.
        When ``Q`` is singular, ``null(Q)`` is added to the no-shrink subspace;
        see the :mod:`nustattools.stats.shrinkage` module docstring for the
        details.

    Returns
    -------
    delta : numpy.ndarray
        The shrinkage estimate of the mean, with the same shape as ``x``.

    Notes
    -----
    The estimator first transforms the problem to *canonical form* (diagonal
    covariance, identity loss), which is lossless, and applies the direction
    there.  The Bayes-rule direction ``a_j = d_j/(d_j + gamma)`` is
    proportional to variance: high-variance coordinates are shrunk more,
    unlike Berger's estimator (which shrinks inversely proportional to
    variance).  The minimax constant ``c*(D, A) = tr(DA) - 2 lambda_max(DA)``
    ensures that the risk never exceeds ``tr(D)`` for ``0 <= strength <= 2``.

    When ``gamma = 0``, the direction reduces to ``A = I`` and the estimator
    becomes a Berger-type estimator with ``c* = tr(D) - 2 max(d)``.  For
    finite ``gamma``, the direction interpolates between this and the identity
    (no shrinkage) as ``gamma`` increases.

    Examples
    --------

    >>> import numpy as np
    >>> import nustattools.stats.shrinkage as sh
    >>> rng = np.random.default_rng(0)
    >>> x = rng.normal(size=5)
    >>> sh.tan_bayes(x).shape
    (5,)

    """

    if gamma < 0:
        msg = "gamma must be non-negative."
        raise ValueError(msg)

    return _estimate(
        x,
        cov,
        Q,
        _tan_bayes_canonical,
        strength=strength,
        gamma=gamma,
        positive=positive,
        offset=offset,
        dirs=dirs,
    )


def _bayes_canonical(x: NDArray[Any], d: NDArray[Any], *, gamma: float) -> NDArray[Any]:
    """Bayes rule in canonical form under the homoscedastic prior Gamma = gamma I.

    The canonical problem has diagonal covariance ``D = diag(d)`` and identity
    loss.  Under the prior ``theta* ~ N(0, gamma I)`` the posterior mean (Bayes
    rule) is:

    .. math:: \\delta_j = \\frac{\\gamma}{d_j + \\gamma} \\, x_j^*.

    ``x`` has shape ``(..., p)`` with coordinate variances ``d`` of shape
    ``(p,)``.  ``gamma >= 0`` is the prior scale:

    - ``gamma = 0``: degenerate prior (point mass at zero); the estimate is zero.
    - ``gamma = inf``: flat prior; the estimate is the identity ``delta = x``.
    - ``0 < gamma < inf``: proper prior; coordinates with larger variance
      ``d_j`` are shrunk less (the shrinkage factor ``gamma / (d_j + gamma)``
      decreases with ``d_j``).

    """

    if gamma == 0.0:
        return np.zeros_like(x)
    if not np.isfinite(gamma):
        return x
    factor = gamma / (d + gamma)
    return cast(NDArray[Any], factor * x)


def bayes(
    x: ArrayLike,
    cov: ArrayLike | None = None,
    *,
    Q: ArrayLike | None = None,
    gamma: float = 1.0,
    offset: ArrayLike | None = None,
    dirs: ArrayLike | None = None,
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
    gamma : float, default=1.0
        Non-negative prior scale in the homoscedastic prior
        :math:`\\theta \\sim N(0, \\gamma I)` (in the canonical coordinates).
        Must be ``>= 0``.  ``gamma = 0`` gives the degenerate estimate
        ``delta = 0``; ``gamma = inf`` gives the identity estimate
        ``delta = x``; intermediate values interpolate between the two.
    offset : array_like, default=None
        A point of shape ``(p,)`` towards which to shrink.  Defaults to zero,
        i.e. shrinking towards the origin.
    dirs : array_like, default=None
        A matrix of shape ``(p, k)`` whose columns span the affine direction
        of shrinkage.  If given, the estimate shrinks towards the affine
        subspace ``offset + span(dirs)``: the component in the subspace is kept
        and the residual ``(I - P) (x - offset)`` (with ``P`` the
        covariance-metric projector) is shrunk towards zero in the complement.
        If ``None``, the estimate shrinks towards the single point ``offset``.
        When ``Q`` is singular, ``null(Q)`` is added to the no-shrink subspace;
        see the :mod:`nustattools.stats.shrinkage` module docstring for the
        details.

    Returns
    -------
    delta : numpy.ndarray
        The shrinkage estimate of the mean, with the same shape as ``x``.

    Notes
    -----
    The estimator first transforms the problem to *canonical form* (diagonal
    covariance, identity loss), which is lossless, and applies the Bayes rule
    there.  In canonical coordinates the posterior mean under
    :math:`\\theta^* \\sim N(0, \\gamma I)` is

    .. math:: \\delta_j = \\frac{\\gamma}{d_j + \\gamma} \\, x_j^*,

    where ``d_j`` is the variance of the ``j``-th canonical coordinate.
    Coordinates with larger variance are shrunk less, which is the opposite of
    Berger's minimax estimator (which shrinks inversely proportional to
    variance).

    The Bayes rule is generally *not* minimax: its risk exceeds the minimax
    risk ``trace(Q @ cov)`` when the true mean is far from the prior mean.
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

    if gamma < 0:
        msg = "gamma must be non-negative."
        raise ValueError(msg)

    return _estimate(
        x,
        cov,
        Q,
        _bayes_canonical,
        gamma=gamma,
        offset=offset,
        dirs=dirs,
    )


def shrink(
    x: ArrayLike,
    cov: ArrayLike | None = None,
    *,
    Q: ArrayLike | None = None,
    method: str = "berger",
    offset: ArrayLike | None = None,
    dirs: ArrayLike | None = None,
    **kwargs: Any,
) -> NDArray[Any]:
    """Shrink an observed multivariate normal mean towards an affine subspace.

    Convenience front-end that dispatches to a named shrinkage estimator after
    transforming the problem to canonical form (a lossless change of
    coordinates that makes the covariance diagonal and the loss the identity,
    so the estimator only has to shrink independent coordinates of varying
    variance).

    Parameters
    ----------
    x : array_like
        Observed data.  A single vector of shape ``(p,)`` or a stack of
        observations of shape ``(..., p)``.
    cov : array_like, default=None
        The known covariance matrix of ``x``, of shape ``(p, p)``.  Defaults
        to the identity matrix.
    Q : array_like, default=None
        The known loss matrix, of shape ``(p, p)``.  May be positive
        semi-definite; see the :mod:`nustattools.stats.shrinkage` module
        docstring for how the loss-free null space is handled.  Defaults to the
        identity.
    method : str, default="berger"
        Which estimator to use.  Available: ``"berger"``, ``"tan"``,
        ``"berger_mb"``, ``"tan_bayes"`` and ``"bayes"``.
    offset : array_like, default=None
        A point of shape ``(p,)`` towards which to shrink.  Defaults to zero.
    dirs : array_like, default=None
        A matrix of shape ``(p, k)`` whose columns span the affine direction
        of shrinkage.  If given, the estimate shrinks towards the affine
        subspace ``offset + span(dirs)``.  When ``Q`` is singular, ``null(Q)``
        is added to the no-shrink subspace; see the
        :mod:`nustattools.stats.shrinkage` module docstring for the details.
    **kwargs
        Additional keyword arguments passed to the estimator.

    Returns
    -------
    delta : numpy.ndarray
        The shrinkage estimate of the mean, with the same shape as ``x``.

    """

    return _resolve_method(method)(x, cov, Q=Q, offset=offset, dirs=dirs, **kwargs)


_METHODS: dict[str, Callable[..., NDArray[Any]]] = {
    "berger": berger,
    "tan": tan,
    "berger_mb": berger_mb,
    "tan_bayes": tan_bayes,
    "bayes": bayes,
}


def _resolve_method(name: str) -> Callable[..., NDArray[Any]]:
    """Look up a shrinkage estimator by name.

    Raises ``ValueError`` if *name* is not a registered method.

    """

    try:
        return _METHODS[name]
    except KeyError as e:
        msg = f"Unknown shrinkage method '{name}'."
        raise ValueError(msg) from e
