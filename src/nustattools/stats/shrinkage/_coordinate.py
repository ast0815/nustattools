"""Coordinate-wise shrinkage estimators for a multivariate normal mean.

This private module implements the estimators that shrink each canonical
coordinate towards the prior independently: the minimax estimators
:func:`tan` and :func:`minimax_bayes`, and the Bayes-rule estimator
:func:`tan_bayes`, together with their canonical-form implementations
(:func:`_tan_canonical`, :func:`_minimax_bayes_canonical` and
:func:`_tan_bayes_canonical`).  Each estimator only ever needs to shrink a
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
    _check_strength,
    _estimate_coordinate,
    _prior_diagonal,
)


def _tan_canonical(
    y: NDArray[Any],
    d: NDArray[Any],
    *,
    positive: bool,
    strength: float,
    gamma: float,
    pi: NDArray[Any] | None = None,
) -> NDArray[Any]:
    """Tan's improved minimax estimator in canonical form.

    Implements [Tan2015]_, Theorem 2.  The estimator segments the coordinates
    into two groups by Bayes importance: high-importance coordinates are
    shrunk inversely proportional to their variance (Berger direction),
    low-importance coordinates in the direction of the Bayes rule.  It belongs
    to the minimax class :math:`(I - \\lambda A) y` with a nonnegative
    diagonal direction matrix :math:`A = \\operatorname{diag}(a)`, chosen
    independently of the data to approximately minimize the Bayes risk;
    :math:`A^\\dagger` denotes this optimal choice, with
    :math:`A_0^\\dagger` / :math:`A_\\infty^\\dagger` its two extreme limits
    below.

    ``y`` has shape ``(..., p)`` with coordinate variances ``d`` of shape
    ``(p,)``.  The coordinates are ranked by the Bayes importance
    :math:`d_j^* = d_j^2/(d_j + \\gamma \\pi_j)`, where ``pi`` (default all
    ones) is the canonical diagonal of the explicit prior covariance and the
    effective per-coordinate prior variance is :math:`\\gamma \\pi_j`:

    - :math:`\\gamma = 0` (:math:`A_0^\\dagger`): the ranking reduces to
      :math:`d_j`, independent of the prior shape, and the low-importance
      coordinates receive equal shrinkage weight.
    - :math:`0 < \\gamma < \\infty`: the ranking follows :math:`d_j^*` and
      the low-importance coordinates are shrunk in the Bayes-rule direction
      :math:`d_j/(d_j + \\gamma \\pi_j)`.
    - :math:`\\gamma = \\infty` (:math:`A_\\infty^\\dagger`): the ranking
      reduces to :math:`d_j^2/\\pi_j` (the flat-prior ranking :math:`d_j^2`
      for :math:`\\pi_j = 1`) and the low-importance coordinates are shrunk
      proportional to their variance.

    ``strength`` scales the minimax constant :math:`c^*`; the estimator is
    minimax for :math:`0 \\le \\mathrm{strength} \\le 2`.  The coordinates are
    in canonical order: ``d`` is non-increasing (variance decreasing) and
    ``pi`` is its aligned prior diagonal, non-increasing within each block of
    (numerically-)equal ``d``.

    """

    p_eff = len(d)
    if p_eff < 3:
        return y

    # Bayes importance d* = d^2/(d+gamma*pi), weight (d+gamma*pi)/d^2 and the
    # low-importance Bayes-rule direction a = d/(d+gamma*pi) (see Corollary 3).
    # An explicit prior shape pi makes the effective prior variance
    # gamma * pi (per coordinate).  Only the gamma=0 limit is
    # shape-independent (d* -> d); the gamma=inf limit below ranks coordinates
    # by d^2/pi (shape-dependent), reducing to d^2 for the default
    # homoscedastic shape.
    pi = _prior_diagonal(d, pi)
    if gamma == 0.0:
        d_star = d
        weight = 1.0 / d
        low_a = np.ones(p_eff)
    elif not np.isfinite(gamma):
        # gamma -> inf with a fixed prior shape pi (guaranteed > 0): the
        # effective per-coordinate prior variance gamma*pi_j outgrows the
        # coordinate variance, so d*_j -> d_j^2/(gamma*pi_j), weight ->
        # gamma*pi_j/d_j^2 and low_a -> d_j/(gamma*pi_j).  The common gamma
        # factors cancel in the segmentation, a_star and the shrinkage ratio,
        # so they are dropped here: d_star = d^2/pi, weight = pi/d^2,
        # low_a = d/pi.  For the default homoscedastic shape (pi_j = 1) this is
        # the classic A†_inf (d^2, 1/d^2, d).
        d_star = d**2 / pi
        weight = pi / d**2
        low_a = d / pi
    else:
        d_plus_g = d + gamma * pi
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

    # c*(D, A†) = M_nu = (nu-2)^2 / S + sum_{j>nu} d*_j (per Corollary 3: for
    # a diagonal A the low-importance part of tr(DA†) - 2*lambda_max(DA†) is
    # sum_{j>nu} d_j^2/(d_j + gamma*pi_j) = sum_{j>nu} d*_j, since
    # lambda_max(DA†) = (nu-2)/S by the segmentation bound; the gamma=inf
    # branch above makes d*_j = d_j^2/pi_j, the shape-aware limit).
    c_star_val = (nu - 2) ** 2 / S
    if nu < p_eff:
        c_star_val += np.sum(d_star_sorted[nu:])

    # Apply estimator: delta_j = (1 - strength * c* * a*_j / (a*^2 . y^2))_+ * y_j.
    # a_star is indexed by descending-importance sorted position, so y must be
    # reordered to match before applying and then mapped back.
    y_sorted = y[..., order]
    s_val = np.sum(a_star**2 * y_sorted**2, axis=-1)
    c_actual = strength * c_star_val
    factor = 1.0 - c_actual * a_star / s_val[..., None]
    if positive:
        factor = np.maximum(factor, 0.0)
    delta_sorted = cast(NDArray[Any], factor * y_sorted)
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
    prior_cov: ArrayLike | None = None,
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
        Shrinkage strength as a fraction of the optimal value :math:`c^*`.
        Must be in :math:`[0, 2]`.  ``strength = 0`` gives the identity
        estimator, ``strength = 1`` the optimal minimax estimator, and
        ``strength = 2`` the boundary of the minimax class.
    gamma : float, default=0.0
        Non-negative prior scale controlling the Bayes importance segmentation
        [Tan2015]_.  Must be :math:`\\ge 0`.  Only a scalar float is
        supported.
    offset : array_like, default=None
        A point of shape ``(p,)`` towards which to shrink.  Defaults to zero,
        i.e. shrinking towards the origin.
    dirs : array_like, default=None
        A matrix of shape ``(p, k)`` whose columns span the affine direction
        of shrinkage.  See the :mod:`nustattools.stats.shrinkage` module
        docstring for details.
    prior_cov : array_like, default=None
        The prior covariance matrix :math:`\\Gamma`, of shape ``(p, p)``, in
        the same coordinates as ``x``, for the Gaussian prior
        :math:`\\theta \\sim N(\\vec o, \\gamma \\Gamma)`.  Defaults to
        :math:`Q^{-1}` (a homoscedastic prior in canonical coordinates).  See
        the :mod:`nustattools.stats.shrinkage` module docstring for details.

    Returns
    -------
    delta : numpy.ndarray
        The shrinkage estimate of the mean, with the same shape as ``x``.

    Notes
    -----
    This is an implementation of the estimator described in [Tan2015]_, Theorem
    2. The estimator first transforms the problem to *canonical form* (diagonal
    covariance, identity loss), which is lossless. It then segments the
    coordinates into two groups based on Bayes "importance" [Tan2015]_.
    High-importance coordinates are shrunk inversely proportional to their
    variance (like Berger's estimator), while low-importance coordinates are
    shrunk in the direction of the Bayes rule (shrinkage proportional to
    variance).  This yields both minimaxity and effective risk reduction,
    whereas Berger's estimator shrinks low-variance coordinates too
    aggressively and the Bayes rule is generally non-minimax.

    The automatic scaling in the estimator leads to some special cases
    for the prior:

    - Prior proportional to :math:`\\Sigma`: If the prior shape :math:`\\Gamma`
      is proportional to the data covariance, the actual prior scale
      :math:`\\gamma` no longer matters, as it is lost in the automatic
      scaling.

      This is also true in the special case when :math:`\\gamma = 0`, since
      this sets the overall prior to 0, which is always proportional to the
      data covariance.

      This "heteroscedastic" case is denoted with :math:`A^\\dagger_0` in
      [Tan2015]_. To quote:

          Then coordinates with high variances are shrunk inversely in
          proportion to their variances, whereas coordinates with low variances
          are shrunk symmetrically. For :math:`\\Gamma = 0`, the proposed
          method has a purely frequentist interpretation: it seeks to minimize
          the upper bound on the pointwise risk of [the estimator] at
          :math:`\\vec \\theta = 0`.

    - A very diffuse, homoscedastic prior: For a homoscedastic prior shape
      in the canonical coordinates, the estimator approaches a limit as
      :math:`\\gamma \\to \\infty`. This case is denoted by
      :math:`A^\\dagger_\\infty`.

          Then coordinates with low (or high) variances are shrunk directly (or
          inversely) in proportion to their variances

    - Homoscedastic prior and data covariance: If the data covariance is
      homoscedastic, i.e. if :math:`\\Sigma \\propto Q^{-1}`, and the prior
      is proportional to it :math:`\\Gamma \\propto \\Sigma`, the ``tan``
      estimator reduces to the James-Stein estimator in the canonical space,
      regardless of the scales of the covariances.

    Examples
    --------

    >>> import numpy as np
    >>> import nustattools.stats.shrinkage as sh
    >>> rng = np.random.default_rng(0)
    >>> x = rng.normal(size=5)
    >>> sh.tan(x).shape
    (5,)

    """

    _check_strength(strength)
    if isinstance(cast(Any, gamma), str) or callable(gamma) or np.ndim(gamma) > 0:
        msg = (
            "gamma must be a scalar float for this estimator; "
            "callable, string, and per-observation array gamma are not supported."
        )
        raise TypeError(msg)
    _check_gamma_nonnegative(gamma)

    return _estimate_coordinate(
        x=x,
        cov=cov,
        q=Q,
        canonical=_tan_canonical,
        positive=positive,
        gamma=gamma,
        strength=strength,
        offset=offset,
        dirs=dirs,
        prior_cov=prior_cov,
    )


def _minimax_bayes_canonical(
    y: NDArray[Any],
    d: NDArray[Any],
    *,
    positive: bool,
    strength: float,
    gamma: float,
    pi: NDArray[Any] | None = None,
) -> NDArray[Any]:
    """Berger's improved minimax estimator :math:`\\delta^{\\mathrm{MB}}` in
    canonical form.

    Implements [Tan2015]_, Equation (8) (Berger's (1982) estimator, reviewed in
    Tan2015, Section 2) under the prior
    :math:`\\vec \\theta \\sim N(0, \\gamma \\Gamma)`.

    ``y`` has shape ``(..., p)`` with coordinate variances ``d`` of shape
    ``(p,)``.  ``strength`` scales the shrinkage constants
    :math:`(k-2)_+` (with :math:`k` the 1-based importance rank, coordinates
    sorted by decreasing Bayes importance):
    ``strength = 1`` recovers Tan's version and ``strength = 2`` Berger's
    original :math:`2(k-2)_+`; minimaxity holds for
    :math:`0 \\le \\mathrm{strength} \\le 2`.
    ``gamma`` is the (finite, non-negative) prior scale.  ``pi`` (default all
    ones) is the canonical diagonal of an explicit prior covariance, making
    the effective per-coordinate prior variance :math:`\\gamma \\pi_j`.  The
    coordinates are in canonical order: ``d`` is non-increasing and ``pi`` is
    its aligned prior diagonal, non-increasing within each block of
    (numerically-)equal ``d``.

    """

    p_eff = len(d)
    if p_eff < 3:
        return y

    # Bayes importance d* = d^2/(d+gamma), Bayes-rule weight w = d/(d+gamma)
    # and the cumulative shrinkage statistic S_k = sum_{l<=k} y_l^2/(d_l+gamma).
    # An explicit prior shape pi makes the effective per-coordinate prior
    # variance gamma * pi (replacing gamma).  For gamma >= 0 these are all
    # well-defined, with gamma=0 the Bhattacharya limit (d* = d, w = 1).  As
    # gamma -> inf, w -> 0 and the estimator reduces to the identity
    # (delta = y), so no separate limit is needed.
    pi = _prior_diagonal(d, pi)
    d_plus_g = d + gamma * pi
    d_star = d**2 / d_plus_g
    weight = d / d_plus_g

    # Sort by decreasing Bayes importance and reorder y to match.
    order = np.argsort(d_star)[::-1]
    d_star_sorted = d_star[order]
    weight_sorted = weight[order]
    y_sorted = y[..., order]

    # S_k = sum_{l<=k} y_l^2 / (d_l + gamma), an increasing cumulative sum.
    s_cum = np.cumsum(y_sorted**2 / d_plus_g[order], axis=-1)

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

    # delta_j = y_j * (1 - w_j * B_j), with the optional positive part.
    factor = 1.0 - weight_sorted * bracket
    if positive:
        factor = np.maximum(factor, 0.0)
    delta_sorted = cast(NDArray[Any], factor * y_sorted)
    delta = np.empty_like(delta_sorted)
    delta[..., order] = delta_sorted
    return delta


def minimax_bayes(
    x: ArrayLike,
    cov: ArrayLike | None = None,
    *,
    Q: ArrayLike | None = None,
    positive: bool = True,
    strength: float = 1.0,
    gamma: float = 1.0,
    offset: ArrayLike | None = None,
    dirs: ArrayLike | None = None,
    prior_cov: ArrayLike | None = None,
) -> NDArray[Any]:
    """Berger's improved minimax shrinkage estimator :math:`\\delta^{\\mathrm{MB}}`.

    This is Berger's (1982) estimator reviewed in [Tan2015]_, Section 2,
    Equation (8), for a multivariate normal mean under a normal prior
    :math:`\\vec \\theta \\sim N(\\vec o, \\gamma \\Gamma)`.  It combines
    the Bayes rule (shrinkage proportional to variance) with a minimax
    shrinkage magnitude that keeps the estimator minimax over the whole
    parameter space.

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
        Shrinkage strength that scales the factor :math:`(k-2)_+` in the paper,
        where k is the coordinate's 1-based rank by decreasing Bayes importance.
        At ``strength = 1`` it yields Tan's version and ``strength = 2``
        Berger's original.  Must be in :math:`[0, 2]`.
    gamma : float, default=1.0
        Non-negative prior scale; see the :mod:`nustattools.stats.shrinkage`
        module docstring for details. This estimator only accepts single
        ``float`` values.
    offset : array_like, default=None
        A point of shape ``(p,)`` towards which to shrink.  Defaults to zero,
        i.e. shrinking towards the origin.
    dirs : array_like, default=None
        A matrix of shape ``(p, k)`` whose columns span the affine direction
        of shrinkage.  See the :mod:`nustattools.stats.shrinkage` module
        docstring for details.
    prior_cov : array_like, default=None
        The prior covariance matrix :math:`\\Gamma`, of shape ``(p, p)``, in
        the same coordinates as ``x``, for the Gaussian prior
        :math:`\\theta \\sim N(\\vec o, \\gamma \\Gamma)`.  Defaults to
        :math:`Q^{-1}` (a homoscedastic prior in canonical coordinates).  See
        the :mod:`nustattools.stats.shrinkage` module docstring for details.

    Returns
    -------
    delta : numpy.ndarray
        The shrinkage estimate of the mean, with the same shape as ``x``.

    Examples
    --------

    >>> import numpy as np
    >>> import nustattools.stats.shrinkage as sh
    >>> rng = np.random.default_rng(0)
    >>> x = rng.normal(size=5)
    >>> sh.minimax_bayes(x).shape
    (5,)

    """

    _check_strength(strength)
    if isinstance(cast(Any, gamma), str) or callable(gamma) or np.ndim(gamma) > 0:
        msg = (
            "gamma must be a scalar float for this estimator; "
            "callable, string, and per-observation array gamma are not supported."
        )
        raise TypeError(msg)
    _check_gamma_nonnegative(gamma)

    return _estimate_coordinate(
        x=x,
        cov=cov,
        q=Q,
        canonical=_minimax_bayes_canonical,
        positive=positive,
        gamma=gamma,
        strength=strength,
        offset=offset,
        dirs=dirs,
        prior_cov=prior_cov,
    )


def _tan_bayes_canonical(
    y: NDArray[Any],
    d: NDArray[Any],
    *,
    positive: bool,
    strength: float,
    gamma: float | NDArray[Any],
    pi: NDArray[Any] | None = None,
) -> NDArray[Any]:
    """Shrinkage estimator :math:`\\delta_{A,c}` in canonical form with the
    Bayes-rule shrinkage direction.

    Implements [Tan2015]_, Section 3, Equation (9): the estimator
    :math:`\\delta_{A,c} = (I - c A / (y^T A^2 y)) y` where
    :math:`A = \\operatorname{diag}(a)` is the Bayes-rule shrinkage direction
    with :math:`a_j = d_j/(d_j + \\gamma \\pi_j)`.

    ``y`` has shape ``(..., p)`` with coordinate variances ``d`` of shape
    ``(p,)``.  ``strength`` scales the minimax constant
    :math:`c^*(D, A) = \\operatorname{tr}(D A) - 2 \\lambda_{\\max}(D A)`:
    the estimator is minimax for
    :math:`0 \\le \\mathrm{strength} \\le 2`.  ``gamma`` is the prior scale (a
    scalar is broadcast across observations and ``(...)`` array values give
    one scale per observation); ``pi`` (default all ones) is the canonical
    diagonal of an explicit prior covariance, making the effective
    per-coordinate prior variance :math:`\\gamma \\pi_j`.  The coordinates are
    in canonical order: ``d`` is non-increasing and ``pi`` is its aligned
    prior diagonal, non-increasing within each block of (numerically-)equal
    ``d``.

    The Bayes-rule direction :math:`a_j` is proportional to variance:
    high-variance coordinates are shrunk more (like Berger's estimator).
    Limits:

    - zero prior scale: :math:`a_j = 1` (:math:`A = I`), reducing to the
      Berger direction with :math:`c^* = \\operatorname{tr}(D) - 2 \\max_j d_j`.
    - infinite prior scale: :math:`a_j \\to d_j/\\pi_j` up to a common scalar.
      The estimator :math:`\\delta_{A,c}` is invariant under a scalar
      rescaling of :math:`A` (the factor cancels between :math:`c^*`,
      :math:`A` and the quadratic form :math:`y^T A^2 y`), so the limit is the
      fixed direction :math:`A \\sim \\operatorname{diag}(d/\\pi)` with
      :math:`c^* = c^*(D, \\operatorname{diag}(d/\\pi))` (shrinkage
      proportional to variance), not the identity; the estimator still
      reduces to the identity when that :math:`c^* \\le 0`.

    """

    p_eff = len(d)
    if p_eff < 3:
        return y

    pi = _prior_diagonal(d, pi)
    g = np.asarray(gamma, dtype=float)[..., None]
    a = d / (d + g * pi)
    # gamma -> inf: a_j = d_j/(d_j + gamma*pi_j) is proportional to d_j/pi_j
    # (the common 1/gamma factor).  Since delta_{A,c} is invariant under a
    # scalar rescaling of A (the factor cancels between c*, A and the quadratic
    # form y^T A^2 y), the limit keeps a_j = d_j/pi_j instead of collapsing to
    # a = 0.
    a = np.where(np.isinf(g), d / pi, a)
    da = d * a
    c_star_val = np.sum(da, axis=-1) - 2.0 * np.max(da, axis=-1)
    bad = c_star_val <= 0.0
    s_val = np.sum(a**2 * y**2, axis=-1)
    c_actual = strength * c_star_val
    with np.errstate(divide="ignore", invalid="ignore"):
        factor = 1.0 - c_actual[..., None] * a / s_val[..., None]
    if positive:
        factor = np.maximum(factor, 0.0)
    return np.where(bad[..., None], y, factor * y)


def tan_bayes(
    x: ArrayLike,
    cov: ArrayLike | None = None,
    *,
    Q: ArrayLike | None = None,
    positive: bool = True,
    strength: float = 1.0,
    gamma: float | str | GammaCallable | NDArray[Any] = 1.0,
    offset: ArrayLike | None = None,
    dirs: ArrayLike | None = None,
    prior_cov: ArrayLike | None = None,
) -> NDArray[Any]:
    """Shrinkage estimator :math:`\\delta_{A,c}` with the Bayes-rule direction.

    Applies [Tan2015]_, Section 3, Equation (9) -- the class of minimax
    estimators :math:`\\delta_{A,c} = (I - c A / (\\vec x^T A^T Q A \\vec x))
    \\vec x` -- with the shrinkage-direction matrix :math:`A` fixed to the
    Bayes rule under the prior :math:`\\theta \\sim N(\\vec o, \\gamma
    \\Gamma)`. In canonical form :math:`A = \\operatorname{diag}(a)` with
    :math:`a_j = d_j/(d_j + \\gamma \\pi_j)`.

    Unlike :func:`tan` (which optimises :math:`A` by approximately minimising
    the Bayes risk among all minimax estimators), this estimator uses the
    Bayes-rule direction directly. The shrinkage magnitude is controlled by
    :math:`c\\, c^*(D, A)` where :math:`c` is the ``strength``, and
    :math:`c^*(D, A) = \\operatorname{Tr}[D A] - 2 \\lambda_{\\max}(D A)`. The
    estimator is minimax for :math:`0 \\le c \\le 2`.

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
        :math:`c^*(D, A) = \\operatorname{tr}(D A) - 2 \\lambda_{\\max}(D A)`.
        ``strength = 0`` gives the
        identity estimator, ``strength = 1`` the optimal minimax value, and
        ``strength = 2`` the boundary of the minimax class.  Values outside
        :math:`[0, 2]` are accepted but the estimator is no longer guaranteed
        minimax.
    gamma : float, str, callable, or numpy.ndarray, default=1.0
        Non-negative prior scale; see the :mod:`nustattools.stats.shrinkage`
        module docstring for the accepted forms.
    offset : array_like, default=None
        A point of shape ``(p,)`` towards which to shrink.  Defaults to zero,
        i.e. shrinking towards the origin.
    dirs : array_like, default=None
        A matrix of shape ``(p, k)`` whose columns span the affine direction
        of shrinkage.  See the :mod:`nustattools.stats.shrinkage` module
        docstring for details.
    prior_cov : array_like, default=None
        The prior covariance matrix :math:`\\Gamma`, of shape ``(p, p)``, in
        the same coordinates as ``x``, for the Gaussian prior
        :math:`\\theta \\sim N(\\vec o, \\gamma \\Gamma)`.  Defaults to
        :math:`Q^{-1}` (a homoscedastic prior in canonical coordinates).  See
        the :mod:`nustattools.stats.shrinkage` module docstring for details.

    Returns
    -------
    delta : numpy.ndarray
        The shrinkage estimate of the mean, with the same shape as ``x``.

    Notes
    -----

    The estimator first transforms the problem to *canonical form* (diagonal
    covariance, identity loss), which is lossless, and applies the direction
    there.  The Bayes-rule direction :math:`a_j = d_j/(d_j + \\gamma \\pi_j)`
    is proportional to variance: high-variance coordinates are shrunk more,
    unlike Berger's estimator (which shrinks inversely proportional to
    variance).  The minimax constant :math:`c^*(D, A) = \\operatorname{tr}(D A)
    - 2 \\lambda_{\\max}(D A)` ensures that the risk never exceeds
    :math:`\\operatorname{tr}(D)` for :math:`0 \\le \\mathrm{strength} \\le
    2`.

    Examples
    --------

    >>> import numpy as np
    >>> import nustattools.stats.shrinkage as sh
    >>> rng = np.random.default_rng(0)
    >>> x = rng.normal(size=5)
    >>> sh.tan_bayes(x).shape
    (5,)

    """

    _check_gamma_nonnegative(gamma)

    return _estimate_coordinate(
        x=x,
        cov=cov,
        q=Q,
        canonical=_tan_bayes_canonical,
        positive=positive,
        gamma=gamma,
        strength=strength,
        offset=offset,
        dirs=dirs,
        prior_cov=prior_cov,
    )
