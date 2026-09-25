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
    x: NDArray[Any],
    d: NDArray[Any],
    *,
    positive: bool,
    strength: float,
    gamma: float,
    pi: NDArray[Any] | None = None,
) -> NDArray[Any]:
    """Tan's improved minimax estimator in canonical form.

    Implements [Tan2015]_, Theorem 2.  The estimator automatically segments
    coordinates into two groups based on Bayes "importance":

    - **High-importance** coordinates are shrunk inversely proportional to
      their variance (Berger direction).
    - **Low-importance** coordinates are shrunk in the direction of the Bayes
      rule.

    ``x`` has shape ``(..., p)`` with coordinate variances ``d`` of shape
    ``(p,)``.  ``strength`` scales the shrinkage constant :math:`c^*`: the
    estimator is minimax for :math:`0 \\le \\mathrm{strength} \\le 2`.

    The estimator belongs to the minimax class :math:`(I - \\lambda A) x`,
    where :math:`A = \\operatorname{diag}(a_1, \\dots, a_p)` is a nonnegative
    diagonal *shrinkage-direction* matrix chosen, independently of the data, to
    approximately minimize the Bayes risk (see [Tan2015]_, Theorem 2);
    :math:`A^\\dagger` denotes this optimal choice, and
    :math:`A_0^\\dagger` / :math:`A_\\infty^\\dagger` its two extreme limits
    below.

    ``gamma`` selects the prior :math:`\\theta \\sim N(0, \\gamma I)` in the
    Bayes-risk criterion of [Tan2015]_, Section 3.3, i.e. a homoscedastic prior
    in the *canonical* coordinate space with scale ``gamma``.  The prior enters
    through the Bayes importance :math:`d_j^* = d_j^2/(d_j + \\gamma)`.  Any
    non-negative ``gamma`` is accepted:

    - ``gamma = 0`` (:math:`A_0^\\dagger`): the limit of a prior proportional to the
      covariance, i.e. :math:`\\Gamma \\propto \\operatorname{diag}(d)`
      in canonical form (so :math:`d_j^* \\propto d_j`).  Low-importance
      coordinates receive equal shrinkage weight.
    - :math:`0 < \\gamma < \\infty`: a homoscedastic prior of scale ``gamma``
      in the canonical coordinates, ranking coordinates by
      :math:`d_j^2/(d_j + \\gamma)`.
      Low-importance coordinates are shrunk in the Bayes-rule direction
      :math:`d_j/(d_j + \\gamma)`.
    - :math:`\\gamma = \\infty` (:math:`A_\\infty^\\dagger`): the limit of a
      prior whose scale
      :math:`\\Gamma \\propto \\gamma \\Theta` outgrows the coordinate
      variances, ranking coordinates by :math:`d_j^* \\propto d_j^2/\\pi_j`
      (for the default flat prior shape :math:`\\pi_j = 1` this reduces to the
      classic flat-canonical-prior :math:`d_j^2`).  Low-importance coordinates
      are shrunk proportional to their variance.

    ``pi`` (the canonical diagonal of an explicit prior covariance ``Theta``,
    default all ones) gives the base prior shape scaled by ``gamma``: the
    effective per-coordinate prior variance is :math:`\\gamma \\pi_j`,
    replacing ``gamma`` throughout the Bayes-rule direction and importance
    above.  As ``gamma`` increases the relative importance ordering of the
    coordinates ranges from :math:`d_j` (:math:`\\gamma = 0`,
    shape-independent) through :math:`d_j^2/(d_j + \\gamma \\pi_j)` to
    :math:`d_j^2/\\pi_j` (:math:`\\gamma = \\infty`; for the default
    homoscedastic shape :math:`\\pi_j = 1` this is the flat-prior
    :math:`d_j^2` ranking).

    The coordinates are in canonical order: ``d`` is non-increasing (variance
    decreasing) and ``pi`` is its aligned prior diagonal, non-increasing within
    each block of (numerically-)equal ``d``.

    """

    p_eff = len(d)
    if p_eff < 3:
        return x

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
        [Tan2015]_.  Must be :math:`\\ge 0`.  The estimator first
        transforms the problem to *canonical form*, in which the covariance is
        the diagonal matrix :math:`D = \\operatorname{diag}(d)` and the loss is
        the identity, so :math:`d_j` below is the variance of the
        :math:`j`-th canonical coordinate (the
        transformed problem has the same risk, so the transform is always
        lossless).

        The prior is homoscedastic in the canonical space, :math:`\\Gamma
        \\propto \\gamma I`, entering through the Bayes importance
        :math:`d_j^* = d_j^2/(d_j + \\gamma)`:

        - :math:`\\gamma = 0` (:math:`A_0^\\dagger`): coordinates are ranked by
          their variance :math:`d_j`.
        - :math:`\\gamma = \\infty` (:math:`A_\\infty^\\dagger`): coordinates
          are ranked by :math:`d_j^2/\\pi_j` (for the default homoscedastic
          prior shape :math:`\\pi_j = 1`, by :math:`d_j^2`).
        - intermediate :math:`\\gamma`: coordinates are ranked by
          :math:`d_j^2/(d_j + \\gamma \\pi_j)`, so the importance ordering
          ranges continuously between :math:`d_j` and :math:`d_j^2/\\pi_j` as
          ``gamma`` grows.

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
        coordinates, and that diagonal :math:`\\operatorname{diag}(\\pi)`
        replaces the implicit identity of the
        homoscedastic prior: ``gamma`` still scales the prior, now per
        coordinate, through the Bayes importance
        :math:`d_j^* = d_j^2/(d_j + \\gamma \\pi_j)`, so the
        :math:`\\gamma = 0` limit is independent of the shape while the
        :math:`\\gamma = \\infty` limit ranks coordinates by
        :math:`d_j^2/\\pi_j`.  ``Theta`` must
        be symmetric positive definite and must be diagonalizable in the
        canonical coordinates (automatic for ``cov`` proportional to
        :math:`Q^{-1}`); see the :mod:`nustattools.stats.shrinkage` module
        docstring.

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
    x: NDArray[Any],
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
    Tan2015, Section 2) for the canonical problem where the covariance is the
    diagonal matrix :math:`D = \\operatorname{diag}(d)` and the loss is the
    identity, under the
    homoscedastic prior :math:`\\theta \\sim N(0, \\gamma I)`.

    ``x`` has shape ``(..., p)`` with coordinate variances ``d`` of shape
    ``(p,)``.  ``strength`` scales the shrinkage constant :math:`(k-2)_+`:
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
        return x

    # Bayes importance d* = d^2/(d+gamma), Bayes-rule weight w = d/(d+gamma)
    # and the cumulative shrinkage statistic S_k = sum_{l<=k} x_l^2/(d_l+gamma).
    # An explicit prior shape pi makes the effective per-coordinate prior
    # variance gamma * pi (replacing gamma).  For gamma >= 0 these are all
    # well-defined, with gamma=0 the Bhattacharya limit (d* = d, w = 1).  As
    # gamma -> inf, w -> 0 and the estimator reduces to the identity
    # (delta = X), so no separate limit is needed.
    pi = _prior_diagonal(d, pi)
    d_plus_g = d + gamma * pi
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


def minimax_bayes(
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
    """Berger's improved minimax shrinkage estimator :math:`\\delta^{\\mathrm{MB}}`.

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
        Shrinkage strength as a fraction of the critical value :math:`(k-2)_+`.
        ``strength = 0`` gives the identity estimator, ``strength = 1`` is
        Tan's version (with
        the constant :math:`(k-2)_+`) and ``strength = 2`` Berger's original
        version (with :math:`2(k-2)_+`).  Must be in :math:`[0, 2]`.
    gamma : float, default=0.0
        Non-negative prior scale in the homoscedastic prior
        :math:`\\theta \\sim N(0, \\gamma I)` (in the canonical coordinates).
        Must be :math:`\\ge 0`.  :math:`\\gamma = 0` corresponds to the
        limiting Bhattacharya estimator; larger ``gamma`` shrinks coordinates
        more strongly in the direction of the Bayes rule.  Because the Bayes
        weight :math:`d_j/(d_j + \\gamma)` vanishes as :math:`\\gamma \\to
        \\infty`, the
        estimator reduces to the identity there, so no infinite-gamma parameter
        is supported.
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
    there.  Unlike :func:`tan`, which approximates the minimax optimal
    shrinkage direction, this estimator uses Berger's explicit minimax
    magnitude in the direction of the Bayes rule [Tan2015]_, Section 2.

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
    x: NDArray[Any],
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
    :math:`\\delta_{A,c} = (I - c A / (x^T A^2 x)) x` where
    :math:`A = \\operatorname{diag}(a)` is the
    Bayes-rule shrinkage direction with :math:`a_j = d_j/(d_j + \\gamma)`.

    ``x`` has shape ``(..., p)`` with coordinate variances ``d`` of shape
    ``(p,)``.  ``strength`` scales the minimax constant
    :math:`c^*(D, A) = \\operatorname{tr}(D A) - 2 \\lambda_{\\max}(D A)`:
    the estimator is minimax for
    :math:`0 \\le \\mathrm{strength} \\le 2`.  ``gamma`` is the prior scale; a
    scalar is broadcast
    across observations and ``(...,)`` array values give one scale per
    observation (matching the batch dims of ``x``).  ``pi`` (default all ones)
    is the canonical diagonal of an explicit prior covariance, making the
    effective per-coordinate prior variance :math:`\\gamma \\pi_j` and hence
    :math:`a_j = d_j/(d_j + \\gamma \\pi_j)`.  The coordinates are in canonical
    order: ``d`` is non-increasing and ``pi`` is its aligned prior diagonal,
    non-increasing within each block of (numerically-)equal ``d``.

    The Bayes-rule direction :math:`a_j` is proportional to variance:
    high-variance
    coordinates are shrunk more (like Berger's estimator), while low-variance
    coordinates are shrunk less.  As the prior scale varies:

    - zero prior scale: :math:`a_j = 1` (:math:`A = I`), reducing to the Berger
      direction with :math:`c^* = \\operatorname{tr}(D) - 2 \\max_j d_j`.
    - infinite prior scale: :math:`a_j \\to d_j/\\pi_j` up to a common scalar.
      The estimator :math:`\\delta_{A,c}` is invariant under a scalar
      rescaling of :math:`A`
      (the factor cancels between :math:`c^*`, :math:`A` and the quadratic form
      :math:`x^T A^2 x`), so the limit is the fixed direction
      :math:`A \\sim \\operatorname{diag}(d/\\pi)`
      with :math:`c^* = c^*(D, \\operatorname{diag}(d/\\pi))` (shrinkage
      proportional to variance),
      not the identity; the estimator still reduces to the identity when that
      :math:`c^* \\le 0`.

    """

    p_eff = len(d)
    if p_eff < 3:
        return x

    pi = _prior_diagonal(d, pi)
    g = np.asarray(gamma, dtype=float)[..., None]
    a = d / (d + g * pi)
    # gamma -> inf: a_j = d_j/(d_j + gamma*pi_j) is proportional to d_j/pi_j
    # (the common 1/gamma factor).  Since delta_{A,c} is invariant under a
    # scalar rescaling of A (the factor cancels between c*, A and the quadratic
    # form x^T A^2 x), the limit keeps a_j = d_j/pi_j instead of collapsing to
    # a = 0.
    a = np.where(np.isinf(g), d / pi, a)
    da = d * a
    c_star_val = np.sum(da, axis=-1) - 2.0 * np.max(da, axis=-1)
    bad = c_star_val <= 0.0
    s_val = np.sum(a**2 * x**2, axis=-1)
    c_actual = strength * c_star_val
    with np.errstate(divide="ignore", invalid="ignore"):
        factor = 1.0 - c_actual[..., None] * a / s_val[..., None]
    if positive:
        factor = np.maximum(factor, 0.0)
    return np.where(bad[..., None], x, factor * x)


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

    Applies [Tan2015]_, Section 3, Equation (9) — the class of minimax
    estimators :math:`\\delta_{A,c} = (I - c A / (x^T A^T Q A x)) x` — with the
    shrinkage-direction matrix :math:`A` fixed to the Bayes rule under the
    homoscedastic prior :math:`\\theta \\sim N(0, \\gamma I)` in canonical
    coordinates.  In canonical form (diagonal covariance
    :math:`D = \\operatorname{diag}(d)`,
    identity loss), :math:`A = \\operatorname{diag}(a)` with
    :math:`a_j = d_j/(d_j + \\gamma)`.

    Unlike :func:`tan` (which optimises :math:`A` by approximately minimising
    the
    Bayes risk among all minimax estimators), this estimator uses the
    Bayes-rule direction directly.  The shrinkage magnitude is controlled by
    :math:`\\mathrm{strength}\\, c^*(D, A)` where
    :math:`c^*(D, A) = \\operatorname{tr}(D A) - 2 \\lambda_{\\max}(D A)`.
    The estimator is minimax for :math:`0 \\le \\mathrm{strength} \\le 2`.

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
        module docstring for the accepted forms (scalar, ``"empirical"``,
        per-observation array, factory string like ``"max_rel_risk(0.1)"``,
        or callable) and the per-observation shape
        contract.  Controls the Bayes-rule shrinkage direction
        :math:`a_j = d_j/(d_j + \\gamma)`:

        - :math:`\\gamma = 0`: :math:`a_j = 1` (:math:`A = I`), the Berger
          direction with :math:`c^* = \\operatorname{tr}(D) - 2 \\max_j d_j`.
        - :math:`\\gamma = \\infty`: :math:`a_j \\to d_j/\\pi_j` up to a common
          scalar.  Since :math:`\\delta_{A,c}` is invariant under a scalar
          rescaling of :math:`A` (the
          factor cancels between :math:`c^*`, :math:`A` and the quadratic form
          :math:`x^T A^T Q A x`), the limit is the fixed direction
          :math:`A \\sim \\operatorname{diag}(d/\\pi)` with
          :math:`c^* = c^*(D, \\operatorname{diag}(d/\\pi))` (shrinkage
          proportional to variance); the estimator reduces to the identity
          only when that :math:`c^* \\le 0`.
        - intermediate :math:`\\gamma`: coordinates with larger variance
          :math:`d_j` are shrunk more (proportional to
          :math:`d_j/(d_j + \\gamma)`).

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
    covariance, identity loss), which is lossless, and applies the direction
    there.  The Bayes-rule direction :math:`a_j = d_j/(d_j + \\gamma)` is
    proportional to variance: high-variance coordinates are shrunk more,
    unlike Berger's estimator (which shrinks inversely proportional to
    variance).  The minimax constant
    :math:`c^*(D, A) = \\operatorname{tr}(D A) - 2 \\lambda_{\\max}(D A)`
    ensures that the risk never exceeds :math:`\\operatorname{tr}(D)` for
    :math:`0 \\le \\mathrm{strength} \\le 2`.
    See the ``gamma`` parameter above for the :math:`\\gamma = 0` and
    :math:`\\gamma = \\infty` limits of the direction.

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
