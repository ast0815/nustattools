"""Individual shrinkage estimators and the ``shrink`` front-end.

This private module defines the concrete minimization estimators
(:func:`berger` and :func:`tan`), their canonical-form implementations
(:func:`_berger_canonical`, :func:`_tan_canonical`), the :func:`shrink`
front-end that dispatches to a named estimator, and the ``_METHODS`` registry
used by :func:`shrink` and the risk-estimation helpers.

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

    Implements [Tan2015]_, Equations (6): with :math:`S = x^T D^{-2} x`,

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
        The known loss matrix, of shape ``(p, p)``.  Defaults to the identity,
        i.e. squared-error loss.
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

    Returns
    -------
    delta : numpy.ndarray
        The shrinkage estimate of the mean, with the same shape as ``x``.

    Notes
    -----
    Coordinates with small variances are shrunk more strongly.  See
    [Tan2015]_, Section 2.

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

    ``gamma`` selects the prior:

    - ``gamma = 0``: no prior (``A†_0``).  Bayes importance equals the
      variance ``d_j``.  Low-importance coordinates receive equal shrinkage
      weight.
    - ``gamma = inf``: flat homoscedastic prior (``A†_∞``).  Bayes importance
      is proportional to ``d_j²``.  Low-importance coordinates are shrunk
      proportional to their variance.

    """

    p_eff = len(d)
    if p_eff < 3:
        return x

    # Bayes importance d* and per-coordinate weight (d_j + gamma_j)/d_j^2.
    if gamma == 0.0:
        d_star = d
        weight = 1.0 / d
        low_a = np.ones(p_eff)
    else:  # gamma == inf
        d_star = d**2
        weight = 1.0 / d**2
        low_a = d.copy()

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

    # c*(D, A†) = (nu-2)^2 / S + sum_{j>nu} d_j * low_a_j^2 * (d_j + gamma_j) / d_j
    # For gamma=0: low_a=1, term = d_j
    # For gamma=inf (scaled): low_a=d_j, term = d_j^2
    c_star_val = (nu - 2) ** 2 / S
    if nu < p_eff:
        if gamma == 0.0:
            c_star_val += np.sum(d_sorted[nu:])
        else:
            c_star_val += np.sum(d_sorted[nu:] ** 2)

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
        The known loss matrix, of shape ``(p, p)``.  Defaults to the identity,
        i.e. squared-error loss.
    positive : bool, default=True
        Use the positive-part estimator, which dominates the plain one.
    strength : float, default=1.0
        Shrinkage strength as a fraction of the optimal value ``c*``.  Must
        be in ``[0, 2]``.  ``strength = 0`` gives the identity estimator,
        ``strength = 1`` the optimal minimax estimator, and ``strength = 2``
        the boundary of the minimax class.
    gamma : float, default=0.0
        Prior specification controlling the Bayes importance segmentation.
        Must be ``0`` or ``inf``:

        - ``gamma = 0``: no prior (``A†_0``).  Coordinates are ranked by
          their variance ``d_j``.
        - ``gamma = inf``: flat homoscedastic prior (``A†_∞``).  Coordinates
          are ranked by ``d_j²``.

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

    Returns
    -------
    delta : numpy.ndarray
        The shrinkage estimate of the mean, with the same shape as ``x``.

    Notes
    -----
    The estimator automatically segments coordinates into two groups based on
    Bayes "importance" [Tan2015]_.  High-importance coordinates are shrunk
    inversely proportional to their variance (like Berger's estimator), while
    low-importance coordinates are shrunk in the direction of the Bayes rule.
    This yields both minimaxity and effective risk reduction, whereas Berger's
    estimator shrinks low-variance coordinates too aggressively and the Bayes
    rule is generally non-minimax.

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
    if gamma not in (0.0, float("inf")):
        msg = "gamma must be 0 or inf."
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
    transforming the problem to canonical form.

    Parameters
    ----------
    x : array_like
        Observed data.  A single vector of shape ``(p,)`` or a stack of
        observations of shape ``(..., p)``.
    cov : array_like, default=None
        The known covariance matrix of ``x``, of shape ``(p, p)``.  Defaults
        to the identity matrix.
    Q : array_like, default=None
        The known loss matrix, of shape ``(p, p)``.  Defaults to the identity.
    method : str, default="berger"
        Which estimator to use.  Available: ``"berger"`` and ``"tan"``.
    offset : array_like, default=None
        A point of shape ``(p,)`` towards which to shrink.  Defaults to zero.
    dirs : array_like, default=None
        A matrix of shape ``(p, k)`` whose columns span the affine direction
        of shrinkage.  If given, the estimate shrinks towards the affine
        subspace ``offset + span(dirs)``.
    **kwargs
        Additional keyword arguments passed to the estimator.

    Returns
    -------
    delta : numpy.ndarray
        The shrinkage estimate of the mean, with the same shape as ``x``.

    """

    try:
        estimator = _METHODS[method]
    except KeyError as e:
        msg = f"Unknown shrinkage method '{method}'."
        raise ValueError(msg) from e
    return estimator(x, cov, Q=Q, offset=offset, dirs=dirs, **kwargs)


_METHODS: dict[str, Callable[..., NDArray[Any]]] = {
    "berger": berger,
    "tan": tan,
}
