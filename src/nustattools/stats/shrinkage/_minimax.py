"""Berger's minimax shrinkage estimator for a multivariate normal mean.

This private module implements the minimax shrinkage estimator
:func:`berger` and its canonical-form implementation
:func:`_berger_canonical`.  The estimator only ever needs to shrink a vector
towards zero with independent coordinates of varying variance (the canonical
form); the shared machinery in
:mod:`nustattools.stats.shrinkage._core` validates and canonicalizes the
general problem.

"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._core import _check_strength, _estimate


def _berger_canonical(
    y: NDArray[Any],
    d: NDArray[Any],
    positive: bool,
    strength: float = 1.0,
) -> NDArray[Any]:
    """Berger's minimax estimator in canonical form.

    Implements [Tan2015]_, Equation (6): with :math:`s = y^T D^{-2} y` and
    :math:`c = \\mathrm{strength}\\,(p_{\\mathrm{eff}} - 2)`,

    .. math:: \\delta_j = \\left(1 - \\frac{c}{d_j S}\\right)_+ y_j.

    ``y`` has shape ``(..., p)`` with coordinate variances ``d`` of shape
    ``(p,)``; the estimator is minimax for
    :math:`0 \\le \\mathrm{strength} \\le 2`.  Berger's estimator involves no
    prior.

    """

    p_eff = len(d)
    c = strength * (p_eff - 2)
    if c <= 0:
        return y
    dinv = 1.0 / d
    s = np.sum(y**2 * dinv**2, axis=-1)
    factor = 1.0 - c * dinv / s[..., None]
    if positive:
        factor = np.maximum(factor, 0.0)
    return cast(NDArray[Any], factor * y)


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

    In canonical coordinates the estimator reads
    :math:`\\delta_j = (1 - c/(d_j s))_+\\, y_j`, where
    :math:`s = \\sum_j y_j^2 / d_j^2` and
    :math:`c = \\mathrm{strength}\\,(p_{\\mathrm{eff}} - 2)`, with
    :math:`p_{\\mathrm{eff}}` the effective number of dimensions.
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
        Shrinkage strength as a fraction of the optimal value
        :math:`c^* = p_{\\mathrm{eff}} - 2` (where ``p_eff`` is the effective
        dimension).  ``strength = 0``
        gives the identity estimator, ``strength = 1`` the optimal minimax
        estimator, and ``strength = 2`` the boundary of the minimax class.
        Must be in ``[0, 2]``.
    offset : array_like, default=None
        A point of shape ``(p,)`` towards which to shrink.  Defaults to zero,
        i.e. shrinking towards the origin.
    dirs : array_like, default=None
        A matrix of shape ``(p, k)`` whose columns span the affine direction
        of shrinkage.  See the :mod:`nustattools.stats.shrinkage` module
        docstring for details.

    Returns
    -------
    delta : numpy.ndarray
        The shrinkage estimate of the mean, with the same shape as ``x``.

    Notes
    -----
    The estimator first transforms the problem to *canonical form* (diagonal
    covariance, identity loss), which is lossless, and applies this estimator
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

    _check_strength(strength)

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
