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
    x: NDArray[Any],
    d: NDArray[Any],
    positive: bool,
    strength: float = 1.0,
) -> NDArray[Any]:
    """Berger's minimax estimator in canonical form.

    Implements [Tan2015]_, Equations (6), for the canonical problem where the
    covariance is the diagonal matrix :math:`D = \\operatorname{diag}(d)` and
    the loss is the identity: with :math:`S = x^T D^{-2} x`,

    .. math:: \\delta_j = \\left(1 - \\frac{c}{d_j S}\\right)_+ x_j.

    ``x`` has shape ``(..., p)`` with coordinate variances ``d`` of shape
    ``(p,)``.  ``strength`` controls the shrinkage magnitude as a fraction of
    the optimal value :math:`c = p_{\\mathrm{eff}} - 2` (where ``p_eff`` is
    the effective dimension); the estimator is minimax for
    :math:`0 \\le \\mathrm{strength} \\le 2`.  Berger's estimator does
    not involve a prior.

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
        of shrinkage.  If given, the estimate shrinks towards the affine
        subspace :math:`\\mathrm{offset} + \\operatorname{span}(\\mathrm{dirs})`:
        the component in the subspace is kept and the residual
        :math:`(I - P)(x - \\mathrm{offset})` (with :math:`P` the
        covariance-metric projector) is shrunk towards zero in the complement.
        If ``None``, the estimate shrinks towards the single point ``offset``.
        When ``Q`` is singular, the null space of ``Q`` is added to the
        no-shrink subspace; see the :mod:`nustattools.stats.shrinkage` module
        docstring for the details.

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

    Berger's estimator is purely frequentist: it involves no Gaussian prior and
    has no ``gamma`` or ``prior_cov`` argument, so passing either raises a
    :class:`TypeError`.

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
