"""Shrinkage estimators for a multivariate normal mean.

This module implements shrinkage estimators for the problem of estimating the
mean ``theta`` of :math:`X \\sim N(\\theta, \\Sigma)` under the loss
:math:`(\\delta - \\theta)^\\mathrm{T} Q (\\delta - \\theta)`.

Following [Tan2015]_, the general problem can always be transformed into the
canonical form where ``Sigma`` is diagonal, ``Q`` is the identity matrix and
the loss reduces to the sum of squared errors.  The public estimators accept a
general covariance matrix ``cov`` and loss matrix ``Q``, canonicalize the
problem internally, and transform the result back.  This keeps the
per-estimator implementations simple: they only ever need to shrink a vector
towards zero with independent coordinates of varying variance.

References
----------

.. [Tan2015] Z. Tan, "Improved minimax estimation of a multivariate normal mean
    under heteroscedasticity," Bernoulli 21(1), 574-603 (2015),
    https://arxiv.org/abs/1505.07607

"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _canonicalize(
    cov: NDArray[Any], q: NDArray[Any]
) -> tuple[NDArray[Any], NDArray[Any], NDArray[Any]]:
    """Transform to canonical form (diagonal covariance, identity loss).

    Returns ``(B, Binv, D)`` such that ``Q = B^T B``, ``B cov B^T = D`` (with
    ``D`` diagonal) and ``Binv = inv(B)``.  Row vectors ``x`` transform to the
    canonical coordinates as ``x_star = x @ B.T`` and back as
    ``x = x_star @ Binv.T``.  See [Tan2015]_, Section 3.2.

    """

    # C with Q = C^T C (upper triangular from the Cholesky factor of Q)
    c = np.linalg.cholesky(q).T
    # Diagonalize C Sigma C^T.  eigh gives o^T (C Sigma C^T) o = diag(d), so
    # the orthogonal O with O (C Sigma C^T) O^T = diag(d) is O = o^T.
    d, o = np.linalg.eigh(c @ cov @ c.T)
    b = o.T @ c
    binv = np.linalg.inv(b)
    return b, binv, d


def _berger_canonical(
    x: NDArray[Any], d: NDArray[Any], c: float, positive: bool
) -> NDArray[Any]:
    """Berger's minimax estimator in canonical form.

    Implements [Tan2015]_, Equations (6): with :math:`S = x^T D^{-2} x`,

    .. math:: \\delta_j = \\left(1 - \\frac{c}{d_j S}\\right)_+ x_j.

    ``x`` has shape ``(..., p)`` with coordinate variances ``d`` of shape
    ``(p,)``.  ``c`` is minimax for ``0 <= c <= 2(p - 2)``.

    """

    dinv = 1.0 / d
    s = np.sum(x**2 * dinv**2, axis=-1)
    factor = 1.0 - c * dinv / s[..., None]
    if positive:
        factor = np.maximum(factor, 0.0)
    return cast(NDArray[Any], factor * x)


def _validate_sympd(a: ArrayLike, shape: tuple[int, int], name: str) -> NDArray[Any]:
    """Validate that ``a`` is symmetric positive definite with the given shape."""

    aa = np.asarray(a)
    if aa.shape != shape:
        msg = f"{name} must have shape {shape}, got {aa.shape}."
        raise ValueError(msg)
    if not np.allclose(aa, aa.T):
        msg = f"{name} must be symmetric."
        raise ValueError(msg)
    try:
        np.linalg.cholesky(aa)
    except np.linalg.LinAlgError as e:
        msg = f"{name} must be positive definite."
        raise ValueError(msg) from e
    return aa


def _validate(
    x: ArrayLike, cov: ArrayLike | None, q: ArrayLike | None
) -> tuple[NDArray[Any], NDArray[Any], NDArray[Any], int]:
    """Validate and resolve the common estimator inputs.

    Returns ``(x, cov, q, p)`` where ``x`` has shape ``(..., p)``, ``cov`` and
    ``q`` are symmetric positive definite ``(p, p)`` matrices, and ``cov`` is
    the identity if it was not given, as is ``q``.

    """

    xa = np.asarray(x)
    if xa.ndim < 1:
        msg = "x must have at least one dimension."
        raise ValueError(msg)
    p = xa.shape[-1]
    if cov is None:
        cova = np.eye(p)
    else:
        cova = _validate_sympd(cov, (p, p), "covariance matrix")
    if q is None:
        qa = np.eye(p)
    else:
        qa = _validate_sympd(q, (p, p), "loss matrix Q")
    return xa, cova, qa, p


def _estimate(
    x: ArrayLike,
    cov: ArrayLike | None,
    q: ArrayLike | None,
    canonical_estimator: Callable[..., NDArray[Any]],
    **kwargs: Any,
) -> NDArray[Any]:
    """Run a canonical-form estimator on the given problem.

    Validates and canonicalizes the common inputs, applies
    ``canonical_estimator`` to the canonical data, and transforms the estimate
    back to the original coordinates.  ``canonical_estimator`` must have the
    signature ``canonical(x_star, d, **kwargs)``, where ``x_star`` has shape
    ``(..., p)`` and ``d`` holds the coordinate variances ``(p,)``; it returns
    the canonical-form estimate with shape ``(..., p)``.

    """

    xa, cova, qa, _ = _validate(x, cov, q)
    b, binv, d = _canonicalize(cova, qa)
    delta_star = canonical_estimator(xa @ b.T, d, **kwargs)
    return cast(NDArray[Any], delta_star @ binv.T)


def berger(
    x: ArrayLike,
    cov: ArrayLike | None = None,
    *,
    Q: ArrayLike | None = None,
    positive: bool = True,
    c: float | None = None,
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
    c : float, default=None
        The shrinkage constant.  Defaults to the optimal value ``c = p - 2``.
        The estimator is minimax for ``0 <= c <= 2 (p - 2)``.

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
    >>> import nustattools.stats as s
    >>> rng = np.random.default_rng(0)
    >>> x = rng.normal(size=5)
    >>> s.berger(x).shape
    (5,)

    """

    xa = np.asarray(x)
    p = xa.shape[-1]

    if c is None:
        c = float(p - 2)
    if c < 0:
        msg = "Shrinkage constant c must be non-negative."
        raise ValueError(msg)

    return _estimate(x, cov, Q, _berger_canonical, c=c, positive=positive)


def shrink(
    x: ArrayLike,
    cov: ArrayLike | None = None,
    *,
    Q: ArrayLike | None = None,
    method: str = "berger",
    **kwargs: Any,
) -> NDArray[Any]:
    """Shrink an observed multivariate normal mean towards zero.

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
        Which estimator to use.  Currently only ``"berger"`` is available.
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
    return estimator(x, cov, Q=Q, **kwargs)


_METHODS: dict[str, Callable[..., NDArray[Any]]] = {
    "berger": berger,
}


__all__ = ["berger", "shrink"]
