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


def _resolve_q(q: ArrayLike | None, p: int) -> NDArray[Any]:
    """Return ``Q`` as a validated positive definite matrix."""

    if q is None:
        return np.eye(p)
    qa = np.asarray(q)
    if qa.shape != (p, p):
        msg = f"Loss matrix Q must have shape {(p, p)}, got {qa.shape}."
        raise ValueError(msg)
    if not np.allclose(qa, qa.T):
        msg = "Loss matrix Q must be symmetric."
        raise ValueError(msg)
    # Validate positive definiteness (also gives a clean error message)
    try:
        np.linalg.cholesky(qa)
    except np.linalg.LinAlgError as e:
        msg = "Loss matrix Q must be positive definite."
        raise ValueError(msg) from e
    return qa


def berger(
    x: ArrayLike,
    cov: ArrayLike,
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
    cov : array_like
        The known covariance matrix of ``x``, of shape ``(p, p)``.  Must be
        symmetric and positive definite.
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
    >>> s.berger(x, cov=np.eye(5)).shape
    (5,)

    """

    xa: NDArray[Any] = np.asarray(x)
    cova: NDArray[Any] = np.asarray(cov)
    p = xa.shape[-1]

    if xa.ndim < 1:
        msg = "x must have at least one dimension."
        raise ValueError(msg)
    if cova.shape != (p, p):
        msg = f"covariance matrix must have shape {(p, p)}, got {cova.shape}."
        raise ValueError(msg)
    if not np.allclose(cova, cova.T):
        msg = "covariance matrix must be symmetric."
        raise ValueError(msg)
    try:
        np.linalg.cholesky(cova)
    except np.linalg.LinAlgError as e:
        msg = "covariance matrix must be positive definite."
        raise ValueError(msg) from e

    qa = _resolve_q(Q, p)

    if c is None:
        c = float(p - 2)
    if c < 0:
        msg = "Shrinkage constant c must be non-negative."
        raise ValueError(msg)

    b, binv, d = _canonicalize(cova, qa)
    x_star = xa @ b.T
    delta_star = _berger_canonical(x_star, d, c, positive)
    return cast(NDArray[Any], delta_star @ binv.T)


def shrink(
    x: ArrayLike,
    cov: ArrayLike,
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
    cov : array_like
        The known covariance matrix of ``x``, of shape ``(p, p)``.
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

    if method == "berger":
        return berger(x, cov, Q=Q, **kwargs)
    msg = f"Unknown shrinkage method '{method}'."
    raise ValueError(msg)


__all__ = ["berger", "shrink"]
