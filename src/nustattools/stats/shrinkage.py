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

The estimators may shrink towards an arbitrary affine subspace
``offset + P (x - offset)`` where ``P`` is a user-supplied projection matrix
(the projection metric — orthogonal in the original coordinates or in the
whitened space — is encoded in ``P``).  Shrinking towards such a subspace
reduces the effective dimension: the component in the subspace is kept and the
residual is shrunk towards zero in the orthogonal complement, following the
reduction in [Tan2016]_.

References
----------

.. [Tan2015] Z. Tan, "Improved minimax estimation of a multivariate normal mean
    under heteroscedasticity," Bernoulli 21(1), 574-603 (2015),
    https://arxiv.org/abs/1505.07607

.. [Tan2016] Z. Tan, "Steinized empirical Bayes estimation of heteroscedastic
    data," Statistica Sinica 26(3), 1219-1248 (2016),
    https://doi.org/10.5705/ss.202014.0069

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
    x: NDArray[Any], d: NDArray[Any], positive: bool, c: float | None = None
) -> NDArray[Any]:
    """Berger's minimax estimator in canonical form.

    Implements [Tan2015]_, Equations (6): with :math:`S = x^T D^{-2} x`,

    .. math:: \\delta_j = \\left(1 - \\frac{c}{d_j S}\\right)_+ x_j.

    ``x`` has shape ``(..., p)`` with coordinate variances ``d`` of shape
    ``(p,)``.  ``c`` defaults to the optimal value ``p - 2`` where ``p`` is the
    effective dimension ``len(d)`` (which may be smaller than the ambient
    dimension when shrinking towards a subspace); by default the estimator is
    minimax for ``0 <= c <= 2 (p - 2)``.

    """

    if c is None:
        c = float(len(d) - 2)
        if c <= 0:
            # The effective dimension is too small for shrinkage towards zero
            # to be valid; leave the input unchanged (no shrinkage).
            return x
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


def _validate_projection(projection: ArrayLike, p: int) -> NDArray[Any]:
    """Validate a projection matrix and return it as an array of shape ``(p, p)``."""

    pp = np.asarray(projection, dtype=float)
    if pp.shape != (p, p):
        msg = f"projection must have shape {(p, p)}, got {pp.shape}."
        raise ValueError(msg)
    if not np.allclose(pp @ pp, pp):
        msg = "projection must be idempotent (P @ P = P)."
        raise ValueError(msg)
    return pp


def _affine_reduce(
    y: NDArray[Any], d: NDArray[Any], p: NDArray[Any]
) -> tuple[NDArray[Any], NDArray[Any], NDArray[Any], NDArray[Any]]:
    """Decompose ``y`` relative to the projection ``P`` onto an affine direction.

    Returns ``(kept, eta, d_perp, l2)``.  ``kept = P y`` is the component lying
    in the direction (kept unshrunk).  The residual ``(I - P) y`` lives in the
    orthogonal complement ``S_perp`` of the direction; ``l2`` is an
    orthonormal basis of ``S_perp`` in which the residual covariance is
    diagonal, ``eta`` are the coordinates of the residual in that basis and
    ``d_perp`` the corresponding reduced (diagonal) variances.  This reduces
    the effective dimension of the shrinkage problem from ``len(d)`` to
    ``len(d_perp)``.

    """

    p_perp = np.eye(p.shape[0]) - p
    # Residual covariance (I - P) D (I - P)^T; its range is S_perp.  Its
    # positive eigenvalues give the reduced variances and the top eigenvectors
    # an orthonormal basis that diagonalizes the residual.
    m = p_perp @ np.diag(d) @ p_perp.T
    lam, vecs = np.linalg.eigh(m)
    keep = lam > 1e-12
    l2 = vecs[:, keep]
    d_perp = lam[keep]
    kept = y @ p.T
    eta = (y - kept) @ l2
    return kept, eta, d_perp, l2


def _estimate(
    x: ArrayLike,
    cov: ArrayLike | None,
    q: ArrayLike | None,
    canonical_estimator: Callable[..., NDArray[Any]],
    *,
    offset: ArrayLike | None = None,
    projection: ArrayLike | None = None,
    **kwargs: Any,
) -> NDArray[Any]:
    """Run a canonical-form estimator on the given problem.

    Validates and canonicalizes the common inputs, applies
    ``canonical_estimator`` to the canonical data, and transforms the estimate
    back to the original coordinates.  ``canonical_estimator`` must have the
    signature ``canonical(x_star, d, **kwargs)``, where ``x_star`` has shape
    ``(..., p)`` and ``d`` holds the coordinate variances ``(p,)``; it returns
    the canonical-form estimate with shape ``(..., p)``.

    The estimate shrinks towards the point ``offset`` (default zero) or, when
    ``projection`` (an idempotent ``(p, p)`` matrix) is given, towards the
    affine subspace spanned by ``projection`` through ``offset``.  In the
    latter case the residual ``(I - projection) (x - offset)`` is shrunk in the
    orthogonal complement, so ``canonical_estimator`` receives the reduced
    data ``(eta, d_perp)`` whose length is the effective dimension.

    """

    xa, cova, qa, p = _validate(x, cov, q)
    b, binv, d = _canonicalize(cova, qa)
    x_star = xa @ b.T
    if offset is None:
        offset_star = np.zeros(p)
    else:
        o = np.asarray(offset, dtype=float)
        if o.shape != (p,):
            msg = f"offset must have shape {(p,)}, got {o.shape}."
            raise ValueError(msg)
        offset_star = o @ b.T
    y = x_star - offset_star
    if projection is None:
        delta_star = canonical_estimator(y, d, **kwargs) + offset_star
    else:
        proj = _validate_projection(projection, p)
        proj_star = b @ proj @ binv
        kept, eta, d_perp, l2 = _affine_reduce(y, d, proj_star)
        reduced = canonical_estimator(eta, d_perp, **kwargs)
        delta_star = offset_star + kept + reduced @ l2.T
    return cast(NDArray[Any], delta_star @ binv.T)


def berger(
    x: ArrayLike,
    cov: ArrayLike | None = None,
    *,
    Q: ArrayLike | None = None,
    positive: bool = True,
    c: float | None = None,
    offset: ArrayLike | None = None,
    projection: ArrayLike | None = None,
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
        The shrinkage constant.  Defaults to ``p_eff - 2``, where ``p_eff`` is
        the effective dimension (``p`` or, with ``projection``, the dimension
        of the orthogonal complement).  The estimator is minimax for
        ``0 <= c <= 2 (p_eff - 2)``.
    offset : array_like, default=None
        A point of shape ``(p,)`` towards which to shrink.  Defaults to zero,
        i.e. shrinking towards the origin.
    projection : array_like, default=None
        An idempotent ``(p, p)`` matrix ``P`` projecting onto a subspace.  If
        given, the estimate shrinks towards the affine subspace
        ``offset + P (x - offset)``: the component in the direction is kept and
        the residual ``(I - P) (x - offset)`` is shrunk in the orthogonal
        complement.  If ``None``, the estimate shrinks towards the single
        point ``offset``.

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

    if c is not None and c < 0:
        msg = "Shrinkage constant c must be non-negative."
        raise ValueError(msg)

    return _estimate(
        x,
        cov,
        Q,
        _berger_canonical,
        c=c,
        positive=positive,
        offset=offset,
        projection=projection,
    )


def shrink(
    x: ArrayLike,
    cov: ArrayLike | None = None,
    *,
    Q: ArrayLike | None = None,
    method: str = "berger",
    offset: ArrayLike | None = None,
    projection: ArrayLike | None = None,
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
        Which estimator to use.  Currently only ``"berger"`` is available.
    offset : array_like, default=None
        A point of shape ``(p,)`` towards which to shrink.  Defaults to zero.
    projection : array_like, default=None
        An idempotent ``(p, p)`` matrix ``P`` projecting onto a subspace.  If
        given, the estimate shrinks towards the affine subspace
        ``offset + P (x - offset)``.
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
    return estimator(x, cov, Q=Q, offset=offset, projection=projection, **kwargs)


_METHODS: dict[str, Callable[..., NDArray[Any]]] = {
    "berger": berger,
}


__all__ = ["berger", "shrink"]
