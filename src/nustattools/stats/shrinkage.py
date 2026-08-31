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
``offset + span(dirs)``, where ``dirs`` is a matrix whose columns span the
affine direction.  Following [Tan2016]_, Section 3.3, the projection onto this
subspace is built in the *covariance* (precision) metric: for a matrix
``V = dirs_star`` in canonical coordinates, the projector is
``P = V (V^T D^{-1} V)^{-1} V^T D^{-1}``.  This makes the fitted component
``P y`` and the residual ``(I - P) y`` statistically uncorrelated, so their
risks add and each can be improved independently.  The component in the
subspace is kept and the residual is shrunk towards zero; because the residual
basis ``l2`` is taken orthonormal (eigenvectors of the symmetric residual
covariance), the change of coordinates is an isometry of the squared-error
loss, so the reduced shrinkage conserves the full loss exactly.

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


def _validate_dirs(dirs: ArrayLike, p: int) -> NDArray[Any]:
    """Validate spanning vectors and return them as an array of shape ``(p, k)``.

    ``dirs`` must have shape ``(p, k)`` with ``k >= 1`` and full column rank
    (its columns span the affine direction of shrinkage, i.e. they form a
    basis of the subspace).  Returns the matrix of shape ``(p, k)``.

    """

    dv = np.asarray(dirs, dtype=float)
    if dv.ndim != 2 or dv.shape[0] != p or dv.shape[1] < 1:
        msg = f"dirs must have shape ({p}, k) with k >= 1, got shape {dv.shape}."
        raise ValueError(msg)
    if not np.all(np.isfinite(dv)):
        msg = "dirs must contain only finite values."
        raise ValueError(msg)
    rank = np.linalg.matrix_rank(dv)
    if rank < dv.shape[1]:
        msg = (
            "dirs must have linearly independent columns (full column rank), "
            f"got rank {rank} for {dv.shape[1]} columns."
        )
        raise ValueError(msg)
    return dv


def _dirs_projection(v: NDArray[Any], d: NDArray[Any]) -> NDArray[Any]:
    """Covariance-metric projector onto the column space of ``v``.

    In canonical coordinates (diagonal covariance ``D = diag(d)``, identity
    loss), returns ``P = V (V^T D^{-1} V)^{-1} V^T D^{-1}``, the projection
    orthogonal in the precision metric ``D^{-1}``.  Such a projection makes
    ``P y`` and ``(I - P) y`` uncorrelated for ``y ~ (I, D)``, so their risks
    separate (see [Tan2016]_, Section 3.3).

    """

    dinv: NDArray[Any] = np.diag(1.0 / d)
    g: NDArray[Any] = v.T @ dinv @ v
    inv_g: NDArray[Any] = np.linalg.inv(g)
    return cast(NDArray[Any], v @ inv_g @ v.T @ dinv)


def _subspace_reduce(
    y: NDArray[Any], d: NDArray[Any], v: NDArray[Any]
) -> tuple[NDArray[Any], NDArray[Any], NDArray[Any], NDArray[Any]]:
    """Decompose ``y`` relative to the affine direction spanned by ``v``.

    Returns ``(kept, eta, d_perp, l2)``.  ``kept = P y`` is the component lying
    in the direction (kept unshrunk), with ``P`` the covariance-metric
    projector of :func:`_dirs_projection`.  The residual ``(I - P) y`` lives in
    the complement ``S_perp``; ``l2`` is an orthonormal basis of ``S_perp`` in
    which the residual covariance is diagonal, ``eta`` the coordinates of the
    residual in that basis and ``d_perp`` the reduced (diagonal) variances.
    This reduces the effective dimension of the shrinkage problem from ``len(d)``
    to ``len(d_perp)``.

    Since ``l2`` holds the orthonormal eigenvectors of the symmetric residual
    covariance ``(I - P) D (I - P)^T``, ``l2^T l2 = I``: the change of basis is
    an isometry of the squared-error loss, so shrinking ``eta`` conserves the
    full-dimensional loss exactly.

    """

    p = _dirs_projection(v, d)
    p_perp = np.eye(d.shape[0]) - p
    # Residual covariance (I - P) D (I - P)^T; its range is S_perp.  Its
    # positive eigenvalues give the reduced variances and its (orthonormal)
    # eigenvectors a basis that diagonalizes the residual.
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
    dirs: ArrayLike | None = None,
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
    ``dirs`` (a matrix whose columns span the affine direction) is given,
    towards the affine subspace ``offset + span(dirs)``.  In the latter case the
    projection is built in the covariance (precision) metric so the fitted and
    residual components are uncorrelated, and the residual
    ``(I - P) (x - offset)`` is shrunk in the complement.  The residual problem
    is itself a canonical normal problem (diagonal covariance ``d_perp``,
    identity loss) and is solved by recursing into ``_estimate`` with no
    subspace or offset, so the effective dimension becomes ``len(d_perp)``.

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
    if dirs is None:
        delta_star = canonical_estimator(y, d, **kwargs) + offset_star
    else:
        v = b @ _validate_dirs(dirs, p)
        kept, eta, d_perp, l2 = _subspace_reduce(y, d, v)
        # The residual (eta, diag(d_perp), identity loss) is itself a canonical
        # normal problem with no subspace and no offset; solve it recursively.
        reduced = _estimate(
            eta, np.diag(d_perp), np.eye(d_perp.shape[0]), canonical_estimator, **kwargs
        )
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
    c : float, default=None
        The shrinkage constant.  Defaults to ``p_eff - 2``, where ``p_eff`` is
        the effective dimension (``p`` or, with ``dirs``, the dimension of the
        orthogonal complement).  The estimator is minimax for
        ``0 <= c <= 2 (p_eff - 2)``.
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
        Which estimator to use.  Currently only ``"berger"`` is available.
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
}


__all__ = ["berger", "shrink"]
