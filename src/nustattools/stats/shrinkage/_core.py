"""Shared infrastructure for the canonical-form shrinkage estimators.

This private module contains the machinery common to all shrinkage estimators
in :mod:`nustattools.stats.shrinkage`: validation of the common inputs,
transformation to canonical form (diagonal covariance ``D``, identity loss),
the affine-subspace projector, and the :func:`_estimate` front-end that runs a
canonical-form estimator on a general problem and transforms the result back.

Nothing in this module is public; estimators and the risk-estimation helpers in
the sibling modules build on it.

"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

# np.finfo(float).eps trips a known pylint numpy false positive (E1101).
_EPSILON: float = np.finfo(float).eps  # pylint: disable=no-member


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


def _zero_eigenvalue_tolerance(w: NDArray[Any], p: int) -> float:
    """Numerically-zero eigenvalue threshold for a size-``p`` symmetric matrix.

    An eigenvalue of ``w`` is treated as zero when it lies below this tolerance,
    which scales the machine epsilon by ``p`` so the threshold grows with the
    matrix size.

    """

    maxw = float(np.max(np.abs(w))) if w.size else 0.0
    return p * _EPSILON * max(maxw, 1.0)


def _validate_sympsd(a: ArrayLike, shape: tuple[int, int], name: str) -> NDArray[Any]:
    """Validate that ``a`` is symmetric positive semi-definite with the given shape."""

    aa = np.asarray(a)
    if aa.shape != shape:
        msg = f"{name} must have shape {shape}, got {aa.shape}."
        raise ValueError(msg)
    if not np.allclose(aa, aa.T):
        msg = f"{name} must be symmetric."
        raise ValueError(msg)
    if shape[0] == 0:
        return aa
    w = np.linalg.eigvalsh(aa)
    if np.min(w) < -_zero_eigenvalue_tolerance(w, shape[0]):
        msg = f"{name} must be positive semi-definite (all eigenvalues >= 0)."
        raise ValueError(msg)
    return aa


def _validate(
    x: ArrayLike, cov: ArrayLike | None, q: ArrayLike | None
) -> tuple[NDArray[Any], NDArray[Any], NDArray[Any], int]:
    """Validate and resolve the common estimator inputs.

    Returns ``(x, cov, q, p)`` where ``x`` has shape ``(..., p)``, ``cov`` is a
    symmetric positive definite ``(p, p)`` matrix and ``q`` is a symmetric
    positive semi-definite ``(p, p)`` matrix; ``cov`` is the identity if it was
    not given, as is ``q``.

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
        qa = _validate_sympsd(q, (p, p), "loss matrix Q")
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
    loss), returns ``P = V (V^T D^{-1} V)^{-1} V^T D^{-1}``, where ``V = v``
    is the ``(p, k)`` spanning matrix, the projection orthogonal in the
    precision metric ``D^{-1}``.  Such a projection makes
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


def _merge_dirs(dirs: ArrayLike | None, q: NDArray[Any], p: int) -> NDArray[Any]:
    """Return a basis of ``span(dirs) union null(q)``.

    The union of the user-supplied ``dirs`` columns and the null space of the
    loss matrix ``q`` spans the full set of directions that must not be shrunk:
    the user's directions are kept at their data value, and the null space of
    ``q`` carries no loss, so both are left untouched.  Only the spanned space
    matters, not the particular representative columns, so a (column-)orthonormal
    independent basis of the union is returned.  ``dirs`` must already be
    validated (see :func:`_validate_dirs`); this only widens its span with
    ``null(q)``.

    """

    cols: list[NDArray[Any]] = []
    if dirs is not None:
        cols.append(np.asarray(dirs, dtype=float))
    w, v = np.linalg.eigh(q)
    tol = _zero_eigenvalue_tolerance(w, p)
    cols.append(v[:, w <= tol])
    if len(cols) == 1 and cols[0].shape[1] < 1:
        return np.empty((p, 0))
    stacked = np.concatenate(cols, axis=1)
    u, s, _ = np.linalg.svd(stacked, full_matrices=False)
    keep = s > _zero_eigenvalue_tolerance(s, max(s.size, 1))
    return u[:, keep]


def _estimate_split(
    x: NDArray[Any],
    cov: NDArray[Any],
    q: NDArray[Any],
    canonical_estimator: Callable[..., NDArray[Any]],
    *,
    offset: ArrayLike | None,
    dirs: NDArray[Any],
    **kwargs: Any,
) -> NDArray[Any]:
    """Estimate on a singular-loss problem by splitting before canonicalizing.

    ``q`` is positive semi-definite but singular.  The directions ``dirs`` span
    ``span(user dirs) + null(q)`` (see :func:`_merge_dirs`), so ``q`` is
    strictly positive definite on the covariance-metric complement of
    ``span(dirs)``: any residual there is :math:`\\Sigma^{-1}`-orthogonal to
    ``null(q)`` and hence cannot itself lie in ``null(q)``.  This function
    therefore splits the data into the part lying in ``span(dirs)`` (kept at
    its covariance-metric data value) and the ``Sigma^{-1}``-orthogonal
    residual, then solves the strictly-positive-definite residual problem with
    :func:`_estimate_pd` and recombines.  The inputs must already be validated
    (see :func:`_validate`); ``dirs`` must span the full no-shrink set.

    """

    p = cov.shape[0]
    v_all = np.asarray(dirs, dtype=float)
    cinv = np.linalg.inv(cov)
    g = v_all.T @ cinv @ v_all
    p_mat = v_all @ np.linalg.solve(g, v_all.T @ cinv)
    if offset is None:
        kept_off = np.zeros(p)
        y = x
    else:
        o = np.asarray(offset, dtype=float)
        if o.shape != (p,):
            msg = f"offset must have shape {(p,)}, got {o.shape}."
            raise ValueError(msg)
        kept_off = o @ p_mat.T
        y = x - o
    kept = y @ p_mat.T
    p_perp = np.eye(p) - p_mat
    m = p_perp @ cov @ p_perp.T
    lam, vecs = np.linalg.eigh(m)
    tol = _zero_eigenvalue_tolerance(lam, p)
    keep = lam > tol
    l2 = vecs[:, keep]
    d_perp = lam[keep]
    q_comp = l2.T @ q @ l2
    eta = (y - kept) @ l2
    delta_comp = _estimate_pd(
        eta, np.diag(d_perp), q_comp, canonical_estimator, **kwargs
    )
    return cast(NDArray[Any], kept_off + kept + delta_comp @ l2.T)


def _estimate_pd(
    x: ArrayLike,
    cov: ArrayLike,
    q: ArrayLike,
    canonical_estimator: Callable[..., NDArray[Any]],
    *,
    offset: ArrayLike | None = None,
    dirs: ArrayLike | None = None,
    **kwargs: Any,
) -> NDArray[Any]:
    """Apply a canonical-form estimator to a strictly positive-definite problem.

    ``q`` must be strictly positive definite (and ``cov`` symmetric positive
    definite, ``x`` a row vector of shape ``(..., p)``), so the problem can be
    canonicalized with :func:`_canonicalize` and ``canonical_estimator`` applied
    directly.  The estimate shrinks towards the point ``offset`` (default zero)
    or, when ``dirs`` (a matrix whose columns span the affine direction) is
    given, towards the affine subspace ``offset + span(dirs)``.

    In the latter case the projection is built in the covariance (precision)
    metric (see :func:`_subspace_reduce`), so the fitted and residual components
    are uncorrelated, and the residual ``(I - P) (x - offset)`` is shrunk in the
    complement.  The residual problem is itself a canonical normal problem with
    diagonal covariance ``d_perp`` and identity loss, so it is solved by
    recursing into :func:`_estimate_pd`; the effective dimension of the
    shrinkage problem becomes ``len(d_perp)``.

    """

    xa = np.asarray(x, dtype=float)
    cova = np.asarray(cov, dtype=float)
    qa = np.asarray(q, dtype=float)
    p = qa.shape[0]
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
        reduced = _estimate_pd(
            eta, np.diag(d_perp), np.eye(d_perp.shape[0]), canonical_estimator, **kwargs
        )
        delta_star = offset_star + kept + reduced @ l2.T
    return cast(NDArray[Any], delta_star @ binv.T)


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

    Validates the common inputs and dispatches to the appropriate solver.  If
    ``q`` is strictly positive definite the problem is solved directly with
    :func:`_estimate_pd`.  If ``q`` is singular (only positive semi-definite),
    the loss-free null space of ``q`` is merged into the no-shrink directions
    and :func:`_estimate_split` separates the problem along the covariance
    metric before canonicalizing the (strictly positive-definite) residual.
    ``canonical_estimator`` must have the signature
    ``canonical(x_star, d, **kwargs)``, where ``x_star`` has shape ``(..., p)``
    and ``d`` holds the coordinate variances ``(p,)``; it returns the
    canonical-form estimate with shape ``(..., p)``.

    The estimate shrinks towards the point ``offset`` (default zero) or, when
    ``dirs`` (a matrix whose columns span the affine direction) is given,
    towards the affine subspace ``offset + span(dirs)``.  For a singular ``q``
    the null space of ``q`` is treated as an additional set of no-shrink
    directions, so the covariance-metric projection of the estimate onto
    ``span(dirs) + null(q)`` equals that of the data.  See :func:`_estimate_pd`
    and :func:`_estimate_split`.

    """

    xa, cova, qa, _ = _validate(x, cov, q)
    p = qa.shape[0]
    w = np.linalg.eigvalsh(qa)
    if np.all(w > _zero_eigenvalue_tolerance(w, p)):
        return _estimate_pd(
            xa, cova, qa, canonical_estimator, offset=offset, dirs=dirs, **kwargs
        )
    dirs_all = None if dirs is None else _validate_dirs(dirs, p)
    dirs_full = _merge_dirs(dirs_all, qa, p)
    return _estimate_split(
        xa,
        cova,
        qa,
        canonical_estimator,
        offset=offset,
        dirs=dirs_full,
        **kwargs,
    )
