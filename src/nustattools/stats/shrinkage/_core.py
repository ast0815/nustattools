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


def _empirical_gamma(
    d: NDArray[Any], pi: NDArray[Any], y: NDArray[Any]
) -> NDArray[Any]:
    """Return the per-observation empirical prior scale ``||y / sqrt(pi)||^2 / p_eff``.

    ``y`` has shape ``(..., p_eff)`` (the centered canonical data) and ``pi``
    has shape ``(p_eff,)`` (the canonical diagonal of the prior covariance,
    all ones for the default homoscedastic prior).  The result has shape
    ``(...,)``: one prior scale for each observation, computed from the
    squared norm over the trailing (coordinate) axis only, after normalising
    each coordinate by the corresponding prior scale ``sqrt(pi_j)``.

    """

    p_eff = len(d)
    safe_pi = np.where(pi > 0, pi, 1.0)
    return cast(NDArray[Any], np.sum(y**2 / safe_pi, axis=-1) / p_eff)


_EMPIRICAL_GAMMA_PRESETS: dict[
    str, Callable[[NDArray[Any], NDArray[Any], NDArray[Any]], NDArray[Any]]
] = {
    "empirical": _empirical_gamma,
}


def _resolve_data_gamma(
    f: Callable[[NDArray[Any], NDArray[Any], NDArray[Any]], NDArray[Any]],
    d: NDArray[Any],
    pi: NDArray[Any],
    y: NDArray[Any],
) -> NDArray[Any]:
    """Call a data-to-scale function ``f(d, pi, y)`` and validate its output shape.

    The callable must return a real-valued array of prior scales that is either
    a singular scalar (a single scale shared by every observation, natural when
    it is computed only from ``d`` / ``pi`` and not the data ``y``) or an
    array whose shape equals ``y.shape[:-1]`` (one prior scale per observation,
    i.e. exactly the leading batch dimensions of the canonical data ``y``).
    Any other shape would collapse independent observations and is rejected.
    The returned value must be non-negative.  The validated prior scales are
    returned as a ``float`` array (a 0-d array for a scalar).

    """

    try:
        out = np.asarray(f(d, pi, y), dtype=float)
    except Exception as e:
        msg = "gamma callable must return a real-valued array of prior scales."
        raise TypeError(msg) from e
    if out.ndim != 0 and out.shape != y.shape[:-1]:
        msg = (
            f"gamma callable must return a scalar or an array of shape "
            f"{y.shape[:-1]} matching the batch dimensions of the data, got "
            f"shape {out.shape}."
        )
        raise ValueError(msg)
    if np.any(out < 0):
        msg = (
            "gamma callable returned negative prior scales; gamma must be non-negative."
        )
        raise ValueError(msg)
    return out


def _canonicalize(
    cov: NDArray[Any], q: NDArray[Any]
) -> tuple[NDArray[Any], NDArray[Any], NDArray[Any]]:
    """Transform to canonical form (diagonal covariance, identity loss).

    Returns ``(B, Binv, D)`` such that ``Q = B^T B``, ``B cov B^T = D`` (with
    ``D`` diagonal) and ``Binv = inv(B)``.  ``D = diag(d)`` with ``d``
    non-increasing: the canonical coordinates are ordered by *decreasing*
    variance, so coordinate ``0`` has the largest variance (matching the
    risk-curve axis convention).  Row vectors ``x`` transform to the
    canonical coordinates as ``x_star = x @ B.T`` and back as
    ``x = x_star @ Binv.T``.  See [Tan2015]_, Section 3.2.

    """

    # C with Q = C^T C (upper triangular from the Cholesky factor of Q)
    c = np.linalg.cholesky(q).T
    # Diagonalize C Sigma C^T.  eigh gives o^T (C Sigma C^T) o = diag(d), so
    # the orthogonal O with O (C Sigma C^T) O^T = diag(d) is O = o^T.  The
    # eigenvalues come out ascending; permute the frame so d is descending.
    d, o = np.linalg.eigh(c @ cov @ c.T)
    perm = np.argsort(d)[::-1]
    o = o[:, perm]
    d = d[perm]
    b = o.T @ c
    binv = np.linalg.inv(b)
    return b, binv, d


def _cov_proportional_to_qinv(cova: NDArray[Any], qa: NDArray[Any]) -> bool:
    """True when ``cov`` is numerically proportional to ``Q^{-1}``.

    With ``C`` the Cholesky/triangular factor ``Q = C^T C``, the canonical
    matrix ``M = C cov C^T`` is a scalar multiple of the identity exactly when
    ``cov = c Q^{-1}``, in which case the canonical variances all coincide and
    the canonical-frame rotation is unconstrained.  When ``cov`` and ``Q`` are
    supplied as numerically-inverted matrices (e.g. ``Q = inv(cov)``) the
    computed ``M`` deviates from that identity by the roundoff of the inversion
    and matrix products (roughly ``eps * cond(cov)``), so the comparison uses a
    relative tolerance of ``sqrt(eps)``, covering condition numbers up to about
    ``1/sqrt(eps)``.

    """

    c = np.linalg.cholesky(qa).T
    m = c @ cova @ c.T
    p = cova.shape[0]
    scale = float(np.trace(m)) / p
    tol = np.sqrt(_EPSILON) * scale
    return bool(np.max(np.abs(m - scale * np.eye(p))) <= tol)


def _group_degenerate(d: NDArray[Any], tol: float) -> list[list[int]]:
    """Partition coordinates into groups of (near-)equal canonical variance.

    ``d`` holds the canonical coordinate variances ``(p,)``.  Coordinates are
    grouped (in descending ``d`` order) when their variance gaps do not exceed
    ``tol``.  Only *within* a such a group is the canonical form defined up to an
    orthogonal rotation: any rotation inside a group of exactly-equal variances
    keeps ``B cov B^T`` diagonal, so within-group (and only within-group)
    freedom can be spent on diagonalizing a second matrix.

    """

    order = np.argsort(d)[::-1]
    groups: list[list[int]] = []
    current: list[int] = [int(order[0])]
    for idx in order[1:]:
        # Descending order, so d[prev] >= d[idx]: the gap is d[prev] - d[idx].
        if d[current[-1]] - d[idx] <= tol:
            current.append(int(idx))
        else:
            groups.append(current)
            current = [int(idx)]
    groups.append(current)
    return groups


def _canonicalize_prior(
    b: NDArray[Any],
    d: NDArray[Any],
    prior_cov: NDArray[Any],
    *,
    free_rotation: bool = False,
) -> tuple[NDArray[Any], NDArray[Any]]:
    """Rotate the canonical frame so the prior covariance becomes diagonal.

    ``b`` and ``d`` are the canonicalization results of :func:`_canonicalize`
    (``B cov B^T = diag(d)`` and ``Q = B^T B``).  ``prior_cov`` is the
    symmetric positive-definite prior covariance in the *original* coordinates.
    This function spends the rotational freedom of the canonical form -- any
    orthogonal rotation within a group of (near-)equal canonical variances
    ``d`` leaves ``B cov B^T`` diagonal -- to make
    ``b_rot @ prior_cov @ b_rot.T`` diagonal as well, and returns that diagonal
    as ``pi`` together with the rotated ``b_rot``.

    The returned ``pi`` is aligned with the ``d`` from :func:`_canonicalize`
    (already non-increasing in the frame it rotates); within each block of
    (numerically-)equal ``d`` that the frame rotates the entries are ordered
    non-increasing as well.  When the prior is already diagonal the same
    ordering is achieved by permuting the runs of *exactly*-equal ``d``
    (a permutation is an orthogonal rotation, so ``B cov B^T = diag(d)`` is
    preserved); the remaining coordinates keep their given diagonal order.

    When all canonical variances coincide (in particular for a data covariance
    proportional to ``Q^{-1}``, where ``D`` is a scalar multiple of the
    identity) the rotation is unconstrained and *any* positive-definite
    ``prior_cov`` is accepted.  Otherwise the prior must already be diagonal in
    the canonical coordinates up to the covariance-degenerate groups: it may
    couple coordinates *inside* a degenerate group (the within-group rotation
    then disentangles them) but not coordinates with differing variances.  A
    genuinely non-diagonalizable prior raises ``ValueError``.

    When ``free_rotation`` is true the rotation is declared unconstrained so
    the prior is diagonalized by a full rotation regardless of the ``d``
    values; the caller should set it only after detecting ``cov``
    proportional to ``Q^{-1}`` to numerical roundoff via
    :func:`_cov_proportional_to_qinv`.

    """

    p = prior_cov.shape[0]
    w = b @ prior_cov @ b.T
    zero_tol = _zero_eigenvalue_tolerance(np.linalg.eigvalsh(w), p)
    if np.allclose(w, np.diag(np.diag(w)), rtol=0.0, atol=zero_tol):
        # The prior is already diagonal in the canonical coordinates, so no
        # rotation is needed.  Only the frame's *ordering* is a free choice:
        # coordinates with exactly-equal canonical variances may be permuted
        # without disturbing ``B cov B^T = diag(d)`` (a permutation is an
        # orthogonal rotation), so permute each run of equal ``d`` so that
        # ``pi`` is non-increasing inside it.  Runs of merely near-equal (but
        # distinct) ``d`` keep their given diagonal order.
        pi = np.diag(w).copy()
        rot = np.eye(p)
        j = 0
        while j < p:
            k = j + 1
            while k < p and d[k] == d[j]:
                k += 1
            if k - j > 1:
                blk = np.arange(j, k)
                desc = np.argsort(pi[blk])[::-1]
                rot[np.ix_(blk, blk)] = np.eye(k - j)[desc]
                pi[blk] = pi[blk][desc]
            j = k
        return pi, rot @ b
    if free_rotation:
        groups = [list(range(p))]
    else:
        tol = _zero_eigenvalue_tolerance(d, p)
        groups = _group_degenerate(d, tol)
        for i, gi in enumerate(groups):
            for gj in groups[i + 1 :]:
                if not np.allclose(w[np.ix_(gi, gj)], 0.0, rtol=0.0, atol=zero_tol):
                    msg = (
                        "prior covariance matrix cannot be diagonalized in the "
                        "canonical coordinates: it couples canonical coordinates "
                        "with differing variances.  When the data covariance is "
                        "proportional to Q^{-1} this is automatic; otherwise "
                        "choose a prior that is diagonal in canonical space."
                    )
                    raise ValueError(msg)
    rot = np.eye(p)
    pi = np.empty(p)
    for group in groups:
        # Sort the positions ascending: the group lists are built by walking
        # ``d`` in descending order, so their members need not be position-
        # ordered.  eigh returns ascending eigenvalues; assigning them in
        # *descending* order to the ascending positions makes ``pi``
        # non-increasing in the flattened array (every such assignment is an
        # orthogonal rotation of the frame, so it is still a valid within-block
        # rotation).
        grp = np.sort(np.asarray(group))
        evals, evecs = np.linalg.eigh(w[np.ix_(grp, grp)])
        desc = np.argsort(evals)[::-1]
        rot[np.ix_(grp, grp)] = evecs[:, desc].T
        pi[grp] = evals[desc]
    if np.any(pi <= 0):
        msg = "prior covariance matrix must be positive definite (in canonical coordinates)."
        raise ValueError(msg)
    return pi, rot @ b


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
    prior_cov: ArrayLike | None = None,
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
    # The prior only acts where shrinkage happens: restrict it to the
    # covariance-metric complement of the no-shrink directions.  Since l2 is
    # an orthonormal basis spanning that complement, the restricted covariance
    # is l2^T prior_cov l2 (the covariance-metric projection of the prior onto
    # the complement is its restriction there).
    if prior_cov is not None:
        pc = _validate_sympd(prior_cov, (p, p), "prior covariance matrix")
        prior_comp = l2.T @ pc @ l2
    else:
        prior_comp = None
    delta_comp = _estimate_pd(
        eta,
        np.diag(d_perp),
        q_comp,
        canonical_estimator,
        prior_cov=prior_comp,
        **kwargs,
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
    prior_cov: ArrayLike | None = None,
    **kwargs: Any,
) -> NDArray[Any]:
    """Apply a canonical-form estimator to a strictly positive-definite problem.

    ``q`` must be strictly positive definite (and ``cov`` symmetric positive
    definite, ``x`` a row vector of shape ``(..., p)``), so the problem can be
    canonicalized with :func:`_canonicalize` and ``canonical_estimator`` applied
    directly.  The estimate shrinks towards the point ``offset`` (default zero)
    or, when ``dirs`` (a matrix whose columns span the affine direction) is
    given, towards the affine subspace ``offset + span(dirs)``.

    ``prior_cov`` is the prior covariance in the *original* coordinates; when
    given, the canonical frame is rotated (whenever the canonical variances
    allow; see :func:`_canonicalize_prior`) so that the prior is diagonal in the
    canonical coordinates, and its diagonal ``pi`` is threaded to the
    canonical estimator as the shape of the prior (scaled by ``gamma``).
    Without it the prior reduces to the homoscedastic ``gamma I`` of the current
    implementation.

    In the latter case the projection is built in the covariance (precision)
    metric (see :func:`_subspace_reduce`), so the fitted and residual components
    are uncorrelated, and the residual ``(I - P) (x - offset)`` is shrunk in the
    complement.  The residual problem is itself a canonical normal problem with
    diagonal covariance ``d_perp`` and identity loss, so it is solved by
    recursing into :func:`_estimate_pd`; the effective dimension of the
    shrinkage problem becomes ``len(d_perp)``.  In that recursion the prior is
    restricted to the complement, i.e. passed as the covariance-metric
    projection of ``prior_cov`` onto the residual subspace, and canonicalized
    again by the recursive solve.

    When ``gamma`` is a named preset (currently only ``"empirical"``, inferred
    per observation as ``||y / sqrt(pi)||^2 / p_eff``) or a callable
    ``f(d, pi, y)``, it is resolved from the canonical data actually shrunk,
    per observation; see the :mod:`nustattools.stats.shrinkage` module docstring
    for the accepted forms and the per-observation shape contract.  When the
    problem is subspace-split (``dirs`` given) the residual
    ``(eta, diag(d_perp))`` is solved by a recursive :func:`_estimate_pd` in
    which the preset name or callable is left unresolved, so the per-observation
    scale is derived from the reduced residual ``eta`` with effective dimension
    ``len(d_perp)`` (rather than the full data).  The same holds when this
    function is entered from :func:`_estimate_split` (singular loss): ``x`` is
    already the loss-free complement residual, so the preset scale reflects
    that reduced problem.

    """

    xa = np.asarray(x, dtype=float)
    cova = np.asarray(cov, dtype=float)
    qa = np.asarray(q, dtype=float)
    p = qa.shape[0]
    b, binv, d = _canonicalize(cova, qa)
    if prior_cov is None:
        pi = np.ones(p)
    else:
        pc = _validate_sympd(prior_cov, (p, p), "prior covariance matrix")
        free = _cov_proportional_to_qinv(cova, qa)
        pi, b = _canonicalize_prior(b, d, pc, free_rotation=free)
        binv = np.linalg.inv(b)
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
    if (
        "gamma" in kwargs
        and isinstance(kwargs["gamma"], str)
        and kwargs["gamma"] not in _EMPIRICAL_GAMMA_PRESETS
    ):
        names = ", ".join(_EMPIRICAL_GAMMA_PRESETS)
        msg = f"Unknown gamma specification '{kwargs['gamma']}'; use a number or one of: {names}."
        raise ValueError(msg)
    # A named preset or callable prior scale is resolved per observation: each
    # draw gets its own scale (e.g. ||y_i / sqrt(pi)||^2 / p_eff).  In the plain
    # (no-dirs) path it is resolved here and passed to the canonical estimator
    # as an array, which vectorizes over the batch.  In the dirs path below the
    # preset name / callable is left unresolved so the recursive solve derives
    # it from the reduced residual actually shrunk (eta, len(d_perp)).
    if dirs is None:
        g = kwargs.get("gamma")
        if isinstance(g, str):
            resolved = _EMPIRICAL_GAMMA_PRESETS[g](d, pi, y)
        elif callable(g):
            resolved = _resolve_data_gamma(g, d, pi, y)
        else:
            resolved = None
        if resolved is not None:
            kwargs["gamma"] = float(resolved) if resolved.ndim == 0 else resolved
        if prior_cov is not None:
            kwargs["pi"] = pi
        delta_star = canonical_estimator(y, d, **kwargs) + offset_star
    else:
        v = b @ _validate_dirs(dirs, p)
        kept, eta, d_perp, l2 = _subspace_reduce(y, d, v)
        # The reduced problem's prior covariance: the canonical prior
        # diag(pi) restricted to the residual subspace in the orthonormal
        # residual basis l2 is l2^T diag(pi) l2; pass it on so the recursive
        # solve canonicalizes (and where possible diagonalizes) it.  With no
        # explicit prior the reduced problem inherits none (the homoscedastic
        # default).
        reduced_prior: NDArray[Any] | None = (
            None if prior_cov is None else l2.T @ np.diag(pi) @ l2
        )
        # The residual (eta, diag(d_perp), identity loss) is itself a canonical
        # normal problem with no subspace and no offset; solve it recursively.
        # gamma="empirical" is intentionally left unresolved so the recursive
        # solve derives it from the residual actually shrunk (eta, len(d_perp)).
        reduced = _estimate_pd(
            eta,
            np.diag(d_perp),
            np.eye(d_perp.shape[0]),
            canonical_estimator,
            prior_cov=reduced_prior,
            **kwargs,
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
    prior_cov: ArrayLike | None = None,
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
    canonical-form estimate with shape ``(..., p)``.  ``prior_cov`` is an
    optional prior covariance in the original coordinates; see
    :func:`_estimate_pd`.

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
            xa,
            cova,
            qa,
            canonical_estimator,
            offset=offset,
            dirs=dirs,
            prior_cov=prior_cov,
            **kwargs,
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
        prior_cov=prior_cov,
        **kwargs,
    )
