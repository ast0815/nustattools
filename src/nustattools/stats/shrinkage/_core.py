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

from ._empirical_prior import (
    _EMPIRICAL_GAMMA_PRESETS,
    _EPSILON,
    _FACTORY_PATTERN,
    _parse_gamma_factory,
)

#: Alias for a gamma prior-scale callable ``f(d, pi, y)``; see the
#: :mod:`nustattools.stats.shrinkage` module docstring for the contract.
GammaCallable = Callable[[NDArray[Any], NDArray[Any], NDArray[Any]], NDArray[Any]]


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

    Returns :math:`(B, B^{-1}, D)` such that :math:`Q = B^T B`,
    :math:`B \\, \\mathrm{cov}\\, B^T = D` (with
    :math:`D` diagonal) and :math:`B^{-1}` the inverse of :math:`B`.
    :math:`D = \\operatorname{diag}(d)` with :math:`d`
    non-increasing: the canonical coordinates are ordered by *decreasing*
    variance, so coordinate ``0`` has the largest variance (matching the
    risk-curve axis convention).  Row vectors ``x`` transform to the
    canonical coordinates as :math:`x^\\star = x B^T` and back as
    :math:`x = x^\\star (B^{-1})^T`.  See [Tan2015]_, Section 3.2.

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
    """True when ``cov`` is numerically proportional to :math:`Q^{-1}`.

    With :math:`C` the Cholesky/triangular factor :math:`Q = C^T C`, the
    canonical matrix :math:`M = C \\, \\mathrm{cov}\\, C^T` is a scalar multiple
    of the identity exactly when :math:`\\mathrm{cov} = c Q^{-1}`, in which
    case the canonical variances all coincide and
    the canonical-frame rotation is unconstrained.  When ``cov`` and ``Q`` are
    supplied as numerically-inverted matrices (e.g. :math:`Q =
    \\mathrm{inv}(\\mathrm{cov})`) the
    computed :math:`M` deviates from that identity by the roundoff of the
    inversion
    and matrix products (roughly :math:`\\varepsilon\\,
    \\mathrm{cond}(\\mathrm{cov})`), so the comparison uses a
    relative tolerance of :math:`\\sqrt{\\varepsilon}`, covering condition
    numbers up to about :math:`1/\\sqrt{\\varepsilon}`.

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
    keeps :math:`B \\, \\mathrm{cov}\\, B^T` diagonal, so within-group (and
    only within-group)
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
    (:math:`B \\, \\mathrm{cov}\\, B^T = \\operatorname{diag}(d)` and
    :math:`Q = B^T B`).  ``prior_cov`` is the
    symmetric positive-definite prior covariance in the *original* coordinates.
    This function spends the rotational freedom of the canonical form -- any
    orthogonal rotation within a group of (near-)equal canonical variances
    ``d`` leaves :math:`B \\, \\mathrm{cov}\\, B^T` diagonal -- to make
    :math:`B_{\\mathrm{rot}}\\, \\mathrm{prior\\_cov}\\, B_{\\mathrm{rot}}^T`
    diagonal as well, and returns that diagonal
    as ``pi`` together with the rotated ``b_rot``.

    The returned ``pi`` is aligned with the ``d`` from :func:`_canonicalize`
    (already non-increasing in the frame it rotates); within each block of
    (numerically-)equal ``d`` that the frame rotates the entries are ordered
    non-increasing as well.  When the prior is already diagonal the same
    ordering is achieved by permuting the runs of *exactly*-equal ``d``
    (a permutation is an orthogonal rotation, so
    :math:`B \\, \\mathrm{cov}\\, B^T = \\operatorname{diag}(d)` is
    preserved); the remaining coordinates keep their given diagonal order.

    A prior proportional to :math:`Q^{-1}` (e.g.
    :math:`\\mathrm{prior\\_cov} = \\mathrm{inv}(Q)` for an
    ill-conditioned and possibly ridge-regularised ``Q``) is the homoscedastic
    canonical prior: its :math:`w` is a scalar multiple of the identity up to
    the
    roundoff of the inversion, so :math:`\\pi` is a constant and no rotation is
    applied.  This is detected from :math:`w` with the same
    :math:`\\sqrt{\\varepsilon}`-relative
    tolerance as :func:`_cov_proportional_to_qinv`, so it covers numerically
    ill-conditioned ``Q`` whose :math:`\\varepsilon\\,
    \\mathrm{cond}(Q)` roundoff would defeat an
    :math:`\\varepsilon`-scale diagonal test.

    When all canonical variances coincide (in particular for a data covariance
    proportional to :math:`Q^{-1}`, where :math:`D` is a scalar multiple of
    the
    identity) the rotation is unconstrained and *any* positive-definite
    ``prior_cov`` is accepted.  Otherwise the prior must already be diagonal in
    the canonical coordinates up to the covariance-degenerate groups: it may
    couple coordinates *inside* a degenerate group (the within-group rotation
    then disentangles them) but not coordinates with differing variances.  A
    genuinely non-diagonalizable prior raises ``ValueError``.

    When ``free_rotation`` is true the rotation is declared unconstrained so
    the prior is diagonalized by a full rotation regardless of the ``d``
    values; the caller should set it only after detecting ``cov``
    proportional to :math:`Q^{-1}` to numerical roundoff via
    :func:`_cov_proportional_to_qinv`.

    """

    p = prior_cov.shape[0]
    w = b @ prior_cov @ b.T
    # When ``prior_cov`` is proportional to ``Q^{-1}`` the canonical prior
    # ``w = B prior_cov B^T`` is a scalar multiple of the identity.  Recognise
    # that structure from ``w`` itself (it never couples coordinates and needs
    # no rotation), with the same ``sqrt(eps)``-relative, roundoff-aware
    # tolerance :func:`_cov_proportional_to_qinv` uses: for an ill-conditioned
    # ``Q`` (e.g. ``Q = inv(prior_cov)`` with a near-singular penalty matrix)
    # the numerically-computed ``w`` deviates from ``scale * I`` by roughly
    # ``eps * cond(Q)``, which can exceed the eps-scale diagonal check below.
    scale = float(np.trace(w)) / p
    if np.max(np.abs(w - scale * np.eye(p))) <= np.sqrt(_EPSILON) * scale:
        return np.full(p, scale), b
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
                        "choose a prior that is diagonal in canonical space.  "
                        "If `prior_cov` was meant to be the default homoscedastic "
                        "prior (proportional to Q^{-1}), the rejected coupling is "
                        "numerical roundoff from inverting twice (e.g. "
                        "`Q = inv(prior_cov)` with `prior_cov` itself built by "
                        "an earlier `inv`): pass Q as the matrix whose inverse "
                        "is `prior_cov` (e.g. `prior_cov = inv(M)` with "
                        "`Q = M`), or omit `prior_cov` (which defaults to "
                        "Q^{-1})."
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


def _canonical_frame(
    cov: NDArray[Any],
    q: NDArray[Any],
    prior_cov: ArrayLike | None,
) -> tuple[NDArray[Any], NDArray[Any], NDArray[Any], NDArray[Any]]:
    """Return the canonical frame ``(B, Binv, d, pi)`` shared by estimators.

    Composes :func:`_canonicalize` and, for a given ``prior_cov``,
    :func:`_canonicalize_prior` into the frame the estimators actually shrink
    in: ``B`` and ``Binv = inv(B)`` are the (rotated) frame, ``d`` the
    non-increasing canonical variances and ``pi`` its aligned prior diagonal
    (all ones without a prior; non-increasing within each block of
    (numerically-)equal ``d`` when the frame rotated a prior).  Callers that
    canonicalize data or directions and map back to the original space with
    ``Binv`` must use this single frame, so their coordinate ``j`` is exactly
    the estimator's canonical coordinate ``j``.

    """

    b, binv, d = _canonicalize(cov, q)
    if prior_cov is None:
        return b, binv, d, np.ones(d.shape[0])
    pc = _validate_sympd(prior_cov, (d.shape[0], d.shape[0]), "prior covariance matrix")
    free = _cov_proportional_to_qinv(cov, q)
    pi, b = _canonicalize_prior(b, d, pc, free_rotation=free)
    return b, np.linalg.inv(b), d, pi


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


def _projector(v: NDArray[Any], cov: NDArray[Any]) -> NDArray[Any]:
    """Covariance-metric projector onto the column space of ``v``.

    Returns :math:`P = V (V^T C^{-1} V)^{-1} V^T C^{-1}`, where :math:`V = v`
    is the
    :math:`(p, k)` spanning matrix and :math:`C = \\mathrm{cov}` the covariance:
    the projection
    orthogonal in the precision metric :math:`C^{-1}`.  Such a projection makes
    :math:`P y` and :math:`(I - P) y` uncorrelated for
    :math:`y \\sim (0, C)`, so their risks
    separate (see [Tan2016]_, Section 3.3).

    """

    cinv: NDArray[Any] = np.linalg.inv(cov)
    g: NDArray[Any] = v.T @ cinv @ v
    return v @ np.linalg.solve(g, v.T @ cinv)


def _reduce_dirs(
    y: NDArray[Any], cov: NDArray[Any], v: NDArray[Any]
) -> tuple[NDArray[Any], NDArray[Any], NDArray[Any], NDArray[Any], NDArray[Any]]:
    """Decompose ``y`` relative to the affine direction spanned by ``v``.

    ``cov`` is the covariance of ``y`` in its current coordinates (general
    symmetric positive definite, e.g. :math:`\\operatorname{diag}(d)` in
    canonical coordinates).
    Returns ``(kept, eta, d_perp, l2, pmat)``.  :math:`\\mathrm{kept} = P y`
    is the
    component lying in the direction (kept unshrunk), with ``P = pmat`` the
    covariance-metric projector of :func:`_projector`.  The residual
    :math:`(I - P) y` lives in the complement ``S_perp``; ``l2`` is an
    orthonormal basis of ``S_perp`` in which the residual covariance is
    diagonal, ``eta`` the coordinates of the residual in that basis and
    ``d_perp`` the reduced (diagonal) variances.  This reduces the effective
    dimension of the shrinkage problem from ``p`` to ``len(d_perp)``.

    Since ``l2`` holds the orthonormal eigenvectors of the symmetric residual
    covariance :math:`(I - P) C (I - P)^T`, :math:`l_2^T l_2 = I`: the change
    of basis is
    an isometry of the squared-error loss, so shrinking ``eta`` conserves the
    full-dimensional loss exactly.  Coordinates whose residual variance is
    numerically zero (the kept subspace itself) are dropped with the
    scale-aware tolerance of :func:`_zero_eigenvalue_tolerance`.

    """

    pmat = _projector(v, cov)
    p_perp = np.eye(cov.shape[0]) - pmat
    # Residual covariance (I - P) C (I - P)^T; its range is S_perp.  Its
    # positive eigenvalues give the reduced variances and its (orthonormal)
    # eigenvectors a basis that diagonalizes the residual.
    m = p_perp @ cov @ p_perp.T
    lam, vecs = np.linalg.eigh(m)
    keep = lam > _zero_eigenvalue_tolerance(lam, cov.shape[0])
    l2 = vecs[:, keep]
    d_perp = lam[keep]
    kept = y @ pmat.T
    eta = (y - kept) @ l2
    return kept, eta, d_perp, l2, pmat


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


def _check_prior_diagonalizable(
    prior: NDArray[Any],
    cov: NDArray[Any],
    q: NDArray[Any],
) -> None:
    """Reject a residual prior that the recursive solve cannot diagonalize.

    The ``dirs`` (and singular-loss) paths reduce the problem to a residual
    canonical normal problem that is solved by recursing into
    :func:`_estimate_pd`, which canonicalizes the reduced ``prior`` together
    with the residual covariance ``cov`` and loss ``q`` (see
    :func:`_canonical_frame`).  A ``prior`` that couples the residual
    coordinates with differing variances would surface there as the generic
    coupling :class:`ValueError`; this helper runs the identical check up front
    so the caller can raise a message that names the interaction with the
    no-shrink directions instead.  A residual problem of at most one coordinate
    is trivially diagonalizable and needs no check.

    """

    if prior.shape[0] <= 1:
        return
    try:
        _canonical_frame(cov, q, prior)
    except ValueError as e:
        msg = (
            "the prior covariance matrix restricted to the residual "
            "(complement of the no-shrink directions) subspace cannot be "
            "diagonalized: it couples residual canonical coordinates with "
            "differing variances once the covariance-metric projection of the "
            "no-shrink directions has been removed.  This is automatic when "
            "`prior_cov` is proportional to `cov` or to `Q**-1`, or when `cov` "
            "is proportional to `Q**-1`; otherwise choose a `prior_cov` whose "
            "canonical diagonal respects the residual coupling (e.g. fewer "
            "no-shrink directions)."
        )
        raise ValueError(msg) from e


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
    :math:`\\operatorname{span}(\\text{user dirs}) + \\operatorname{null}(q)`
    (see :func:`_merge_dirs`), so ``q`` is
    strictly positive definite on the covariance-metric complement of
    :math:`\\operatorname{span}(\\mathrm{dirs})`: any residual there is
    :math:`\\Sigma^{-1}`-orthogonal to
    ``null(q)`` and hence cannot itself lie in ``null(q)``.  This function
    therefore splits the data into the part lying in
    :math:`\\operatorname{span}(\\mathrm{dirs})` (kept at
    its covariance-metric data value) and the :math:`\\Sigma^{-1}`-orthogonal
    residual, then solves the strictly-positive-definite residual problem with
    :func:`_estimate_pd` and recombines.  The inputs must already be validated
    (see :func:`_validate`); ``dirs`` must span the full no-shrink set.

    """

    p = cov.shape[0]
    v_all = np.asarray(dirs, dtype=float)
    if offset is None:
        y = x
    else:
        o = np.asarray(offset, dtype=float)
        if o.shape != (p,):
            msg = f"offset must have shape {(p,)}, got {o.shape}."
            raise ValueError(msg)
        y = x - o
    kept, eta, d_perp, l2, pmat = _reduce_dirs(y, cov, v_all)
    kept_off = np.zeros(p) if offset is None else o @ pmat.T
    q_comp = l2.T @ q @ l2
    # The prior only acts where shrinkage happens: restrict it to the
    # covariance-metric complement of the no-shrink directions.  The residuals
    # live in ``range(l2)``, an orthonormal basis of that complement, so the
    # residual prior is the covariance-metric projection of ``prior_cov`` onto
    # the complement, expressed in the residual coordinates
    # ``l2^T (I - P) prior_cov (I - P)^T l2`` (a plain restriction ``l2^T
    # prior_cov l2`` would ignore the projected geometry).  A projection that
    # couples the residual canonical coordinates with differing variances is
    # rejected with a directed message before the recursive solve.
    if prior_cov is not None:
        pc = _validate_sympd(prior_cov, (p, p), "prior covariance matrix")
        p_perp = np.eye(p) - pmat
        prior_comp = l2.T @ p_perp @ pc @ p_perp.T @ l2
        _check_prior_diagonalizable(prior_comp, np.diag(d_perp), q_comp)
    else:
        prior_comp = None
    if d_perp.shape[0] == 0:
        delta_comp = np.zeros_like(eta)
    else:
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
    given, towards the affine subspace
    :math:`\\mathrm{offset} + \\operatorname{span}(\\mathrm{dirs})`.

    ``prior_cov`` is the prior covariance in the *original* coordinates; when
    given, the canonical frame is rotated (whenever the canonical variances
    allow; see :func:`_canonicalize_prior`) so that the prior is diagonal in the
    canonical coordinates, and its diagonal :math:`\\pi` is threaded to the
    canonical estimator as the shape of the prior (scaled by :math:`\\gamma`).
    Without it the prior reduces to the homoscedastic :math:`\\gamma I` of the
    current implementation.

    In the latter case the projection is built in the covariance (precision)
    metric (see :func:`_reduce_dirs`), so the fitted and residual components
    are uncorrelated, and the residual :math:`(I - P) (x - \\mathrm{offset})`
    is shrunk in the
    complement.  The residual problem is itself a canonical normal problem with
    diagonal covariance ``d_perp`` and identity loss, so it is solved by
    recursing into :func:`_estimate_pd`; the effective dimension of the
    shrinkage problem becomes ``len(d_perp)``.  In that recursion the prior is
    restricted to the residual subspace to become the reduced prior, and
    canonicalized again by the recursive solve: for an isotropic canonical
    prior (``prior_cov`` proportional to :math:`Q^{-1}`) it is passed as the
    homoscedastic :math:`s I` in the orthonormal residual basis, otherwise as
    the
    covariance-metric projection
    :math:`l_2^T (I - P) \\operatorname{diag}(\\pi) (I - P)^T l_2` of the
    canonical prior onto the
    residual subspace.  A non-isotropic prior whose projection couples the
    residual coordinates with differing variances cannot be diagonalized there
    and is rejected with a :class:`ValueError` (see
    :func:`_check_prior_diagonalizable`).

    When ``gamma`` is a named preset (currently only ``"empirical"``, inferred
    per observation as :math:`\\| y / \\sqrt{\\pi} \\|^2 / p_{\\mathrm{eff}}`)
    or a callable
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
    b, binv, d, pi = _canonical_frame(cova, qa, prior_cov)
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
    if "gamma" in kwargs and isinstance(kwargs["gamma"], str):
        g_str = kwargs["gamma"]
        if g_str not in _EMPIRICAL_GAMMA_PRESETS:
            if _FACTORY_PATTERN.match(g_str):
                kwargs["gamma"] = _parse_gamma_factory(g_str)
            else:
                names = ", ".join(_EMPIRICAL_GAMMA_PRESETS)
                msg = f"Unknown gamma specification '{g_str}'; use a number or one of: {names}."
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
        kept, eta, d_perp, l2, pmat = _reduce_dirs(y, np.diag(d), v)
        # The reduced problem's prior, restricted to the residual subspace in
        # the orthonormal residual basis l2.  The data residuals (I - P) y
        # live in the covariance-metric complement of span(dirs), so the
        # residual prior is the covariance-metric projection
        # l2^T (I - P) diag(pi) (I - P)^T l2 of the canonical prior onto that
        # subspace; the recursive solve canonicalizes it again.  An isotropic
        # canonical prior (prior_cov proportional to Q^{-1}) is the
        # homoscedastic residual prior s I in the orthonormal residual basis
        # (matching the no-dirs homoscedastic default), so it needs no
        # projection.  A non-isotropic prior whose projection couples the
        # residual coordinates with differing variances cannot be diagonalized
        # by the recursive solve and is rejected up front.  With no explicit
        # prior the reduced problem inherits none (the homoscedastic default).
        reduced_prior: NDArray[Any] | None
        if prior_cov is None:
            reduced_prior = None
        elif np.max(np.abs(pi - pi[0])) <= np.sqrt(_EPSILON) * float(np.mean(pi)):
            reduced_prior = np.eye(d_perp.shape[0]) * float(np.mean(pi))
        else:
            pp = np.eye(p) - pmat
            proj = l2.T @ pp @ np.diag(pi) @ pp.T @ l2
            _check_prior_diagonalizable(proj, np.diag(d_perp), np.eye(d_perp.shape[0]))
            reduced_prior = proj
        # The residual (eta, diag(d_perp), identity loss) is itself a canonical
        # normal problem with no subspace and no offset; solve it recursively.
        # gamma="empirical" is intentionally left unresolved so the recursive
        # solve derives it from the residual actually shrunk (eta, len(d_perp)).
        # When the no-shrink directions span everything there is no residual to
        # shrink and the reduced estimate vanishes.
        if d_perp.shape[0] == 0:
            reduced = np.zeros_like(eta)
        else:
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
    towards the affine subspace
    :math:`\\mathrm{offset} + \\operatorname{span}(\\mathrm{dirs})`.  For a
    singular ``q``
    the null space of ``q`` is treated as an additional set of no-shrink
    directions, so the covariance-metric projection of the estimate onto
    :math:`\\operatorname{span}(\\mathrm{dirs}) + \\operatorname{null}(q)`
    equals that of the data.  See :func:`_estimate_pd`
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


def _check_strength(strength: float) -> None:
    """Raise ``ValueError`` unless ``0 <= strength <= 2``."""

    if strength < 0 or strength > 2:
        msg = "strength must be in [0, 2]."
        raise ValueError(msg)


def _check_gamma_nonnegative(
    gamma: float | str | Callable[..., Any] | NDArray[Any],
) -> None:
    """Raise ``ValueError`` if ``gamma`` contains negative values.

    String presets and callables are left to be resolved downstream; only
    numeric (scalar or array) values are validated here.

    """

    if (
        not isinstance(gamma, str)
        and not callable(gamma)
        and np.any(np.asarray(gamma) < 0)
    ):
        msg = "gamma must be non-negative."
        raise ValueError(msg)


def _prior_diagonal(d: NDArray[Any], pi: NDArray[Any] | None) -> NDArray[Any]:
    """Return the canonical prior diagonal ``pi``, defaulting to all ones."""

    return pi if pi is not None else np.ones(d.shape[0])


def _estimate_coordinate(
    x: ArrayLike,
    cov: ArrayLike | None,
    q: ArrayLike | None,
    canonical: Callable[..., NDArray[Any]],
    *,
    positive: bool,
    strength: float,
    gamma: float | str | Callable[..., NDArray[Any]] | NDArray[Any],
    offset: ArrayLike | None = None,
    dirs: ArrayLike | None = None,
    prior_cov: ArrayLike | None = None,
) -> NDArray[Any]:
    """Dispatch a coordinate-wise estimator to the shared solver.

    The coordinate-family estimators (``tan``, ``minimax_bayes``,
    ``tan_bayes`` and ``robust_bayes``) all share this exact set of keyword
    arguments.  Repackaging them here keeps the wrapper functions thin and
    lets the public estimators pass ``canonical`` as a plain positional-style
    argument instead of threading ``canonical_estimator`` backwards through
    :func:`_estimate`.

    """

    return _estimate(
        x,
        cov,
        q,
        canonical,
        strength=strength,
        gamma=gamma,
        positive=positive,
        offset=offset,
        dirs=dirs,
        prior_cov=prior_cov,
    )
