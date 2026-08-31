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

from collections.abc import Callable, Sequence
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


_Estimator = Callable[..., NDArray[Any]] | str


def _normalize_estimators(
    estimators: _Estimator | Sequence[_Estimator],
) -> tuple[bool, list[_Estimator]]:
    """Return ``(is_single, list)`` for an estimator spec.

    A single callable or method name becomes ``(True, [estimators])``; a
    sequence of them becomes ``(False, list(estimators))``.

    """

    if isinstance(estimators, str) or callable(estimators):
        return True, [estimators]
    if isinstance(estimators, (list, tuple)):
        return False, list(estimators)
    msg = "estimators must be a callable, a method name, or a sequence of these."
    raise TypeError(msg)


def estimate_risk(
    theta: ArrayLike,
    cov: ArrayLike,
    estimators: _Estimator | Sequence[_Estimator],
    *,
    Q: ArrayLike | None = None,
    n_reps: int = 10_000,
    seed: int | np.random.Generator | None = None,
    **kwargs: Any,
) -> NDArray[Any]:
    """Estimate the risk of shrinkage estimators by Monte Carlo.

    For the normal-mean problem ``x ~ N(theta, cov)`` with loss
    ``(delta - theta)^T Q (delta - theta)``, the risk of an estimator ``delta``
    is ``R = E[(delta(x) - theta)^T Q (delta(x) - theta)]``, which generally has
    no closed form for shrinkage estimators.  This function estimates it by
    averaging the quadratic loss over ``n_reps`` samples ``x`` drawn from
    ``N(theta, cov)``.

    If an ``estimators`` *sequence* is given, all estimators are evaluated on
    the *same* Monte Carlo samples, so that any difference between their
    estimated risks reflects genuine performance differences rather than
    sampling noise.

    Parameters
    ----------
    theta : array_like
        The true mean, of shape ``(p,)``.
    cov : array_like
        The known covariance matrix of ``x``, of shape ``(p, p)``.  Must be
        symmetric and positive definite.
    estimators : callable or str, or sequence of these
        The estimator(s) to evaluate.  A callable is applied as
        ``estimator(x, cov, Q=Q, **kwargs)``; a string is resolved through the
        registry of known methods (see :func:`shrink`).  To compare estimators
        with different parameters, pass callables, e.g.
        ``functools.partial(berger, c=3.0)``.
    Q : array_like, default=None
        The known loss matrix, of shape ``(p, p)``.  Defaults to the identity.
    n_reps : int, default=10000
        Number of Monte Carlo draws.  Must be at least 2 so that the standard
        error is finite.
    seed : int or numpy.random.Generator, default=None
        Seed for the random number generator, for reproducible results.
    **kwargs
        Additional keyword arguments passed to every estimator (e.g. ``c``,
        ``positive``, ``offset``, ``dirs``).

    Returns
    -------
    risk : numpy.ndarray
        Each estimator contributes a row ``[risk, standard error]``, where
        ``risk`` is the Monte Carlo mean of the quadratic loss and ``standard
        error`` is its Monte Carlo standard error ``std(loss) / sqrt(n_reps)``.
        If ``estimators`` is a single estimator the result has shape ``(2,)``;
        if a sequence, shape ``(len(estimators), 2)``, in the given order.

    Examples
    --------
    Estimate the risk of Berger's estimator and compare it with the risk of the
    raw (identity) estimator, using a shared set of samples.

    >>> import functools
    >>> import numpy as np
    >>> import nustattools.stats as s
    >>> rng = np.random.default_rng(0)
    >>> theta = np.array([1.0, 0.0, 0.0])
    >>> estimators = [
    ...     functools.partial(s.shrink, c=2.0),
    ...     functools.partial(s.shrink, c=0.0),
    ... ]
    >>> s.estimate_risk(theta, np.eye(3), estimators, n_reps=2000, seed=0).shape
    (2, 2)

    """

    if n_reps < 2:
        msg = "n_reps must be an integer >= 2."
        raise ValueError(msg)

    theta_arr = np.asarray(theta, dtype=float)
    if theta_arr.ndim != 1:
        msg = f"theta must be a 1-D vector, got shape {theta_arr.shape}."
        raise ValueError(msg)
    p = theta_arr.shape[0]
    cova = _validate_sympd(cov, (p, p), "cov")
    qa = np.eye(p) if Q is None else _validate_sympd(Q, (p, p), "Q")

    is_single, est_list = _normalize_estimators(estimators)

    gen = np.random.default_rng(seed)
    x = gen.multivariate_normal(theta_arr, cova, size=n_reps)

    results = []
    for est in est_list:
        if isinstance(est, str):
            try:
                fn = _METHODS[est]
            except KeyError as e:
                msg = f"Unknown shrinkage method '{est}'."
                raise ValueError(msg) from e
        else:
            fn = est
        delta = fn(x, cova, Q=qa, **kwargs)
        d = delta - theta_arr
        loss = np.einsum("ni,ij,nj->n", d, qa, d)
        risk: float = float(np.mean(loss))
        se: float = float(np.std(loss, ddof=1) / np.sqrt(n_reps))
        results.append((risk, se))

    result: NDArray[Any] = np.array(results, dtype=float)
    if is_single:
        return cast(NDArray[Any], result[0])
    return result


_Direction = str | int | np.integer | NDArray[Any]


def _resolved_labels(
    est_list: list[_Estimator],
    estimator_labels: Sequence[str] | None,
) -> list[str]:
    """Return one display label per estimator.

    Explicit labels are validated against the number of estimators.  Otherwise
    a callable's ``__name__`` is used, falling back to its position.

    """

    if estimator_labels is not None:
        labels = list(estimator_labels)
        if len(labels) != len(est_list):
            msg = "estimator_labels must match the number of estimators."
            raise ValueError(msg)
        return labels
    labels = []
    for index, est in enumerate(est_list):
        if isinstance(est, str):
            labels.append(est)
            continue
        if callable(est):
            name = getattr(est, "__name__", None)
            if isinstance(name, str):
                labels.append(name)
                continue
        labels.append(str(index))
    return labels


def _canonical_directions(
    directions: _Direction | Sequence[_Direction],
    d: NDArray[Any],
    b: NDArray[Any],
    direction_labels: Sequence[str] | None,
) -> list[tuple[str, NDArray[Any]]]:
    """Resolve direction specs into unit canonical-space directions.

    Each entry of ``directions`` is one of:

    * a name: ``"uniform"`` (``u* ~ 1``), ``"proportional"`` (``u* ~ sqrt(d)``,
      constant signal-to-noise ratio) or ``"inverse"`` (``u* ~ 1/sqrt(d)``);
    * an integer ``j``: the ``j``-th canonical axis, ordered by *decreasing*
      variance (``j = 0`` is the largest variance and ``j = -1`` the
      smallest);
    * an array of the original dimension: mapped to canonical space as
      ``u* ~ raw @ B.T``.

    Every direction is normalized so that ``||u*|| = 1``, so that the distance
    axis agrees with the canonical Euclidean norm of the mean.

    """

    p = d.shape[0]
    if isinstance(directions, (list, tuple)):
        items: Sequence[_Direction] = list(directions)
    else:
        items = [cast(_Direction, directions)]

    descend = np.argsort(d)[::-1]
    result: list[tuple[str, NDArray[Any]]] = []
    for index, raw in enumerate(items):
        if isinstance(raw, str):
            if raw == "uniform":
                u = np.ones(p)
            elif raw == "proportional":
                u = np.sqrt(d)
            elif raw == "inverse":
                u = 1.0 / np.sqrt(d)
            else:
                msg = f"Unknown direction '{raw}'."
                raise ValueError(msg)
            default_label = raw
        elif isinstance(raw, (int, np.integer)):
            axis = int(raw)
            if not -p <= axis < p:
                msg = f"Direction axis {axis} out of range for dimension {p}."
                raise ValueError(msg)
            u = np.zeros(p)
            u[descend[axis]] = 1.0
            default_label = f"axis {axis}"
        else:
            vec = np.asarray(raw, dtype=float)
            if vec.shape != (p,):
                msg = f"Direction vector must have shape ({p},), got {vec.shape}."
                raise ValueError(msg)
            u = vec @ b.T
            default_label = f"dir {index}"
        norm = float(np.linalg.norm(u))
        if norm <= 0.0:
            msg = f"Direction '{default_label}' has zero canonical norm."
            raise ValueError(msg)
        label = (
            direction_labels[index] if direction_labels is not None else default_label
        )
        result.append((label, u / norm))
    return result


def _as_magnitudes(distances: tuple[float, float, int] | ArrayLike) -> NDArray[Any]:
    """Expand a magnitude specification into an array of distances.

    A ``(start, stop, num)`` triple is expanded with ``np.linspace``; otherwise
    the argument is used as an array of magnitudes.

    """

    if (
        isinstance(distances, (list, tuple))
        and len(distances) == 3
        and isinstance(distances[2], (int, np.integer))
    ):
        start, stop, num = distances
        if int(num) < 1:
            msg = "The number of distances must be >= 1."
            raise ValueError(msg)
        return np.linspace(start, stop, int(num))
    arr = np.asarray(distances, dtype=float)
    if arr.ndim != 1 or arr.size == 0 or not np.all(np.isfinite(arr)):
        msg = "distances must be a finite 1-D array of magnitudes."
        raise ValueError(msg)
    if np.any(arr < 0.0):
        msg = "distances must be non-negative."
        raise ValueError(msg)
    return arr


def estimate_risk_curve(
    cov: ArrayLike,
    estimators: _Estimator | Sequence[_Estimator],
    *,
    Q: ArrayLike | None = None,
    directions: _Direction | Sequence[_Direction] = (
        "uniform",
        "proportional",
        "inverse",
    ),
    distances: tuple[float, float, int] | ArrayLike = (0.0, 10.0, 41),
    n_reps: int = 10_000,
    seed: int | np.random.Generator | None = None,
    estimator_labels: Sequence[str] | None = None,
    direction_labels: Sequence[str] | None = None,
    **kwargs: Any,
) -> list[dict[str, Any]]:
    """Sweep the estimated risk of estimators against the true mean.

    The mean is moved along one or more *directions* in canonical space, at
    increasing *distances* ``t = ||theta_star||``, and :func:`estimate_risk` is
    evaluated at each ``(direction, distance)`` pair.  The result is a list of
    records, one per ``(direction, distance, estimator)``, suitable for wrapping
    in ``pandas.DataFrame`` for e.g. a seaborn plot.

    Parameters
    ----------
    cov : array_like
        The known covariance matrix of ``x``, of shape ``(p, p)``.  Must be
        symmetric and positive definite.
    estimators : callable or str, or sequence of these
        The estimator(s) to evaluate; see :func:`estimate_risk`.
    Q : array_like, default=None
        The known loss matrix, of shape ``(p, p)``.  Defaults to the identity.
    directions : str, int, array_like, or sequence of these, default=\
('uniform', 'proportional', 'inverse')
        The directions along which to move the mean.  A name selects a built-in
        canonical-space direction (``"uniform"``, ``"proportional"`` ~
        ``sqrt(d)``, ``"inverse"`` ~ ``1/sqrt(d)``); an integer ``j`` selects
        the ``j``-th canonical axis ordered by decreasing variance (``j = 0``
        is the largest variance and ``j = -1`` the smallest); an array of shape
        ``(p,)`` is a raw direction
        in the original space, mapped to canonical space as ``u* ~ raw @ B.T``.
        A list or tuple combines several directions.  Every direction is
        normalized so that ``||u*|| = 1``.
    distances : array_like, or (start, stop, num), default=(0.0, 10.0, 41)
        The magnitudes ``t`` at which to evaluate the risk.  A ``(start, stop,
        num)`` triple is expanded with ``np.linspace``; otherwise the argument
        is used directly as an array of (non-negative) magnitudes.
    n_reps : int, default=10000
        Number of Monte Carlo draws per ``(direction, distance)`` pair.
    seed : int or numpy.random.Generator, default=None
        Seed for the random number generator, for reproducible results.
    estimator_labels : sequence of str, default=None
        Optional display labels, one per estimator, used as ``estimator`` in
        each record.  Defaults to the estimator ``__name__`` (or index).
    direction_labels : sequence of str, default=None
        Optional display labels, one per direction, used as ``direction`` in
        each record.  Defaults to the direction name, axis index, or position.
    **kwargs
        Additional keyword arguments passed to every estimator (as in
        :func:`estimate_risk`).

    Returns
    -------
    list of dict
        One record per ``(direction, distance, estimator)`` with keys
        ``direction``, ``distance``, ``mahalanobis``, ``estimator``, ``risk``,
        ``se`` and ``risk_ratio = risk / trace(Q @ cov)`` (the minimax risk of
        the raw estimator ``delta0 = x``, so a value <= 1 indicates minimaxity).
        ``mahalanobis = sqrt(theta^T cov^-1 theta)`` is the Mahalanobis distance
        of the true mean (same for every estimator at a given sweep point).

    Examples
    --------
    Evaluate two estimators along the uniform direction, ready for a plot.

    >>> import functools
    >>> import numpy as np
    >>> import nustattools.stats as s
    >>> records = s.estimate_risk_curve(
    ...     np.diag([1.0, 2.0, 3.0]),
    ...     [functools.partial(s.shrink, c=2.0), functools.partial(s.shrink, c=0.0)],
    ...     directions="uniform",
    ...     distances=(0.0, 10.0, 3),
    ...     n_reps=2000,
    ...     seed=0,
    ... )
    >>> len(records)
    6
    >>> all("distance" in r for r in records)
    True

    """

    if n_reps < 2:
        msg = "n_reps must be an integer >= 2."
        raise ValueError(msg)

    cv = np.asarray(cov)
    if cv.ndim != 2 or cv.shape[0] != cv.shape[1]:
        msg = f"cov must be a square matrix, got shape {cv.shape}."
        raise ValueError(msg)
    p = cv.shape[0]
    cova = _validate_sympd(cov, (p, p), "cov")
    qa = np.eye(p) if Q is None else _validate_sympd(Q, (p, p), "Q")

    baseline: float = float(np.trace(qa @ cova))
    b, binv, d = _canonicalize(cova, qa)
    pinv: NDArray[Any] = np.linalg.inv(cova)

    dirs = _canonical_directions(directions, d, b, direction_labels)
    _, est_list = _normalize_estimators(estimators)
    est_labels = _resolved_labels(est_list, estimator_labels)
    magnitudes = _as_magnitudes(distances)

    records: list[dict[str, Any]] = []
    for direction, u_star in dirs:
        for t in magnitudes:
            theta = (t * u_star) @ binv.T
            mahalanobis: float = float(np.sqrt(theta @ pinv @ theta))
            full = estimate_risk(
                theta,
                cova,
                est_list,
                Q=qa,
                n_reps=n_reps,
                seed=seed,
                **kwargs,
            )
            for est_label, (risk, se) in zip(est_labels, full, strict=True):
                records.append(
                    {
                        "direction": direction,
                        "distance": float(t),
                        "mahalanobis": mahalanobis,
                        "estimator": est_label,
                        "risk": float(risk),
                        "se": float(se),
                        "risk_ratio": float(risk) / baseline,
                    }
                )
    return records


_METHODS: dict[str, Callable[..., NDArray[Any]]] = {
    "berger": berger,
}


__all__ = ["berger", "estimate_risk", "estimate_risk_curve", "shrink"]
