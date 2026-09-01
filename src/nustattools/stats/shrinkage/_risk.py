"""Monte-Carlo risk estimation helpers for the shrinkage estimators.

This private module contains :func:`estimate_risk` and
:func:`estimate_risk_curve`, which estimate the quadratic loss of a shrinkage
estimator (or compare several on shared samples) by Monte Carlo, together with
their private helpers (:func:`_normalize_estimators`, :func:`_resolved_labels`,
:func:`_canonical_directions`, :func:`_as_magnitudes`).

"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._core import _canonicalize, _validate_sympd
from ._estimators import _METHODS

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
        ``functools.partial(berger, strength=0.5)``.
    Q : array_like, default=None
        The known loss matrix, of shape ``(p, p)``.  Defaults to the identity.
    n_reps : int, default=10000
        Number of Monte Carlo draws.  Must be at least 2 so that the standard
        error is finite.
    seed : int or numpy.random.Generator, default=None
        Seed for the random number generator, for reproducible results.
    **kwargs
        Additional keyword arguments passed to every estimator (e.g.
        ``strength``, ``positive``, ``offset``, ``dirs``).

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
    ...     functools.partial(s.shrink, strength=1.0),
    ...     functools.partial(s.shrink, strength=0.0),
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

    result = _risk_from_samples(x, theta_arr, cova, qa, est_list, n_reps, **kwargs)
    if is_single:
        return cast(NDArray[Any], result[0])
    return result


def _risk_from_samples(
    x: NDArray[Any],
    theta: NDArray[Any],
    cova: NDArray[Any],
    qa: NDArray[Any],
    est_list: list[_Estimator],
    n_reps: int,
    **kwargs: Any,
) -> NDArray[Any]:
    """Estimate risk and standard error from a batch of pre-drawn samples.

    ``x`` has shape ``(n_reps, p)`` and holds draws from ``N(theta, cova)``.
    Each estimator yields a row ``[risk, standard error]``.  The quadratic
    loss is evaluated as ``sum(d * (d @ qa), axis=1)``, which avoids the
    ``(p, p)`` einsum factorization and is faster for the typical small
    ``p`` of a risk sweep.

    """

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
        d = delta - theta
        loss = np.sum(d * (d @ qa), axis=1)
        risk = float(np.mean(loss))
        se = float(np.std(loss, ddof=1) / np.sqrt(n_reps))
        results.append((risk, se))
    return np.array(results, dtype=float)


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
    increasing *distances* ``t = ||theta_star||``, and the risk of each
    estimator is evaluated at each ``(direction, distance)`` pair (as in
    :func:`estimate_risk`).  The result is a list of records, one per
    ``(direction, distance, estimator)``, suitable for wrapping in
    ``pandas.DataFrame`` for e.g. a seaborn plot.

    The *canonical space* is reached by the lossless change of coordinates that
    diagonalizes the covariance to ``D = diag(d)`` and reduces the loss to the
    identity; ``theta_star`` and ``u*`` below are the mean and direction in
    those coordinates.

    Because the covariance is fixed across the sweep, the Monte Carlo noise is
    drawn *once* as ``N(0, cov)`` and translated to each sweep point.  All
    points therefore share the same draws (common random numbers): this avoids
    re-drawing per point and sharply reduces the sampling noise on the risk
    curve, so differences between neighbouring points are less polluted by
    Monte Carlo error.

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
        Number of Monte Carlo draws.  The same ``n_reps`` draws are shared by
        every ``(direction, distance)`` pair.
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
    ...     [functools.partial(s.shrink, strength=1.0), functools.partial(s.shrink, strength=0.0)],
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
    # Draw the noise once and translate it to each sweep point.  Because the
    # covariance is fixed, ``N(theta, cov)`` is ``N(0, cov) + theta``, so all
    # sweep points share the *same* Monte Carlo draws (common random numbers).
    # This both avoids re-drawing per point and makes neighbouring points'
    # estimates strongly correlated, greatly reducing the sampling noise on the
    # risk curve.
    gen = np.random.default_rng(seed)
    x0 = gen.multivariate_normal(np.zeros(p), cova, size=n_reps)
    for direction, u_star in dirs:
        for t in magnitudes:
            theta = (t * u_star) @ binv.T
            mahalanobis: float = float(np.sqrt(theta @ pinv @ theta))
            full = _risk_from_samples(
                x0 + theta, theta, cova, qa, est_list, n_reps, **kwargs
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
