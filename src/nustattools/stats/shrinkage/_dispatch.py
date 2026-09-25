"""The :func:`shrink` front-end and the estimator registry.

This private module implements the :func:`shrink` convenience front-end that
dispatches to a named shrinkage estimator, together with the ``_METHODS``
registry and the :func:`_resolve_method` lookup used by :func:`shrink` and the
risk-estimation helpers in :mod:`nustattools.stats.shrinkage._risk`.

"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from numpy.typing import ArrayLike, NDArray

from ._bayes import bayes, robust_bayes
from ._coordinate import minimax_bayes, tan, tan_bayes
from ._minimax import berger


def shrink(
    x: ArrayLike,
    cov: ArrayLike | None = None,
    *,
    Q: ArrayLike | None = None,
    method: str = "berger",
    offset: ArrayLike | None = None,
    dirs: ArrayLike | None = None,
    prior_cov: ArrayLike | None = None,
    **kwargs: Any,
) -> NDArray[Any]:
    """Shrink an observed multivariate normal mean towards an affine subspace.

    Convenience front-end that dispatches to a named shrinkage estimator after
    transforming the problem to canonical form (a lossless change of
    coordinates that makes the covariance diagonal and the loss the identity,
    so the estimator only has to shrink independent coordinates of varying
    variance).

    Parameters
    ----------
    x : array_like
        Observed data.  A single vector of shape ``(p,)`` or a stack of
        observations of shape ``(..., p)``.
    cov : array_like, default=None
        The known covariance matrix of ``x``, of shape ``(p, p)``.  Defaults
        to the identity matrix.
    Q : array_like, default=None
        The known loss matrix, of shape ``(p, p)``.  May be positive
        semi-definite; see the :mod:`nustattools.stats.shrinkage` module
        docstring for how the loss-free null space is handled.  Defaults to the
        identity.
    method : str, default="berger"
        Which estimator to use.  Available: ``"berger"``, ``"tan"``,
        ``"minimax_bayes"``, ``"tan_bayes"``, ``"robust_bayes"`` and ``"bayes"``.
    offset : array_like, default=None
        A point of shape ``(p,)`` towards which to shrink.  Defaults to zero.
    dirs : array_like, default=None
        A matrix of shape ``(p, k)`` whose columns span the affine direction
        of shrinkage.  If given, the estimate shrinks towards the affine
        subspace :math:`\\mathrm{offset} + \\operatorname{span}(\\mathrm{dirs})`.
        When ``Q`` is singular, the null space of ``Q``
        is added to the no-shrink subspace; see the
        :mod:`nustattools.stats.shrinkage` module docstring for the details.
    prior_cov : array_like, default=None
        The prior covariance matrix, of shape ``(p, p)``, in the same
        coordinates as ``x`` — the fixed covariance of the prior
        :math:`\\theta \\sim N(0, \\gamma \\Theta)`.  Defaults to
        :math:`Q^{-1}`
        (the current homoscedastic prior in canonical coordinates).  When
        given, it must be symmetric positive definite and diagonalizable in the
        canonical coordinates (automatic for ``cov`` proportional to
        :math:`Q^{-1}`); see the :mod:`nustattools.stats.shrinkage` module
        docstring.  Only the Bayes-rule estimators (:func:`~nustattools.stats.shrinkage.bayes`,
        :func:`~nustattools.stats.shrinkage.robust_bayes`, :func:`~nustattools.stats.shrinkage.tan_bayes`) and the gamma-based
        minimax estimators (:func:`~nustattools.stats.shrinkage.tan`, :func:`~nustattools.stats.shrinkage.minimax_bayes`) accept it.
        :func:`~nustattools.stats.shrinkage.berger`, which involves no prior, rejects it (and any
        ``gamma``): passing ``prior_cov`` with ``method="berger"`` (the
        default) raises a :class:`TypeError`.
    **kwargs
        Additional keyword arguments passed to the estimator, e.g.
        ``strength``, ``positive``, or ``gamma`` (for the Bayes-rule
        estimators).  See the :mod:`nustattools.stats.shrinkage` module
        docstring for the accepted forms of ``gamma``.

    Returns
    -------
    delta : numpy.ndarray
        The shrinkage estimate of the mean, with the same shape as ``x``.

    """

    fn = _resolve_method(method)
    if prior_cov is not None:
        kwargs["prior_cov"] = prior_cov
    return fn(x, cov, Q=Q, offset=offset, dirs=dirs, **kwargs)


_METHODS: dict[str, Callable[..., NDArray[Any]]] = {
    "berger": berger,
    "tan": tan,
    "minimax_bayes": minimax_bayes,
    "tan_bayes": tan_bayes,
    "robust_bayes": robust_bayes,
    "bayes": bayes,
}


def _resolve_method(name: str) -> Callable[..., NDArray[Any]]:
    """Look up a shrinkage estimator by name.

    Raises ``ValueError`` if *name* is not a registered method.

    """

    try:
        return _METHODS[name]
    except KeyError as e:
        msg = f"Unknown shrinkage method '{name}'."
        raise ValueError(msg) from e
