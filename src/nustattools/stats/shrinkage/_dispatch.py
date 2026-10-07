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
from ._linear import matmul
from ._minimax import berger


def shrink(
    x: ArrayLike,
    cov: ArrayLike | None = None,
    *,
    Q: ArrayLike | None = None,
    method: str = "tan",
    offset: ArrayLike | None = None,
    dirs: ArrayLike | None = None,
    prior_cov: ArrayLike | None = None,
    **kwargs: Any,
) -> NDArray[Any]:
    """Shrink an observed multivariate normal mean towards an affine subspace.

    Convenience front-end that dispatches to a named shrinkage estimator.

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
        The known loss matrix, of shape ``(p, p)``.  May be positive
        semi-definite; see the :mod:`nustattools.stats.shrinkage` module
        docstring for how the loss-free null space is handled.  Defaults to the
        identity, i.e. squared-error loss.
    method : str, default="tan"
        Which estimator to use.  Available: :func:`~nustattools.stats.shrinkage.bayes`,
        :func:`~nustattools.stats.shrinkage.berger`,
        :func:`~nustattools.stats.shrinkage.minimax_bayes`,
        :func:`~nustattools.stats.shrinkage.robust_bayes`,
        :func:`~nustattools.stats.shrinkage.matmul`,
        :func:`~nustattools.stats.shrinkage.tan`, and
        :func:`~nustattools.stats.shrinkage.tan_bayes`.
        The method name matches the name of the estimator function.
    offset : array_like, default=None
        A point of shape ``(p,)`` towards which to shrink.  Defaults to zero,
        i.e. shrinking towards the origin.
    dirs : array_like, default=None
        A matrix of shape ``(p, k)`` whose columns span the affine direction
        of shrinkage.  See the :mod:`nustattools.stats.shrinkage` module
        docstring for details.
    prior_cov : array_like, default=None
        The prior covariance matrix :math:`\\Gamma`, of shape ``(p, p)``, in
        the same coordinates as ``x``, for the Gaussian prior
        :math:`\\theta \\sim N(\\vec o, \\gamma \\Gamma)`.  Defaults to
        :math:`Q^{-1}` (a homoscedastic prior in canonical coordinates).  See
        the :mod:`nustattools.stats.shrinkage` module docstring for
        details.
    **kwargs
        Additional keyword arguments passed to the estimator, e.g.
        ``strength``, ``positive`` or ``gamma``.  See the
        :mod:`nustattools.stats.shrinkage` module docstring for the accepted
        forms of ``gamma``.

    Returns
    -------
    delta : numpy.ndarray
        The shrinkage estimate of the mean, with the same shape as ``x``.

    Examples
    --------

    >>> import numpy as np
    >>> import nustattools.stats.shrinkage as sh
    >>> rng = np.random.default_rng(0)
    >>> x = rng.normal(size=5)
    >>> sh.shrink(x).shape
    (5,)

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
    "matmul": matmul,
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
