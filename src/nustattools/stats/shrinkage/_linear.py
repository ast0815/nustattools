"""Linear-transformation estimators in canonical form.

This private module implements the :func:`matmul` linear-transformation
estimator, which applies an arbitrary user-supplied matrix in the original
coordinates, together with its canonical-form implementation
(:func:`_matmul_canonical`) and the dominating-improvement helper
(:func:`_enhance_canonical`).

"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._core import _canonical_frame, _estimate, _validate


def _matmul_canonical(
    y: NDArray[Any],
    _d: NDArray[Any],
    *,
    A: NDArray[Any],
) -> NDArray[Any]:
    """Matrix multiply in canonical form.

    ``A`` has shape ``(p, p)`` and is the user-supplied matrix transformed to
    canonical coordinates (:math:`A^* = B A B^{-1}`), applied to the centered
    canonical data ``y`` as :math:`\\delta = A y`.

    """

    return cast(NDArray[Any], y @ A.T)


_ENHANCE_CONDITION_FAILED = (
    "`enhance=True` requires the Eldar (2006) dominance condition "
    "`lambda_max(D^{-1/2} (D (I - A^*)^T (I - A^*) D)^{1/2} D^{-1/2}) <= 1` "
    "to hold in the canonical coordinates (eq. (91)). It does not for "
    "this A, so the analytical improvement is not guaranteed to "
    "dominate and is not applied."
)


def _enhance_canonical(g: NDArray[Any], d: NDArray[Any]) -> NDArray[Any]:
    """Dominating linear improvement of ``g`` in the canonical coordinates.

    Applies the closed-form improvement of [Eldar2006]_ (Theorem 9) to the
    canonical ``(p, p)`` matrix ``g`` whose coordinates have canonical
    variances ``d``.  Writing :math:`E = (I - g)^T (I - g)` for the loss
    matrix of the bias and :math:`D = \\operatorname{diag}(d)` for the
    canonical covariance, it returns

    .. math::

        g^* = I - (D E D)^{1/2}\\, D^{-1},

    where a positive-semidefinite square root is taken.  Pointwise
    :math:`(I - g^*)^T (I - g^*) = E`, so the bias term is left unchanged
    while the variance :math:`\\operatorname{tr}(g D g^T)` is not increased
    provided the dominance condition

    .. math::

        \\lambda_{\\max}\\left(D^{-1/2} (D E D)^{1/2} D^{-1/2}\\right) \\le 1

    (eq. (91)) holds.  When the canonical variances are isotropic (``cov``
    proportional to :math:`Q^{-1}`) the construction reduces to the classical
    improvement of [Cohen1966]_ (Theorem 2.1), :math:`I - E^{1/2}`, which
    dominates unconditionally, so the condition is not enforced there.

    Parameters
    ----------
    g : numpy.ndarray
        The canonical transformation matrix of shape ``(p, p)``.
    d : numpy.ndarray
        The canonical variances of shape ``(p,)`` (usually ordered decreasing).

    Returns
    -------
    g_enh : numpy.ndarray
        The improved matrix of shape ``(p, p)``.

    Raises
    ------
    ValueError
        If the canonical variances are genuinely heteroscedastic and the
        dominance condition (91) of [Eldar2006]_ fails.

    """

    p = g.shape[0]
    c = np.eye(p) - g
    a = c.T @ c
    evals, evecs = np.linalg.eigh(np.diag(d) @ a @ np.diag(d))
    s = (evecs * np.sqrt(np.maximum(evals, 0.0))) @ evecs.T
    if not np.allclose(d, d[0]):
        # Condition (91): lambda_max(D^{-1/2} (D A D)^{1/2} D^{-1/2}) <= 1,
        # the D-weighted analogue of lambda_max(A) <= 1 in the isotropic case.
        winv = np.diag(1.0 / np.sqrt(d))
        cond = np.linalg.eigvalsh(winv @ s @ winv)[-1]
        if cond > 1.0 + 1e-9:
            raise ValueError(_ENHANCE_CONDITION_FAILED)
    return cast(NDArray[Any], np.eye(p) - s @ np.diag(1.0 / d))


def matmul(
    x: ArrayLike,
    cov: ArrayLike | None = None,
    *,
    Q: ArrayLike | None = None,
    A: ArrayLike,
    offset: ArrayLike | None = None,
    dirs: ArrayLike | None = None,
    enhance: bool = False,
) -> NDArray[Any]:
    """Apply a user-specified linear transformation to the data.

    Computes :math:`\\vec \\delta = \\vec o + A (\\vec x - \\vec o)` where
    :math:`\\vec o` is the ``offset``: like the shrinkage target of the other
    estimators, it shifts the coordinate zero, so the point is added back after
    applying ``A``.  With the default offset of zero the estimator is the plain
    linear map :math:`\\vec \\delta = A \\vec x`.

    ``A`` has shape ``(p, p)`` and need not have any special properties (e.g.
    invertibility, symmetry, or positive-definiteness). Ensuring it does
    something useful is the user's responsibility.

    Parameters
    ----------
    x : array_like
        Observed data.  A single vector of shape ``(p,)`` or a stack of
        observations of shape ``(..., p)``.  The transformation is applied to
        each observation over the last axis.
    cov : array_like, default=None
        The known covariance matrix of ``x``, of shape ``(p, p)``.  Must be
        symmetric and positive definite.  Defaults to the identity matrix.
    Q : array_like, default=None
        The known loss matrix, of shape ``(p, p)``.  Defaults to the identity.
    A : array_like
        The transformation matrix, of shape ``(p, p)``.  Must contain only
        finite values.
    offset : array_like, default=None
        The point of shape ``(p,)`` the transformation is applied around, i.e.
        it shifts the coordinate zero like the shrinkage target of the other
        estimators.  It is subtracted from ``x`` before applying ``A`` and
        added back afterwards.  Defaults to zero.
    dirs : array_like, default=None
        Not supported.  Must be ``None``; passing a value raises
        :class:`ValueError`.
    enhance : bool, default=False
        If ``True``, replace ``A`` by a linear estimator that dominates it in
        the canonical coordinates (the construction of [Eldar2006]_,
        Theorem 9): it leaves the bias term unchanged while not increasing the
        variance, provided a dominance condition holds.  See *Notes*.

    Returns
    -------
    delta : numpy.ndarray
        The transformed estimate, with the same shape as ``x``.

    Notes
    -----
    ``enhance`` replaces :math:`A^*` in the canonical coordinates by the
    dominating linear estimator of [Eldar2006]_ (Theorem 9).  Writing :math:`D
    = \\operatorname{diag}(d)` for the diagonal canonical covariance of the
    coordinates and :math:`E = (I - A^*)^T (I - A^*)` for the loss matrix of
    the bias, it returns

    .. math::

        A_{\\mathrm{enh}} = I - (D E D)^{1/2}\\, D^{-1}.

    Pointwise,

    .. math::

        (I - A_{\\mathrm{enh}})^T (I - A_{\\mathrm{enh}}) = E,

    so the bias term :math:`\\theta^T E \\theta` is left unchanged for every
    true mean :math:`\\theta` while the variance
    :math:`\\operatorname{tr}(A_{\\mathrm{enh}} D A_{\\mathrm{enh}}^T)` is not
    increased whenever the dominance condition holds:

    .. math::

        \\lambda_{\\max}\\left(D^{-1/2} (D E D)^{1/2} D^{-1/2}\\right) \\le 1
        \\qquad \\text{(Eldar 2006, eq. (91))}.

    Under this condition the estimate dominates :math:`A` in the quadratic
    risk.  When the canonical variances are isotropic (``cov`` proportional to
    :math:`Q^{-1}`) the construction reduces to the classical improvement of
    [Cohen1966]_ (Theorem 2.1), :math:`I - E^{1/2}`, which dominates
    unconditionally, so the condition is not enforced.  For a genuinely
    heteroscedastic problem with the condition violated, ``enhance=True``
    raises :class:`ValueError` instead of silently degrading the estimate. When
    :math:`A^*` is symmetric in the canonical coordinates with all eigenvalues
    in :math:`[0, 1]` (a shrinkage kernel), ``enhance`` is a no-op *only* if
    :math:`A^*` commutes with :math:`D`. Otherwise the construction still
    reduces its variance in the :math:`D`-metric.

    """

    if dirs is not None:
        msg = "matmul does not support dirs; pass dirs=None."
        raise ValueError(msg)
    aa = np.asarray(A, dtype=float)
    if aa.ndim != 2:
        msg = f"A must be a 2-D matrix, got {aa.ndim}-D."
        raise ValueError(msg)
    xa, cova, qa, p = _validate(x, cov, Q)
    if aa.shape != (p, p):
        msg = f"A must have shape ({p}, {p}), got {aa.shape}."
        raise ValueError(msg)
    if not np.all(np.isfinite(aa)):
        msg = "A must contain only finite values."
        raise ValueError(msg)
    # The offset shifts the coordinate zero: _estimate (like _estimate_pd for
    # the shrinkage estimators) centers the canonical data on the offset and
    # adds it back afterwards, yielding delta = offset + A @ (x - offset).
    b, binv, d, _pi = _canonical_frame(cova, qa, None)
    a_star = b @ aa @ binv
    if enhance:
        a_star = _enhance_canonical(a_star, d)
    return _estimate(
        xa,
        cov,
        Q,
        _matmul_canonical,
        A=a_star,
        offset=offset,
    )
