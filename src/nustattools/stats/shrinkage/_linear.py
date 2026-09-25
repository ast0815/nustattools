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
    x: NDArray[Any],
    _d: NDArray[Any],
    *,
    A: NDArray[Any],
) -> NDArray[Any]:
    """Matrix multiply in canonical form.

    ``A`` has shape ``(p, p)`` and is the user-supplied matrix transformed to
    canonical coordinates
    (:math:`A^* = B A B^{-1}`).  Applied as
    :math:`\\delta = A x` where ``x`` is the centered canonical data.

    """

    return cast(NDArray[Any], x @ A.T)


_ENHANCE_CONDITION_FAILED = (
    "enhance=True requires the Eldar (2006) dominance condition "
    "lambda_max(D^{-1/2} (D A D)^{1/2} D^{-1/2}) <= 1 to hold in the canonical "
    "coordinates (eq. (91)); it does not for this A, so the analytical "
    "improvement is not guaranteed to dominate and is not applied."
)


def _enhance_canonical(g: NDArray[Any], d: NDArray[Any]) -> NDArray[Any]:
    """Dominating linear improvement of ``g`` in the canonical coordinates.

    Applies the closed-form improvement of [Eldar2006]_ (Theorem 9) to the
    canonical ``(p, p)`` matrix ``g`` whose coordinates have canonical
    variances ``d``.  Writing :math:`E = (g - I)^T (g - I)` for the loss
    matrix of the bias and :math:`D = \\operatorname{diag}(d)` for the
    canonical covariance, it returns

    .. math::

        g^* = I - (D E D)^{1/2}\\, D^{-1},

    where a positive-semidefinite square root is taken.  Pointwise
    :math:`(g^* - I)^T (g^* - I) = E`, so the bias term is left unchanged
    while the variance :math:`\\operatorname{tr}(g D g^T)` is not increased
    provided the dominance condition

    .. math::

        \\lambda_{\\max}\\left(D^{-1/2} (D E D)^{1/2} D^{-1/2}\\right) \\le 1

    (eq. (91)) holds.  When the canonical variances are isotropic (``cov``
    proportional to :math:`Q^{-1}`) the construction reduces to the classical
    improvement of [Cohen1966]_ (Theorem 2.1), :math:`I - E^{1/2}`, which
    dominates unconditionally, so the condition is not enforced there.  In the
    general heteroscedastic case a violation raises :class:`ValueError` rather
    than silently degrading the estimate.

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
    c = g - np.eye(p)
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
    """Apply a user-specified linear transformation to centered data.

    Computes :math:`\\delta = A (x - \\mathrm{offset})` where ``A`` is a matrix
    in the original coordinate system.  Internally the problem is canonicalized
    (diagonal covariance, identity loss) and ``A`` is transformed to canonical
    coordinates :math:`A^* = B A B^{-1}` before application, so the result is
    independent of the coordinate choice.

    ``A`` has shape ``(p, p)`` and need not have any special properties (e.g.
    invertibility, symmetry, or positive-definiteness); ensuring it does
    something useful is the user's responsibility.

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
    A : array_like
        The transformation matrix, of shape ``(p, p)``.  Must contain only
        finite values.
    offset : array_like, default=None
        A point of shape ``(p,)`` subtracted from ``x`` before applying ``A``.
        Defaults to zero.
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
    dominating linear estimator of [Eldar2006]_ (Theorem 9).  Writing
    :math:`D = \\operatorname{diag}(d)` for the diagonal canonical covariance
    of the coordinates and
    :math:`E = (A^* - I)^T (A^* - I)` for the loss matrix of the bias, it
    returns

    .. math::

        A_{\\mathrm{enh}} = I - (D E D)^{1/2}\\, D^{-1},

    the positive-semidefinite square root of :math:`D E D`.  Pointwise,

    .. math::

        (A_{\\mathrm{enh}} - I)^T (A_{\\mathrm{enh}} - I) = E,

    so the bias term :math:`\\theta^T E \\theta` is left unchanged for every
    true mean :math:`\\theta` while the variance
    :math:`\\operatorname{tr}(A_{\\mathrm{enh}} D A_{\\mathrm{enh}}^T)` is not
    increased whenever the dominance condition holds:

    .. math::

        \\lambda_{\\max}\\left(D^{-1/2} (D E D)^{1/2} D^{-1/2}\\right) \\le 1
        \\qquad \\text{(Eldar 2006, eq. (91))}.

    Under this condition the estimate dominates :math:`A` exactly in the
    quadratic risk, not merely heuristically.  When the canonical variances
    are isotropic (``cov`` proportional to :math:`Q^{-1}`) the construction
    reduces to the classical improvement of [Cohen1966]_ (Theorem 2.1),
    :math:`I - E^{1/2}`,
    which dominates unconditionally, so the condition is not enforced.  For a
    genuinely heteroscedastic problem with the condition violated,
    ``enhance=True`` raises :class:`ValueError` instead of silently degrading
    the estimate.  When :math:`A` is already symmetric in the canonical
    coordinates with all eigenvalues in :math:`[0, 1]` (a shrinkage kernel),
    ``enhance`` is a no-op only if :math:`A^*` also commutes with :math:`D`,
    i.e. the estimator is admissible; otherwise the construction still reduces
    its variance in the :math:`D`-metric.

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
    # Handle offset ourselves: subtract before canonicalizing so that the
    # canonical estimator receives A @ (x - offset) without the offset being
    # added back (which _estimate_pd does for shrinkage estimators).
    if offset is not None:
        o = np.asarray(offset, dtype=float)
        if o.shape != (p,):
            msg = f"offset must have shape {(p,)}, got {o.shape}."
            raise ValueError(msg)
        xa = xa - o
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
    )
