"""Convenience functions to regularize data sets with a likelihood described by
a mean and covariance matrix."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .shrinkage import bayes

# Relative diagonal jitter added to the shape-difference loss matrix to keep
# it numerically invertible when the shape projection is rank-deficient
# (e.g. when the input is already on the model shape).
_DIAGONAL_INFLATION = 1e-6

# Relative-residual tolerance for the ``max_abs_risk`` gamma factory's Newton
# refinement, matching the closed-form start to machine precision.
_MAX_ABS_RISK_RTOL = 1e-3

_MSG_COV_SHAPE = "cov must be square with shape (k, k)."
_MSG_COV_SQUARE_MISMATCH = "cov shape {got} does not match x trailing dimension {k}."
_MSG_X_DIM = "x must have at least one dimension."
_MSG_COV_NOT_SYM = "cov must be symmetric within float tolerance."
_MSG_COV_NOT_PD = "cov must be positive definite."
_MSG_MODEL_SHAPE = "model must have shape (k,) matching x trailing dimension."
_MSG_DELTA_CHI2 = "delta_chi2 must be > 0."
_MSG_FINITE = "all inputs must be finite."


def regularize(
    x: ArrayLike,
    cov: ArrayLike,
    model: ArrayLike | None = None,
    delta_chi2: float = 1.0,
) -> tuple[NDArray[Any], NDArray[Any]]:
    """Regularize a data set with given MLE and covariance.

    If ``model`` is given, the data is regularized towards the model shape,
    otherwise towards a flat prior (all ones). By default, the shift of the
    data is limited to a ``delta_chi2`` of 1, so the regularized data is
    guaranteed to lie within all single-bin 68% CI error bars of the original
    data.

    Parameters
    ----------
    x : array-like
        The data vector(s) to regularize, shape ``(..., k)``.
    cov : array-like
        The covariance matrix describing the uncertainties of ``x``, shape
        ``(k, k)``. Must be symmetric and positive definite.
    model : array-like, optional
        The model shape to regularize towards, shape ``(k,)``. Defaults to a
        flat prior (all ones), which preserves the overall normalization but
        pulls every component towards equality.
    delta_chi2 : float, default=1.0
        The upper bound on the squared Mahalanobis distance of the
        regularized result from the unregularized data. Must be positive.

    Returns
    -------
    reg_x : numpy.ndarray
        The regularized data, shape ``(..., k)``.
    err : numpy.ndarray
        Asymmetric error bars around ``reg_x`` shaped like ``reg_x``, ready
        for plotting. ``err[0]`` is the lower error bar (``reg_x - err[0]``
        is the lower edge of the original 1-sigma interval relative to the
        regularized point) and ``err[1]`` is the upper error bar
        (``reg_x + err[1]`` is the upper edge). The original error bar
        centered on ``x`` is shifted so the regularized value lies inside it.

    Raises
    ------
    ValueError
        If ``cov`` is not square, not symmetric, not positive definite, or
        has a shape incompatible with ``x``; if ``model`` is given and does
        not match the trailing dimension of ``x``; if ``delta_chi2`` is not
        positive; or if any input contains non-finite values.

    Examples
    --------

    >>> import numpy as np
    >>> from nustattools.stats import regularize
    >>> rng = np.random.default_rng(0)
    >>> x = rng.normal(size=5)
    >>> cov = np.eye(5)
    >>> reg_x, err = regularize(x, cov, model=np.array([1.0, 2.0, 3.0, 2.0, 1.0]))
    >>> reg_x.shape
    (5,)
    >>> err.shape
    (2, 5)
    >>> bool(np.all(reg_x - err[0] <= x))
    True
    >>> bool(np.all(x <= reg_x + err[1]))
    True

    Notes
    -----

    Applies a post-hoc regularization as described in [Koch2022]_. The penalty
    matrix ``Q`` penalizes shape differences from the model.
    The multiplication

    .. math::

        \\vec y = D_M^{-1} \\vec{x}

    where `D_M` is a diagonal matrix with the entries of the ``model`` as the
    diagonal elements, first scales all data points relative to the model, so
    that data with all ``y_i`` identical has the same shape as the model.

    Then, with ``k = x.shape[-1]``, the projection

    .. math::

        P \\vec y = \\left(I_k - \\tfrac{1}{k} \\mathbf{1}\\mathbf{1}^T\\right) \\vec y

    yields a vector that is the difference of the actual ``y`` with the model
    with the average scaling factor over all data points.

    The penalty term is the L_2 norm of that vector so the full penalty matrix is

    .. math::

        Q = D_M^{-1} P^T P D_M^{-1}

    The regularization strength is chosen such that the squared
    Mahalanobis distance of the regularized result from the unregularized
    data is at most ``delta_chi2``. If the model shape is already closer
    to the unregularized data than this distance, the regularized data
    coincides with the model shape.

    The implementation uses the Bayes shrinkage estimator
    :func:`nustattools.stats.shrinkage.bayes` with the risk metric being the
    data covariance, and the inverse of ``Q`` as the prior covariance. In order
    to do this, a small relative diagonal jitter is added to make ``Q``
    numerically invertible.     The ``max_abs_risk`` empirical gamma factory is
    then used to set the per-observation prior scale from the data and ensure that
    the shrunk data is moved by the required ``delta_chi2``.

    """

    if delta_chi2 <= 0:
        raise ValueError(_MSG_DELTA_CHI2)

    x_arr = np.asarray(x, dtype=float)
    cov_arr = np.asarray(cov, dtype=float)
    if x_arr.ndim < 1:
        raise ValueError(_MSG_X_DIM)
    k = x_arr.shape[-1]
    if cov_arr.shape != (k, k):
        if cov_arr.ndim != 2 or cov_arr.shape[0] != cov_arr.shape[1]:
            raise ValueError(_MSG_COV_SHAPE)
        raise ValueError(_MSG_COV_SQUARE_MISMATCH.format(got=cov_arr.shape, k=k))
    if not np.all(np.isfinite(x_arr)) or not np.all(np.isfinite(cov_arr)):
        raise ValueError(_MSG_FINITE)
    if not np.allclose(cov_arr, cov_arr.T, rtol=1e-5, atol=1e-8):
        raise ValueError(_MSG_COV_NOT_SYM)
    eigvals = np.linalg.eigvalsh(cov_arr)
    if float(eigvals.min()) <= 0:
        raise ValueError(_MSG_COV_NOT_PD)

    if model is None:
        model_arr = np.ones(k)
    else:
        model_arr = np.asarray(model, dtype=float)
        if model_arr.shape != (k,):
            raise ValueError(_MSG_MODEL_SHAPE)
        if not np.all(np.isfinite(model_arr)):
            raise ValueError(_MSG_FINITE)

    # Penalise shape differences from the model, but not normalisation
    # differences: project out the all-ones direction from diag(1/model).
    shape_diff_projection = (np.eye(k) - 1.0 / k) @ np.diag(1.0 / model_arr)
    penalty_matrix = shape_diff_projection.T @ shape_diff_projection

    # Inflate the diagonal a tiny bit to keep the matrix numerically
    # invertible when the projection is rank-deficient.
    penalty_matrix = penalty_matrix + np.diag(
        np.diag(penalty_matrix) * _DIAGONAL_INFLATION
    )

    prior_cov = np.linalg.inv(penalty_matrix)

    reg_x = bayes(
        x_arr,
        cov_arr,
        Q=np.linalg.inv(cov_arr),
        prior_cov=prior_cov,
        gamma=f"max_abs_risk({delta_chi2}, {_MAX_ABS_RISK_RTOL})",
    )

    err = np.sqrt(np.diag(cov_arr))
    shift = reg_x - x_arr
    err_lo = err + shift
    err_up = err - shift

    return reg_x, np.array((err_lo, err_up))


__all__ = ["regularize"]
