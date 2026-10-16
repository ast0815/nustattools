"""Convenience functions to regularize data sets with a likelihood described by
a mean and covariance matrix."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .shrinkage import bayes

# Relative-residual tolerance for the ``max_abs_risk`` gamma factory's Newton
# refinement, matching the closed-form start to machine precision.
_MAX_ABS_RISK_RTOL = 1e-3

_MSG_COV_SHAPE = "cov must be square with shape (k, k)."
_MSG_COV_SQUARE_MISMATCH = "cov shape {got} does not match x trailing dimension {k}."
_MSG_X_DIM = "x must have at least one dimension."
_MSG_COV_NOT_SYM = "cov must be symmetric within float tolerance."
_MSG_COV_NOT_PD = "cov must be positive definite."
_MSG_MODEL_SHAPE = "model must have shape (k,) matching x trailing dimension."
_MSG_MODEL_NONZERO = "model must not contain zero entries; x is divided by the model."
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
    data is limited to a ``delta_chi2`` of 1, so the regularized data lies
    within all single-bin 68% CI error bars of the original data -- to within
    the 1e-3 relative residual of the Newton refinement that sets the
    per-observation prior scale (see Notes).

    Parameters
    ----------
    x : array-like
        The data vector(s) to regularize, shape ``(..., k)``.
    cov : array-like
        The covariance matrix describing the uncertainties of ``x``, shape
        ``(k, k)``. Must be symmetric and positive definite.
    model : array-like, optional
        The model shape to regularize towards, shape ``(k,)``. Defaults to a
        flat prior (all ones).
    delta_chi2 : float, default=1.0
        The upper bound on the squared Mahalanobis distance of the
        regularized result from the unregularized data. Must be positive.

    Returns
    -------
    reg_x : numpy.ndarray
        The regularized data, shape ``(..., k)``.
    err : numpy.ndarray
        Asymmetric error bars around ``reg_x``, ready for plotting, with
        shape ``(2, *reg_x.shape)``: the leading axis holds the lower
        (``err[0]``) and upper (``err[1]``) error bar. ``err[0]`` is the
        lower error bar (``reg_x - err[0]`` is the lower edge of the
        original 1-sigma interval relative to the regularized point) and
        ``err[1]`` is the upper error bar (``reg_x + err[1]`` is the upper
        edge). The original error bar centered on ``x`` is shifted so the
        regularized value lies inside it.

    Raises
    ------
    ValueError
        If ``cov`` is not square, not symmetric, not positive definite, or
        has a shape incompatible with ``x``; if ``model`` is given and does
        not match the trailing dimension of ``x`` or contains a zero entry;
        if ``delta_chi2`` is not positive; or if any input contains
        non-finite values.

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
    matrix ``G`` (named apart from the loss matrix ``Q`` of
    :mod:`nustattools.stats.shrinkage`) penalizes shape differences from the
    model.
    The multiplication

    .. math::

        \\vec y = D_M^{-1} \\vec{x}

    where `D_M` is a diagonal matrix with the entries of the ``model`` as the
    diagonal elements, first scales all data points relative to the model. So
    data with all ``y_i`` identical has the same shape as the model.

    Then, with ``k = x.shape[-1]``, the penalty term is set to the squared
    differences between data points:

    .. math::

        \\sum_{i<j} (y_i - y_j)^2 = \\sum_{i<j} y_i^2 + y_j^2 - 2 y_i y_j
            = \\vec{y}^T G \\vec{y},

    with

    .. math::

        G = k I_k - \\vec{1} \\vec{1}^T = \\left( \\begin{array}{cccc}
                k-1 & -1 & -1 & \\cdots \\\\
                -1 & k-1 & -1 & \\cdots \\\\
                -1 & -1 & k-1 & \\cdots \\\\
                \\vdots & \\vdots & \\vdots & \\ddots
            \\end{array}\\right) .

    This penalises differences between all possible combinations of data bins,
    so it works with unordered or N-dimensional data.  The penalty matrix is
    only positive *semi*-definite: its null space is the model direction, which
    is exactly the normalization that must not be penalised.

    The regularization strength is chosen such that the squared
    Mahalanobis distance of the regularized result from the unregularized
    data is at most ``delta_chi2``. If the model shape is already closer
    to the unregularized data than this distance, the regularized data
    coincides with the model shape.
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
    # The symmetry tolerance is relative to the matrix norm so the verdict does
    # not depend on the unit system: with an absolute floor, a matrix that is
    # asymmetric by a factor of two in relative terms is accepted once every
    # entry falls below it.  An empty covariance (k == 0) has no norm.
    if not np.allclose(
        cov_arr,
        cov_arr.T,
        rtol=1e-5,
        atol=1e-8 * (float(np.max(np.abs(cov_arr))) if cov_arr.size else 0.0),
    ):
        raise ValueError(_MSG_COV_NOT_SYM)
    eigvals = np.linalg.eigvalsh(cov_arr)
    if eigvals.size and float(eigvals.min()) <= 0:
        # Vacuously positive definite when there are no eigenvalues (k == 0);
        # the reduced check for that case is the early return below.
        raise ValueError(_MSG_COV_NOT_PD)

    if model is None:
        model_arr = np.ones(k)
    else:
        model_arr = np.asarray(model, dtype=float)
        if model_arr.shape != (k,):
            raise ValueError(_MSG_MODEL_SHAPE)
        if not np.all(np.isfinite(model_arr)):
            raise ValueError(_MSG_FINITE)
        if np.any(model_arr == 0.0):
            # 1/model overflows to inf here and surfaces much later as a
            # bogus "prior covariance matrix must be symmetric".
            raise ValueError(_MSG_MODEL_NONZERO)

    if k == 0:
        # Zero bins: the (already validated) estimator limit is the identity
        # (`bayes` returns the input) and `err` has nothing to quote, so
        # return without running the shape machinery below, which would die
        # in reductions over the empty covariance.
        return x_arr.copy(), np.empty((2, *x_arr.shape))

    # Penalise shape differences from the model, but not normalisation
    # differences.  The penalty matrix is positive semi-definite (positive
    # definite on the shape subspace, exactly zero along the normalised model
    # direction), so its inverse needs a completion along that direction.  The
    # `dirs` machinery below keeps the model direction at its data value and
    # never sees the completion, so its magnitude is immaterial; it is chosen
    # on the scale of the proper penalty eigenvalues (whose sum is the trace).
    model_hat = model_arr / np.linalg.norm(model_arr)
    penalty_matrix = (
        np.diag(1.0 / model_arr) @ (k * np.eye(k) - 1.0) @ np.diag(1.0 / model_arr)
    )
    completion = np.trace(penalty_matrix) / k
    if not completion > 0:  # k == 1: the penalty is the zero matrix
        completion = 1.0
    prior_cov = np.linalg.inv(
        penalty_matrix + completion * np.outer(model_hat, model_hat)
    )

    reg_x = bayes(
        x_arr,
        cov_arr,
        Q=np.linalg.inv(cov_arr),
        prior_cov=prior_cov,
        dirs=model_hat[:, np.newaxis],
        gamma=f"max_abs_risk({delta_chi2}, {_MAX_ABS_RISK_RTOL})",
    )

    err = np.sqrt(np.diag(cov_arr))
    shift = reg_x - x_arr
    err_lo = err + shift
    err_up = err - shift

    return reg_x, np.array((err_lo, err_up))


__all__ = ["regularize"]
