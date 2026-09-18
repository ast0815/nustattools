"""Convenience functions to regularize data sets with a likelihood described by
a mean and covairance matrix."""

import numpy as np
from .shrinkage import bayes


def regularize(x, cov, model=None, delta_chi2=1.0):
    """Regularize a data set with given MLE and covariance.

    If `model` is given, the data is regularized towards the model shape,
    otherwise towards a flat prior. By default, the shift of the data is limited to a delta
    chi^2 of 1, so the regularized data is guaranteed to be within all
    single-bin 68% CI error bars.

    Notes
    -----

    Applies a post-hoc regularization as described in [arXiv:2207.02125]. The
    penalty matrix `Q` is chosen as [insert math from code]. The regularization
    strength is chosen such that the squared M-distance of the regularised
    result from the unregularized result is at most equal to `delta_chi2`. If
    the model shape is closer to the unregularized data than this distance, the
    regularized data will have exactly the model shape.

    The implementation uses the [insert reference to `bayes` shrinkage method]
    using the `max_abs_risk` empirical gamma factor.

    """

    k = x.shape[-1]
    if model is None:
        model = np.ones(k)

    # This is the opinionated part: Penalise shape differences from the model,
    # but not normalisation differences.
    shape_diff_projection = (np.eye(k) - 1 / k) @ np.diag(1 / model)
    penalty_matrix = shape_diff_projection.T @ shape_diff_projection

    # Need to inflate diagonal a tiny bit to make matrix invertible
    penalty_matrix += np.diag(np.diag(penalty_matrix) * 1e-6)

    # The prior used for the bayes shrinkage
    prior_cov = np.linalg.inv(penalty_matrix)

    # Get the regularized data points, shifted according to the Bayes rule to a
    # delta chi^2 of 1. This ensures that no matter what, the regularized data
    # is is within all error bars of the original covariance matrix.
    reg_x = bayes(
        x, cov, Q=np.linalg.inv(cov), prior_cov=prior_cov, gamma=f"max_abs_risk({delta_chi2}, 1e-3)"
    )

    # Calculate the size of asymmetric error bars for easy plotting
    err = np.sqrt(np.diag(cov))
    shift = reg_x - x
    err_lo = err + shift
    err_up = err - shift

    return reg_x, np.array( (err_lo, err_up) )

__all__ = ["regularize"]
