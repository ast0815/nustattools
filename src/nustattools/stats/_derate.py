"""Derate a covariance to accommodate unknown correlations."""

from __future__ import annotations

from typing import Any

import numpy as np
from numba import njit
from numpy.typing import ArrayLike, NDArray
from scipy.linalg import block_diag, sqrtm
from scipy.stats import chi2


@njit()  # type: ignore[misc]
def _fix(cov: NDArray[Any]) -> NDArray[Any]:
    changed = True
    while changed:
        changed = False
        for k in range(len(cov)):
            for j in range(k + 1, len(cov)):
                # pivot point k, m
                pivot = np.sign(cov[k, j])
                if not np.isfinite(pivot):
                    continue
                if np.all(np.isfinite(cov[k, :])) and np.all(np.isfinite(cov[:, j])):
                    continue
                for m in range(len(cov)):
                    if (np.isfinite(cov[k, m]) and not np.isfinite(cov[m, j])) and (
                        cov[k, m] != 0 or pivot != 0
                    ):
                        cov[j, m] = cov[m, j] = np.sign(cov[k, m] * pivot)
                        changed = True
                    elif (not np.isfinite(cov[k, m]) and np.isfinite(cov[m, j])) and (
                        cov[m, j] != 0 or pivot != 0
                    ):
                        cov[m, k] = cov[k, m] = np.sign(cov[m, j] * pivot)
                        changed = True
    return cov


def fill_max_correlation(cor: ArrayLike, target: ArrayLike) -> NDArray[Any]:
    """Fill the correlation matrix with elements to achieve maximum correlation.

    Try to match the signs in `target`.

    Only replaces elements in `cor` that are ``np.nan``.
    """

    cora = np.array(cor)
    target = np.asarray(target)

    # Check and fix connections to other elements
    cora = _fix(cora)

    priority = np.unravel_index(
        np.argsort(np.abs(target), axis=None)[::-1], target.shape
    )

    for i, j in zip(*priority):
        if np.isfinite(cora[i, j]):
            continue

        # Set the new element
        t = 1 if target[i, j] == 0 else np.sign(target[i, j])

        cora[i, j] = cora[j, i] = t

        # Check and fix connections to other elements
        cora = _fix(cora)

    return cora


def get_blocks(cov: NDArray[Any]) -> list[int]:
    """Determine the sizes of known block matrices.

    Assumes the matrix is symmetric.

    """

    blocks = []
    n = 1

    # Find blocks by looking at NaNs
    nans = np.isnan(cov)
    trans = np.any(nans[:-1] ^ nans[1:], axis=1)

    # Find blocks by looking at zeros on diagonal
    zeros = np.diag(cov) == 0
    trans |= zeros[:-1] ^ zeros[1:]

    for j in range(cov.shape[0] - 1):
        if trans[j]:
            blocks.append(n)
            n = 1
        else:
            n += 1

    # Add last block
    blocks.append(n)

    return blocks


def get_whitening_transform(cov: NDArray[Any]) -> NDArray[Any]:
    """Get the blockwise whitening matrix."""

    blocks = get_blocks(cov)
    W_l = []
    i = 0
    for n in blocks:
        c = cov[i : i + n, :][:, i : i + n]
        if np.all(c == 0):
            W_l.append(np.zeros_like(c))
        else:
            W_l.append(np.linalg.inv(sqrtm(c)))
        i += n

    return np.asarray(block_diag(*W_l))


def derate_covariance(
    cov: list[NDArray[Any]] | NDArray[Any],
    *,
    jacobian: ArrayLike | None = None,
    sigma: float = 3.0,
    accuracy: float = 0.01,
) -> float:
    """Derate the covariance of some data to account for unknown correlations.

    See TODO: Ref to paper.

    Parameters
    ----------
    cov : numpy.ndarray or list of numpy.ndarray
        The covariance matrix of the data or a list of covariances that add up
        to the total. Unknown covariance blocks must be ``np.nan``. Off
        diagonal blocks may only be ``0'' or ``np.nan''. Diagonal blocks must
        not be ``np.nan''.
    jacobian : numpy.ndarray, default=None
        Jacobian matrix of the model prediction wrt the best-fit parameters.
    sigma : float, default=3.
        The desired confidence level up to which the derated covariance should
        be conservative, expressed in standard-normal standard deviations. E.g.
        ``sigma=3.`` corresponds to ``CL=0.997``.
    accuracy : float, default=0.01
        The derating factor is calculated using numerical sampling. This parameter
        determines how many samples to throw. Lower values mean more samples.

    Returns
    -------
    a : float
        The derating factor for the total covariance.

    """

    # Make sure we have a list of covariances
    if isinstance(cov, list):
        covl = [np.asarray(item) for item in cov]
    else:
        covl = [np.asarray(cov)]

    # Assumed covariance
    # All unknown elements set to 0.
    cov_0_l = [np.nan_to_num(c) for c in covl]
    cov_0 = np.sum(cov_0_l, axis=0)
    cov_0_inv = np.linalg.inv(cov_0)

    # If no Jacobian is specified, assume we cover full parameter space
    n_data = covl[0].shape[0]
    if jacobian is None:
        jacobian = np.eye(n_data)
    else:
        jacobian = np.asarray(jacobian)

    # Transform to whitened coordinate systems and calculate "nightmare_cov"
    # covariance, then transform back
    nightmare_cov = np.zeros_like(cov_0)
    for c, c0 in zip(covl, cov_0_l):
        # Determine the whitening transform for each covariance
        W = get_whitening_transform(c)
        # NaNs turn everything into NaN, use zeroed covs
        cor = W @ c0 @ W.T
        # Set unknowns back to NaN
        cor[np.isnan(c)] = np.nan
        # Set almost 0 to 0
        cor[np.abs(cor) < 1e-15] = 0.0
        # Assumed total covariance in whitened coordinates
        S = W @ cov_0 @ W.T
        Si = np.linalg.inv(S)
        A = W @ jacobian
        Q = np.linalg.inv(A.T @ Si @ A) @ A.T @ Si
        P = A @ Q
        T = Si @ P
        cor_nightmare = fill_max_correlation(cor, T)
        Wi = np.linalg.inv(W)
        cov_nightmare = Wi @ cor_nightmare @ Wi.T
        nightmare_cov = nightmare_cov + cov_nightmare

    # Desired significance
    alpha = chi2.sf(sigma**2, df=1)

    # Assumed critical value in parameter space
    n_param = jacobian.shape[1]
    crit_0 = chi2.isf(alpha, df=n_param)

    # Nightmare critical value from random throws
    rng = np.random.default_rng()
    # Matrix that solves the least squares problem
    # Uses assumed covariance
    parameter_estimator = (
        np.linalg.inv(jacobian.T @ cov_0_inv @ jacobian) @ jacobian.T @ cov_0_inv
    )
    # Assumed covariance in parameter space
    assumed_parameter_cov = parameter_estimator @ cov_0 @ parameter_estimator.T
    assumed_parameter_cov_inv = np.linalg.inv(assumed_parameter_cov)
    # Actual nightmare_cov covariance
    nightmare_parameter_cov = (
        parameter_estimator @ nightmare_cov @ parameter_estimator.T
    )
    # Estimate necessary precision
    # var = alpha(1-alpha) / (n f(crit_0)**2) =!= (crit_0 * rel_error)**2
    n_throws = (
        int(
            (alpha * (1.0 - alpha))
            / (chi2.pdf(crit_0, df=n_param) ** 2 * (crit_0 * accuracy) ** 2)
        )
        + 1
    )
    throws = rng.multivariate_normal(
        mean=[0.0] * n_param, cov=nightmare_parameter_cov, size=n_throws
    )

    dist = np.einsum("ai,ij,aj->a", throws, assumed_parameter_cov_inv, throws)
    crit_nightmare = -np.quantile(-dist, alpha)

    derate = crit_nightmare / crit_0

    derate = max(1.0, derate)

    return float(derate)


__all__ = ["derate_covariance"]
