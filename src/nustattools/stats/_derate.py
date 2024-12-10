"""Derate a covariance to accommodate unknown correlations."""

from __future__ import annotations

from typing import Any
from warnings import warn

import numpy as np
from numba import njit
from numpy.typing import ArrayLike, NDArray
from scipy.linalg import block_diag, sqrtm, svd
from scipy.stats import chi2

from .gx2.functions import gx2inv


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


def make_positive_definite(cov: NDArray[Any]) -> NDArray[Any]:
    """Make a covariance matrix positive definite.

    Ensures that it is symmetric and adds small numbers to the diagonal until
    it is positive definite.
    """

    if np.all(cov == 0):
        msg = "Matrix block is all zeros! At least specify the diagonal elements!"
        raise ValueError(msg)

    # Make sure matrix is symmetric
    c = (cov + cov.T) / 2

    # Make sure it is positive definite
    power = -18
    epsilon = 0.0
    while True:
        try:
            t = c + np.eye(len(c)) * epsilon
            # Check that it can be Cholesky decomposed and inverted
            np.linalg.cholesky(t)
            np.linalg.inv(t)
        except np.linalg.LinAlgError:
            power += 1
            epsilon = np.min(np.diag(c)[np.diag(c) > 0]) * 10**power
        else:
            c = c + np.eye(len(c)) * epsilon
            break

    if power > -18:
        msg = f"Had to increase covariance diagonal by {epsilon * 100} to make it positive definite."
        warn(msg, stacklevel=3)

    return np.asarray(c)


def get_whitening_transform(
    cov: NDArray[Any],
    transform: str = "zca_aligned",
    projection: NDArray[Any] | None = None,
) -> tuple[NDArray[Any], NDArray[Any]]:
    """Get the blockwise whitening matrix and inverse."""

    tokens = transform.split("_")
    if len(tokens) > 2:
        msg = f"Unknown whitening transform '{transform}'."
        raise ValueError(msg)

    blocks = get_blocks(cov)
    W_l = []
    Wi_l = []
    i = 0
    for n in blocks:
        c = cov[i : i + n, :][:, i : i + n]

        if np.all(c == 0):
            msg = "Matrix block is all zeros! At least specify the diagonal elements!"
            raise ValueError(msg)

        c = make_positive_definite(c)
        # Determine transformation
        if tokens[0] in ("mahalanobis", "zca"):
            sc = sqrtm(c)
            W = np.linalg.inv(sc)
            Wi = sc
            W_l.append(W)
            Wi_l.append(Wi)
        elif tokens[0] in ("zca-cor",):
            V = np.sqrt(np.diag(c))
            Vi = 1 / V
            V = np.diag(V)
            Vi = np.diag(Vi)
            cor = Vi @ c @ Vi
            sc = sqrtm(cor)
            W_l.append(np.linalg.inv(sc) @ Vi)
            Wi_l.append(V @ sc)
        elif tokens[0] in ("cholesky",):
            W = np.linalg.cholesky(np.linalg.inv(c)).T
            W_l.append(W)
            Wi_l.append(np.linalg.inv(W))
        else:
            msg = f"Unknown whitening transform '{transform}'."
            raise ValueError(msg)

        i += n

    W = block_diag(*W_l)
    Wi = block_diag(*Wi_l)

    Rt_l = []
    if len(tokens) == 2:
        if tokens[1] == "aligned":
            # Align whitened coordinates with image space
            P = W @ projection @ Wi
            i = 0
            for n in blocks:
                A = P[i : i + n, :][:, i : i + n]
                U, _, _ = np.linalg.svd(A)
                Rt_l.append(U)

                i += n

            Rt = block_diag(*Rt_l)
            Wi = Wi @ Rt
            W = Rt.T @ W
        else:
            msg = f"Unknown whitening transform '{transform}'."
            raise ValueError(msg)

    return np.asarray(W), np.asarray(Wi)


def derate_covariance(
    cov: list[NDArray[Any]] | NDArray[Any],
    *,
    jacobian: ArrayLike | None = None,
    sigma: float = 3.0,
    return_dict: dict[str, Any] | None = None,
    whitening: str = "zca_aligned",
    method: str = "gx2",
    precision: float = 0.01,
    max_batch_size: int = 10_000,
    goodness_of_fit: bool = False,
) -> float:
    """Derate the covariance of some data to account for unknown correlations.

    See [Koch2024]_.

    Parameters
    ----------
    cov : numpy.ndarray or list of numpy.ndarray
        The covariance matrix of the data or a list of covariances that add up
        to the total. Unknown covariance blocks must be ``np.nan``. Off
        diagonal blocks may only be ``0`` or ``np.nan``. Diagonal blocks must
        not be ``np.nan``.
    jacobian : numpy.ndarray, default=None
        Jacobian matrix of the model prediction wrt the best-fit parameters.
        If no jacobian is provided, the identity matrix will be used.
    sigma : float, default=3.
        The desired confidence level up to which the derated covariance should
        be conservative, expressed in standard-normal standard deviations. E.g.
        ``sigma=3.`` corresponds to ``CL=0.997``.
    return_dict : dict, optional
        If specified, the nightmare covariance and thrown data samples are
        added to this dictionary for detailed studies outside the function.
    whitening : str, default="zca_aligned"
        Specify which method to use for the whitening transform.
    method : str, default="gx2"
        Either ``gx2`` or ``mc``. The former calculates the p-value directly,
        the latter uses Monte Carlo methods.
    precision : float, default=0.01
        If the derating factor is calculated using numerical sampling (MC), this
        parameter determines how many samples to throw. Lower values mean more
        samples.
    max_batch_size : int, default=10_000
        If the derating factor is calculated using numerical sampling (MC), this
        parameter determines how many samples to are thrown at once. This is
        repeated until the total number for the required precision is reached.
    goodness_of_fit : bool, default=False
        The derating factor for the Goodness of Fit test is different from the
        derating factor for model parameter estimation. If this parameter is
        set to `True`, The provided model `jacobian` will be used to construct
        the null space of the model in the data space. This is then used to
        calculate the necessary derating factor for the Goodness of Fit or
        Composite Hypothesis test.

    Returns
    -------
    a : float
        The derating factor for the total covariance.

    Notes
    -----

    The basic available whitening transforms are:

    ``mahalanobis`` or ``zca``
        ``W = sqrtm(inv(cov))``

    ``zca-cor``
        ``W = sqrtm(inv(cor)) @ diag(1/sqrt(diag(cov)))``

    ``cholesky``
        ``W = cholesky(inv(cov)).T``

    See [Kessy2015]_.

    If ``_aligned`` is appended to a basic transform, an additional rotation
    matrix is prepended, which aligns the whitened coordinate axes with the
    model parameter space given by `jacobian`. See [Koch2024]_.

    References
    ----------

    .. [Kessy2015] Kessy, Agnan / Lewin, Alex / Strimmer, Korbinian
       "Optimal whitening and decorrelation",
       The American Statistician 2018, Vol. 72, No. 4, pp. 309-314 , Vol. 72, No. 4,
       Informa UK Limited, p. 309-314, https://arxiv.org/abs/1512.00809

    .. [Koch2024] L. Koch "Hypothesis tests and model parameter estimation on
       data sets1 with missing correlation information",
       https://arxiv.org/abs/2410.22333

    """

    # Make sure we have a list of covariances
    if isinstance(cov, list):
        covl = [np.asarray(item) for item in cov]
    else:
        covl = [np.asarray(cov)]

    # Assumed covariance
    # All unknown elements set to 0.
    # Make sure they are positive definite
    cov_0_l = [make_positive_definite(np.nan_to_num(c)) for c in covl]
    cov_0 = np.sum(cov_0_l, axis=0)
    cov_0_inv = np.linalg.inv(cov_0)

    # If no Jacobian is specified, assume we cover full parameter space
    n_data = covl[0].shape[0]
    if jacobian is None:
        jacobian = np.eye(n_data)
    else:
        jacobian = np.asarray(jacobian)

    if goodness_of_fit:
        # Calculate base of null space using SVD
        U, _, _ = svd(jacobian)
        # First k columns are image space, rest are null space
        k = jacobian.shape[1]
        jacobian = np.asarray(U[:, k:])

    # Projection matrix in original coordinates
    S = make_positive_definite(cov_0)
    Si = np.linalg.inv(S)
    A = jacobian
    Q = np.linalg.inv(make_positive_definite(A.T @ Si @ A)) @ A.T @ Si
    P = A @ Q

    # Target matrix
    T = Si @ P

    # Transform to whitened coordinate systems and calculate "nightmare_cov"
    # covariance, then transform back
    nightmare_cov = np.zeros_like(cov_0)
    for c, c0 in zip(covl, cov_0_l):
        # Whitened correlation is identity matrix
        cor = np.eye(len(c0))
        # Set unknowns back to NaN
        cor[np.isnan(c)] = np.nan

        # Determine the whitening transform for each covariance
        W, Wi = get_whitening_transform(c, transform=whitening, projection=P)

        # Target matrix in whitened coordinates
        Txi = Wi.T @ T @ Wi

        # Assumed total covariance in whitened coordinates
        cor_nightmare = fill_max_correlation(cor, Txi)

        # Transform back to non-whitened coordinates
        cov_nightmare = Wi @ cor_nightmare @ Wi.T
        nightmare_cov = nightmare_cov + cov_nightmare

    # Desired significance
    alpha = chi2.sf(sigma**2, df=1)

    # Assumed critical value in parameter space
    n_param = jacobian.shape[1]
    crit_0 = chi2.isf(alpha, df=n_param)

    # Actual critical value in parameter space
    crit_nightmare: np.floating[Any] | float = 0.0
    if method == "gx2":
        # Generalised chi-squared
        Si_theta = A.T @ Si @ A
        V_theta = Q @ nightmare_cov @ Q.T
        H = Si_theta @ V_theta
        weights = np.real(np.linalg.eigvals(H))
        # Get rid of zeros and numerical artefacts
        weights_l = sorted(weights[weights > 1e-2])
        del weights
        ndofs = list(np.ones_like(weights_l, dtype=int))
        # Combine identical weights
        i = 0
        while i + 1 < len(weights_l):
            if abs(weights_l[i] - weights_l[i + 1]) < 1e-3:
                weights_l[i] = (
                    weights_l[i] * ndofs[i] + weights_l[i + 1] * ndofs[i + 1]
                ) / (ndofs[i] + ndofs[i + 1])
                ndofs[i] += ndofs[i + 1]
                del weights_l[i + 1]
                del ndofs[i + 1]
            else:
                i += 1
        tol = alpha / 1000.0
        crit_nightmare = gx2inv(
            alpha,
            np.array(weights_l, dtype=float),
            np.array(ndofs, dtype=int),
            np.zeros_like(weights_l),
            0,
            0,
            AbsTol=tol,
            RelTol=tol,
            side="upper",
        )
    elif method == "mc":
        # Nightmare critical value from random throws
        rng = np.random.default_rng()
        # Matrix that solves the least squares problem
        # Uses assumed covariance
        parameter_estimator = (
            np.linalg.inv(jacobian.T @ cov_0_inv @ jacobian) @ jacobian.T @ cov_0_inv
        )
        # Assumed covariance in parameter space
        assumed_parameter_cov = parameter_estimator @ cov_0 @ parameter_estimator.T
        assumed_parameter_cov_inv = np.linalg.inv(
            make_positive_definite(assumed_parameter_cov)
        )
        # Actual nightmare_cov covariance
        nightmare_parameter_cov = (
            parameter_estimator @ nightmare_cov @ parameter_estimator.T
        )
        # Estimate necessary precision
        # var = alpha(1-alpha) / (n f(crit_0)**2) =!= (crit_0 * rel_error)**2
        n_throws = (
            int(
                (alpha * (1.0 - alpha))
                / (chi2.pdf(crit_0, df=n_param) ** 2 * (crit_0 * precision) ** 2)
            )
            + 1
        )
        # Throw in batches and average to avoid huge memory footprint
        n_batches = (n_throws // max_batch_size) + 1
        batch_size = (n_throws // n_batches) + 1
        if batch_size * alpha < 2:
            msg = f"Batch size of {batch_size} is too small for significance level {alpha}."
            raise RuntimeError(msg)
        throws = dist = None
        for _ in range(n_batches):
            del throws, dist
            throws = rng.multivariate_normal(
                mean=[0.0] * n_param, cov=nightmare_parameter_cov, size=batch_size
            )

            dist = np.einsum("ai,ij,aj->a", throws, assumed_parameter_cov_inv, throws)

            crit_nightmare += -np.quantile(-dist, alpha)

        crit_nightmare /= n_batches
    else:
        msg = f"Unknown method {method}!"
        raise ValueError(msg)

    derate = crit_nightmare / crit_0

    if derate.ndim > 0:
        derate = derate[0]

    derate = max(1.0, derate)

    if return_dict is not None:
        return_dict["nightmare_cov"] = nightmare_cov
        if method == "mc":
            return_dict["throws"] = throws
        return_dict["W"] = W
        return_dict["Wi"] = Wi
        return_dict["Q"] = Q

    return float(derate)


__all__ = ["derate_covariance"]
