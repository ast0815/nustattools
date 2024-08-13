from __future__ import annotations

from typing import Any

import numpy as np
from numba import njit
from numpy.typing import ArrayLike, NDArray


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


def derate_covariance(
    cov: list[NDArray[Any]] | NDArray[Any],
    *,
    jacobian: ArrayLike | None = None,
    sigma: float = 3.0,
) -> float:
    """Derate the covariance of some data to account for unknown correaltions.

    See TODO: Ref to paper.

    Parameters
    ----------
    cov : ndarray or list of ndarrays
        The covariance matrix of the data or a list of covariances that add up
        to the total. Unknown covariances must be ``np.nan``.
    jacobian : ndarray, default=None
        Jacobian matrix of the model prediction wrt to the best-fit parameters.
    sigma : float, default=3.
        The desired confidence level up to which the derated covariance should
        be conservative, expressed in standard-normal stadard deviations. E.g.
        ``sigma=3.`` corresponds to ``CL=0.997``.

    Returns
    -------
    a : float
        The derating factor for the total covariance.

    """

    if isinstance(cov, list):
        covl = [np.asarray(item) for item in cov]
    else:
        covl = [np.asarray(cov)]

    if jacobian is None:
        jacobian = np.eye(covl[0].shape[0])

    return sigma / 3.0
