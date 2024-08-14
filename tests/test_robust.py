from __future__ import annotations

import numpy as np

import nustattools.robust as r


def test_derate_unity_covariance():
    cov = np.eye(7)
    assert np.abs(r.derate_covariance(cov, sigma=1, accuracy=0.001) - 1.0) < 0.01


def test_derate_single_covariance():
    cov = np.array(
        [
            [2.0, 1.0, np.nan, np.nan],
            [1.0, 2.0, np.nan, np.nan],
            [np.nan, np.nan, 3.0, 2.0],
            [np.nan, np.nan, 2.0, 3.0],
        ]
    )
    assert np.abs(r.derate_covariance(cov, sigma=2) - 1.63) < 0.05
