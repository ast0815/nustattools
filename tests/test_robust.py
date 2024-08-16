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
    assert np.abs(r.derate_covariance(cov, sigma=2) - 1.29) < 0.1


def test_derate_multi_covariance():
    cov1 = np.array(
        [
            [2.0, 1.0, np.nan, np.nan],
            [1.0, 2.0, np.nan, np.nan],
            [np.nan, np.nan, 3.0, 2.0],
            [np.nan, np.nan, 2.0, 3.0],
        ]
    )
    cov2 = np.eye(4)
    cov3 = np.array(
        [
            [2.0, 1.0, 0.0, np.nan],
            [1.0, 2.0, 0.0, np.nan],
            [0.0, 0.0, 3.0, np.nan],
            [np.nan, np.nan, np.nan, 3.0],
        ]
    )
    assert np.abs(r.derate_covariance([cov1, cov2, cov3], sigma=2) - 1.16) < 0.1


def test_derate_single_covariance_fit():
    cov = np.block(
        [
            [np.eye(5), np.full((5, 5), np.nan)],
            [np.full((5, 5), np.nan), np.eye(5)],
        ]
    )
    A = np.array(
        [
            [2, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, -1, 1, -1, 1, -1, 1, -1, 1, -1],
        ],
        dtype=float,
    ).T
    A = A / np.sqrt(np.sum(A**2, axis=0, keepdims=True))
    assert (
        np.abs(r.derate_covariance(cov, jacobian=A, sigma=3, accuracy=0.001) - 1.827)
        < 0.02
    )


def test_derate_multi_covariance_fit():
    cov1 = np.array(
        [
            [2.0, 1.0, np.nan, np.nan],
            [1.0, 2.0, np.nan, np.nan],
            [np.nan, np.nan, 3.0, 2.0],
            [np.nan, np.nan, 2.0, 3.0],
        ]
    )
    cov2 = np.eye(4)
    cov3 = np.array(
        [
            [2.0, 1.0, 0.0, np.nan],
            [1.0, 2.0, 0.0, np.nan],
            [0.0, 0.0, 3.0, np.nan],
            [np.nan, np.nan, np.nan, 3.0],
        ]
    )
    A = np.array(
        [
            [2, 1, 1, 1],
            [1, -1, 1, -1],
        ],
        dtype=float,
    ).T
    A = A / np.sqrt(np.sum(A**2, axis=0, keepdims=True))
    assert (
        np.abs(r.derate_covariance([cov1, cov2, cov3], jacobian=A, sigma=2) - 1.54)
        < 0.1
    )
