from __future__ import annotations

import numpy as np
import pytest

import nustattools.stats as s
from nustattools.stats._regularize import (
    _MAX_ABS_RISK_RTOL,
)


def _make_diagonal_cov(k: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.diag(rng.uniform(0.5, 2.0, size=k))


def _make_correlated_cov(k: int, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    u = rng.normal(size=(k, 2))
    return u @ u.T + (k + 1) * np.eye(k)


def test_default_model_is_flat():
    x = np.array([1.0, 4.0, 1.0])
    cov = np.eye(3) * 0.5
    reg_x_flat, err_flat = s.regularize(x, cov, model=None)
    reg_x_ones, err_ones = s.regularize(x, cov, model=np.ones(3))
    assert np.array_equal(reg_x_flat, reg_x_ones)
    assert np.array_equal(err_flat, err_ones)


def test_return_shapes_1d():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    cov = _make_correlated_cov(5)
    reg_x, err = s.regularize(x, cov)
    assert reg_x.shape == x.shape
    assert err.shape == (2, 5)
    assert reg_x.dtype == float
    assert err.dtype == float


def test_return_shapes_batched():
    x = np.zeros((3, 4))
    x[0] = np.linspace(1, 4, 4)
    x[1] = np.linspace(4, 1, 4)
    x[2] = np.array([2.0, 2.0, 2.0, 2.0])
    cov = _make_correlated_cov(4)
    reg_x, err = s.regularize(x, cov)
    assert reg_x.shape == x.shape
    assert err.shape == (2, 3, 4)


def test_regularized_value_lies_within_original_error_bars():
    rng = np.random.default_rng(42)
    x = rng.normal(size=8)
    cov = _make_correlated_cov(8, seed=2)
    reg_x, err = s.regularize(x, cov, model=np.linspace(1, 8, 8))
    assert np.all(reg_x - err[0] <= x + 1e-10)
    assert np.all(x <= reg_x + err[1] + 1e-10)


def test_error_bars_nonnegative():
    rng = np.random.default_rng(7)
    x = rng.normal(size=6)
    cov = _make_diagonal_cov(6, seed=3)
    reg_x, err = s.regularize(x, cov, delta_chi2=2.5)
    assert np.all(err[0] >= -1e-10)
    assert np.all(err[1] >= -1e-10)
    assert reg_x.shape == x.shape


@pytest.mark.parametrize("delta_chi2", [0.1, 1.0, 5.0, 10.0])
def test_delta_chi2_caps_mahalanobis_distance(delta_chi2):
    rng = np.random.default_rng(11)
    x = rng.normal(size=6)
    cov = _make_correlated_cov(6, seed=4)
    reg_x, _ = s.regularize(x, cov, delta_chi2=delta_chi2)
    sinv = np.linalg.inv(cov)
    diff = reg_x - x
    m2 = float(diff @ sinv @ diff)
    # The max_abs_risk Newton refinement is driven by _MAX_ABS_RISK_RTOL, so
    # the cap is delta_chi2 * (1 + rtol) up to a small numerical margin.
    assert m2 <= delta_chi2 * (1.0 + _MAX_ABS_RISK_RTOL) + 1e-6


def test_very_small_delta_chi2_keeps_data():
    """A tiny delta_chi2 caps the shift so tightly that reg_x ≈ x."""
    x = np.array([1.0, 2.0, 3.0, 4.0])
    cov = np.eye(4)
    reg_x, _ = s.regularize(x, cov, delta_chi2=1e-6)
    assert np.allclose(reg_x, x, atol=1e-3)


def test_large_delta_chi2_shrinks_heavily():
    """A large delta_chi2 lets the shift grow, pulling data toward the prior mean."""
    rng = np.random.default_rng(5)
    x = rng.normal(size=6) * 5.0
    cov = np.eye(6)
    reg_x_small, _ = s.regularize(x, cov, delta_chi2=0.01)
    reg_x_large, _ = s.regularize(x, cov, delta_chi2=1000.0)
    # The smaller cap keeps the result closer to x; the larger cap shifts it.
    assert float(np.linalg.norm(reg_x_small - x)) < float(
        np.linalg.norm(reg_x_large - x)
    )


def test_deterministic_output():
    rng = np.random.default_rng(13)
    x = rng.normal(size=5)
    cov = _make_diagonal_cov(5, seed=6)
    out1 = s.regularize(x, cov, model=np.array([1.0, 2.0, 3.0, 2.0, 1.0]))
    out2 = s.regularize(x, cov, model=np.array([1.0, 2.0, 3.0, 2.0, 1.0]))
    assert np.array_equal(out1[0], out2[0])
    assert np.array_equal(out1[1], out2[1])


def test_invalid_delta_chi2_raises():
    x = np.array([1.0, 2.0, 3.0])
    cov = np.eye(3)
    with pytest.raises(ValueError, match="delta_chi2 must be > 0"):
        s.regularize(x, cov, delta_chi2=0.0)
    with pytest.raises(ValueError, match="delta_chi2 must be > 0"):
        s.regularize(x, cov, delta_chi2=-1.0)


def test_invalid_cov_shape_raises():
    x = np.array([1.0, 2.0, 3.0])
    cov = np.eye(4)
    with pytest.raises(ValueError, match="cov shape"):
        s.regularize(x, cov)
    cov_rect = np.zeros((3, 5))
    with pytest.raises(ValueError, match="cov must be square"):
        s.regularize(x, cov_rect)


def test_invalid_cov_symmetry_raises():
    x = np.array([1.0, 2.0, 3.0])
    cov = np.array([[1.0, 0.5, 0.0], [0.3, 1.0, 0.0], [0.0, 0.0, 1.0]])
    with pytest.raises(ValueError, match="symmetric"):
        s.regularize(x, cov)


def test_invalid_cov_positive_definite_raises():
    x = np.array([1.0, 2.0, 3.0])
    cov = np.array([[1.0, 0.0, 0.0], [0.0, -0.5, 0.0], [0.0, 0.0, 1.0]])
    with pytest.raises(ValueError, match="positive definite"):
        s.regularize(x, cov)
    cov_singular = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
    with pytest.raises(ValueError, match="positive definite"):
        s.regularize(x, cov_singular)


def test_invalid_model_shape_raises():
    x = np.array([1.0, 2.0, 3.0])
    cov = np.eye(3)
    with pytest.raises(ValueError, match="model must have shape"):
        s.regularize(x, cov, model=np.array([1.0, 2.0]))


def test_invalid_finite_inputs_raise():
    x = np.array([1.0, np.inf, 3.0])
    cov = np.eye(3)
    with pytest.raises(ValueError, match="all inputs must be finite"):
        s.regularize(x, cov)
    x = np.array([1.0, 2.0, 3.0])
    cov = np.array([[np.nan, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    with pytest.raises(ValueError, match="all inputs must be finite"):
        s.regularize(x, cov)
    x = np.array([1.0, 2.0, 3.0])
    cov = np.eye(3)
    with pytest.raises(ValueError, match="all inputs must be finite"):
        s.regularize(x, cov, model=np.array([1.0, np.nan, 3.0]))


def test_invalid_x_dimension_raises():
    cov = np.eye(3)
    with pytest.raises(ValueError, match="x must have at least one dimension"):
        s.regularize(np.float64(1.0), cov)
