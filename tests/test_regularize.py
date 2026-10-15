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


def test_cov_symmetry_gate_is_scale_invariant():
    # The same relative asymmetry must be rejected whatever the unit system:
    # the old absolute tolerance (atol=1e-8) wrongly accepted this matrix once
    # every entry was below ~1e-8.
    x = np.array([1.0, 2.0, 3.0])
    cov = np.array([[1.0, 0.5, 0.0], [0.3, 1.0, 0.0], [0.0, 0.0, 1.0]])
    for scale in (1.0, 1e-10, 1e-20):
        with pytest.raises(ValueError, match="symmetric"):
            s.regularize(x, scale * cov)


def test_tiny_exactly_symmetric_cov_accepted():
    # The relative gate must not over-reject: an exactly symmetric positive
    # definite covariance passes however small its scale.
    x = np.array([1.0, 2.0, 3.0]) * 1e-10
    cov = np.diag([1.0, 0.5, 2.0]) * 1e-20
    data, _ = s.regularize(x, cov, delta_chi2=1e-2)
    assert data.shape == (3,)


def test_zero_bins_returns_input_unregularized():
    # k == 0 is supported throughout (`bayes`' documented limit is the
    # identity), so regularize must return the inputs untouched instead of
    # dying in reductions over the empty covariance.
    x = np.zeros(0)
    cov = np.zeros((0, 0))
    data, err = s.regularize(x, cov, delta_chi2=1.0)
    assert data is not x
    assert data.shape == (0,)
    assert err.shape == (2, 0)
    batched, batched_err = s.regularize(np.zeros((2, 3, 0)), cov, delta_chi2=1.0)
    assert batched.shape == (2, 3, 0)
    assert batched_err.shape == (2, 2, 3, 0)


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


def test_zero_model_entry_raises():
    # The model divides the data, so a zero entry overflowed to inf and the
    # failure surfaced much later as a bogus "prior covariance matrix must be
    # symmetric".
    x = np.array([1.0, 2.0, 3.0])
    cov = np.eye(3)
    with pytest.raises(ValueError, match="model must not contain zero entries"):
        s.regularize(x, cov, model=np.array([0.0, 1.0, 2.0]))


def test_invalid_x_dimension_raises():
    cov = np.eye(3)
    with pytest.raises(ValueError, match="x must have at least one dimension"):
        s.regularize(np.float64(1.0), cov)


def test_shift_has_no_model_direction_component():
    # The covariance-metric `dirs` machinery keeps the model (normalisation)
    # direction at its data value, so the shift between the data and the
    # regularized result must be Mahalanobis-orthogonal to the model direction.
    rng = np.random.default_rng(0)
    u = rng.normal(size=(5, 4))
    cov = u @ u.T + (np.trace(u @ u.T) / 5) * np.eye(5)
    x = rng.normal(size=5) * np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    model = rng.uniform(1.0, 9.0, size=5)
    reg_x, _ = s.regularize(x, cov, model=model)
    sinv = np.linalg.inv(cov)
    assert np.allclose((reg_x - x) @ sinv @ model, 0.0)


def test_data_on_model_shape_is_unchanged():
    # Data already in model shape must be returned untouched: the shape error
    # is zero, so the delta_chi2 budget has nothing to spend it on (previously
    # the diagonal-inflation prior let the budget drain into the
    # normalisation).
    template = np.array([2.0, 3.0, 1.0])
    cov = np.diag(template)
    for delta_chi2 in (0.01, 1.0, 10.0):
        reg_x, _ = s.regularize(
            template.copy(), cov, model=template, delta_chi2=delta_chi2
        )
        assert np.allclose(reg_x, template)


def test_within_budget_data_coincides_with_model_shape():
    # When the Mahalanobis distance to the model shape is smaller than
    # delta_chi2, the result is the shape itself (the closest point of the
    # shape manifold), not a shape-preserving pull at the full budget.
    x = np.array([2.0, 3.0])
    reg_x, _ = s.regularize(x, np.eye(2))
    # Closest flat point to [2, 3] is [2.5, 2.5], at distance^2 = 0.5 <= 1.
    assert np.allclose(reg_x, [2.5, 2.5])


def test_budget_smaller_than_shape_error_moves_partially():
    # With the budget below the distance to the model shape, the shift spends
    # exactly delta_chi2 in Mahalanobis distance, entirely in the shape
    # directions (the component along the model direction is kept).
    x = np.array([2.0, 3.0])
    reg_x, _ = s.regularize(x, np.eye(2), delta_chi2=0.1)
    # The Newton refinement stops at a 1e-3 relative residual, so the shift is
    # delta_chi2 only to within that tolerance.
    assert np.allclose(
        np.sum((reg_x - x) ** 2),
        0.1,
        rtol=0.0,
        atol=0.1 * _MAX_ABS_RISK_RTOL + 1e-9,
    )
    assert np.allclose(np.mean(reg_x), np.mean(x))


def test_flat_data_survives_large_budget():
    # Flat data with a default flat model: previously regularize returned
    # [0, 0] for delta_chi2 = 10 (the whole budget was spent on the
    # normalisation); now the data is the model shape and stays put.
    x = np.array([2.0, 2.0])
    reg_x, _ = s.regularize(x, np.eye(2), delta_chi2=10.0)
    assert np.allclose(reg_x, x)


def test_sum_preserved_for_diagonal_template_covariance():
    # With cov = diag(model) the covariance-metric projection onto the model
    # direction sums to the data sum (1^T cov^-1 model is constant), so both
    # the kept component and the contrast residual preserve the total.
    template = np.array([10.0, 12.0, 15.0, 11.0, 9.0, 13.0])
    cov = np.diag(template)
    reg_on, _ = s.regularize(template.copy(), cov, model=template)
    assert np.isclose(reg_on.sum(), template.sum())
    tilted = template.copy()
    tilted[3:] *= 1.5
    reg_tilted, _ = s.regularize(tilted, cov, model=template)
    assert np.isclose(reg_tilted.sum(), tilted.sum())


def test_single_bin_returns_input():
    # A single bin has no shape degree of freedom: the model direction spans
    # everything, so the regularized data is the data itself.  Previously this
    # died with a raw numpy.linalg.LinAlgError from inverting the zero penalty.
    x = np.array([2.0])
    reg_x, err = s.regularize(x, np.eye(1))
    assert np.allclose(reg_x, x)
    reg_x, err = s.regularize(x, np.eye(1), model=[3.0])
    assert np.allclose(reg_x, x)
    assert err.shape == (2, 1)
