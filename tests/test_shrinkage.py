from __future__ import annotations

import numpy as np
import pytest

import nustattools.stats as s


def rng():
    return np.random.default_rng(42)


def _berger_general_formula(x, cov, q, c):
    """Closed-form Berger estimator in the general (non-canonical) form.

    From [Tan2015]_, Section 3.2: ``delta_{A,c}`` with ``A = Q^-1 Sigma^-1``,
    i.e. ``delta = x - c AX / (x^T A^T Q A x)``.

    """
    x = np.asarray(x, dtype=float)
    cov = np.asarray(cov, dtype=float)
    q = np.asarray(q, dtype=float)
    sinv = np.linalg.inv(cov)
    qinv = np.linalg.inv(q)
    a = qinv @ sinv
    s = float(x @ sinv @ qinv @ sinv @ x)
    return x - c * (a @ x) / s


def test_berger_homoscedastic_equals_james_stein():
    # For D = sigma^2 I and Q = I, Berger with c = p - 2 reduces to
    # (1 - (p-2) sigma^2 / ||x||^2) x (James-Stein).
    sigma = 2.0
    x = rng().normal(size=7)
    expected = (1 - 5 * sigma**2 / np.sum(x**2)) * x
    np.testing.assert_allclose(
        s.berger(x, cov=sigma**2 * np.eye(7), positive=False), expected
    )


def test_berger_positive_part_preserves_sign():
    # Positive part should never flip the sign of any coordinate, only
    # shrink it (possibly to zero).
    x = rng().normal(size=5) * 0.05
    delta = s.berger(x, cov=np.eye(5))
    assert np.all(delta * x >= -1e-12)


def test_berger_zero_shrinkage_is_identity():
    x = rng().normal(size=5)
    np.testing.assert_allclose(s.berger(x, cov=np.eye(5), c=0.0), x)


def test_berger_general_matches_closed_form():
    # The canonicalized computation (default Q=I) must agree with the direct
    # general-form Berger estimator.
    a = rng().normal(size=(5, 5))
    cov = a @ a.T + np.eye(5)
    x = rng().normal(size=5)
    c = 3.0
    np.testing.assert_allclose(
        s.berger(x, cov=cov, positive=False, c=c),
        _berger_general_formula(x, cov, np.eye(5), c),
    )


def test_berger_general_Q_matches_closed_form():
    a = rng().normal(size=(5, 5))
    cov = a @ a.T + np.eye(5)
    b = rng().normal(size=(5, 5))
    q = b @ b.T + np.eye(5)
    x = rng().normal(size=5)
    c = 2.5
    np.testing.assert_allclose(
        s.berger(x, cov=cov, Q=q, positive=False, c=c),
        _berger_general_formula(x, cov, q, c),
    )


def test_berger_broadcasting_shapes():
    p = 5
    x = rng().normal(size=(4, 3, p))
    out = s.berger(x, cov=np.eye(p))
    assert out.shape == (4, 3, p)
    for idx in np.ndindex(4, 3):
        np.testing.assert_allclose(
            out[idx], s.berger(x[idx], cov=np.eye(p)), rtol=1e-12
        )


def test_berger_minimaxity():
    # Berger's (non-positive-part) estimator is minimax: its risk is never
    # greater than tr(cov), for any true mean. Evaluate at theta = 0 where the
    # signal is strongest.
    p = 5
    cov = np.diag([4.0, 2.0, 1.0, 0.5, 0.25])
    n = 200_000
    gen = rng()
    xs = gen.multivariate_normal(np.zeros(p), cov, size=n)
    loss = np.sum(s.berger(xs, cov=cov, positive=False) ** 2, axis=1)
    assert np.mean(loss) <= np.trace(cov) + 0.02
    # The positive-part version further reduces risk.
    lossp = np.sum(s.berger(xs, cov=cov, positive=True) ** 2, axis=1)
    assert np.mean(lossp) <= np.mean(loss) - 0.05


def test_shrink_dispatch():
    x = rng().normal(size=5)
    np.testing.assert_allclose(s.shrink(x, np.eye(5)), s.berger(x, np.eye(5)))


def test_shrink_unknown_method():
    with pytest.raises(ValueError, match="Unknown shrinkage method"):
        s.shrink(rng().normal(size=3), np.eye(3), method="nope")


def test_berger_shape_error():
    with pytest.raises(ValueError, match="covariance matrix must have shape"):
        s.berger(rng().normal(size=3), np.eye(4))


def test_berger_nonsymmetric_error():
    with pytest.raises(ValueError, match="must be symmetric"):
        s.berger(rng().normal(size=2), np.array([[1.0, 0.5], [0.0, 1.0]]))


def test_berger_not_pd_error():
    with pytest.raises(ValueError, match="positive definite"):
        s.berger(rng().normal(size=3), np.zeros((3, 3)))


def test_berger_negative_c_error():
    with pytest.raises(ValueError, match="non-negative"):
        s.berger(rng().normal(size=3), np.eye(3), c=-1.0)
