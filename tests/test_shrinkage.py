from __future__ import annotations

import numpy as np
import pytest

import nustattools.stats as s
from nustattools.stats import shrinkage as _shrinkage


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
        _shrinkage.berger(x, cov=sigma**2 * np.eye(7), positive=False), expected
    )


def test_berger_positive_part_preserves_sign():
    # Positive part should never flip the sign of any coordinate, only
    # shrink it (possibly to zero).
    x = rng().normal(size=5) * 0.05
    delta = _shrinkage.berger(x, cov=np.eye(5))
    assert np.all(delta * x >= -1e-12)


def test_berger_default_cov_is_identity():
    x = rng().normal(size=6)
    np.testing.assert_allclose(
        _shrinkage.berger(x), _shrinkage.berger(x, cov=np.eye(6))
    )
    assert _shrinkage.berger(x).shape == (6,)


def test_shrink_default_cov_is_identity():
    x = rng().normal(size=6)
    np.testing.assert_allclose(s.shrink(x), s.shrink(x, cov=np.eye(6)))


def test_berger_zero_shrinkage_is_identity():
    x = rng().normal(size=5)
    np.testing.assert_allclose(_shrinkage.berger(x, cov=np.eye(5), c=0.0), x)


def test_berger_general_matches_closed_form():
    # The canonicalized computation (default Q=I) must agree with the direct
    # general-form Berger estimator.
    a = rng().normal(size=(5, 5))
    cov = a @ a.T + np.eye(5)
    x = rng().normal(size=5)
    c = 3.0
    np.testing.assert_allclose(
        _shrinkage.berger(x, cov=cov, positive=False, c=c),
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
        _shrinkage.berger(x, cov=cov, Q=q, positive=False, c=c),
        _berger_general_formula(x, cov, q, c),
    )


def test_berger_broadcasting_shapes():
    p = 5
    x = rng().normal(size=(4, 3, p))
    out = _shrinkage.berger(x, cov=np.eye(p))
    assert out.shape == (4, 3, p)
    for idx in np.ndindex(4, 3):
        np.testing.assert_allclose(
            out[idx], _shrinkage.berger(x[idx], cov=np.eye(p)), rtol=1e-12
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
    loss = np.sum(_shrinkage.berger(xs, cov=cov, positive=False) ** 2, axis=1)
    assert np.mean(loss) <= np.trace(cov) + 0.02
    # The positive-part version further reduces risk.
    lossp = np.sum(_shrinkage.berger(xs, cov=cov, positive=True) ** 2, axis=1)
    assert np.mean(lossp) <= np.mean(loss) - 0.05


def test_shrink_dispatch():
    x = rng().normal(size=5)
    np.testing.assert_allclose(s.shrink(x, np.eye(5)), _shrinkage.berger(x, np.eye(5)))


def test_shrink_unknown_method():
    with pytest.raises(ValueError, match="Unknown shrinkage method"):
        s.shrink(rng().normal(size=3), np.eye(3), method="nope")


def test_berger_shape_error():
    with pytest.raises(ValueError, match="covariance matrix must have shape"):
        _shrinkage.berger(rng().normal(size=3), np.eye(4))


def test_berger_nonsymmetric_error():
    with pytest.raises(ValueError, match="must be symmetric"):
        _shrinkage.berger(rng().normal(size=2), np.array([[1.0, 0.5], [0.0, 1.0]]))


def test_berger_not_pd_error():
    with pytest.raises(ValueError, match="positive definite"):
        _shrinkage.berger(rng().normal(size=3), np.zeros((3, 3)))


def test_berger_negative_c_error():
    with pytest.raises(ValueError, match="non-negative"):
        _shrinkage.berger(rng().normal(size=3), np.eye(3), c=-1.0)


def test_estimate_canonicalizes_and_decanonicalizes():
    # The shared _estimate wrapper must canonicalize the inputs, pass the
    # canonical data to the estimator, and transform the result back.  Use a
    # synthetic canonical estimator so the transform is exercised for an
    # estimator that is not the built-in Berger one.
    calls = []

    def dummy_canonical(x_star, d, *, scale):
        calls.append((np.array(x_star), np.array(d)))
        return scale * x_star

    a = rng().normal(size=(4, 4))
    cov = a @ a.T + np.eye(4)
    b = rng().normal(size=(4, 4))
    q = b @ b.T + np.eye(4)
    xa = rng().normal(size=(2, 4))

    _shrinkage._estimate(xa, cov, q, dummy_canonical, scale=2.0)
    (x_star, d) = calls[0]

    # The estimator receives the canonical data from the shared transform.
    bmat, _, d_from_transform = _shrinkage._canonicalize(cov, q)
    np.testing.assert_allclose(x_star, xa @ bmat.T)
    np.testing.assert_allclose(d, d_from_transform)

    # Applying the identity canonical estimator recovers the original data, so
    # the decanonicalization exactly inverts the canonicalization.
    out_identity = _shrinkage._estimate(xa, cov, q, lambda xs, _dd: xs)
    np.testing.assert_allclose(out_identity, xa)


def _projection(basis):
    """Euclidean-orthogonal projection matrix onto the span of ``basis``.

    Only used to build the *expected* values in the identity-loss closed-form
    tests, where it coincides with the covariance-metric projector.

    """
    u, _ = np.linalg.qr(np.asarray(basis, dtype=float))
    return u @ u.T


def test_berger_point_offset_equals_shift():
    # Shrinking towards a point ``t`` (no dirs) must equal ``t +`` shrinking
    # ``x - t`` towards zero.
    gen = rng()
    p = 6
    x = gen.normal(size=p)
    t = gen.normal(size=p)
    np.testing.assert_allclose(
        _shrinkage.berger(x, offset=t), t + _shrinkage.berger(x - t), rtol=1e-12
    )


def test_berger_full_dirs_is_identity():
    # dirs spanning the whole space leave nothing to shrink, so the result is
    # the input regardless of the offset.
    gen = rng()
    p = 6
    x = gen.normal(size=p)
    offset = gen.normal(size=p)
    np.testing.assert_allclose(_shrinkage.berger(x, dirs=np.eye(p)), x, atol=1e-12)
    np.testing.assert_allclose(
        _shrinkage.berger(x, dirs=np.eye(p), offset=offset), x, atol=1e-12
    )


def test_berger_dirs_c_zero_recovers_x():
    # With c = 0 Berger performs no shrinkage, so the subspace machinery must
    # reconstruct the input exactly for any direction and offset.
    gen = rng()
    p = 6
    x = gen.normal(size=p)
    offset = gen.normal(size=p)
    v = gen.normal(size=(p, 2))
    np.testing.assert_allclose(_shrinkage.berger(x, dirs=v, c=0.0), x, rtol=1e-9)
    np.testing.assert_allclose(
        _shrinkage.berger(x, dirs=v, offset=offset, c=0.0), x, rtol=1e-9
    )


def test_berger_small_complement_is_identity():
    # When the orthogonal complement has dimension 2 the optimal c = p_eff - 2
    # = 0, so there is no shrinkage and the estimate is the input.
    gen = rng()
    p = 6
    x = gen.normal(size=p)
    v = gen.normal(size=(p, 4))  # complement dimension 2
    np.testing.assert_allclose(_shrinkage.berger(x, dirs=v), x, atol=1e-12)


def test_berger_subspace_matches_james_stein_identity():
    # With cov = Q = I and dirs spanning a 2-dimensional subspace of R^6,
    # Berger reduces to the James-Stein estimator shrunk towards the subspace
    # (Lehmann & Casella, Ex. 6.2): the component in the subspace is kept and
    # the residual is shrunk by 1 - p_eff/||r||^2 with p_eff = 4.
    gen = rng()
    p = 6
    x = gen.normal(size=p)
    v = gen.normal(size=(p, 2))
    proj = _projection(v)
    residual = (np.eye(p) - proj) @ x
    expected = proj @ x + (1 - 2 / np.sum(residual**2)) * residual
    np.testing.assert_allclose(
        _shrinkage.berger(x, dirs=v, positive=False), expected, rtol=1e-10
    )


def test_berger_affine_subspace_matches_shifted_identity():
    # Shrinking towards an offset subspace must keep the projection onto the
    # shifted subspace and shrink the residual (I - P)(x - offset).
    gen = rng()
    p = 6
    x = gen.normal(size=p)
    offset = gen.normal(size=p)
    v = gen.normal(size=(p, 2))
    proj = _projection(v)
    y = x - offset
    residual = (np.eye(p) - proj) @ y
    expected = offset + proj @ y + (1 - 2 / np.sum(residual**2)) * residual
    np.testing.assert_allclose(
        _shrinkage.berger(x, dirs=v, offset=offset, positive=False),
        expected,
        rtol=1e-10,
    )


def test_berger_subspace_keeps_projected_component():
    # The component of the estimate along the projected direction must equal
    # the projection of the data; only the orthogonal residual is shrunk.
    gen = rng()
    p = 6
    x = gen.normal(size=p)
    v = gen.normal(size=(p, 2))
    proj = _projection(v)
    delta = _shrinkage.berger(x, dirs=v)
    np.testing.assert_allclose(proj @ delta, proj @ x, rtol=1e-12)
    resid = (np.eye(p) - proj) @ delta
    raw = (np.eye(p) - proj) @ x
    assert np.linalg.norm(resid) <= np.linalg.norm(raw)


def test_berger_subspace_reduces_general_cov():
    # A nontrivial general covariance / loss pair must still support shrinking
    # towards a subspace. The affine component of the estimate is kept exactly
    # as in the data, and the orthogonal residual is shrunk towards zero.
    gen = rng()
    p = 6
    x = gen.normal(size=p)
    a = gen.normal(size=(p, p))
    cov = a @ a.T + np.eye(p)
    bmat = gen.normal(size=(p, p))
    q = bmat @ bmat.T + np.eye(p)
    v = gen.normal(size=(p, 2))

    b, _, d = _shrinkage._canonicalize(cov, q)
    v_star = b @ v
    x_star = x @ b.T
    pmat = _shrinkage._dirs_projection(v_star, d)

    delta = _shrinkage.berger(x, cov=cov, Q=q, dirs=v)
    assert delta.shape == (p,)
    delta_star = delta @ b.T
    # The affine (projected) component of the estimate equals that of the data.
    np.testing.assert_allclose(pmat @ delta_star, pmat @ x_star, atol=1e-10)
    # The orthogonal residual is shrunk towards zero, so its norm does not grow.
    assert np.linalg.norm((np.eye(p) - pmat) @ delta_star) <= np.linalg.norm(
        (np.eye(p) - pmat) @ x_star
    )


def test_shrink_dispatches_offset_dirs():
    gen = rng()
    p = 6
    x = gen.normal(size=p)
    offset = gen.normal(size=p)
    v = gen.normal(size=(p, 2))
    np.testing.assert_allclose(
        s.shrink(x, offset=offset, dirs=v),
        _shrinkage.berger(x, offset=offset, dirs=v),
        rtol=1e-12,
    )


def test_dirs_shape_error():
    with pytest.raises(ValueError, match="dirs must have shape"):
        _shrinkage.berger(rng().normal(size=3), dirs=np.eye(4))
    with pytest.raises(ValueError, match="dirs must have shape"):
        _shrinkage.berger(rng().normal(size=3), dirs=rng().normal(size=(3, 0)))


def test_dirs_dependent_columns_error():
    # A linearly dependent set does not span a fixed subspace (not a basis) and
    # yields a singular Gram matrix, so it must be rejected.
    dep = np.column_stack([np.ones(3), np.ones(3)])
    with pytest.raises(ValueError, match="linearly independent"):
        _shrinkage.berger(rng().normal(size=3), dirs=dep)


def test_berger_offset_shape_error():
    with pytest.raises(ValueError, match="offset must have shape"):
        _shrinkage.berger(rng().normal(size=3), offset=np.ones(4))


def test_subspace_reduce_structure():
    # _subspace_reduce must return an orthonormal basis of the complement, the
    # kept (projected) part, reduced variances, and residual coordinates such
    # that the residual is reconstructed as eta @ l2.T.
    gen = rng()
    p = 6
    y = gen.normal(size=p)
    d = np.sort(gen.uniform(0.5, 3.0, p))[::-1]
    v = gen.normal(size=(p, 2))
    kept, eta, d_perp, l2 = _shrinkage._subspace_reduce(y, d, v)
    assert l2.shape == (p, 4)
    assert d_perp.shape == (4,)
    np.testing.assert_allclose(l2.T @ l2, np.eye(4), atol=1e-12)
    residual = y - kept
    np.testing.assert_allclose(eta @ l2.T, residual, atol=1e-10)
    assert np.all(d_perp > 0)


def test_dirs_projection_uncorrelates():
    # The covariance-metric projector must make fitted and residual
    # uncorrelated: P D (I - P)^T = 0. Without this, the risk of the two
    # components would not separate and independent shrinkage would be invalid.
    gen = rng()
    p = 7
    d = np.sort(gen.uniform(0.5, 4.0, p))[::-1]
    v = gen.normal(size=(p, 3))
    pmat = _shrinkage._dirs_projection(v, d)
    cross = pmat @ np.diag(d) @ (np.eye(p) - pmat).T
    np.testing.assert_allclose(cross, np.zeros((p, p)), atol=1e-10)
    # The projector is idempotent.
    np.testing.assert_allclose(pmat @ pmat, pmat, atol=1e-10)


def test_subspace_reduce_loss_isometry():
    # The reduced problem must be isometric to the full-space residual loss:
    # l2 orthonormal AND eta's covariance exactly diag(d_perp). Together these
    # guarantee the recursion minimizes the same squared-error loss as the
    # full-dimensional problem (loss conservation).
    gen = rng()
    p = 7
    d = np.sort(gen.uniform(0.5, 4.0, p))[::-1]
    v = gen.normal(size=(p, 2))
    _kept, _eta, d_perp, l2 = _shrinkage._subspace_reduce(gen.normal(size=p), d, v)
    # l2 orthonormal -> change of basis is an isometry of the Euclidean loss.
    np.testing.assert_allclose(l2.T @ l2, np.eye(l2.shape[1]), atol=1e-12)
    # eta has exactly the diagonal covariance reported as d_perp.
    pmat = _shrinkage._dirs_projection(v, d)
    m = (np.eye(p) - pmat) @ np.diag(d) @ (np.eye(p) - pmat).T
    np.testing.assert_allclose(l2.T @ m @ l2, np.diag(d_perp), atol=1e-10)


def test_subspace_loss_conserved_by_recursion():
    # Loss conservation (independence): because the covariance-metric
    # projection is used, the affine (kept) component and the residual are
    # uncorrelated, so the total loss of the estimator is exactly the sum of
    # the kept-component loss and the residual-shrinkage loss, with zero cross
    # term. Verify this as a Pythagorean identity when the truth lies in the
    # direction of shrinkage (residual truth is zero).
    gen = rng()
    p = 6
    v = gen.normal(size=(p, 2))
    proj = _projection(v)
    truth = proj @ gen.normal(size=p)  # lies in the subspace
    x = truth + gen.normal(size=p)
    delta = _shrinkage.berger(x, dirs=v, positive=False)
    # The affine component of the estimate equals that of the data...
    np.testing.assert_allclose(proj @ delta, proj @ x, rtol=1e-12)
    # ...so its loss is the noise projected onto the subspace.
    kept_loss = np.sum((proj @ x - proj @ truth) ** 2)
    # The residual is James-Stein shrunk and its truth is zero.
    residual = (np.eye(p) - proj) @ x
    shrunk = (1 - 2 / np.sum(residual**2)) * residual
    residual_loss = np.sum(shrunk**2)
    # The cross term between the kept and residual components vanishes
    # (independent shrinkage), so the total loss is the exact sum.
    total_loss = np.sum((delta - truth) ** 2)
    np.testing.assert_allclose(total_loss, kept_loss + residual_loss, rtol=1e-6)
