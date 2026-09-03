from __future__ import annotations

import functools

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


def _tan_general_formula(x, cov, q, gamma, strength=1.0, positive=False):
    """Direct implementation of Tan's estimator (Corollary 3) for any gamma >= 0.

    Independent of :func:`nustattools.stats.shrinkage.tan`: diagonalizes the
    problem in the Q-metric and implements the paper's algorithm for arbitrary
    non-negative ``gamma``.  The two limit cases ``gamma=0`` and ``gamma=inf``
    use the special-cased formulas; finite positive ``gamma`` uses the general
    finite-gamma formulas with Bayes importance ``d_j* = d_j**2/(d_j+gamma)``.

    See [Tan2015]_, Corollary 3 and the algorithm in Section 3.3.

    """
    x = np.asarray(x, dtype=float)
    cov = np.asarray(cov, dtype=float)
    q = np.asarray(q, dtype=float)
    c = np.linalg.cholesky(q).T
    d, o = np.linalg.eigh(c @ cov @ c.T)
    b = o.T @ c
    binv = np.linalg.inv(b)
    x_star = x @ b.T

    p = len(d)
    if gamma == 0.0:
        d_star = d
        weight = 1.0 / d
        low_a = np.ones(p)
    elif not np.isfinite(gamma):
        d_star = d**2
        weight = 1.0 / d**2
        low_a = d
    else:
        d_plus_g = d + gamma
        d_star = d**2 / d_plus_g
        weight = d_plus_g / d**2
        low_a = d / d_plus_g
    order = np.argsort(d_star)[::-1]
    d_sorted = d[order]
    d_star_sorted = d_star[order]
    cw = np.cumsum(weight[order])
    nu = p
    for k in range(3, p):
        if (k - 2) / cw[k - 1] > d_star_sorted[k]:
            nu = k
            break
    s = cw[nu - 1]
    a = np.empty(p)
    a[:nu] = (nu - 2) / (s * d_sorted[:nu])
    a[nu:] = low_a[order[nu:]]
    c_star = (nu - 2) ** 2 / s
    if nu < p:
        if not np.isfinite(gamma):
            c_star += np.sum(d_sorted[nu:] ** 2)
        else:
            c_star += np.sum(d_star_sorted[nu:])

    x_sorted = x_star[..., order]
    s_val = np.sum(a**2 * x_sorted**2, axis=-1)
    factor = 1.0 - strength * c_star * a / s_val[..., None]
    if positive:
        factor = np.maximum(factor, 0.0)
    delta_sorted = factor * x_sorted
    delta_star = np.empty_like(delta_sorted)
    delta_star[..., order] = delta_sorted
    return delta_star @ binv.T


def _mb_general_formula(x, cov, q, gamma, strength=1.0, positive=False):
    """Direct implementation of Berger's delta^MB estimator for any gamma >= 0.

    Independent of :func:`nustattools.stats.shrinkage.berger_mb`: diagonalizes
    the problem in the Q-metric and implements [Tan2015]_, Equation (8) under
    the homoscedastic prior ``Gamma = gamma I`` (so ``gamma_j = gamma``).  The
    shrinkage constant is ``strength * (k - 2)_+``.

    """
    x = np.asarray(x, dtype=float)
    cov = np.asarray(cov, dtype=float)
    q = np.asarray(q, dtype=float)
    c = np.linalg.cholesky(q).T
    d, o = np.linalg.eigh(c @ cov @ c.T)
    b = o.T @ c
    binv = np.linalg.inv(b)
    x_star = x @ b.T

    p = len(d)
    d_plus_g = d + gamma
    d_star = d**2 / d_plus_g
    weight = d / d_plus_g
    order = np.argsort(d_star)[::-1]
    d_star_sorted = d_star[order]
    weight_sorted = weight[order]
    x_sorted = x_star[..., order]

    s_cum = np.cumsum(x_sorted**2 / d_plus_g[order], axis=-1)
    c_k = strength * np.maximum(np.arange(p) - 1, 0.0)
    m_k = np.minimum(1.0, c_k / s_cum)
    d_next = np.concatenate((d_star_sorted[1:], np.zeros(1)))
    t_k = (d_star_sorted - d_next) * m_k
    bracket = np.cumsum(t_k[..., ::-1], axis=-1)[..., ::-1] / d_star_sorted

    factor = 1.0 - weight_sorted * bracket
    if positive:
        factor = np.maximum(factor, 0.0)
    delta_sorted = factor * x_sorted
    delta_star = np.empty_like(delta_sorted)
    delta_star[..., order] = delta_sorted
    return delta_star @ binv.T


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
    np.testing.assert_allclose(_shrinkage.berger(x, cov=np.eye(5), strength=0.0), x)


def test_berger_general_matches_closed_form():
    # The canonicalized computation (default Q=I) must agree with the direct
    # general-form Berger estimator.
    a = rng().normal(size=(5, 5))
    cov = a @ a.T + np.eye(5)
    x = rng().normal(size=5)
    c = 3.0  # with p=5, c = strength * (p-2) = strength * 3, so strength=1.0
    strength = c / 3.0
    np.testing.assert_allclose(
        _shrinkage.berger(x, cov=cov, positive=False, strength=strength),
        _berger_general_formula(x, cov, np.eye(5), c),
    )


def test_berger_general_Q_matches_closed_form():
    a = rng().normal(size=(5, 5))
    cov = a @ a.T + np.eye(5)
    b = rng().normal(size=(5, 5))
    q = b @ b.T + np.eye(5)
    x = rng().normal(size=5)
    c = 2.5  # with p=5, c = strength * (p-2) = strength * 3, so strength = 2.5/3
    strength = c / 3.0
    np.testing.assert_allclose(
        _shrinkage.berger(x, cov=cov, Q=q, positive=False, strength=strength),
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


def test_berger_out_of_range_strength_error():
    for bad in (-1.0, 3.0):
        with pytest.raises(ValueError, match="strength"):
            _shrinkage.berger(rng().normal(size=3), np.eye(3), strength=bad)


def test_estimate_degenerate_empty_dimension():
    # p = 0 is a degenerate but valid problem: the empty loss matrix passes PSD
    # validation (zero-size guard) and the pipeline returns an empty estimate.
    delta = _shrinkage._estimate(np.empty((2, 0)), None, np.eye(0), lambda xs, _dd: xs)
    assert delta.shape == (2, 0)


def test_merge_dirs_union_of_user_dirs_and_null_q():
    # _merge_dirs returns an orthonormal independent basis of span(dirs) +
    # null(Q); only the spanned space matters, not the representative columns.
    gen = rng()
    p = 6
    q = _psd_q(4, p, gen)
    w, v = np.linalg.eigh(q)
    u_r = v[:, w > 1e-10]
    null_basis = v[:, w < 1e-10]
    range_col = u_r[:, 0]
    null_col = null_basis[:, 0]
    # A single range direction column plus a null direction column.
    basis = _shrinkage._merge_dirs(np.column_stack([range_col, null_col]), q, p)
    # Expected span: the two user columns plus the rest of null(Q).
    expected = np.column_stack([range_col, null_basis])
    # The basis spans the same space as 'expected' (full column rank projection).
    assert basis.shape == (p, 3)
    np.testing.assert_allclose(basis.T @ basis, np.eye(3), rtol=1e-9, atol=1e-10)
    proj = basis @ np.linalg.solve(basis.T @ basis, basis.T)
    exp_proj = expected @ np.linalg.solve(expected.T @ expected, expected.T)
    np.testing.assert_allclose(proj @ exp_proj, exp_proj, rtol=1e-8, atol=1e-10)


def test_merge_dirs_without_user_dirs_is_null_q_basis():
    # With dirs=None, _merge_dirs spans exactly null(Q).
    gen = rng()
    p = 6
    q = _psd_q(4, p, gen)
    basis = _shrinkage._merge_dirs(None, q, p)
    w, v = np.linalg.eigh(q)
    null_basis = v[:, w < 1e-10]
    assert basis.shape == (p, null_basis.shape[1])
    # Every basis column lies in null(Q), i.e. Q annihilates it.
    np.testing.assert_allclose(q @ basis, np.zeros_like(basis), rtol=1e-8, atol=1e-10)


def test_merge_dirs_pd_q_with_no_dirs_is_empty():
    # For a strictly positive-definite Q with no user dirs there is no null
    # space, so the no-shrink span is empty (shape (p, 0)).
    p = 4
    q = np.eye(p)
    basis = _shrinkage._merge_dirs(None, q, p)
    assert basis.shape == (p, 0)


def test_merge_dirs_redundant_user_dirs_are_deduplicated():
    # User dirs columns that overlap null(Q) (or each other) are handled by the
    # span basis: duplicate directions do not inflate the result.
    gen = rng()
    p = 6
    q = _psd_q(4, p, gen)
    w, v = np.linalg.eigh(q)
    null_col = v[:, w < 1e-10][:, 0]
    # Two identical user columns in null(Q); merged span must equal null(Q) alone.
    dirs = np.column_stack([null_col, 2.0 * null_col])
    basis = _shrinkage._merge_dirs(dirs, q, p)
    null_only = _shrinkage._merge_dirs(None, q, p)
    assert basis.shape == null_only.shape


def test_estimate_split_keeps_no_shrink_projection_and_recovers_data():
    # _estimate_split keeps the covariance-metric projection of the data onto
    # span(dirs) (= span(user dirs) + null(Q)) at its data value.  When the span
    # covers the whole space there is nothing left to shrink, so the estimate
    # equals the data -- a robust end-to-end check of the unified dirs handling.
    gen = rng()
    p = 7
    q = _psd_q(5, p, gen)
    cov = gen.normal(size=(p, p))
    cov = cov @ cov.T + np.eye(p)
    x = gen.normal(size=p)
    w, v = np.linalg.eigh(q)
    u_r = v[:, w > 1e-10]
    # dirs spanning u_r plus the null(Q) basis covers the whole space.
    dirs = u_r
    merge = _shrinkage._merge_dirs(dirs, q, p)
    assert merge.shape == (p, p)
    for scale in (0.0, 1.0):
        delta = _shrinkage._estimate_split(
            x,
            cov,
            q,
            lambda xs, _dd, _s=scale: _s * xs,
            offset=None,
            dirs=merge,
        )
        np.testing.assert_allclose(delta, x, rtol=1e-8, atol=1e-10)


def test_estimate_split_shrinks_only_complement_of_no_shrink_span():
    # With dirs = null(Q) only, the estimator acts only on the range of Q: the
    # covariance-metric projection onto null(Q) is kept at the data value while
    # the range part is shrunk.  Checked via the shrink-towards scale.
    gen = rng()
    p = 6
    q = _psd_q(4, p, gen)
    cov = np.eye(p)
    x = gen.normal(size=p)
    merge = _shrinkage._merge_dirs(None, q, p)
    w, v = np.linalg.eigh(q)
    null_basis = v[:, w < 1e-10]
    null_proj = null_basis @ null_basis.T
    delta = _shrinkage._estimate_split(
        x, cov, q, lambda xs, _dd: 0.5 * xs, offset=None, dirs=merge
    )
    # With cov = I the covariance metric is Euclidean, so the kept part is the
    # Euclidean projection onto null(Q); the estimate keeps it at the data value.
    np.testing.assert_allclose(null_proj @ delta, null_proj @ x, rtol=1e-8, atol=1e-10)
    # The range (complement) part is shrunk by half.
    range_part = (np.eye(p) - null_proj) @ delta
    range_x = (np.eye(p) - null_proj) @ x
    np.testing.assert_allclose(range_part, 0.5 * range_x, rtol=1e-8, atol=1e-10)


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
    np.testing.assert_allclose(_shrinkage.berger(x, dirs=v, strength=0.0), x, rtol=1e-9)
    np.testing.assert_allclose(
        _shrinkage.berger(x, dirs=v, offset=offset, strength=0.0), x, rtol=1e-9
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


def test_estimate_risk_single_returns_pair():
    # A single estimator is squeezed to a (2,) [risk, standard error] pair.
    gen = rng()
    p = 3
    theta = gen.normal(size=p)
    res = _shrinkage.estimate_risk(
        theta, np.eye(p), functools.partial(s.shrink, strength=1.0), n_reps=2000, seed=0
    )
    assert res.shape == (2,)
    assert res[0] > 0  # risk is positive
    assert res[1] > 0  # standard error is positive


def test_estimate_risk_sequence_returns_rows():
    # A sequence of estimators yields one [risk, se] row per estimator, in order.
    gen = rng()
    p = 3
    theta = gen.normal(size=p)
    est = [
        functools.partial(s.shrink, strength=1.0),
        functools.partial(s.shrink, strength=0.0),
        s.shrink,
    ]
    res = _shrinkage.estimate_risk(theta, np.eye(p), est, n_reps=2000, seed=0)
    assert res.shape == (3, 2)


def test_estimate_risk_single_element_sequence_keeps_dim():
    # A sequence with a single entry keeps the leading estimator dimension.
    gen = rng()
    p = 3
    theta = gen.normal(size=p)
    res = _shrinkage.estimate_risk(
        theta,
        np.eye(p),
        [functools.partial(s.shrink, strength=1.0)],
        n_reps=2000,
        seed=0,
    )
    assert res.shape == (1, 2)
    single = _shrinkage.estimate_risk(
        theta, np.eye(p), functools.partial(s.shrink, strength=1.0), n_reps=2000, seed=0
    )
    np.testing.assert_allclose(res[0], single, rtol=1e-12)


def test_estimate_risk_identity_matches_trace():
    # With c=0 Berger is the identity, so the risk is trace(Q @ cov).  For
    # cov = Q = I that is p, and each draw is chi-squared_p, whose sample mean
    # has standard error sqrt(2p / n_reps).
    gen = rng()
    p = 4
    theta = gen.normal(size=p)
    n_reps = 20_000
    res = _shrinkage.estimate_risk(
        theta,
        np.eye(p),
        functools.partial(s.shrink, strength=0.0),
        n_reps=n_reps,
        seed=0,
    )
    np.testing.assert_allclose(res[0], p, rtol=0.02)
    expected_se = np.sqrt(2.0 * p / n_reps)
    np.testing.assert_allclose(res[1], expected_se, rtol=0.1)


def test_estimate_risk_general_loss_matches_trace():
    # The identity estimator's risk is trace(Q @ cov) for general Q and cov.
    gen = rng()
    p = 4
    theta = gen.normal(size=p)
    a = gen.normal(size=(p, p))
    cov = a @ a.T + np.eye(p)
    bmat = gen.normal(size=(p, p))
    q = bmat @ bmat.T + np.eye(p)
    expected = np.trace(q @ cov)
    res = _shrinkage.estimate_risk(
        theta,
        cov,
        functools.partial(s.shrink, strength=0.0),
        Q=q,
        n_reps=20_000,
        seed=0,
    )
    np.testing.assert_allclose(res[0], expected, rtol=0.03)


def test_estimate_risk_berger_beats_identity():
    # Shrinking a nontrivial mean must reduce the estimated risk below the
    # raw (identity) baseline.
    gen = rng()
    p = 6
    theta = gen.normal(size=p)
    shrinker = _shrinkage.estimate_risk(
        theta,
        np.eye(p),
        functools.partial(s.shrink, strength=1.0),
        n_reps=20_000,
        seed=0,
    )
    identity = _shrinkage.estimate_risk(
        theta,
        np.eye(p),
        functools.partial(s.shrink, strength=0.0),
        n_reps=20_000,
        seed=0,
    )
    assert shrinker[0] < identity[0]


def test_estimate_risk_shares_samples():
    # All estimators in a sequence are evaluated on the same draws, so each row
    # equals the single-estimator result built from the same seed.
    gen = rng()
    p = 3
    theta = gen.normal(size=p)
    n_reps = 2000
    est = [
        functools.partial(s.shrink, strength=1.0),
        functools.partial(s.shrink, strength=0.0),
    ]
    multi = _shrinkage.estimate_risk(theta, np.eye(p), est, n_reps=n_reps, seed=7)
    for i, e in enumerate(est):
        single = _shrinkage.estimate_risk(theta, np.eye(p), e, n_reps=n_reps, seed=7)
        np.testing.assert_allclose(multi[i], single, rtol=1e-12)


def test_estimate_risk_method_name_equals_callable():
    # Passing the method name as a string must give the same result as passing
    # the corresponding callable.
    gen = rng()
    p = 3
    theta = gen.normal(size=p)
    by_name = _shrinkage.estimate_risk(theta, np.eye(p), "berger", n_reps=2000, seed=0)
    by_fn = _shrinkage.estimate_risk(
        theta, np.eye(p), _shrinkage.berger, n_reps=2000, seed=0
    )
    np.testing.assert_allclose(by_name, by_fn, rtol=1e-12)


def test_estimate_risk_invalid_method_raises():
    with pytest.raises(ValueError, match="Unknown shrinkage method"):
        _shrinkage.estimate_risk(
            rng().normal(size=3), np.eye(3), "not-a-method", n_reps=100
        )
    with pytest.raises(ValueError, match="Unknown shrinkage method"):
        _shrinkage.estimate_risk(
            rng().normal(size=3), np.eye(3), ["berger", "nope"], n_reps=100
        )


def test_estimate_risk_n_reps_must_be_at_least_two():
    gen = rng()
    theta = gen.normal(size=3)
    for n in (0, 1, -5):
        with pytest.raises(ValueError, match="n_reps must be an integer >= 2"):
            _shrinkage.estimate_risk(
                theta, np.eye(3), functools.partial(s.shrink, strength=1.0), n_reps=n
            )


def test_estimate_risk_seed_determinism():
    # The same seed reproduces the same draws; a different seed (almost surely)
    # gives a different result.
    gen = rng()
    p = 3
    theta = gen.normal(size=p)
    est = functools.partial(s.shrink, strength=1.0)
    a = _shrinkage.estimate_risk(theta, np.eye(p), est, n_reps=500, seed=3)
    b = _shrinkage.estimate_risk(theta, np.eye(p), est, n_reps=500, seed=3)
    np.testing.assert_allclose(a, b, rtol=0.0, atol=0.0)
    c = _shrinkage.estimate_risk(theta, np.eye(p), est, n_reps=500, seed=4)
    assert not np.allclose(a[0], c[0])


def test_estimate_risk_bad_shapes_raise():
    with pytest.raises(ValueError, match="theta must be a 1-D vector"):
        _shrinkage.estimate_risk(
            rng().normal(size=(3, 3)), np.eye(3), s.shrink, n_reps=100
        )
    with pytest.raises(ValueError, match="cov must have shape"):
        _shrinkage.estimate_risk(rng().normal(size=3), np.eye(4), s.shrink, n_reps=100)
    with pytest.raises(ValueError, match="Q must have shape"):
        _shrinkage.estimate_risk(
            rng().normal(size=3), np.eye(3), s.shrink, Q=np.eye(4), n_reps=100
        )


def test_estimate_risk_curve_record_shape():
    # One record per (direction, distance, estimator); single estimator gives
    # one record per (direction, distance).
    cov = np.diag([1.0, 2.0, 3.0])
    est = functools.partial(s.shrink, strength=1.0)
    recs = _shrinkage.estimate_risk_curve(
        cov, est, directions="uniform", distances=(0.0, 10.0, 5), n_reps=500, seed=0
    )
    assert len(recs) == 5
    assert set(recs[0]) == {
        "direction",
        "distance",
        "mahalanobis",
        "estimator",
        "risk",
        "se",
        "risk_ratio",
    }


def test_estimate_risk_curve_mahalanobis():
    # The mahalanobis field is sqrt(theta^T cov^-1 theta), the same for every
    # estimator at a given sweep point, and agrees with a direct computation.
    cov = np.diag([1.0, 3.0, 2.0])
    est = [
        functools.partial(s.shrink, strength=1.0),
        functools.partial(s.shrink, strength=0.0),
    ]
    recs = _shrinkage.estimate_risk_curve(
        cov,
        est,
        Q=np.diag([2.0, 1.0, 1.0]),
        directions="proportional",
        distances=[1.0, 3.0],
        n_reps=50,
        seed=0,
    )
    pinv = np.linalg.inv(np.asarray(cov, dtype=float))
    _b, binv, d = _shrinkage._canonicalize(
        np.asarray(cov, dtype=float), np.asarray(np.diag([2.0, 1.0, 1.0]), dtype=float)
    )
    refs = {}
    for r in recs:
        refs.setdefault(r["distance"], r["mahalanobis"])
        assert r["mahalanobis"] == refs[r["distance"]]
    for key, val in refs.items():
        u = np.sqrt(d)
        u = u / np.linalg.norm(u)
        theta = (float(key) * u) @ binv.T
        np.testing.assert_allclose(val, np.sqrt(theta @ pinv @ theta), rtol=1e-6)


def test_estimate_risk_curve_negative_axis():
    # A negative axis selects from the largest-first ordering: -1 is the
    # smallest variance and differs from axis 0 (the largest variance).
    cov = np.diag([1.0, 3.0, 2.0])
    est = functools.partial(s.shrink, strength=1.0)
    recs = _shrinkage.estimate_risk_curve(
        cov, est, directions=-1, distances=[0.0, 1.0], n_reps=200, seed=0
    )
    assert recs[0]["direction"] == "axis -1"
    _b, binv, d = _shrinkage._canonicalize(cov, np.eye(3))
    smallest = int(np.argsort(d)[::-1][-1])
    raw = np.eye(3)[smallest] @ binv
    by_raw = _shrinkage.estimate_risk_curve(
        cov, est, directions=[raw], distances=[0.0, 1.0], n_reps=200, seed=0
    )
    for a, b_ in zip(recs, by_raw, strict=True):
        np.testing.assert_allclose(a["risk"], b_["risk"], rtol=1e-12)
    axis0 = _shrinkage.estimate_risk_curve(
        cov, est, directions=0, distances=[1.0], n_reps=200, seed=0
    )
    assert not np.allclose(recs[0]["risk"], axis0[0]["risk"])


def test_estimate_risk_curve_matches_brute_force():
    # The sweep draws N(0, cov) once and translates it to each (direction,
    # distance) pair, so re-evaluating each point on that same shared draw must
    # reproduce the recorded risk and standard error exactly.
    cov = np.diag([1.0, 3.0, 2.0])
    q = np.diag([2.0, 1.0, 1.0])
    est = functools.partial(s.shrink, strength=1.0)
    n_reps = 800
    seed = 11
    recs = _shrinkage.estimate_risk_curve(
        cov,
        est,
        Q=q,
        directions=["proportional", 1],
        distances=[1.0, 3.0, 5.0],
        n_reps=n_reps,
        seed=seed,
    )
    _b, binv, d = _shrinkage._canonicalize(
        np.diag([1.0, 3.0, 2.0]), np.diag([2.0, 1.0, 1.0])
    )
    descend = np.argsort(d)[::-1]
    rng = np.random.default_rng(seed)
    x0 = rng.multivariate_normal(np.zeros(3), cov, size=n_reps)
    for r in recs:
        if r["direction"] == "proportional":
            u_star = np.sqrt(d)
        else:  # axis 1
            u_star = np.zeros(3)
            u_star[descend[1]] = 1.0
        u_star = u_star / np.linalg.norm(u_star)
        theta = (r["distance"] * u_star) @ binv.T
        x = x0 + theta
        delta = est(x, cov, Q=q)
        diff = delta - theta
        loss = np.einsum("ni,ij,nj->n", diff, q, diff)
        np.testing.assert_allclose(r["risk"], np.mean(loss), rtol=1e-10)
        np.testing.assert_allclose(
            r["se"], np.std(loss, ddof=1) / np.sqrt(n_reps), rtol=1e-10
        )


def test_estimate_risk_curve_distance_is_canonical_norm():
    # The distance entry equals the canonical Euclidean norm of the mean.
    cov = np.diag([1.0, 2.0, 3.0])
    est = functools.partial(s.shrink, strength=1.0)
    recs = _shrinkage.estimate_risk_curve(
        cov, est, directions="uniform", distances=[2.0, 4.0], n_reps=200, seed=0
    )
    assert [r["distance"] for r in recs] == [2.0, 4.0]


def test_estimate_risk_curve_risk_ratio():
    # risk_ratio = risk / trace(Q @ cov), and a minimax estimator keeps it <= 1.
    cov = np.diag([1.0, 2.0, 5.0])
    est = "berger"
    q = np.diag([2.0, 1.0, 1.0])
    baseline = float(np.trace(q @ cov))
    recs = _shrinkage.estimate_risk_curve(
        cov,
        est,
        Q=q,
        directions="proportional",
        distances=[0.0, 2.0],
        n_reps=2000,
        seed=0,
    )
    for r in recs:
        np.testing.assert_allclose(r["risk_ratio"], r["risk"] / baseline, rtol=1e-12)
        assert r["risk_ratio"] <= 1.0 + 1e-6


def test_estimate_risk_curve_axis_ordering():
    # Axis 0 is the coordinate with the largest variance.  It must match a raw
    # vector whose canonical image is the largest-variance coordinate.
    cov = np.diag([1.0, 3.0, 2.0])
    est = functools.partial(s.shrink, strength=1.0)
    recs = _shrinkage.estimate_risk_curve(
        cov, est, directions=0, distances=[0.0, 1.0], n_reps=200, seed=0
    )
    assert recs[0]["direction"] == "axis 0"
    _b, binv, d = _shrinkage._canonicalize(cov, np.eye(3))
    largest = int(np.argsort(d)[::-1][0])
    raw = np.eye(3)[largest] @ binv
    same_axis = _shrinkage.estimate_risk_curve(
        cov, est, directions=[raw], distances=[0.0, 1.0], n_reps=200, seed=0
    )
    for a, b_ in zip(recs, same_axis, strict=True):
        np.testing.assert_allclose(a["risk"], b_["risk"], rtol=1e-12)


def test_estimate_risk_curve_raw_vector_matches_canonical_when_identity():
    # For cov = Q = I the canonical and original space coincide, so a raw vector
    # proportional to sqrt(d) reproduces the named "proportional" direction.
    cov = np.eye(3)
    est = functools.partial(s.shrink, strength=1.0)
    d = np.ones(3)
    prop = _shrinkage.estimate_risk_curve(
        cov, est, directions="proportional", distances=[0.0, 2.0], n_reps=300, seed=0
    )
    raw = _shrinkage.estimate_risk_curve(
        cov, est, directions=[np.sqrt(d)], distances=[0.0, 2.0], n_reps=300, seed=0
    )
    for rp, rr in zip(prop, raw, strict=True):
        np.testing.assert_allclose(rp["risk"], rr["risk"], rtol=1e-12)


def test_estimate_risk_curve_estimator_labels():
    # Explicit estimator labels are used verbatim; a single callable defaults to
    # its __name__; a functools.partial falls back to its position.
    cov = np.diag([1.0, 2.0, 3.0])
    recs = _shrinkage.estimate_risk_curve(
        cov,
        functools.partial(s.shrink, strength=1.0),
        directions="uniform",
        distances=[1.0],
        n_reps=100,
        seed=0,
        estimator_labels=["my shrinker"],
    )
    assert recs[0]["estimator"] == "my shrinker"
    recs = _shrinkage.estimate_risk_curve(
        cov, "berger", directions="uniform", distances=[1.0], n_reps=100, seed=0
    )
    assert recs[0]["estimator"] == "berger"


def test_estimate_risk_curve_direction_labels():
    cov = np.diag([1.0, 2.0, 3.0])
    recs = _shrinkage.estimate_risk_curve(
        cov,
        "berger",
        directions=["uniform", "inverse"],
        distances=[1.0],
        n_reps=100,
        seed=0,
        direction_labels=["a", "b"],
    )
    assert [r["direction"] for r in recs] == ["a", "b"]


def test_estimate_risk_curve_mixed_directions_counts():
    # Mixed named / axis / vector directions all contribute records in order.
    cov = np.diag([1.0, 2.0, 3.0])
    est = functools.partial(s.shrink, strength=1.0)
    recs = _shrinkage.estimate_risk_curve(
        cov,
        est,
        directions=["uniform", 0, np.array([1.0, 0.0, 0.0])],
        distances=[1.0, 2.0],
        n_reps=200,
        seed=0,
    )
    assert [r["direction"] for r in recs] == [
        "uniform",
        "uniform",
        "axis 0",
        "axis 0",
        "dir 2",
        "dir 2",
    ]


def test_estimate_risk_curve_validation_errors():
    cov = np.diag([1.0, 2.0, 3.0])
    est = functools.partial(s.shrink, strength=1.0)
    with pytest.raises(ValueError, match="Unknown direction"):
        _shrinkage.estimate_risk_curve(
            cov, est, directions="bogus", distances=[1.0], n_reps=100
        )
    with pytest.raises(ValueError, match="out of range"):
        _shrinkage.estimate_risk_curve(
            cov, est, directions=3, distances=[1.0], n_reps=100
        )
    with pytest.raises(ValueError, match="zero canonical norm"):
        _shrinkage.estimate_risk_curve(
            cov, est, directions=[np.zeros(3)], distances=[1.0], n_reps=100
        )
    with pytest.raises(ValueError, match="must have shape"):
        _shrinkage.estimate_risk_curve(
            cov, est, directions=[np.ones(4)], distances=[1.0], n_reps=100
        )
    with pytest.raises(ValueError, match="non-negative"):
        _shrinkage.estimate_risk_curve(
            cov, est, directions="uniform", distances=[-1.0], n_reps=100
        )
    with pytest.raises(ValueError, match="estimator_labels"):
        _shrinkage.estimate_risk_curve(
            cov,
            est,
            directions="uniform",
            distances=[1.0],
            n_reps=100,
            estimator_labels=["a", "b"],
        )
    with pytest.raises(ValueError, match="must be a square matrix"):
        _shrinkage.estimate_risk_curve(
            np.eye(4)[:3], est, directions="uniform", distances=[1.0], n_reps=100
        )
    with pytest.raises(ValueError, match="n_reps must be"):
        _shrinkage.estimate_risk_curve(
            cov, est, directions="uniform", distances=[1.0], n_reps=1
        )


def test_tan_default_cov_is_identity():
    x = rng().normal(size=6)
    np.testing.assert_allclose(_shrinkage.tan(x), _shrinkage.tan(x, cov=np.eye(6)))
    assert _shrinkage.tan(x).shape == (6,)


def test_tan_default_similar_to_berger_homoscedastic():
    # In the homoscedastic case D = sigma^2 I, both estimators reduce to a
    # James-Stein type shrinkage and should give comparable estimates.
    sigma = 1.0
    x = rng().normal(size=8)
    t = _shrinkage.tan(x, cov=sigma**2 * np.eye(8), positive=False)
    b = _shrinkage.berger(x, cov=sigma**2 * np.eye(8), positive=False)
    # Both must reproduce the sign pattern of x (no sign flips) and shrink.
    assert np.all(t * x >= -1e-12)
    assert np.all(b * x >= -1e-12)


def test_tan_zero_strength_is_identity():
    x = rng().normal(size=5)
    np.testing.assert_allclose(_shrinkage.tan(x, strength=0.0), x, atol=1e-12)


def test_tan_shape():
    p = 5
    x = rng().normal(size=p)
    assert _shrinkage.tan(x).shape == (p,)
    assert _shrinkage.tan(x, gamma=float("inf")).shape == (p,)
    assert _shrinkage.tan(x, gamma=1.0).shape == (p,)


def test_tan_broadcasting_shapes():
    p = 5
    x = rng().normal(size=(4, 3, p))
    out = _shrinkage.tan(x, cov=np.eye(p))
    assert out.shape == (4, 3, p)
    for idx in np.ndindex(4, 3):
        np.testing.assert_allclose(
            out[idx], _shrinkage.tan(x[idx], cov=np.eye(p)), rtol=1e-12
        )


def test_tan_small_dim_is_identity():
    # For p < 3 there is no shrinkage; the estimate is the input.
    x = rng().normal(size=2)
    np.testing.assert_allclose(_shrinkage.tan(x), x, atol=1e-12)
    np.testing.assert_allclose(_shrinkage.tan(x, gamma=float("inf")), x, atol=1e-12)
    np.testing.assert_allclose(_shrinkage.tan(x, gamma=1.0), x, atol=1e-12)


def test_tan_positive_dominates_plain():
    # The positive-part estimator must have lower (or equal) risk than the
    # plain one.
    rngg = rng()
    cov = np.diag([4.0, 2.0, 1.0, 0.5, 0.25])
    theta = np.array([0.0, 0.0, 0.0, 2.0, 2.0])
    xs = rngg.multivariate_normal(theta, cov, size=100_000)
    plain_loss = np.sum(
        (_shrinkage.tan(xs, cov=cov, positive=False) - theta) ** 2, axis=1
    )
    pos_loss = np.sum((_shrinkage.tan(xs, cov=cov, positive=True) - theta) ** 2, axis=1)
    assert np.mean(pos_loss) <= np.mean(plain_loss) + 0.05


def test_tan_minimaxity():
    # Tan's estimator is minimax: its risk is never greater than tr(cov), for
    # any true mean.  Evaluate at theta = 0 where the signal is strongest.
    p = 5
    cov = np.diag([4.0, 2.0, 1.0, 0.5, 0.25])
    n = 200_000
    xs = rng().multivariate_normal(np.zeros(p), cov, size=n)
    for gamma in (0.0, 1.0, float("inf")):
        loss = np.sum(_shrinkage.tan(xs, cov=cov, gamma=gamma) ** 2, axis=1)
        assert np.mean(loss) <= np.trace(cov) + 0.05


def test_tan_beats_berger_low_variance_truth():
    # Berger shrinks low-variance (low-importance) coordinates too
    # aggressively (inversely proportional to variance).  When the truth
    # concentrates in the low-variance coordinates, Tan's estimator should
    # reduce the risk more than Berger.
    rngg = rng()
    cov = np.diag([4.0, 2.0, 1.0, 0.5, 0.25])
    theta = np.array([0.0, 0.0, 0.0, 2.0, 2.0])
    xs = rngg.multivariate_normal(theta, cov, size=100_000)
    risk_tan = np.mean(
        np.sum((_shrinkage.tan(xs, cov=cov, gamma=0.0) - theta) ** 2, axis=1)
    )
    risk_berg = np.mean(np.sum((_shrinkage.berger(xs, cov=cov) - theta) ** 2, axis=1))
    assert risk_tan < risk_berg - 0.1


def test_tan_gamma_two_special_cases_differ():
    # gamma=0 (A†_0) and gamma=inf (A†_∞) produce different shrinkage
    # directions under heteroscedasticity, so in general the estimates differ.
    d = np.linspace(0.5, 3.0, 6)
    x = rng().normal(size=6) * np.sqrt(d)
    a = _shrinkage.tan(x, cov=np.diag(d), gamma=0.0)
    b = _shrinkage.tan(x, cov=np.diag(d), gamma=float("inf"))
    assert not np.allclose(a, b, atol=1e-8)


def test_tan_finite_gamma_shrinks_and_interpolates():
    # A finite positive gamma must actually shrink (not return x) and, at a
    # given x, lie between the gamma=0 and gamma=inf estimates interpolated
    # through the gamma parameter.  Values of gamma approaching 0 (resp. inf)
    # approach the corresponding limit estimate.
    d = np.diag([4.0, 2.0, 1.0, 0.5, 0.25])
    x = rng().normal(size=5)
    est0 = _shrinkage.tan(x, cov=d, gamma=0.0)
    esti = _shrinkage.tan(x, cov=d, gamma=float("inf"))
    est_mid = _shrinkage.tan(x, cov=d, gamma=1.0)
    # Finite gamma must differ from the input (i.e. actually shrink).
    assert not np.allclose(est_mid, x, atol=1e-8)
    # gamma->0 and gamma->inf limits agree with the special cases.
    np.testing.assert_allclose(_shrinkage.tan(x, cov=d, gamma=1e-8), est0, rtol=1e-3)
    np.testing.assert_allclose(_shrinkage.tan(x, cov=d, gamma=1e8), esti, rtol=1e-3)


def test_tan_strength_validation():
    for bad in (-1.0, 3.0):
        with pytest.raises(ValueError, match="strength"):
            _shrinkage.tan(rng().normal(size=5), strength=bad)


def test_tan_gamma_validation():
    with pytest.raises(ValueError, match="gamma"):
        _shrinkage.tan(rng().normal(size=5), gamma=-1.0)


def test_tan_point_offset_equals_shift():
    # Shrinking towards a point t (no dirs) must equal t + shrinking x - t
    # towards zero.
    gen = rng()
    p = 6
    x = gen.normal(size=p)
    t = gen.normal(size=p)
    np.testing.assert_allclose(
        _shrinkage.tan(x, offset=t), t + _shrinkage.tan(x - t), rtol=1e-10
    )


def test_tan_full_dirs_is_identity():
    # dirs spanning the whole space leave nothing to shrink, so the result is
    # the input regardless of offset.
    gen = rng()
    p = 6
    x = gen.normal(size=p)
    np.testing.assert_allclose(_shrinkage.tan(x, dirs=np.eye(p)), x, atol=1e-12)


def test_tan_dirs_small_complement_is_identity():
    # When the orthogonal complement has dimension < 3 there is no shrinkage
    # in the complement, so the estimate is the input.
    gen = rng()
    p = 6
    x = gen.normal(size=p)
    v = gen.normal(size=(p, 4))  # complement dimension 2
    np.testing.assert_allclose(_shrinkage.tan(x, dirs=v), x, atol=1e-12)


def test_tan_dirs_c_zero_recovers_x():
    gen = rng()
    p = 6
    x = gen.normal(size=p)
    v = gen.normal(size=(p, 2))
    np.testing.assert_allclose(_shrinkage.tan(x, dirs=v, strength=0.0), x, rtol=1e-9)


def test_tan_subspace_keeps_projected_component():
    # The component of the estimate along the projected direction must equal
    # the projection of the data; only the orthogonal residual is shrunk.
    gen = rng()
    p = 6
    x = gen.normal(size=p)
    v = gen.normal(size=(p, 2))
    proj = _projection(v)
    delta = _shrinkage.tan(x, dirs=v)
    np.testing.assert_allclose(proj @ delta, proj @ x, rtol=1e-12)


def test_tan_gamma_zero_matches_paper_reference():
    # tan(x, gamma=0) must equal the paper's A†_0 estimator (Tan2015 Cor. 3)
    # computed independently.  The covariance is deliberately non-monotonic so
    # the eigensystem is a non-trivial permutation, which is what a naive
    # coordinate-order bug would mix up.
    a = rng().normal(size=(5, 5))
    cov = a @ a.T + np.diag([3.0, 0.5, 1.0, 4.0, 2.0])
    gen = rng()
    xs = gen.multivariate_normal(np.zeros(5), cov, size=30)
    for positive in (True, False):
        for strength in (0.5, 1.0):
            got = _shrinkage.tan(
                xs, cov=cov, gamma=0.0, positive=positive, strength=strength
            )
            ref = _tan_general_formula(
                xs, cov, np.eye(5), 0.0, strength=strength, positive=positive
            )
            np.testing.assert_allclose(got, ref, atol=1e-8)


def test_tan_gamma_inf_matches_paper_reference():
    # tan(x, gamma=inf) must equal the paper's A†_∞ limit (Tan2015 Cor. 3),
    # computed independently via the gamma->inf formulas.
    a = rng().normal(size=(5, 5))
    cov = a @ a.T + np.diag([3.0, 0.5, 1.0, 4.0, 2.0])
    gen = rng()
    xs = gen.multivariate_normal(np.zeros(5), cov, size=30)
    for positive in (True, False):
        for strength in (0.5, 1.0):
            got = _shrinkage.tan(
                xs, cov=cov, gamma=float("inf"), positive=positive, strength=strength
            )
            ref = _tan_general_formula(
                xs, cov, np.eye(5), float("inf"), strength=strength, positive=positive
            )
            np.testing.assert_allclose(got, ref, rtol=1e-6)


def test_tan_reference_broadcasts():
    # The elementwise reference must agree across stacked (broadcast) inputs
    # too, exercising the ordering logic under `...` indexing.
    a = rng().normal(size=(4, 4))
    cov = a @ a.T + np.eye(4)
    gen = rng()
    xs = gen.normal(size=(6, 4))
    got = _shrinkage.tan(xs, cov=cov, gamma=0.0)
    ref = _tan_general_formula(xs, cov, np.eye(4), 0.0, positive=True)
    np.testing.assert_allclose(got, ref, rtol=1e-6, atol=1e-8)


def test_tan_general_gamma_matches_reference():
    # tan(x, gamma=g) must equal the general-gamma reference for finite g > 0.
    a = rng().normal(size=(5, 5))
    cov = a @ a.T + np.diag([3.0, 0.5, 1.0, 4.0, 2.0])
    gen = rng()
    xs = gen.multivariate_normal(np.zeros(5), cov, size=30)
    for gamma in (0.1, 1.0, 10.0, 100.0):
        for positive in (True, False):
            for strength in (0.5, 1.0):
                got = _shrinkage.tan(
                    xs, cov=cov, gamma=gamma, positive=positive, strength=strength
                )
                ref = _tan_general_formula(
                    xs, cov, np.eye(5), gamma, strength=strength, positive=positive
                )
                np.testing.assert_allclose(got, ref, rtol=1e-6, atol=1e-8)


def test_shrink_dispatches_tan():
    x = rng().normal(size=5)
    np.testing.assert_allclose(s.shrink(x, np.eye(5), method="tan"), _shrinkage.tan(x))
    np.testing.assert_allclose(
        s.shrink(x, np.eye(5), method="tan", gamma=float("inf")),
        _shrinkage.tan(x, gamma=float("inf")),
    )


def test_berger_mb_default_cov_is_identity():
    x = rng().normal(size=6)
    np.testing.assert_allclose(
        _shrinkage.berger_mb(x), _shrinkage.berger_mb(x, cov=np.eye(6))
    )
    assert _shrinkage.berger_mb(x).shape == (6,)


def test_berger_mb_zero_strength_is_identity():
    x = rng().normal(size=5)
    np.testing.assert_allclose(_shrinkage.berger_mb(x, strength=0.0), x, atol=1e-12)


def test_berger_mb_shape():
    p = 5
    x = rng().normal(size=p)
    assert _shrinkage.berger_mb(x).shape == (p,)
    assert _shrinkage.berger_mb(x, gamma=1.0).shape == (p,)


def test_berger_mb_broadcasting_shapes():
    p = 5
    x = rng().normal(size=(4, 3, p))
    out = _shrinkage.berger_mb(x, cov=np.eye(p))
    assert out.shape == (4, 3, p)
    for idx in np.ndindex(4, 3):
        np.testing.assert_allclose(
            out[idx], _shrinkage.berger_mb(x[idx], cov=np.eye(p)), rtol=1e-12
        )


def test_berger_mb_matches_reference():
    # berger_mb must equal the independently implemented Eq. (8) for a
    # deliberately non-monotonic covariance (non-trivial permutation), any
    # gamma >= 0, several strengths and both positive/plain versions.
    a = rng().normal(size=(5, 5))
    cov = a @ a.T + np.diag([3.0, 0.5, 1.0, 4.0, 2.0])
    gen = rng()
    xs = gen.multivariate_normal(np.zeros(5), cov, size=30)
    for gamma in (0.0, 0.1, 1.0, 10.0):
        for strength in (0.5, 1.0, 2.0):
            for positive in (True, False):
                got = _shrinkage.berger_mb(
                    xs, cov=cov, gamma=gamma, strength=strength, positive=positive
                )
                ref = _mb_general_formula(
                    xs, cov, np.eye(5), gamma, strength=strength, positive=positive
                )
                np.testing.assert_allclose(got, ref, rtol=1e-8, atol=1e-10)


def test_berger_mb_reference_broadcasts():
    # The elementwise reference must agree across stacked (broadcast) inputs.
    a = rng().normal(size=(4, 4))
    cov = a @ a.T + np.eye(4)
    gen = rng()
    xs = gen.normal(size=(6, 4))
    got = _shrinkage.berger_mb(xs, cov=cov, gamma=1.0)
    ref = _mb_general_formula(xs, cov, np.eye(4), 1.0)
    np.testing.assert_allclose(got, ref, rtol=1e-8, atol=1e-10)


def test_berger_mb_small_dim_is_identity():
    # For p < 3 the constant (k-2)_+ vanishes for all k, so there is no
    # shrinkage and the estimate is the input.
    x = rng().normal(size=2)
    np.testing.assert_allclose(_shrinkage.berger_mb(x), x, atol=1e-12)
    np.testing.assert_allclose(_shrinkage.berger_mb(x, gamma=1.0), x, atol=1e-12)


def test_berger_mb_positive_dominates_plain():
    # The positive-part estimator must have lower (or equal) risk than the
    # plain one.
    rngg = rng()
    cov = np.diag([4.0, 2.0, 1.0, 0.5, 0.25])
    theta = np.array([0.0, 0.0, 0.0, 2.0, 2.0])
    xs = rngg.multivariate_normal(theta, cov, size=100_000)
    plain_loss = np.sum(
        (_shrinkage.berger_mb(xs, cov=cov, positive=False) - theta) ** 2, axis=1
    )
    pos_loss = np.sum(
        (_shrinkage.berger_mb(xs, cov=cov, positive=True) - theta) ** 2, axis=1
    )
    assert np.mean(pos_loss) <= np.mean(plain_loss) + 0.05


def test_berger_mb_minimaxity():
    # Berger's delta^MB is minimax: its risk is never greater than tr(cov), for
    # any true mean.  Evaluate at theta = 0 where the signal is strongest.
    p = 5
    cov = np.diag([4.0, 2.0, 1.0, 0.5, 0.25])
    n = 200_000
    xs = rng().multivariate_normal(np.zeros(p), cov, size=n)
    for gamma in (0.0, 1.0, 10.0):
        for strength in (0.5, 1.0, 2.0):
            loss = np.sum(
                _shrinkage.berger_mb(xs, cov=cov, gamma=gamma, strength=strength) ** 2,
                axis=1,
            )
            assert np.mean(loss) <= np.trace(cov) + 0.05


def test_berger_mb_strength_validation():
    for bad in (-1.0, 3.0):
        with pytest.raises(ValueError, match="strength"):
            _shrinkage.berger_mb(rng().normal(size=5), strength=bad)


def test_berger_mb_gamma_validation():
    with pytest.raises(ValueError, match="gamma"):
        _shrinkage.berger_mb(rng().normal(size=5), gamma=-1.0)


def test_berger_mb_point_offset_equals_shift():
    # Shrinking towards a point t (no dirs) must equal t + shrinking x - t
    # towards zero.
    gen = rng()
    p = 6
    x = gen.normal(size=p)
    t = gen.normal(size=p)
    np.testing.assert_allclose(
        _shrinkage.berger_mb(x, offset=t), t + _shrinkage.berger_mb(x - t), rtol=1e-10
    )


def test_berger_mb_full_dirs_is_identity():
    # dirs spanning the whole space leave nothing to shrink, so the result is
    # the input regardless of offset.
    gen = rng()
    p = 6
    x = gen.normal(size=p)
    np.testing.assert_allclose(_shrinkage.berger_mb(x, dirs=np.eye(p)), x, atol=1e-12)


def test_berger_mb_dirs_small_complement_is_identity():
    # When the orthogonal complement has dimension < 3 there is no shrinkage
    # in the complement, so the estimate is the input.
    gen = rng()
    p = 6
    x = gen.normal(size=p)
    v = gen.normal(size=(p, 4))  # complement dimension 2
    np.testing.assert_allclose(_shrinkage.berger_mb(x, dirs=v), x, atol=1e-12)


def test_berger_mb_subspace_keeps_projected_component():
    # The component of the estimate along the projected direction must equal
    # the projection of the data; only the orthogonal residual is shrunk.
    gen = rng()
    p = 6
    x = gen.normal(size=p)
    v = gen.normal(size=(p, 2))
    proj = _projection(v)
    delta = _shrinkage.berger_mb(x, dirs=v)
    np.testing.assert_allclose(proj @ delta, proj @ x, rtol=1e-12)


def test_berger_mb_small_complement_survives_general_gamma():
    # A small complement with a moderate gamma still produces shrinkage in the
    # low-Bayes-importance coordinates, so the result differs from the input.
    gen = rng()
    p = 6
    x = gen.normal(size=p)
    v = gen.normal(size=(p, 3))  # complement dimension 3
    delta = _shrinkage.berger_mb(x, dirs=v, gamma=1.0, positive=False)
    assert not np.allclose(delta, x, atol=1e-8)


def test_shrink_dispatches_berger_mb():
    x = rng().normal(size=5)
    np.testing.assert_allclose(
        s.shrink(x, np.eye(5), method="berger_mb"), _shrinkage.berger_mb(x)
    )
    np.testing.assert_allclose(
        s.shrink(x, np.eye(5), method="berger_mb", gamma=1.0),
        _shrinkage.berger_mb(x, gamma=1.0),
    )


def _psd_q(rank, p, gen):
    """Return a symmetric PSD ``(p, p)`` matrix of the given ``rank``."""
    w = np.concatenate([np.linspace(0.5, 3.0, rank), np.zeros(p - rank)])
    v, _ = np.linalg.qr(gen.normal(size=(p, p)))
    q = (v * w) @ v.T
    return (q + q.T) / 2


def test_psd_Q_keeps_null_component_at_data():
    # When Q is singular, the loss-blind null space is treated as a no-shrink
    # direction: the covariance-metric projection of the estimate onto
    # span(null(Q)) must equal that of the data regardless of how much the
    # range is shrunk.  Checked for every estimator with a general cov (the
    # covariance metric is then distinct from the Euclidean one).
    gen = rng()
    p = 7
    q = _psd_q(5, p, gen)
    cov = rng().normal(size=(p, p))
    cov = cov @ cov.T + np.eye(p)
    x = gen.normal(size=p)
    w, v = np.linalg.eigh(q)
    null_basis = v[:, w < 1e-10]
    cinv = np.linalg.inv(cov)
    t = null_basis.T @ cinv @ null_basis
    p_null = null_basis @ np.linalg.solve(t, null_basis.T @ cinv)
    for est in (_shrinkage.berger, _shrinkage.tan, _shrinkage.berger_mb):
        kwargs = {"gamma": float("inf")} if est is _shrinkage.tan else {}
        delta = est(x, cov=cov, Q=q, **kwargs)
        np.testing.assert_allclose(delta @ p_null.T, x @ p_null.T, rtol=1e-6, atol=1e-8)


def test_psd_Q_range_component_is_shrunk():
    # A singular Q must still shrink the range component (where the loss is
    # positive) towards the target, so the estimate moves off the full data.
    gen = rng()
    p = 6
    q = _psd_q(4, p, gen)  # rank 4 -> Berger range dimension 4, c = 2 > 0
    cov = np.eye(p)
    x = gen.normal(size=p)
    delta = _shrinkage.berger(x, cov=cov, Q=q, positive=False)
    assert not np.allclose(delta, x, atol=1e-8)


def test_psd_Q_null_is_auto_included_in_no_shrink_span():
    # null(Q) is automatically part of the no-shrink span, so explicitly adding
    # it in dirs does not change the result (only the span matters).
    gen = rng()
    p = 7
    q = _psd_q(4, p, gen)
    cov = gen.normal(size=(p, p))
    cov = cov @ cov.T + np.eye(p)
    x = gen.normal(size=p)
    w, v = np.linalg.eigh(q)
    null_basis = v[:, w < 1e-10]
    for est in (_shrinkage.berger, _shrinkage.tan, _shrinkage.berger_mb):
        kwargs = {"gamma": float("inf")} if est is _shrinkage.tan else {}
        plain = est(x, cov=cov, Q=q, **kwargs)
        with_null_dirs = est(x, cov=cov, Q=q, dirs=null_basis, **kwargs)
        np.testing.assert_allclose(with_null_dirs, plain, rtol=1e-8, atol=1e-10)


def test_psd_Q_identity_strength_recovers_x():
    # With strength 0 (no shrinkage) the estimate must be the input even for a
    # singular Q.
    gen = rng()
    p = 6
    q = _psd_q(3, p, gen)
    cov = gen.normal(size=(p, p))
    cov = cov @ cov.T + np.eye(p)
    x = gen.normal(size=p)
    np.testing.assert_allclose(
        _shrinkage.berger(x, cov=cov, Q=q, strength=0.0), x, rtol=1e-9
    )


def test_psd_Q_dirs_in_null_are_dropped():
    # dirs columns lying in null(Q) carry no loss and must be dropped, leaving
    # the result identical to the no-dirs estimate.
    gen = rng()
    p = 6
    q = _psd_q(4, p, gen)
    cov = np.eye(p)
    x = gen.normal(size=p)
    w, v = np.linalg.eigh(q)
    null_basis = v[:, w < 1e-10]
    dirs_null = null_basis  # shape (p, 2)
    plain = _shrinkage.berger(x, cov=cov, Q=q)
    with_dirs = _shrinkage.berger(x, cov=cov, Q=q, dirs=dirs_null)
    np.testing.assert_allclose(with_dirs, plain, rtol=1e-9, atol=1e-10)


def test_psd_Q_dirs_mixed_range_and_null_keeps_range_columns():
    # When dirs mixes range and null columns, the null columns are dropped and
    # the range columns still drive the subspace shrinkage.
    gen = rng()
    p = 6
    q = _psd_q(4, p, gen)
    cov = np.eye(p)
    x = gen.normal(size=p)
    w, v = np.linalg.eigh(q)
    range_basis = v[:, w > 1e-10]
    null_basis = v[:, w < 1e-10]
    # A range direction plus a null direction.
    dirs = np.column_stack([range_basis[:, 0], null_basis[:, 0]])
    delta = _shrinkage.berger(x, cov=cov, Q=q, dirs=dirs, positive=False)
    # The estimator ran with the surviving range column: it must differ from
    # both the no-dirs result and the raw data when the range permits shrinkage.
    assert not np.allclose(delta, x, atol=1e-8)
    assert delta.shape == (p,)


def test_psd_Q_linearly_dependent_dirs_raise():
    # User dirs columns are required to be linearly independent (full column
    # rank), regardless of Q, so a dependent dirs set is rejected.
    gen = rng()
    p = 6
    q = _psd_q(4, p, gen)
    cov = np.eye(p)
    x = gen.normal(size=p)
    range_col = gen.normal(size=p)
    dirs = np.column_stack([range_col, 2.0 * range_col])
    with pytest.raises(ValueError, match="linearly independent"):
        _shrinkage.berger(x, cov=cov, Q=q, dirs=dirs)


def test_psd_Q_range_subspace_component_is_kept():
    # The part of the dirs subspace lying in the range of Q behaves exactly as
    # in the SPD case: the covariance-metric
    # projection of the estimate onto span(dirs_r) equals that of the data
    # (P * delta == P * x), while the orthogonal residual is shrunk.  The null
    # component is kept at the data value regardless.
    gen = rng()
    p = 7
    q = _psd_q(5, p, gen)
    cov = gen.normal(size=(p, p))
    cov = cov @ cov.T + np.eye(p)
    for offset in (None, gen.normal(size=p)):
        x = gen.normal(size=p)
        w, v = np.linalg.eigh(q)
        u_r = v[:, w > 1e-10]
        null_basis = v[:, w < 1e-10]
        # Two independent range columns plus a null column.
        dirs = np.column_stack([u_r[:, :2], null_basis[:, 0]])
        for est, kwargs in (
            (_shrinkage.berger, {}),
            (_shrinkage.tan, {"gamma": np.inf}),
            (_shrinkage.berger_mb, {}),
        ):
            delta = est(x, cov=cov, Q=q, offset=offset, dirs=dirs, **kwargs)
            dirs_r = u_r.T @ dirs
            dirs_r = dirs_r[:, np.linalg.norm(dirs_r, axis=0) > 1e-12]
            cov_r = u_r.T @ cov @ u_r
            q_r = np.diag(w[w > 1e-10])
            b_r, _, d_r = _shrinkage._canonicalize(cov_r, q_r)
            p_mat = _shrinkage._dirs_projection(b_r @ dirs_r, d_r)
            x_r = x @ u_r
            delta_r = (delta - (x - x @ (u_r @ u_r.T))) @ u_r
            np.testing.assert_allclose(
                (delta_r @ b_r.T) @ p_mat.T,
                (x_r @ b_r.T) @ p_mat.T,
                rtol=1e-8,
                atol=1e-10,
            )


def test_psd_Q_broadcasting_shapes():
    p = 6
    q = _psd_q(4, p, rng())
    x = rng().normal(size=(3, p))
    out = _shrinkage.berger(x, cov=np.eye(p), Q=q)
    assert out.shape == (3, p)
    for i in range(3):
        np.testing.assert_allclose(
            out[i], _shrinkage.berger(x[i], cov=np.eye(p), Q=q), rtol=1e-12
        )


def test_psd_Q_estimate_risk_matches_trace():
    # The identity estimator's risk under a singular Q is trace(Q @ cov), and
    # estimate_risk must accept a positive semi-definite Q.
    gen = rng()
    p = 5
    theta = gen.normal(size=p)
    cov = gen.normal(size=(p, p))
    cov = cov @ cov.T + np.eye(p)
    q = _psd_q(3, p, gen)
    expected = float(np.trace(q @ cov))
    res = _shrinkage.estimate_risk(
        theta,
        cov,
        functools.partial(s.shrink, strength=0.0),
        Q=q,
        n_reps=20_000,
        seed=0,
    )
    np.testing.assert_allclose(res[0], expected, rtol=0.03)


def test_Q_not_psd_raises():
    # An indefinite Q (a negative eigenvalue) must be rejected.
    q = np.diag([1.0, -1.0])
    with pytest.raises(ValueError, match="positive semi-definite"):
        _shrinkage.berger(rng().normal(size=2), np.eye(2), Q=q)
    with pytest.raises(ValueError, match="positive semi-definite"):
        _shrinkage.estimate_risk(
            rng().normal(size=2), np.eye(2), s.shrink, Q=q, n_reps=100
        )


def test_Q_nonsymmetric_raises_on_psd_path():
    # A non-symmetric Q is rejected by the positive semi-definite validator
    # (the covariance's strict validator is exercised elsewhere).
    q = np.array([[1.0, 0.0], [1.0, 0.0]])
    with pytest.raises(ValueError, match="must be symmetric"):
        _shrinkage.berger(rng().normal(size=2), np.eye(2), Q=q)
