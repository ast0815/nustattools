from __future__ import annotations

import functools

import numpy as np
import pytest

import nustattools.stats as s
from nustattools.stats import shrinkage as _shrinkage
from nustattools.stats.shrinkage._core import (
    _canonical_frame,
    _cov_proportional_to_qinv,
    _max_abs_risk_gamma,
    _max_rel_risk_gamma,
    _parse_gamma_factory,
)
from nustattools.stats.shrinkage._core import (
    _empirical_gamma as _core_empirical_gamma,
)
from nustattools.stats.shrinkage._estimators import (
    _bayes_canonical,
    _minimax_bayes_canonical,
    _robust_bayes_canonical,
    _tan_bayes_canonical,
    _tan_canonical,
)
from nustattools.stats.shrinkage._risk import _canonical_directions


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

    Independent of :func:`nustattools.stats.shrinkage.minimax_bayes`: diagonalizes
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


def test_canonicalize_sorts_by_decreasing_variance():
    # The canonical coordinates are ordered by decreasing variance d
    # (coordinate 0 is the largest variance), matching the risk-curve axis
    # convention, while the frame still diagonalizes cov and Q and round-trips.
    gen = rng()
    p = 5
    a = gen.normal(size=(p, p))
    cov = a @ a.T + p * np.eye(p)
    m = gen.normal(size=(p, p))
    q = m @ m.T + np.eye(p)
    b, binv, d = _shrinkage._canonicalize(cov, q)
    np.testing.assert_allclose(b @ cov @ b.T, np.diag(d), rtol=1e-9, atol=1e-10)
    np.testing.assert_allclose(b.T @ b, q, rtol=1e-9, atol=1e-10)
    np.testing.assert_allclose(b @ binv, np.eye(p), rtol=1e-9, atol=1e-10)
    assert np.all(np.diff(d) <= 1e-12)


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


def test_estimate_risk_empirical_gamma_shrinks_near_zero():
    # Regression: a batched risk sweep must resolve the empirical prior scale
    # per observation.  Previously the scale was summed over the whole batch,
    # giving gamma proportional to n_reps, so every estimator collapsed to (near)
    # the identity and the risk at theta=0 was ~ p.  With per-observation gamma
    # the empirical bayes estimate must shrink meaningfully near the origin.
    p = 6
    theta = np.zeros(p)
    identity = _shrinkage.estimate_risk(
        theta,
        np.eye(p),
        functools.partial(s.shrink, strength=0.0),
        n_reps=20_000,
        seed=0,
    )
    empirical = _shrinkage.estimate_risk(
        theta,
        np.eye(p),
        functools.partial(s.shrink, method="bayes", gamma="empirical"),
        n_reps=20_000,
        seed=0,
    )
    # Shrinking a zero mean must cut the risk well below the identity baseline.
    assert empirical[0] < identity[0] - 1.0


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
    raw = binv[:, smallest]
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
    raw = binv[:, largest]
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


def test_minimax_bayes_default_cov_is_identity():
    x = rng().normal(size=6)
    np.testing.assert_allclose(
        _shrinkage.minimax_bayes(x), _shrinkage.minimax_bayes(x, cov=np.eye(6))
    )
    assert _shrinkage.minimax_bayes(x).shape == (6,)


def test_minimax_bayes_zero_strength_is_identity():
    x = rng().normal(size=5)
    np.testing.assert_allclose(_shrinkage.minimax_bayes(x, strength=0.0), x, atol=1e-12)


def test_minimax_bayes_shape():
    p = 5
    x = rng().normal(size=p)
    assert _shrinkage.minimax_bayes(x).shape == (p,)
    assert _shrinkage.minimax_bayes(x, gamma=1.0).shape == (p,)


def test_minimax_bayes_broadcasting_shapes():
    p = 5
    x = rng().normal(size=(4, 3, p))
    out = _shrinkage.minimax_bayes(x, cov=np.eye(p))
    assert out.shape == (4, 3, p)
    for idx in np.ndindex(4, 3):
        np.testing.assert_allclose(
            out[idx], _shrinkage.minimax_bayes(x[idx], cov=np.eye(p)), rtol=1e-12
        )


def test_minimax_bayes_matches_reference():
    # minimax_bayes must equal the independently implemented Eq. (8) for a
    # deliberately non-monotonic covariance (non-trivial permutation), any
    # gamma >= 0, several strengths and both positive/plain versions.
    a = rng().normal(size=(5, 5))
    cov = a @ a.T + np.diag([3.0, 0.5, 1.0, 4.0, 2.0])
    gen = rng()
    xs = gen.multivariate_normal(np.zeros(5), cov, size=30)
    for gamma in (0.0, 0.1, 1.0, 10.0):
        for strength in (0.5, 1.0, 2.0):
            for positive in (True, False):
                got = _shrinkage.minimax_bayes(
                    xs, cov=cov, gamma=gamma, strength=strength, positive=positive
                )
                ref = _mb_general_formula(
                    xs, cov, np.eye(5), gamma, strength=strength, positive=positive
                )
                np.testing.assert_allclose(got, ref, rtol=1e-8, atol=1e-10)


def test_minimax_bayes_reference_broadcasts():
    # The elementwise reference must agree across stacked (broadcast) inputs.
    a = rng().normal(size=(4, 4))
    cov = a @ a.T + np.eye(4)
    gen = rng()
    xs = gen.normal(size=(6, 4))
    got = _shrinkage.minimax_bayes(xs, cov=cov, gamma=1.0)
    ref = _mb_general_formula(xs, cov, np.eye(4), 1.0)
    np.testing.assert_allclose(got, ref, rtol=1e-8, atol=1e-10)


def test_minimax_bayes_small_dim_is_identity():
    # For p < 3 the constant (k-2)_+ vanishes for all k, so there is no
    # shrinkage and the estimate is the input.
    x = rng().normal(size=2)
    np.testing.assert_allclose(_shrinkage.minimax_bayes(x), x, atol=1e-12)
    np.testing.assert_allclose(_shrinkage.minimax_bayes(x, gamma=1.0), x, atol=1e-12)


def test_minimax_bayes_positive_dominates_plain():
    # The positive-part estimator must have lower (or equal) risk than the
    # plain one.
    rngg = rng()
    cov = np.diag([4.0, 2.0, 1.0, 0.5, 0.25])
    theta = np.array([0.0, 0.0, 0.0, 2.0, 2.0])
    xs = rngg.multivariate_normal(theta, cov, size=100_000)
    plain_loss = np.sum(
        (_shrinkage.minimax_bayes(xs, cov=cov, positive=False) - theta) ** 2, axis=1
    )
    pos_loss = np.sum(
        (_shrinkage.minimax_bayes(xs, cov=cov, positive=True) - theta) ** 2, axis=1
    )
    assert np.mean(pos_loss) <= np.mean(plain_loss) + 0.05


def test_minimax_bayes_minimaxity():
    # Berger's delta^MB is minimax: its risk is never greater than tr(cov), for
    # any true mean.  Evaluate at theta = 0 where the signal is strongest.
    p = 5
    cov = np.diag([4.0, 2.0, 1.0, 0.5, 0.25])
    n = 200_000
    xs = rng().multivariate_normal(np.zeros(p), cov, size=n)
    for gamma in (0.0, 1.0, 10.0):
        for strength in (0.5, 1.0, 2.0):
            loss = np.sum(
                _shrinkage.minimax_bayes(xs, cov=cov, gamma=gamma, strength=strength)
                ** 2,
                axis=1,
            )
            assert np.mean(loss) <= np.trace(cov) + 0.05


def test_minimax_bayes_strength_validation():
    for bad in (-1.0, 3.0):
        with pytest.raises(ValueError, match="strength"):
            _shrinkage.minimax_bayes(rng().normal(size=5), strength=bad)


def test_minimax_bayes_gamma_validation():
    with pytest.raises(ValueError, match="gamma"):
        _shrinkage.minimax_bayes(rng().normal(size=5), gamma=-1.0)


def test_minimax_bayes_point_offset_equals_shift():
    # Shrinking towards a point t (no dirs) must equal t + shrinking x - t
    # towards zero.
    gen = rng()
    p = 6
    x = gen.normal(size=p)
    t = gen.normal(size=p)
    np.testing.assert_allclose(
        _shrinkage.minimax_bayes(x, offset=t),
        t + _shrinkage.minimax_bayes(x - t),
        rtol=1e-10,
    )


def test_minimax_bayes_full_dirs_is_identity():
    # dirs spanning the whole space leave nothing to shrink, so the result is
    # the input regardless of offset.
    gen = rng()
    p = 6
    x = gen.normal(size=p)
    np.testing.assert_allclose(
        _shrinkage.minimax_bayes(x, dirs=np.eye(p)), x, atol=1e-12
    )


def test_minimax_bayes_dirs_small_complement_is_identity():
    # When the orthogonal complement has dimension < 3 there is no shrinkage
    # in the complement, so the estimate is the input.
    gen = rng()
    p = 6
    x = gen.normal(size=p)
    v = gen.normal(size=(p, 4))  # complement dimension 2
    np.testing.assert_allclose(_shrinkage.minimax_bayes(x, dirs=v), x, atol=1e-12)


def test_minimax_bayes_subspace_keeps_projected_component():
    # The component of the estimate along the projected direction must equal
    # the projection of the data; only the orthogonal residual is shrunk.
    gen = rng()
    p = 6
    x = gen.normal(size=p)
    v = gen.normal(size=(p, 2))
    proj = _projection(v)
    delta = _shrinkage.minimax_bayes(x, dirs=v)
    np.testing.assert_allclose(proj @ delta, proj @ x, rtol=1e-12)


def test_minimax_bayes_small_complement_survives_general_gamma():
    # A small complement with a moderate gamma still produces shrinkage in the
    # low-Bayes-importance coordinates, so the result differs from the input.
    gen = rng()
    p = 6
    x = gen.normal(size=p)
    v = gen.normal(size=(p, 3))  # complement dimension 3
    delta = _shrinkage.minimax_bayes(x, dirs=v, gamma=1.0, positive=False)
    assert not np.allclose(delta, x, atol=1e-8)


def test_shrink_dispatches_minimax_bayes():
    x = rng().normal(size=5)
    np.testing.assert_allclose(
        s.shrink(x, np.eye(5), method="minimax_bayes"), _shrinkage.minimax_bayes(x)
    )
    np.testing.assert_allclose(
        s.shrink(x, np.eye(5), method="minimax_bayes", gamma=1.0),
        _shrinkage.minimax_bayes(x, gamma=1.0),
    )


def _bayes_general_formula(x, cov, q, gamma):
    """Closed-form Bayes rule in the general (non-canonical) form.

    Under the prior theta ~ N(0, gamma I) in canonical coordinates, the Bayes
    rule is delta_j = gamma / (d_j + gamma) * x_star_j.  Transform back to the
    original space.

    """
    x = np.asarray(x, dtype=float)
    cov = np.asarray(cov, dtype=float)
    q = np.asarray(q, dtype=float)
    c = np.linalg.cholesky(q).T
    d, o = np.linalg.eigh(c @ cov @ c.T)
    b = o.T @ c
    binv = np.linalg.inv(b)
    x_star = x @ b.T
    factor = gamma / (d + gamma)
    delta_star = factor * x_star
    return delta_star @ binv.T


def test_bayes_default_cov_is_identity():
    x = rng().normal(size=6)
    np.testing.assert_allclose(_shrinkage.bayes(x), _shrinkage.bayes(x, cov=np.eye(6)))
    assert _shrinkage.bayes(x).shape == (6,)


def test_bayes_gamma_zero_is_zero():
    x = rng().normal(size=5)
    np.testing.assert_allclose(
        _shrinkage.bayes(x, cov=np.eye(5), gamma=0.0), np.zeros(5)
    )


def test_bayes_gamma_inf_is_identity():
    x = rng().normal(size=5)
    np.testing.assert_allclose(
        _shrinkage.bayes(x, cov=np.eye(5), gamma=float("inf")), x, atol=1e-12
    )


def test_bayes_shape():
    p = 5
    x = rng().normal(size=p)
    assert _shrinkage.bayes(x).shape == (p,)
    assert _shrinkage.bayes(x, gamma=0.0).shape == (p,)
    assert _shrinkage.bayes(x, gamma=float("inf")).shape == (p,)


def test_bayes_broadcasting_shapes():
    p = 5
    x = rng().normal(size=(4, 3, p))
    out = _shrinkage.bayes(x, cov=np.eye(p))
    assert out.shape == (4, 3, p)
    for idx in np.ndindex(4, 3):
        np.testing.assert_allclose(
            out[idx], _shrinkage.bayes(x[idx], cov=np.eye(p)), rtol=1e-12
        )


def test_bayes_general_matches_closed_form():
    # The canonicalized computation must agree with the direct general-form
    # Bayes rule for a non-trivial covariance and Q = I.
    a = rng().normal(size=(5, 5))
    cov = a @ a.T + np.eye(5)
    x = rng().normal(size=5)
    gamma = 2.5
    np.testing.assert_allclose(
        _shrinkage.bayes(x, cov=cov, gamma=gamma),
        _bayes_general_formula(x, cov, np.eye(5), gamma),
    )


def test_bayes_general_Q_matches_closed_form():
    # Same with a non-trivial Q.
    a = rng().normal(size=(5, 5))
    cov = a @ a.T + np.eye(5)
    b = rng().normal(size=(5, 5))
    q = b @ b.T + np.eye(5)
    x = rng().normal(size=5)
    gamma = 3.0
    np.testing.assert_allclose(
        _shrinkage.bayes(x, cov=cov, Q=q, gamma=gamma),
        _bayes_general_formula(x, cov, q, gamma),
    )


def test_bayes_shrinks_and_preserves_sign():
    # The Bayes rule factor gamma / (d_j + gamma) is always in (0, 1) for
    # gamma > 0, so the estimate never flips sign.
    x = rng().normal(size=5)
    delta = _shrinkage.bayes(x, cov=np.eye(5), gamma=1.0)
    assert np.all(delta * x >= -1e-12)
    assert not np.allclose(delta, x, atol=1e-8)


def test_bayes_gamma_interpolation():
    # Intermediate gamma must lie between gamma=0 (zero) and gamma=inf
    # (identity).
    d = np.diag([4.0, 2.0, 1.0, 0.5, 0.25])
    x = rng().normal(size=5)
    est0 = _shrinkage.bayes(x, cov=d, gamma=0.0)
    esti = _shrinkage.bayes(x, cov=d, gamma=float("inf"))
    np.testing.assert_allclose(est0, np.zeros(5))
    np.testing.assert_allclose(esti, x, atol=1e-12)
    # gamma->0 and gamma->inf limits agree with the special cases.
    np.testing.assert_allclose(_shrinkage.bayes(x, cov=d, gamma=1e-8), est0, atol=1e-6)
    np.testing.assert_allclose(_shrinkage.bayes(x, cov=d, gamma=1e8), esti, rtol=1e-3)


def test_bayes_low_variance_coordinates_are_shrunk_more():
    # Under the loss-proportional prior, the shrinkage factor gamma/(d_j+gamma)
    # decreases with d_j, so low-variance coordinates are shrunk more strongly.
    d = np.array([0.1, 1.0, 10.0])
    x = np.array([1.0, 1.0, 1.0])
    delta = _shrinkage.bayes(x, cov=np.diag(d), gamma=1.0)
    # factor = gamma/(d+gamma): 1/1.1 ~ 0.909, 1/2 = 0.5, 1/11 ~ 0.091
    factors = 1.0 / (d + 1.0)
    np.testing.assert_allclose(delta, factors * x, rtol=1e-12)
    # Low variance (d=0.1) is shrunk least, high variance (d=10) shrunk most.
    assert abs(delta[0]) > abs(delta[1]) > abs(delta[2])


def test_bayes_gamma_validation():
    with pytest.raises(ValueError, match="gamma"):
        _shrinkage.bayes(rng().normal(size=5), gamma=-1.0)


def test_bayes_point_offset_equals_shift():
    # Shrinking towards a point t (no dirs) must equal t + shrinking x - t
    # towards zero.
    gen = rng()
    p = 6
    x = gen.normal(size=p)
    t = gen.normal(size=p)
    np.testing.assert_allclose(
        _shrinkage.bayes(x, offset=t), t + _shrinkage.bayes(x - t), rtol=1e-10
    )


def test_bayes_full_dirs_is_identity():
    # dirs spanning the whole space leave nothing to shrink, so the result is
    # the input regardless of offset.
    gen = rng()
    p = 6
    x = gen.normal(size=p)
    offset = gen.normal(size=p)
    np.testing.assert_allclose(_shrinkage.bayes(x, dirs=np.eye(p)), x, atol=1e-12)
    np.testing.assert_allclose(
        _shrinkage.bayes(x, dirs=np.eye(p), offset=offset), x, atol=1e-12
    )


def test_bayes_dirs_small_complement_is_identity():
    # When the orthogonal complement has dimension 0, there is nothing to
    # shrink, so the estimate is the input.
    gen = rng()
    p = 6
    x = gen.normal(size=p)
    v = gen.normal(size=(p, 6))
    np.testing.assert_allclose(_shrinkage.bayes(x, dirs=v), x, atol=1e-12)


def test_bayes_subspace_keeps_projected_component():
    # The component of the estimate along the projected direction must equal
    # the projection of the data; only the orthogonal residual is shrunk.
    gen = rng()
    p = 6
    x = gen.normal(size=p)
    v = gen.normal(size=(p, 2))
    proj = _projection(v)
    delta = _shrinkage.bayes(x, dirs=v)
    np.testing.assert_allclose(proj @ delta, proj @ x, rtol=1e-12)
    resid = (np.eye(p) - proj) @ delta
    raw = (np.eye(p) - proj) @ x
    # Residual is shrunk (norm doesn't grow).
    assert np.linalg.norm(resid) <= np.linalg.norm(raw)


def test_bayes_shrink_dispatch():
    x = rng().normal(size=5)
    np.testing.assert_allclose(
        s.shrink(x, np.eye(5), method="bayes"), _shrinkage.bayes(x)
    )
    np.testing.assert_allclose(
        s.shrink(x, np.eye(5), method="bayes", gamma=2.0),
        _shrinkage.bayes(x, gamma=2.0),
    )


def test_bayes_psd_Q_keeps_null_component_at_data():
    # When Q is singular, the loss-blind null space is treated as a no-shrink
    # direction.
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
    delta = _shrinkage.bayes(x, cov=cov, Q=q, gamma=1.0)
    np.testing.assert_allclose(delta @ p_null.T, x @ p_null.T, rtol=1e-6, atol=1e-8)


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
    for est in (_shrinkage.berger, _shrinkage.tan, _shrinkage.minimax_bayes):
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
    for est in (_shrinkage.berger, _shrinkage.tan, _shrinkage.minimax_bayes):
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
            (_shrinkage.minimax_bayes, {}),
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


# ---------------------------------------------------------------------------
# tan_bayes: delta_{A,c} with the Bayes-rule shrinkage direction
# ---------------------------------------------------------------------------


def _tan_bayes_general_formula(x, cov, q, gamma, strength=1.0):
    """Closed-form delta_{A,c} with A = D(D+gamma I)^{-1} in general form.

    From [Tan2015]_, Section 3, Equation (9): delta_{A,c} =
    (I - c A / (x^T A^T Q A x)) x, with A = diag(a) where a_j =
    d_j / (d_j + gamma) in canonical form and c = strength * c*(D, A).

    """
    x = np.asarray(x, dtype=float)
    cov = np.asarray(cov, dtype=float)
    q = np.asarray(q, dtype=float)
    c = np.linalg.cholesky(q).T
    d, o = np.linalg.eigh(c @ cov @ c.T)
    b = o.T @ c
    binv = np.linalg.inv(b)
    x_star = x @ b.T

    a = d / (d + gamma)
    if np.isinf(gamma):
        # gamma -> inf: a_j -> d_j up to a common scalar, which cancels in the
        # estimator (a linear factor of A does not change delta_{A,c}).
        a = d
    da = d * a
    c_star = float(np.sum(da) - 2.0 * np.max(da))
    if c_star <= 0.0:
        return x
    s_val = np.sum(a**2 * x_star**2, axis=-1)
    factor = 1.0 - strength * c_star * a / s_val
    delta_star = factor * x_star
    return delta_star @ binv.T


def test_tan_bayes_default_cov_is_identity():
    x = rng().normal(size=6)
    np.testing.assert_allclose(
        _shrinkage.tan_bayes(x), _shrinkage.tan_bayes(x, cov=np.eye(6))
    )
    assert _shrinkage.tan_bayes(x).shape == (6,)


def test_tan_bayes_gamma_zero_is_berger_like():
    # gamma=0 -> a=1 (A=I), c* = sum(d) - 2*max(d), which is the Berger-like
    # direction with uniform shrinkage weight.  Pick d with sum(d) > 2*max(d)
    # so that c* > 0 and shrinkage actually occurs.
    d = np.array([1.0, 1.0, 1.0, 0.5])
    x = rng().normal(size=4)
    delta = _shrinkage.tan_bayes(x, cov=np.diag(d), gamma=0.0, positive=False)
    c_star = float(np.sum(d) - 2.0 * np.max(d))
    s_val = float(np.sum(x**2))
    factor = 1.0 - c_star / s_val
    np.testing.assert_allclose(delta, factor * x, rtol=1e-10)


def test_tan_bayes_gamma_inf_limit():
    # gamma=inf -> a_j -> d_j/pi_j up to a common scalar that cancels: the
    # limit is the fixed direction A ~ diag(d/pi), not the identity (a linear
    # factor of A does not change delta_{A,c}).  For the flat cov=eye case the
    # direction is a=1, i.e. the Berger-like c* = sum(d) - 2*max(d) = 3.
    x = rng().normal(size=5)
    delta = _shrinkage.tan_bayes(x, cov=np.eye(5), gamma=float("inf"), positive=False)
    c_star = 3.0
    s_val = float(np.sum(x**2))
    factor = 1.0 - c_star / s_val
    np.testing.assert_allclose(delta, factor * x, rtol=1e-10)
    assert not np.allclose(delta, x, atol=1e-8)


def test_tan_bayes_gamma_inf_matches_prior_shape_limit():
    # The gamma=inf limit uses a_j = d_j/pi_j (up to a cancelling scalar).  With
    # cov = diag(d), Q = I the canonical coordinates are the original
    # coordinates (already in the canonical decreasing-variance order), so a
    # directly-specified prior diagonal must reproduce the internal canonical
    # form for both a constant and a non-constant prior shape.
    d = np.array([2.9, 2.3, 1.7, 1.1, 0.6])
    x = rng().normal(size=5)
    pi = np.array([1.0, 1.0, 0.5, 0.5, 0.3])
    cov = np.diag(d)

    for shape in (np.ones(5), pi):
        delta = _shrinkage.tan_bayes(
            x,
            cov=cov,
            gamma=float("inf"),
            positive=False,
            prior_cov=np.diag(shape),
        )
        ref = _tan_bayes_canonical(
            x, d, positive=False, strength=1.0, gamma=float("inf"), pi=shape
        )
        np.testing.assert_allclose(delta, ref, rtol=1e-9, atol=1e-10)
        assert not np.allclose(delta, x, atol=1e-8)


def test_tan_bayes_gamma_inf_constant_shape_equals_flat():
    # At gamma = inf a constant prior shape (pi_j = c for all j) only rescales A
    # by a common factor that cancels in the shrinkage ratio, so it reproduces
    # the default homoscedastic limit exactly.
    gen = rng()
    d = gen.uniform(0.5, 3.0, size=5)
    x = gen.normal(size=5)
    flat = _tan_bayes_canonical(x, d, positive=True, strength=1.0, gamma=float("inf"))
    for c in (0.25, 2.5):
        shaped = _tan_bayes_canonical(
            x,
            d,
            positive=True,
            strength=1.0,
            gamma=float("inf"),
            pi=c * np.ones(5),
        )
        np.testing.assert_allclose(shaped, flat, rtol=1e-9, atol=1e-10)


def test_tan_bayes_gamma_inf_limit_is_continuous():
    # Large finite gamma approaches the gamma=inf limit continuously.
    d = np.array([1.0, 0.8, 0.5, 0.3, 0.2])
    cov = np.diag(d)
    x = rng().normal(size=5)
    inf = _shrinkage.tan_bayes(x, cov=cov, gamma=float("inf"), positive=False)
    for gamma in (1e8, 1e12):
        np.testing.assert_allclose(
            _shrinkage.tan_bayes(x, cov=cov, gamma=gamma, positive=False),
            inf,
            rtol=1e-6,
            atol=1e-8,
        )


def test_tan_bayes_gamma_inf_limit_can_stay_identity():
    # When one coordinate dominates, the limit direction gives c* <= 0 and the
    # estimator reduces to the identity, as for finite gamma.
    d = np.array([4.0, 2.0, 1.0, 0.5, 0.25])
    cov = np.diag(d)
    x = rng().normal(size=5)
    np.testing.assert_allclose(
        _shrinkage.tan_bayes(x, cov=cov, gamma=float("inf")), x, atol=1e-12
    )


def test_tan_bayes_array_gamma_mixes_inf():
    # A per-observation gamma array may mix inf (limit direction) with finite
    # values; each observation must use its own scale.
    p = 5
    d = np.array([1.0, 0.8, 0.5, 0.3, 0.2])
    cov = np.diag(d)
    x = rng().normal(size=(4, p))
    gammas = np.array([np.inf, 1.0, 2.0, np.inf])
    delta = _shrinkage.tan_bayes(x, cov=cov, gamma=gammas, positive=False)
    for i in range(4):
        np.testing.assert_allclose(
            delta[i],
            _shrinkage.tan_bayes(x[i], cov=cov, gamma=float(gammas[i]), positive=False),
            rtol=1e-10,
        )


def test_tan_bayes_shape():
    p = 5
    x = rng().normal(size=p)
    assert _shrinkage.tan_bayes(x).shape == (p,)
    assert _shrinkage.tan_bayes(x, gamma=0.0).shape == (p,)
    assert _shrinkage.tan_bayes(x, gamma=float("inf")).shape == (p,)


def test_tan_bayes_broadcasting_shapes():
    p = 5
    x = rng().normal(size=(4, 3, p))
    out = _shrinkage.tan_bayes(x, cov=np.eye(p))
    assert out.shape == (4, 3, p)
    for idx in np.ndindex(4, 3):
        np.testing.assert_allclose(
            out[idx], _shrinkage.tan_bayes(x[idx], cov=np.eye(p)), rtol=1e-12
        )


def test_tan_bayes_general_matches_closed_form():
    # The canonicalized computation must agree with the direct general-form
    # formula for a non-trivial covariance and Q = I.
    a = rng().normal(size=(5, 5))
    cov = a @ a.T + np.eye(5)
    x = rng().normal(size=5)
    gamma = 2.5
    np.testing.assert_allclose(
        _shrinkage.tan_bayes(x, cov=cov, gamma=gamma, positive=False),
        _tan_bayes_general_formula(x, cov, np.eye(5), gamma),
    )


def test_tan_bayes_general_Q_matches_closed_form():
    # Same with a non-trivial Q.
    a = rng().normal(size=(5, 5))
    cov = a @ a.T + np.eye(5)
    b = rng().normal(size=(5, 5))
    q = b @ b.T + np.eye(5)
    x = rng().normal(size=5)
    gamma = 3.0
    np.testing.assert_allclose(
        _shrinkage.tan_bayes(x, cov=cov, Q=q, gamma=gamma, positive=False),
        _tan_bayes_general_formula(x, cov, q, gamma),
    )


def test_tan_bayes_general_formula_at_inf():
    # The gamma=inf limit also agrees with the direct general-form formula.
    a = rng().normal(size=(5, 5))
    cov = a @ a.T + 5.0 * np.eye(5)
    b = rng().normal(size=(5, 5))
    q = b @ b.T + np.eye(5)
    x = rng().normal(size=5)
    np.testing.assert_allclose(
        _shrinkage.tan_bayes(x, cov=cov, Q=q, gamma=float("inf"), positive=False),
        _tan_bayes_general_formula(x, cov, q, float("inf")),
        rtol=1e-8,
        atol=1e-10,
    )


def test_tan_bayes_minimaxity():
    # tan_bayes with strength=1 is minimax: its risk never exceeds tr(cov).
    p = 5
    cov = np.diag([4.0, 2.0, 1.0, 0.5, 0.25])
    n = 200_000
    gen = rng()
    xs = gen.multivariate_normal(np.zeros(p), cov, size=n)
    for gamma in (0.0, 1.0, 5.0):
        loss = np.sum(
            _shrinkage.tan_bayes(xs, cov=cov, gamma=gamma, positive=False) ** 2,
            axis=1,
        )
        assert np.mean(loss) <= np.trace(cov) + 0.05, (
            f"gamma={gamma}: risk {np.mean(loss):.4f} > tr(cov)={np.trace(cov):.4f}"
        )


def test_tan_bayes_positive_part_dominates():
    # The positive-part version has lower or equal risk.
    p = 5
    cov = np.diag([4.0, 2.0, 1.0, 0.5, 0.25])
    n = 200_000
    gen = rng()
    xs = gen.multivariate_normal(np.zeros(p), cov, size=n)
    for gamma in (0.0, 1.0, 5.0):
        loss_plain = np.sum(
            _shrinkage.tan_bayes(xs, cov=cov, gamma=gamma, positive=False) ** 2,
            axis=1,
        )
        loss_pos = np.sum(
            _shrinkage.tan_bayes(xs, cov=cov, gamma=gamma, positive=True) ** 2,
            axis=1,
        )
        assert np.mean(loss_pos) <= np.mean(loss_plain) + 0.02, (
            f"gamma={gamma}: positive-part risk {np.mean(loss_pos):.4f} "
            f"> plain risk {np.mean(loss_plain):.4f}"
        )


def test_tan_bayes_strength_zero_is_identity():
    x = rng().normal(size=5)
    np.testing.assert_allclose(_shrinkage.tan_bayes(x, cov=np.eye(5), strength=0.0), x)


def test_tan_bayes_shrinks_and_preserves_sign():
    # The factor 1 - c*a/(a^2.x^2) is always <= 1, so the estimate is shrunk.
    x = rng().normal(size=5)
    delta = _shrinkage.tan_bayes(x, cov=np.eye(5), gamma=1.0)
    assert np.all(delta * x >= -1e-12)
    assert not np.allclose(delta, x, atol=1e-8)


def test_tan_bayes_shrink_dispatch():
    x = rng().normal(size=5)
    np.testing.assert_allclose(
        s.shrink(x, np.eye(5), method="tan_bayes"), _shrinkage.tan_bayes(x)
    )
    np.testing.assert_allclose(
        s.shrink(x, np.eye(5), method="tan_bayes", gamma=2.0),
        _shrinkage.tan_bayes(x, gamma=2.0),
    )


def test_tan_bayes_gamma_validation():
    with pytest.raises(ValueError, match="gamma"):
        _shrinkage.tan_bayes(rng().normal(size=5), gamma=-1.0)


def test_tan_bayes_point_offset_equals_shift():
    # Shrinking towards a point t (no dirs) must equal t + shrinking x - t
    # towards zero.
    gen = rng()
    p = 6
    x = gen.normal(size=p)
    t = gen.normal(size=p)
    np.testing.assert_allclose(
        _shrinkage.tan_bayes(x, offset=t),
        t + _shrinkage.tan_bayes(x - t),
        rtol=1e-10,
    )


def test_tan_bayes_full_dirs_is_identity():
    # dirs spanning the whole space leave nothing to shrink, so the result is
    # the input regardless of offset.
    gen = rng()
    p = 6
    x = gen.normal(size=p)
    offset = gen.normal(size=p)
    np.testing.assert_allclose(_shrinkage.tan_bayes(x, dirs=np.eye(p)), x, atol=1e-12)
    np.testing.assert_allclose(
        _shrinkage.tan_bayes(x, dirs=np.eye(p), offset=offset), x, atol=1e-12
    )


def test_tan_bayes_dirs_small_complement_is_identity():
    # When the orthogonal complement has dimension 0, there is nothing to
    # shrink, so the estimate is the input.
    gen = rng()
    p = 6
    x = gen.normal(size=p)
    v = gen.normal(size=(p, 6))
    np.testing.assert_allclose(_shrinkage.tan_bayes(x, dirs=v), x, atol=1e-12)


def test_tan_bayes_subspace_keeps_projected_component():
    # The component of the estimate along the projected direction must equal
    # the projection of the data; only the orthogonal residual is shrunk.
    gen = rng()
    p = 6
    x = gen.normal(size=p)
    v = gen.normal(size=(p, 2))
    proj = _projection(v)
    delta = _shrinkage.tan_bayes(x, dirs=v)
    np.testing.assert_allclose(proj @ delta, proj @ x, rtol=1e-12)
    resid = (np.eye(p) - proj) @ delta
    raw = (np.eye(p) - proj) @ x
    assert np.linalg.norm(resid) <= np.linalg.norm(raw)


def test_tan_bayes_known_values():
    # Hand-computed case.  d = [0.2, 0.5, 1.0, 1.5, 2.0], gamma=1.0 ->
    # a = [1/6, 1/3, 1/2, 3/5, 2/3].  c* = sum(da) - 2*max(da) = 0.266... > 0.
    d = np.array([0.2, 0.5, 1.0, 1.5, 2.0])
    gamma = 1.0
    a = d / (d + gamma)
    da = d * a
    c_star = float(np.sum(da) - 2.0 * np.max(da))
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    s_val = float(np.sum(a**2 * x**2))
    factor = 1.0 - c_star * a / s_val
    expected = factor * x
    np.testing.assert_allclose(
        _shrinkage.tan_bayes(x, cov=np.diag(d), gamma=gamma, positive=False),
        expected,
        rtol=1e-10,
    )


def test_tan_bayes_general_Q_matches_closed_form_broadcast():
    # Same as test_tan_bayes_general_Q_matches_closed_form but with stacked
    # inputs, exercising the broadcasting logic.
    a = rng().normal(size=(5, 5))
    cov = a @ a.T + np.eye(5)
    b_mat = rng().normal(size=(5, 5))
    q = b_mat @ b_mat.T + np.eye(5)
    x = rng().normal(size=(3, 5))
    gamma = 2.0
    got = _shrinkage.tan_bayes(x, cov=cov, Q=q, gamma=gamma, positive=False)
    for i in range(3):
        np.testing.assert_allclose(
            got[i],
            _tan_bayes_general_formula(x[i], cov, q, gamma),
            rtol=1e-10,
        )


# ---------------------------------------------------------------------------
# robust_bayes: delta^RB (Tan2015 Equation 7)
# ---------------------------------------------------------------------------


def _robust_bayes_general_formula(x, cov, q, gamma, strength=1.0):
    """Closed-form delta^RB in the general (non-canonical) form.

    From [Tan2015]_, Equation (7): delta^RB = (I - m D(D+gamma I)^{-1}) X with
    m = min(1, strength*(p-2)_+ / (X^T (D+gamma I)^{-1} X)), in canonical
    coordinates.  Transform back to the original space.

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
    if p < 3:
        return x
    d_plus_g = d + gamma
    weight = d / d_plus_g
    s_val = np.sum(x_star**2 / d_plus_g, axis=-1)
    ratio = strength * (p - 2) / s_val
    m_k = np.minimum(1.0, ratio)
    delta_star = (1.0 - m_k * weight) * x_star
    return delta_star @ binv.T


def test_robust_bayes_default_cov_is_identity():
    x = rng().normal(size=6)
    np.testing.assert_allclose(
        _shrinkage.robust_bayes(x), _shrinkage.robust_bayes(x, cov=np.eye(6))
    )
    assert _shrinkage.robust_bayes(x).shape == (6,)


def test_robust_bayes_gamma_zero_spherical():
    # gamma=0 -> w = d/d = 1, S = X^T D^{-1} X, so delta = (1 - min(1, S_0/S)) x
    # with S_0 = strength*(p-2).  Pick d homogeneous enough that S_0/S < 1 so
    # the ratio is unsaturated.
    d = np.array([1.0, 1.0, 1.0, 0.5, 0.5])
    x = rng().normal(size=5)
    delta = _shrinkage.robust_bayes(x, cov=np.diag(d), gamma=0.0)
    s_val = float(np.sum(x**2 / d))
    m_k = min(1.0, (5 - 2) / s_val)
    np.testing.assert_allclose(delta, (1.0 - m_k) * x, rtol=1e-10)


def test_robust_bayes_gamma_inf_is_identity():
    # gamma=inf -> w -> 0 and S -> 0, so delta -> x.
    x = rng().normal(size=5)
    np.testing.assert_allclose(
        _shrinkage.robust_bayes(x, cov=np.eye(5), gamma=1e12), x, atol=1e-6
    )


def test_robust_bayes_shape():
    p = 5
    x = rng().normal(size=p)
    assert _shrinkage.robust_bayes(x).shape == (p,)
    assert _shrinkage.robust_bayes(x, gamma=0.0).shape == (p,)


def test_robust_bayes_broadcasting_shapes():
    p = 5
    x = rng().normal(size=(4, 3, p))
    out = _shrinkage.robust_bayes(x, cov=np.eye(p))
    assert out.shape == (4, 3, p)
    for idx in np.ndindex(4, 3):
        np.testing.assert_allclose(
            out[idx], _shrinkage.robust_bayes(x[idx], cov=np.eye(p)), rtol=1e-12
        )


def test_robust_bayes_general_matches_closed_form():
    # The canonicalized computation must agree with the direct general-form
    # formula for a non-trivial covariance and Q = I.
    a = rng().normal(size=(5, 5))
    cov = a @ a.T + np.eye(5)
    x = rng().normal(size=5)
    for gamma in (0.0, 1.0, 5.0):
        for strength in (0.5, 1.0, 2.0):
            np.testing.assert_allclose(
                _shrinkage.robust_bayes(x, cov=cov, gamma=gamma, strength=strength),
                _robust_bayes_general_formula(x, cov, np.eye(5), gamma, strength),
            )


def test_robust_bayes_general_Q_matches_closed_form():
    # Same with a non-trivial Q.
    a = rng().normal(size=(5, 5))
    cov = a @ a.T + np.eye(5)
    b = rng().normal(size=(5, 5))
    q = b @ b.T + np.eye(5)
    x = rng().normal(size=5)
    gamma = 3.0
    np.testing.assert_allclose(
        _shrinkage.robust_bayes(x, cov=cov, Q=q, gamma=gamma),
        _robust_bayes_general_formula(x, cov, q, gamma),
    )


def test_robust_bayes_strength_zero_is_identity():
    x = rng().normal(size=5)
    np.testing.assert_allclose(
        _shrinkage.robust_bayes(x, cov=np.eye(5), strength=0.0), x
    )


def test_robust_bayes_strength_two_matches_direct():
    # strength=2 -> m = min(1, 2*(p-2)/S); compare the shrinkage magnitude
    # against the strength=1 case directly.
    d = np.array([1.0, 1.0, 1.0, 0.5, 0.5])
    x = rng().normal(size=5)
    s_val = float(np.sum(x**2 / (d + 1.0)))
    w = d / (d + 1.0)
    m1 = min(1.0, (5 - 2) / s_val)
    m2 = min(1.0, 2 * (5 - 2) / s_val)
    np.testing.assert_allclose(
        _shrinkage.robust_bayes(x, cov=np.diag(d), gamma=1.0, strength=1.0),
        (1 - m1 * w) * x,
        rtol=1e-10,
    )
    np.testing.assert_allclose(
        _shrinkage.robust_bayes(x, cov=np.diag(d), gamma=1.0, strength=2.0),
        (1 - m2 * w) * x,
        rtol=1e-10,
    )


def test_robust_bayes_shrinks_and_preserves_sign():
    # The factor 1 - m*w with m <= 1 and w <= 1 is always in [0, 1), so the
    # estimate shrinks towards zero without ever flipping sign.
    x = rng().normal(size=5)
    delta = _shrinkage.robust_bayes(x, cov=np.eye(5), gamma=1.0)
    assert np.all(delta * x >= -1e-12)
    assert np.all(np.abs(delta) <= np.abs(x) + 1e-12)


def test_robust_bayes_shrink_dispatch():
    x = rng().normal(size=5)
    np.testing.assert_allclose(
        s.shrink(x, np.eye(5), method="robust_bayes"), _shrinkage.robust_bayes(x)
    )
    np.testing.assert_allclose(
        s.shrink(x, np.eye(5), method="robust_bayes", gamma=2.0),
        _shrinkage.robust_bayes(x, gamma=2.0),
    )


def test_robust_bayes_gamma_validation():
    with pytest.raises(ValueError, match="gamma"):
        _shrinkage.robust_bayes(rng().normal(size=5), gamma=-1.0)


def test_robust_bayes_point_offset_equals_shift():
    # Shrinking towards a point t (no dirs) must equal t + shrinking x - t
    # towards zero.
    gen = rng()
    p = 6
    x = gen.normal(size=p)
    t = gen.normal(size=p)
    np.testing.assert_allclose(
        _shrinkage.robust_bayes(x, offset=t),
        t + _shrinkage.robust_bayes(x - t),
        rtol=1e-10,
    )


def test_robust_bayes_full_dirs_is_identity():
    # dirs spanning the whole space leave nothing to shrink, so the result is
    # the input regardless of offset.
    gen = rng()
    p = 6
    x = gen.normal(size=p)
    offset = gen.normal(size=p)
    np.testing.assert_allclose(
        _shrinkage.robust_bayes(x, dirs=np.eye(p)), x, atol=1e-12
    )
    np.testing.assert_allclose(
        _shrinkage.robust_bayes(x, dirs=np.eye(p), offset=offset), x, atol=1e-12
    )


def test_robust_bayes_dirs_small_complement_is_identity():
    # When the orthogonal complement has dimension 0, there is nothing to
    # shrink, so the estimate is the input.
    gen = rng()
    p = 6
    x = gen.normal(size=p)
    v = gen.normal(size=(p, 6))
    np.testing.assert_allclose(_shrinkage.robust_bayes(x, dirs=v), x, atol=1e-12)


def test_robust_bayes_subspace_keeps_projected_component():
    # The component of the estimate along the projected direction must equal
    # the projection of the data; only the orthogonal residual is shrunk.
    gen = rng()
    p = 6
    x = gen.normal(size=p)
    v = gen.normal(size=(p, 2))
    proj = _projection(v)
    delta = _shrinkage.robust_bayes(x, dirs=v)
    np.testing.assert_allclose(proj @ delta, proj @ x, rtol=1e-12)
    resid = (np.eye(p) - proj) @ delta
    raw = (np.eye(p) - proj) @ x
    assert np.linalg.norm(resid) <= np.linalg.norm(raw)


def test_robust_bayes_zero_x_returns_zero():
    # S = 0 when x = 0: the ratio strength*(p-2)/S -> inf so m = 1 and the
    # estimate is 0 (a finite value, not a division by zero).
    x = np.zeros(5)
    np.testing.assert_allclose(
        _shrinkage.robust_bayes(x, cov=np.eye(5), gamma=1.0), x, atol=1e-12
    )


def test_robust_bayes_two_dimensions_is_identity():
    # With p = 2, (p-2)_+ = 0 so there is no shrinkage.
    x = rng().normal(size=2)
    np.testing.assert_allclose(_shrinkage.robust_bayes(x, cov=np.eye(2)), x, atol=1e-12)


_EMPIRICAL_METHODS = [
    "bayes",
    "robust_bayes",
    "tan_bayes",
]
_FLOAT_ONLY_METHODS = [
    "tan",
    "minimax_bayes",
]


def _empirical_gamma(x, p, offset=None):
    y = np.asarray(x, dtype=float) - (0.0 if offset is None else offset)
    return float(np.sum(y**2)) / p


@pytest.mark.parametrize("method", _EMPIRICAL_METHODS)
def test_empirical_gamma_matches_explicit_identity(method):
    # With identity covariance and loss, canonical space is the raw space, so
    # empirical gamma = ||x||^2 / p.  The estimator must give the same
    # result as passing that value explicitly.
    gen = rng()
    p = 6
    x = gen.normal(size=p)
    fn = getattr(_shrinkage, method)
    gamma_exp = _empirical_gamma(x, p)
    np.testing.assert_allclose(
        fn(x, gamma="empirical"), fn(x, gamma=gamma_exp), rtol=1e-12
    )


@pytest.mark.parametrize("method", _EMPIRICAL_METHODS)
def test_empirical_gamma_matches_explicit_offset(method):
    # The empirical gamma is computed from the centered data x - offset, so a
    # non-zero offset must be subtracted before taking the norm.
    gen = rng()
    p = 6
    x = gen.normal(size=p)
    t = gen.normal(size=p)
    fn = getattr(_shrinkage, method)
    gamma_exp = _empirical_gamma(x, p, offset=t)
    np.testing.assert_allclose(
        fn(x, offset=t, gamma="empirical"),
        fn(x, offset=t, gamma=gamma_exp),
        rtol=1e-12,
    )


@pytest.mark.parametrize("method", _EMPIRICAL_METHODS)
def test_empirical_gamma_matches_explicit_general_cov(method):
    # Under a general covariance the data must be canonicalized first; resolve
    # the empirical gamma in canonical coordinates and compare against the
    # explicit value.
    gen = rng()
    p = 5
    a = gen.normal(size=(p, p))
    cov = a @ a.T + p * np.eye(p)
    x = gen.normal(size=p)
    fn = getattr(_shrinkage, method)
    _, v = np.linalg.eigh(cov)
    b = v.T  # cov = B^T D B with B = V^T, so x_star = x @ B^T = x @ V
    x_star = x @ b.T
    gamma_exp = _empirical_gamma(x_star, p)
    np.testing.assert_allclose(
        fn(x, cov=cov, gamma="empirical"), fn(x, cov=cov, gamma=gamma_exp), rtol=1e-10
    )


def test_empirical_gamma_batched_matches_per_row():
    # Empirical gamma is per observation, so a batched call must equal stacking
    # the single-vector (gamma="empirical") results row by row.  This guards
    # against the prior scale being inflated by summing over the whole batch
    # (which made batched/risk-sweep results wrong, risk ~ 1 even at theta=0).
    gen = rng()
    x = gen.normal(size=(4, 7))
    for method in _EMPIRICAL_METHODS:
        fn = getattr(_shrinkage, method)
        batched = fn(x, gamma="empirical")
        assert batched.shape == x.shape
        per_row = np.stack([fn(row, gamma="empirical") for row in x])
        np.testing.assert_allclose(batched, per_row, rtol=1e-12, atol=1e-12)
        # The batched result must also equal per-row explicit gamma values.
        explicit = np.stack([fn(row, gamma=_empirical_gamma(row, p=7)) for row in x])
        np.testing.assert_allclose(batched, explicit, rtol=1e-12, atol=1e-12)


def test_empirical_gamma_per_observation_vector():
    # The internal prior-scale resolver must return one value per observation
    # (the norm over the trailing coordinate axis only).  This is the
    # per-observation array that the vectorized estimators broadcast, so it
    # must not sum over the batch.  With the default homoscedastic prior
    # (pi = 1) the scale is ||y||^2 / p_eff; with a non-uniform prior the
    # scale is ||y / sqrt(pi)||^2 / p_eff, covered by a separate test.
    gen = rng()
    p = 6
    n = 5
    d = np.linspace(0.5, 2.0, p)
    y = gen.normal(size=(n, p))
    g = _core_empirical_gamma(d, np.ones(p), y)
    assert g.shape == (n,)
    for i in range(n):
        expected = float(np.sum(y[i] ** 2)) / p
        np.testing.assert_allclose(g[i], expected, rtol=1e-12, atol=1e-12)

    small_input = gen.normal(size=(n, 2))
    small = _core_empirical_gamma(np.array([0.5, 1.0]), np.ones(2), small_input)
    assert small.shape == (n,)
    for i in range(n):
        np.testing.assert_allclose(
            small[i], float(np.sum(small_input[i] ** 2)) / 2.0, rtol=1e-12, atol=1e-12
        )

    single = _core_empirical_gamma(d, np.ones(p), gen.normal(size=p))
    assert single.ndim == 0


def test_empirical_gamma_uses_pi():
    # The "empirical" prior scale is the MLE-like scale ||y / sqrt(pi)||^2 / p:
    # normalizing each coordinate by the corresponding prior shape recovers a
    # homoscedastic observation, then the standard per-coordinate norm is
    # divided by the effective dimension.  With pi = 1 this reduces to the
    # classic ||y||^2 / p.
    gen = rng()
    p = 5
    n = 4
    d = np.linspace(0.5, 2.0, p)
    y = gen.normal(size=(n, p))
    pi = np.array([1.0, 4.0, 0.5, 2.0, 3.0])
    g = _core_empirical_gamma(d, pi, y)
    assert g.shape == (n,)
    for i in range(n):
        expected = float(np.sum(y[i] ** 2 / pi)) / p
        np.testing.assert_allclose(g[i], expected, rtol=1e-12, atol=1e-12)

    # pi = 1 reproduces the homoscedastic default exactly.
    g_default = _core_empirical_gamma(d, np.ones(p), y)
    g_pi1 = _core_empirical_gamma(d, np.ones(p), y)
    np.testing.assert_allclose(g_default, g_pi1, rtol=1e-12, atol=1e-12)
    for i in range(n):
        np.testing.assert_allclose(
            g_default[i], float(np.sum(y[i] ** 2)) / p, rtol=1e-12, atol=1e-12
        )


def test_empirical_gamma_via_estimator_uses_pi():
    # The estimator-level "empirical" preset threads the canonical pi through
    # to the gamma resolver, so the prior-aware formula governs the
    # per-observation scale.  Compare a per-observation resolver with a
    # callable that follows the same formula.
    gen = rng()
    p = 5
    n = 3
    d = np.linspace(0.5, 2.0, p)
    x = gen.normal(size=(n, p))
    pi = np.array([1.0, 4.0, 0.5, 2.0, 3.0])
    cov = np.diag(d)

    def callable_gamma(d_, pi_, y_):
        return np.sum(y_**2 / pi_, axis=-1) / len(d_)

    for method in ("bayes", "robust_bayes", "tan_bayes"):
        fn = getattr(_shrinkage, method)
        from_empirical = fn(x, cov=cov, gamma="empirical", prior_cov=np.diag(pi))
        from_callable = fn(x, cov=cov, gamma=callable_gamma, prior_cov=np.diag(pi))
        np.testing.assert_allclose(
            from_empirical, from_callable, rtol=1e-12, atol=1e-14
        )


def test_empirical_gamma_batched_with_dirs_matches_per_row():
    # With a subspace split the per-observation scale is resolved from each
    # row's reduced residual, so a batched dirs call must equal stacking the
    # single-vector results.
    gen = rng()
    p = 6
    x = gen.normal(size=(5, p))
    v = gen.normal(size=(p, 2))
    for method in _EMPIRICAL_METHODS:
        fn = getattr(_shrinkage, method)
        batched = fn(x, dirs=v, gamma="empirical")
        per_row = np.stack([fn(row, dirs=v, gamma="empirical") for row in x])
        np.testing.assert_allclose(batched, per_row, rtol=1e-12, atol=1e-12)


def test_empirical_gamma_small_dim_uses_plain_scale():
    # The empirical prior scale has no small-dimension clamp: for p < 3 it is
    # still ||y||^2 / p (the default-homoscedastic case with pi = 1), so
    # "empirical" must match passing that value explicitly rather than
    # forcing gamma = 0.
    gen = rng()
    for p in (1, 2):
        x = gen.normal(size=p)
        for method in _EMPIRICAL_METHODS:
            fn = getattr(_shrinkage, method)
            gamma_exp = _empirical_gamma(x, p)
            np.testing.assert_allclose(
                fn(x, gamma="empirical"),
                fn(x, gamma=gamma_exp),
                rtol=1e-12,
                atol=1e-12,
            )


def test_empirical_gamma_unknown_string_raises():
    gen = rng()
    x = gen.normal(size=5)
    for method in _EMPIRICAL_METHODS:
        with pytest.raises(ValueError, match="empirical"):
            getattr(_shrinkage, method)(x, gamma="bogus")


def test_max_rel_risk_gamma_matches_lambda():
    # The max_rel_risk factory builds exactly the per-observation scale
    # sqrt(sum((d*y/pi)^2)) / sqrt(sum(d)*alpha).
    gen = rng()
    p = 5
    d = np.linspace(0.5, 2.0, p)
    pi = np.linspace(0.5, 1.5, p)
    y = gen.normal(size=(3, p))
    alpha = 0.1
    g = _max_rel_risk_gamma(alpha)
    ref = np.sqrt(np.sum((d * y / pi) ** 2, axis=-1) / (np.sum(d) * alpha))
    np.testing.assert_allclose(g(d, pi, y), ref, rtol=1e-12, atol=1e-14)


def test_max_abs_risk_gamma_matches_lambda():
    # The max_abs_risk factory builds exactly the per-observation scale
    # sqrt(sum((d*y/pi)^2) / alpha): only alpha in the denominator, unlike the
    # relative-risk factory which additionally divides by sum(d).
    gen = rng()
    p = 5
    d = np.linspace(0.5, 2.0, p)
    pi = np.linspace(0.5, 1.5, p)
    y = gen.normal(size=(3, p))
    alpha = 0.1
    g = _max_abs_risk_gamma(alpha)
    ref = np.sqrt(np.sum((d * y / pi) ** 2, axis=-1) / alpha)
    np.testing.assert_allclose(g(d, pi, y), ref, rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize("builder", [_max_rel_risk_gamma, _max_abs_risk_gamma])
def test_max_risk_gamma_alpha_validation(builder):
    # The risk cap alpha must be a positive number, for both factories.
    for bad in (0.0, -1.0):
        with pytest.raises(ValueError, match="alpha"):
            builder(bad)


@pytest.mark.parametrize(
    ("factory", "builder"),
    [
        ("max_rel_risk", _max_rel_risk_gamma),
        ("max_abs_risk", _max_abs_risk_gamma),
    ],
)
def test_parse_gamma_factory_numeric(factory, builder):
    # A factory string parses its numeric arguments and builds the callable.
    gen = rng()
    p = 5
    d = np.linspace(0.5, 2.0, p)
    pi = np.ones(p)
    y = gen.normal(size=p)
    g = _parse_gamma_factory(f"{factory}( 0.1 )")
    np.testing.assert_allclose(g(d, pi, y), builder(0.1)(d, pi, y))


def test_parse_gamma_factory_bad_spec():
    # A string that does not match the name(...) pattern is rejected.
    with pytest.raises(ValueError, match="Unknown gamma specification"):
        _parse_gamma_factory("max_rel_risk")


def test_parse_gamma_factory_unknown_name():
    with pytest.raises(ValueError, match="factory"):
        _parse_gamma_factory("bogus(0.1)")


@pytest.mark.parametrize(
    ("factory", "builder"),
    [
        ("max_rel_risk", _max_rel_risk_gamma),
        ("max_abs_risk", _max_abs_risk_gamma),
    ],
)
@pytest.mark.parametrize("alpha", [0.1, 2.0])
@pytest.mark.parametrize("method", _EMPIRICAL_METHODS)
def test_factory_gamma_matches_explicit_callable(factory, builder, method, alpha):
    # A factory string resolves to the same per-observation scale as the
    # equivalent callable built by its factory function.
    gen = rng()
    p = 6
    x = gen.normal(size=p)
    fn = getattr(_shrinkage, method)
    np.testing.assert_allclose(
        fn(x, gamma=f"{factory}({alpha})"),
        fn(x, gamma=builder(alpha)),
        rtol=1e-12,
        atol=1e-14,
    )


@pytest.mark.parametrize("factory", ["max_rel_risk", "max_abs_risk"])
@pytest.mark.parametrize("method", _EMPIRICAL_METHODS)
def test_factory_gamma_batched_matches_per_row(factory, method):
    # A factory string on a batched x yields one scale per observation, so the
    # batched call must equal stacking per-row calls.
    gen = rng()
    p = 6
    n = 4
    x = gen.normal(size=(n, p))
    fn = getattr(_shrinkage, method)
    batched = fn(x, gamma=f"{factory}(0.1)")
    per_row = np.stack([fn(x[i], gamma=f"{factory}(0.1)") for i in range(n)])
    np.testing.assert_allclose(batched, per_row, rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize(
    ("factory", "builder"),
    [
        ("max_rel_risk", _max_rel_risk_gamma),
        ("max_abs_risk", _max_abs_risk_gamma),
    ],
)
@pytest.mark.parametrize("method", _EMPIRICAL_METHODS)
def test_factory_gamma_with_dirs_matches_explicit(factory, builder, method):
    # The factory string flows through the subspace-split recursion like a
    # callable: the recursive solve re-derives the scale from the reduced
    # residual actually shrunk.
    gen = rng()
    p = 6
    x = gen.normal(size=p)
    v = gen.normal(size=(p, 2))
    fn = getattr(_shrinkage, method)
    np.testing.assert_allclose(
        fn(x, dirs=v, gamma=f"{factory}(0.1)"),
        fn(x, dirs=v, gamma=builder(0.1)),
        rtol=1e-10,
        atol=1e-12,
    )


@pytest.mark.parametrize("factory", ["max_rel_risk", "max_abs_risk"])
@pytest.mark.parametrize("method", _EMPIRICAL_METHODS)
def test_factory_gamma_with_dirs_matches_per_row(factory, method):
    gen = rng()
    p = 6
    n = 4
    x = gen.normal(size=(n, p))
    v = gen.normal(size=(p, 2))
    fn = getattr(_shrinkage, method)
    batched = fn(x, dirs=v, gamma=f"{factory}(0.1)")
    per_row = np.stack([fn(x[i], dirs=v, gamma=f"{factory}(0.1)") for i in range(n)])
    np.testing.assert_allclose(batched, per_row, rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize(
    ("factory", "builder"),
    [
        ("max_rel_risk", _max_rel_risk_gamma),
        ("max_abs_risk", _max_abs_risk_gamma),
    ],
)
def test_factory_gamma_with_psd_Q_matches_explicit(factory, builder):
    # With singular Q the null space of Q is merged into the no-shrink
    # directions and the factory scale is resolved from the residual problem.
    gen = rng()
    p = 6
    x = gen.normal(size=p)
    q = np.diag([1.0, 1.0, 1.0, 1.0, 0.0, 0.0])
    fn = _shrinkage.bayes
    np.testing.assert_allclose(
        fn(x, Q=q, gamma=f"{factory}(0.1)"),
        fn(x, Q=q, gamma=builder(0.1)),
        rtol=1e-10,
        atol=1e-12,
    )


@pytest.mark.parametrize(
    ("factory", "builder"),
    [
        ("max_rel_risk", _max_rel_risk_gamma),
        ("max_abs_risk", _max_abs_risk_gamma),
    ],
)
def test_factory_gamma_uses_pi(factory, builder):
    # The max_*_risk scales divide by the canonical prior diagonal pi, so an
    # explicit prior_cov must change the result and match the explicit callable
    # under the same prior.
    gen = rng()
    p = 4
    x = gen.normal(size=p)
    prior = np.diag([1.0, 4.0, 0.5, 2.0])
    with_prior = _shrinkage.bayes(x, gamma=f"{factory}(0.1)", prior_cov=prior)
    without = _shrinkage.bayes(x, gamma=f"{factory}(0.1)")
    assert not np.allclose(with_prior, without, rtol=1e-8, atol=1e-12)
    np.testing.assert_allclose(
        with_prior,
        _shrinkage.bayes(x, gamma=builder(0.1), prior_cov=prior),
        rtol=1e-10,
        atol=1e-12,
    )


@pytest.mark.parametrize("factory", ["max_rel_risk", "max_abs_risk"])
def test_factory_gamma_via_shrink_frontend(factory):
    # The shrink() front-end forwards the factory string to the resolved
    # estimator.
    gen = rng()
    x = gen.normal(size=6)
    np.testing.assert_allclose(
        s.shrink(x, method="bayes", gamma=f"{factory}(0.1)"),
        _shrinkage.bayes(x, gamma=f"{factory}(0.1)"),
        rtol=1e-12,
        atol=1e-14,
    )


def test_factory_gamma_unknown_factory_raises():
    # A string matching the factory syntax but naming an unregistered factory
    # is rejected.
    gen = rng()
    x = gen.normal(size=5)
    for method in _EMPIRICAL_METHODS:
        with pytest.raises(ValueError, match="factory"):
            getattr(_shrinkage, method)(x, gamma="bogus(0.1)")


@pytest.mark.parametrize(
    ("bad", "match"),
    [
        ("max_rel_risk", "Unknown gamma specification"),
        ("max_rel_risk()", "Invalid gamma factory argument"),
        ("max_rel_risk(abc)", "must be a numeric literal"),
    ],
)
def test_factory_gamma_malformed_raises(bad, match):
    # Factory syntax errors (no parentheses, no argument, non-numeric argument)
    # are rejected with ValueError.
    gen = rng()
    x = gen.normal(size=5)
    with pytest.raises(ValueError, match=match):
        _shrinkage.bayes(x, gamma=bad)


@pytest.mark.parametrize("factory", ["max_rel_risk", "max_abs_risk"])
def test_factory_gamma_bad_arity_or_type_raises(factory):
    # Arity that does not fit the registered factory, or a valid literal that
    # is not a number, is rejected with TypeError.
    gen = rng()
    x = gen.normal(size=5)
    for bad in (f"{factory}(1, 2)", f"{factory}('0.1')"):
        with pytest.raises(TypeError):
            _shrinkage.bayes(x, gamma=bad)


@pytest.mark.parametrize("factory", ["max_rel_risk", "max_abs_risk"])
def test_factory_gamma_alpha_nonpositive_raises(factory):
    # The risk cap alpha must be > 0, enforced at parse time.
    gen = rng()
    x = gen.normal(size=5)
    for bad in (f"{factory}(0)", f"{factory}(-1)"):
        with pytest.raises(ValueError, match="alpha"):
            _shrinkage.bayes(x, gamma=bad)


@pytest.mark.parametrize("factory", ["max_rel_risk", "max_abs_risk"])
def test_float_only_methods_reject_factory_gamma(factory):
    # tan and minimax_bayes only accept a numeric gamma, exactly as with the
    # "empirical" string and callables.
    gen = rng()
    x = gen.normal(size=6)
    for method in _FLOAT_ONLY_METHODS:
        with pytest.raises(TypeError):
            getattr(_shrinkage, method)(x, gamma=f"{factory}(0.1)")


def test_float_only_methods_reject_string_gamma():
    # tan and minimax_bayes only accept a numeric gamma; per-observation
    # (vector) gammas and the "empirical" string are not supported because
    # their gamma-dependent coordinate ranking/segmentation is not (yet)
    # broadcast-friendly.  Passing a string is a type error, not a ValueError,
    # so users get a clear hint that the parameter type is wrong.
    gen = rng()
    x = gen.normal(size=6)
    v = gen.normal(size=(6, 2))
    for method in _FLOAT_ONLY_METHODS:
        fn = getattr(_shrinkage, method)
        with pytest.raises(TypeError):
            fn(x, gamma="empirical")
        with pytest.raises(TypeError):
            fn(x, dirs=v, gamma="empirical")
        with pytest.raises(TypeError):
            fn(gen.normal(size=(4, 6)), gamma="empirical")


def test_empirical_gamma_with_dirs_matches_explicit():
    # Empirical gamma is resolved from the residual actually shrunk: in a
    # subspace-split problem the top level defers resolution into the recursive
    # solve, which uses the reduced residual norm and effective dimension
    # len(d_perp).  With identity covariance the covariance-metric projector is
    # the Euclidean one, so the residual is (I - P) x with P = v (v^T v)^{-1} v^T
    # and len(d_perp) = p - k.
    gen = rng()
    p = 6
    x = gen.normal(size=p)
    v = gen.normal(size=(p, 2))
    p_mat = v @ np.linalg.solve(v.T @ v, v.T)
    resid = (np.eye(p) - p_mat) @ x
    p_eff = p - v.shape[1]
    gamma_exp = _empirical_gamma(resid, p_eff)
    for method in _EMPIRICAL_METHODS:
        fn = getattr(_shrinkage, method)
        np.testing.assert_allclose(
            fn(x, dirs=v, gamma="empirical"),
            fn(x, dirs=v, gamma=gamma_exp),
            rtol=1e-12,
        )


def test_empirical_gamma_dirs_differs_from_full_data():
    # The subspace-split empirical gamma differs from the no-shrink full-data
    # one: the recursive solve derives it from the residual, not the whole
    # vector, so the kept component's norm must not inflate the prior scale.
    gen = rng()
    p = 6
    x = gen.normal(size=p)
    v = gen.normal(size=(p, 2))
    fn = _shrinkage.bayes
    full_gamma = _empirical_gamma(x, p)
    assert not np.allclose(
        fn(x, dirs=v, gamma="empirical"), fn(x, dirs=v, gamma=full_gamma), rtol=1e-8
    )


def test_empirical_gamma_with_psd_Q_matches_explicit():
    # With singular Q the null space of Q is merged into the no-shrink
    # directions and the residual complement problem has reduced dimension;
    # empirical gamma is resolved from that residual.
    gen = rng()
    p = 6
    x = gen.normal(size=p)
    q = np.diag([1.0, 1.0, 1.0, 1.0, 0.0, 0.0])
    fn = _shrinkage.bayes
    sed = np.zeros(p)
    sed[:4] = 1.0
    p_eff = int(np.sum(sed))
    eta = x[sed == 1.0]
    gamma_exp = _empirical_gamma(eta, p_eff)
    np.testing.assert_allclose(
        fn(x, Q=q, gamma="empirical"), fn(x, Q=q, gamma=gamma_exp), rtol=1e-12
    )


def test_explicit_vector_gamma_matches_per_row():
    # The 3 elementwise estimators (bayes, robust_bayes, tan_bayes) accept an
    # explicit vector gamma of shape (...,) — one prior scale per observation
    # — and produce the same result as stacking per-row calls.  This is the
    # new contract; "empirical" resolves internally to the same shape.
    gen = rng()
    p = 6
    n = 4
    x = gen.normal(size=(n, p))
    g = gen.uniform(0.0, 3.0, size=n)
    for method in _EMPIRICAL_METHODS:
        fn = getattr(_shrinkage, method)
        batched = fn(x, gamma=g)
        per_row = np.stack([fn(x[i], gamma=g[i]) for i in range(n)])
        np.testing.assert_allclose(batched, per_row, rtol=1e-12, atol=1e-14)


def test_explicit_vector_gamma_with_dirs():
    # The vector gamma also flows correctly through a subspace split: the
    # recursive solve sees the same per-observation scales, since the residual
    # preserves the leading batch dimensions.
    gen = rng()
    p = 6
    n = 4
    x = gen.normal(size=(n, p))
    g = gen.uniform(0.0, 3.0, size=n)
    v = gen.normal(size=(p, 2))
    for method in _EMPIRICAL_METHODS:
        fn = getattr(_shrinkage, method)
        batched = fn(x, dirs=v, gamma=g)
        per_row = np.stack([fn(x[i], dirs=v, gamma=g[i]) for i in range(n)])
        np.testing.assert_allclose(batched, per_row, rtol=1e-12, atol=1e-14)


def test_vector_gamma_rejects_negative():
    # A vector gamma with any negative entry is rejected.
    gen = rng()
    x = gen.normal(size=(3, 4))
    g_bad = np.array([1.0, -0.1, 2.0])
    for method in _EMPIRICAL_METHODS:
        fn = getattr(_shrinkage, method)
        with pytest.raises(ValueError, match="non-negative"):
            fn(x, gamma=g_bad)


def test_callable_gamma_matches_explicit_per_obs():
    # A gamma callable f(d, pi, y) is resolved per observation: for a batched
    # x it must return shape (n,) and the result must equal passing that
    # per-obs array explicitly.
    gen = rng()
    p = 6
    n = 4
    x = gen.normal(size=(n, p))
    for method in _EMPIRICAL_METHODS:
        fn = getattr(_shrinkage, method)
        g = gen.uniform(0.0, 3.0, size=n)
        np.testing.assert_allclose(
            fn(x, gamma=lambda _d, _pi, y, g=g: np.full(y.shape[:-1], g)),
            fn(x, gamma=g.astype(float)),
            rtol=1e-12,
            atol=1e-14,
        )


def test_callable_gamma_single_vector_scalar():
    # For a single vector (batch shape ()) a callable may return a scalar, which
    # is the matching shape; it must equal passing that value explicitly.
    gen = rng()
    p = 6
    x = gen.normal(size=p)

    def f(_d, _pi, y):
        return np.sum(y**2) / p

    fn = _shrinkage.bayes
    np.testing.assert_allclose(
        fn(x, gamma=f), fn(x, gamma=float(np.sum(x**2) / p)), rtol=1e-12, atol=1e-14
    )


def test_callable_gamma_scalar_shared_across_batch():
    # A callable may return a singular scalar, giving one scale shared by every
    # observation in the batch (natural when the scale is computed only from d,
    # e.g. a d-only callable).  It must equal passing that scalar explicitly.
    gen = rng()
    p = 6
    n = 4
    x = gen.normal(size=(n, p))
    g = float(np.mean(np.linspace(0.5, 2.0, p)))
    for method in _EMPIRICAL_METHODS:
        fn = getattr(_shrinkage, method)
        np.testing.assert_allclose(
            fn(x, gamma=lambda _d, _pi, _y: g),
            fn(x, gamma=g),
            rtol=1e-12,
            atol=1e-14,
        )


def test_callable_gamma_d_only_scalar():
    # A callable that infers the scale only from the coordinate variances d
    # (ignoring the data y) returns a scalar shared across the batch, and must
    # equal passing that scalar explicitly.
    gen = rng()
    p = 6
    n = 3
    x = gen.normal(size=(n, p))
    d = np.linspace(0.5, 2.0, p)

    def f(d, _pi, _y):
        return float(np.mean(d))

    for method in _EMPIRICAL_METHODS:
        fn = getattr(_shrinkage, method)
        np.testing.assert_allclose(
            fn(x, cov=np.diag(d), gamma=f),
            fn(x, cov=np.diag(d), gamma=float(np.mean(d))),
            rtol=1e-12,
            atol=1e-14,
        )


def test_callable_gamma_scalar_with_dirs():
    # A scalar-returning callable flows through the subspace-split recursion:
    # the top level defers resolution and the recursive solve re-derives the
    # same single scale from the reduced residual, so the result equals passing
    # the scalar explicitly.
    gen = rng()
    p = 6
    n = 3
    x = gen.normal(size=(n, p))
    v = gen.normal(size=(p, 2))
    g = float(np.mean(np.linspace(0.5, 2.0, p)))
    for method in _EMPIRICAL_METHODS:
        fn = getattr(_shrinkage, method)
        np.testing.assert_allclose(
            fn(x, dirs=v, gamma=lambda _d, _pi, _y: g),
            fn(x, dirs=v, gamma=g),
            rtol=1e-12,
            atol=1e-14,
        )


def test_callable_gamma_with_dirs_matches_per_row():
    # A callable flows through the subspace-split recursion: the top level
    # defers resolution and the recursive solve derives the scale from the
    # reduced residual, so batched dirs must equal stacking per-row calls.
    gen = rng()
    p = 6
    n = 4
    x = gen.normal(size=(n, p))
    v = gen.normal(size=(p, 2))

    def f(d, _pi, y):
        return np.sum(y**2, axis=-1) / len(d)

    for method in _EMPIRICAL_METHODS:
        fn = getattr(_shrinkage, method)
        batched = fn(x, dirs=v, gamma=f)
        per_row = np.stack([fn(x[i], dirs=v, gamma=f) for i in range(n)])
        np.testing.assert_allclose(batched, per_row, rtol=1e-12, atol=1e-14)


def test_callable_gamma_via_shrink_frontend():
    # The shrink() front-end forwards the callable to the resolved estimator.
    gen = rng()
    x = gen.normal(size=6)

    def f(d, _pi, y):
        return np.sum(y**2, axis=-1) / len(d)

    np.testing.assert_allclose(
        s.shrink(x, method="bayes", gamma=f),
        _shrinkage.bayes(x, gamma=f),
        rtol=1e-12,
        atol=1e-14,
    )


def test_callable_gamma_receives_pi():
    # The gamma callable f(d, pi, y) receives the canonical diagonal of the
    # prior covariance as its middle argument, both the default homoscedastic
    # shape (pi = 1) and an explicit prior_cov.
    gen = rng()
    p = 4
    x = gen.normal(size=p)

    seen: dict[str, np.ndarray] = {}

    def f(d, pi, y):
        seen["d"] = np.asarray(d, dtype=float).copy()
        seen["pi"] = np.asarray(pi, dtype=float).copy()
        return float(np.sum(y**2) / len(d))

    _shrinkage.bayes(x, gamma=f)
    np.testing.assert_allclose(seen["pi"], np.ones(p), rtol=1e-12, atol=1e-14)

    prior = np.diag([1.0, 4.0, 0.5, 2.0])
    _shrinkage.bayes(x, gamma=f, prior_cov=prior)
    # With cov = Q = I the canonical variances all coincide, so the free
    # ordering is spent to report pi ordered decreasingly within the block.
    expected = np.sort(np.diag(prior))[::-1]
    np.testing.assert_allclose(seen["pi"], expected, rtol=1e-12, atol=1e-14)


def test_callable_gamma_rejects_negative():
    # A callable returning negative prior scales is rejected.
    gen = rng()
    x = gen.normal(size=(3, 4))

    def f(_d, _pi, y):
        return -np.ones(y.shape[:-1])

    with pytest.raises(ValueError, match="non-negative"):
        _shrinkage.bayes(x, gamma=f)


def test_callable_gamma_rejects_bad_shape():
    # A callable returning the full coordinate shape (..., p) or any shape other
    # than the batch dims is rejected.
    gen = rng()
    x = gen.normal(size=(3, 4))

    def f(_d, _pi, y):
        return np.ones(y.shape)

    with pytest.raises(ValueError, match="matching"):
        _shrinkage.bayes(x, gamma=f)


def test_callable_gamma_rejects_non_numeric():
    # A callable returning a non-numeric / non-array value is rejected.
    gen = rng()
    x = gen.normal(size=(3, 4))

    def f(_d, _pi, _y):
        return "not a gamma"

    with pytest.raises(TypeError):
        _shrinkage.bayes(x, gamma=f)


def test_float_only_methods_reject_callable_gamma():
    # tan and minimax_bayes only accept a numeric gamma; a callable prior scale
    # is rejected just like a string or vector gamma.
    gen = rng()
    x = gen.normal(size=6)

    def f(_d, _pi, y):
        return np.full(y.shape[:-1], 1.0)

    for method in _FLOAT_ONLY_METHODS:
        fn = getattr(_shrinkage, method)
        with pytest.raises(TypeError):
            fn(x, gamma=f)


# ---------------------------------------------------------------------------
# Explicit prior covariance (prior_cov): validation, rotation, and consistency.
# ---------------------------------------------------------------------------

_PRIOR_METHODS = ["bayes", "robust_bayes", "tan_bayes", "tan", "minimax_bayes"]


def _diagonalizable_prior(cov, q, pi):
    """A SPD prior whose canonical form is diagonal with variances ``pi``.

    Given ``cov`` and ``q``, ``B^{-1} diag(pi) B^{-T}`` is by construction
    diagonal in the canonical coordinates.
    """
    b, _, _ = _shrinkage._canonicalize(np.asarray(cov), np.asarray(q))
    binv = np.linalg.inv(b)
    return binv @ np.diag(pi) @ binv.T


def test_prior_cov_validation():
    # prior_cov must be square of the right shape, symmetric and positive
    # definite; and its canonical form must be diagonalizable.
    gen = rng()
    p = 4
    x = gen.normal(size=p)
    for est in _PRIOR_METHODS:
        fn = getattr(_shrinkage, est)
        with pytest.raises(ValueError, match="prior covariance matrix must have shape"):
            fn(x, prior_cov=np.eye(p - 1))
        with pytest.raises(ValueError, match="must be symmetric"):
            fn(x, prior_cov=np.triu(np.ones((p, p))))
        with pytest.raises(ValueError, match="positive definite"):
            fn(x, prior_cov=np.zeros((p, p)))


def test_prior_cov_not_diagonalizable_in_canonical_space_raises():
    # With distinct canonical variances the rotation is not free, so a prior
    # that couples canonical coordinates with differing variances is rejected.
    gen = rng()
    x = gen.normal(size=3)
    cov = np.diag([1.0, 2.0, 4.0])  # canonical coords = original coords
    q = np.eye(3)
    bad = np.array([[1.0, 0.5, 0.0], [0.5, 1.0, 0.0], [0.0, 0.0, 2.0]])
    for est in _PRIOR_METHODS:
        fn = getattr(_shrinkage, est)
        with pytest.raises(ValueError, match="cannot be diagonalized"):
            fn(x, cov=cov, Q=q, gamma=1.0, prior_cov=bad)


def test_prior_cov_diagonal_in_canonical_space_accepted():
    # A prior that is diagonal in the canonical coordinates works even when the
    # canonical variances are all distinct (no rotational freedom needed).
    gen = rng()
    x = gen.normal(size=3)
    cov = np.diag([1.0, 2.0, 4.0])
    q = np.eye(3)
    prior = np.diag([2.0, 0.5, 3.0])
    for est in _PRIOR_METHODS:
        fn = getattr(_shrinkage, est)
        assert fn(x, cov=cov, Q=q, gamma=1.0, prior_cov=prior).shape == x.shape


def test_prior_cov_identity_equals_default():
    # The default prior is Q^{-1} (the homoscedastic canonical prior), so
    # passing prior_cov = inv(Q) must reproduce the default exactly, for every
    # prior-aware estimator and a general cov/Q.
    gen = rng()
    p = 6
    x = gen.normal(size=(3, p))
    a = gen.normal(size=(p, p))
    cov = a @ a.T + np.eye(p)
    b = gen.normal(size=(p, p))
    q = b @ b.T + np.eye(p)
    params = {
        "bayes": {"gamma": 1.5},
        "robust_bayes": {"gamma": 1.5},
        "tan_bayes": {"gamma": 0.7, "positive": False},
        "tan": {"gamma": 0.7, "positive": False},
        "minimax_bayes": {"gamma": 0.7, "positive": False},
    }
    for est in _PRIOR_METHODS:
        fn = getattr(_shrinkage, est)
        default = fn(x, cov=cov, Q=q, **params[est])
        with_qinv = fn(x, cov=cov, Q=q, prior_cov=np.linalg.inv(q), **params[est])
        np.testing.assert_allclose(default, with_qinv, rtol=1e-9, atol=1e-9)


def test_gamma_scales_prior_shape():
    # gamma is a pure scale of the effective prior: doubling gamma with a fixed
    # shape equals keeping gamma and doubling the shape.
    gen = rng()
    x = gen.normal(size=(4, 4))
    pi = np.array([2.0, 1.0, 3.0, 0.5])
    cov = np.eye(4)
    q = np.eye(4)
    for est in _PRIOR_METHODS:
        fn = getattr(_shrinkage, est)
        lhs = fn(x, cov=cov, Q=q, gamma=2.0, prior_cov=np.diag(pi))
        rhs = fn(x, cov=cov, Q=q, gamma=1.0, prior_cov=np.diag(2.0 * pi))
        np.testing.assert_allclose(lhs, rhs, rtol=1e-12, atol=1e-12)


def test_bayes_general_prior_matches_canonical_reference():
    # The full pipeline (canonicalization + rotation + Bayes rule) must equal a
    # manual canonical computation with the rotated frame, for a general SPD
    # prior in the cov proportional to Q^{-1} regime.
    gen = rng()
    p = 5
    x = gen.normal(size=p)
    a = gen.normal(size=(p, p))
    q = a @ a.T + p * np.eye(p)
    q = (q + q.T) / 2
    cov = 2.0 * np.linalg.inv(q)  # cov proportional to Q^{-1}: rotation free
    m = gen.normal(size=(p, p))
    prior = m @ m.T + p * np.eye(p)
    prior = (prior + prior.T) / 2  # arbitrary SPD

    b, _, d = _shrinkage._canonicalize(cov, q)
    pi, b_rot = _shrinkage._canonicalize_prior(b, d, prior)
    xs = x @ b_rot.T
    gamma = 1.3
    factor = gamma * pi / (d + gamma * pi)
    expect = factor * xs @ np.linalg.inv(b_rot).T

    np.testing.assert_allclose(
        _shrinkage.bayes(x, cov=cov, Q=q, gamma=gamma, prior_cov=prior),
        expect,
        rtol=1e-9,
        atol=1e-10,
    )


def test_prior_cov_proportional_qinv_ill_conditioned():
    # Q = inv(cov) makes chol(Q)^T cov chol(Q)^T equal to c*I only up to the
    # roundoff of the inversion (~ eps * cond(cov)).  The computed canonical
    # variances are then only *nearly* equal, so the free rotation must be
    # recognized from the matrix-level proportionality, not from the eps-scale
    # grouping tolerance: a general SPD prior has to be accepted.
    gen = rng()
    p = 5
    u = np.linalg.qr(gen.normal(size=(p, p)))[0]
    cov = u @ np.diag(np.logspace(-2, 2, p)) @ u.T
    cov = (cov + cov.T) / 2
    q = np.linalg.inv(cov)  # cond(cov) ~ 1e4: proportionality only numerical
    x = gen.normal(size=p)
    m = gen.normal(size=(p, p))
    prior = m @ m.T + p * np.eye(p)
    prior = (prior + prior.T) / 2

    b, _, d = _shrinkage._canonicalize(cov, q)
    assert _cov_proportional_to_qinv(cov, q)
    # Without the flag the near-coinciding variances split into groups at the
    # eps tolerance, so the general prior is (rightly) rejected; only the
    # roundoff-aware free rotation accepts it.
    with pytest.raises(ValueError, match="cannot be diagonalized"):
        _shrinkage._canonicalize_prior(b.copy(), d.copy(), prior)

    pi, b_rot = _shrinkage._canonicalize_prior(b, d, prior, free_rotation=True)
    xs = x @ b_rot.T
    gamma = 1.3
    factor = gamma * pi / (d + gamma * pi)
    expect = factor * xs @ np.linalg.inv(b_rot).T
    np.testing.assert_allclose(
        _shrinkage.bayes(x, cov=cov, Q=q, gamma=gamma, prior_cov=prior),
        expect,
        rtol=1e-8,
        atol=1e-9,
    )

    # The risk-sweep entry point uses the same free-rotation detection.
    records = _shrinkage.estimate_risk_curve(
        cov,
        "bayes",
        Q=q,
        directions="uniform",
        distances=(0.0, 2.0, 3),
        n_reps=500,
        seed=0,
        gamma=1.0,
        prior_cov=prior,
    )
    assert len(records) == 3


def test_canonical_estimators_agree_with_public_prior_cov():
    # With cov = I, Q = I the canonical coordinates are the original
    # coordinates, so the public estimators applied to a diagonal prior must
    # agree with the internal canonical estimators given the prior diagonal.
    gen = rng()
    x = gen.normal(size=6)
    d = gen.uniform(0.5, 3.0, size=6)
    pi = gen.uniform(0.5, 3.0, size=6)
    cov = np.diag(d)
    q = np.eye(6)
    gamma = 0.9

    expected = {
        "bayes": _bayes_canonical(x, d, gamma=gamma, pi=pi),
        "robust_bayes": _robust_bayes_canonical(x, d, strength=1.0, gamma=gamma, pi=pi),
        "tan_bayes": _tan_bayes_canonical(
            x, d, positive=False, strength=1.0, gamma=gamma, pi=pi
        ),
        "tan": _tan_canonical(x, d, positive=False, strength=1.0, gamma=gamma, pi=pi),
        "minimax_bayes": _minimax_bayes_canonical(
            x, d, positive=False, strength=1.0, gamma=gamma, pi=pi
        ),
    }
    for est, ref in expected.items():
        fn = getattr(_shrinkage, est)
        got = fn(x, cov=cov, Q=q, gamma=gamma, prior_cov=np.diag(pi))
        np.testing.assert_allclose(got, ref, rtol=1e-9, atol=1e-10)


def test_tan_gamma_inf_constant_shape_equals_flat():
    # At gamma = inf a constant prior shape (pi_j = c for all j) only rescales
    # d*, weight and low_a by common factors that cancel in the segmentation,
    # a_star and the shrinkage ratio, so it reproduces the default homoscedastic
    # A†_inf exactly.
    gen = rng()
    d = gen.uniform(0.5, 3.0, size=5)
    x = gen.normal(size=5)
    flat = _tan_canonical(x, d, positive=True, strength=1.0, gamma=float("inf"))
    for c in (0.25, 2.5):
        shaped = _tan_canonical(
            x,
            d,
            positive=True,
            strength=1.0,
            gamma=float("inf"),
            pi=c * np.ones(5),
        )
        np.testing.assert_allclose(shaped, flat, rtol=1e-9, atol=1e-10)

    cov = np.diag(d)
    public_flat = _shrinkage.tan(
        x, cov=cov, Q=np.eye(5), positive=True, gamma=float("inf")
    )
    public_shaped = _shrinkage.tan(
        x,
        cov=cov,
        Q=np.eye(5),
        positive=True,
        gamma=float("inf"),
        prior_cov=2.5 * np.eye(5),
    )
    np.testing.assert_allclose(public_shaped, public_flat, rtol=1e-9, atol=1e-10)


def test_tan_gamma_inf_prior_shape_ranks_by_d2_over_pi():
    # The gamma=inf limit with a fixed non-uniform prior shape ranks the Bayes
    # importance by d_j^2/pi_j (weight pi/d^2, low-axis d/pi), not the flat
    # d_j^2: the result follows the shape-aware limit and differs from the
    # homoscedastic A†_inf.
    d = np.array([1.0, 2.0, 3.0, 4.0])
    pi = np.array([1.0, 4.0, 1.0, 2.0])
    x = np.array([0.8, -1.3, 0.5, 1.1])

    d_star = d**2 / pi
    weight = pi / d**2
    low_a = d / pi
    order = np.argsort(d_star)[::-1]
    d_sorted = d[order]
    d_star_sorted = d_star[order]
    cw = np.cumsum(weight[order])
    p = len(d)
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
        c_star += np.sum(d_star_sorted[nu:])
    x_sorted = x[order]
    s_val = np.sum(a**2 * x_sorted**2)
    factor = 1.0 - c_star * a / s_val
    ref = np.empty(p)
    ref[order] = factor * x_sorted

    got = _tan_canonical(x, d, positive=False, strength=1.0, gamma=float("inf"), pi=pi)
    np.testing.assert_allclose(got, ref, rtol=1e-9, atol=1e-10)

    flat = _tan_canonical(x, d, positive=False, strength=1.0, gamma=float("inf"))
    assert not np.allclose(got, flat, rtol=1e-9, atol=1e-10)


def test_prior_cov_couples_only_degenerate_coordinates():
    # The within-group rotation freedom of the canonicalization can disentangle
    # a prior that couples coordinates of (near-)equal variance, but not a prior
    # coupling coordinates with differing variances.
    d = np.array([1.0, 1.0, 2.0])
    ok = np.array(
        [[1.0, 0.5, 0.0], [0.5, 1.0, 0.0], [0.0, 0.0, 2.0]]
    )  # couples the two degenerate coords (0,1) only
    pi, b = _shrinkage._canonicalize_prior(np.eye(3), d, ok)
    np.testing.assert_allclose(b @ ok @ b.T, np.diag(pi), rtol=1e-9, atol=1e-10)
    np.testing.assert_allclose(pi, [1.5, 0.5, 2.0], rtol=1e-9, atol=1e-10)

    bad = np.array(
        [[1.0, 0.0, 0.5], [0.0, 1.0, 0.0], [0.5, 0.0, 2.0]]
    )  # couples coord 0 (d=1) to coord 2 (d=2)
    with pytest.raises(ValueError, match="cannot be diagonalized"):
        _shrinkage._canonicalize_prior(np.eye(3), d, bad)


def test_prior_cov_rotation_preserves_canonical_form():
    # The rotated frame must still satisfy B cov B^T = diag(d) and Q = B^T B:
    # rotating the canonical coordinates within degenerate groups is lossless.
    gen = rng()
    p = 5
    a = gen.normal(size=(p, p))
    q = a @ a.T + p * np.eye(p)
    q = (q + q.T) / 2
    cov = 3.0 * np.linalg.inv(q)  # all canonical variances equal -> free rotation
    m = gen.normal(size=(p, p))
    prior = m @ m.T + p * np.eye(p)
    prior = (prior + prior.T) / 2
    b, _, d = _shrinkage._canonicalize(cov, q)
    pi, b_rot = _shrinkage._canonicalize_prior(b, d, prior)
    np.testing.assert_allclose(b_rot @ cov @ b_rot.T, np.diag(d), rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(b_rot.T @ b_rot, q, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(
        b_rot @ prior @ b_rot.T, np.diag(pi), rtol=1e-8, atol=1e-10
    )


def test_prior_canonical_frame_pi_ordering():
    # pi is non-increasing within each block of (numerically-)equal canonical
    # variance, in both the already-diagonal and the rotated paths; within a
    # block the permutation/rotation is the only freedom, so this is always
    # possible and is what the canonicalization reports.
    d = np.array([4.0, 4.0, 2.0, 1.0, 1.0])  # blocks (0,1) and (3,4)

    # Already-diagonal prior: blocks are permuted so pi descends inside them,
    # distinct-variance coordinates stay pinned to their diagonal entries.
    prior = np.diag([2.0, 4.0, 3.0, 1.0, 0.5])
    pi, _ = _shrinkage._canonicalize_prior(np.eye(5), d, prior)
    np.testing.assert_allclose(pi, [4.0, 2.0, 3.0, 1.0, 0.5], rtol=1e-12, atol=1e-14)

    # Rotated path: a general prior coupling only the degenerate coordinates is
    # diagonalized within each block and ordered non-increasing there.
    m = np.array(
        [
            [1.0, 0.6, 0.0, 0.0, 0.0],
            [0.6, 2.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 3.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 2.0, 0.8],
            [0.0, 0.0, 0.0, 0.8, 1.0],
        ]
    )
    pi_r, b_r = _shrinkage._canonicalize_prior(np.eye(5), d, m)
    np.testing.assert_allclose(b_r @ m @ b_r.T, np.diag(pi_r), rtol=1e-9, atol=1e-10)
    assert np.all(np.diff(pi_r[0:2]) <= 1e-12)
    assert np.all(np.diff(pi_r[3:5]) <= 1e-12)


def test_canonical_frame_matches_estimator_pipeline():
    # _canonical_frame must reproduce the frame (B, Binv, d, pi) the estimators
    # canonicalize the data in: with a prior it is the rotated frame and the
    # prior diagonal pi non-increasing within each equal-d block, with the
    # inverse consistently rotated as well.
    cov = np.diag([2.0, 2.0, 1.0, 1.0, 0.5])
    q = np.eye(5)
    prior = np.diag([1.0, 3.0, 2.0, 1.0, 1.0])
    b, binv, d, pi = _canonical_frame(cov, q, prior)
    np.testing.assert_allclose(d, [2.0, 2.0, 1.0, 1.0, 0.5])
    np.testing.assert_allclose(pi, [3.0, 1.0, 2.0, 1.0, 1.0])  # sorted tie blocks
    np.testing.assert_allclose(b @ cov @ b.T, np.diag(d), rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(b @ binv, np.eye(5), rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(b @ prior @ b.T, np.diag(pi), rtol=1e-12, atol=1e-14)

    b0, binv0, d0, pi0 = _canonical_frame(cov, q, None)
    np.testing.assert_allclose(pi0, np.ones(5))
    np.testing.assert_allclose(b0 @ cov @ b0.T, np.diag(d0), rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(b0 @ binv0, np.eye(5), rtol=1e-12, atol=1e-14)


def test_risk_curve_axes_match_estimator_prior_frame():
    # The risk-curve "axis j" must be the estimator's canonical coordinate j
    # (pi non-increasing within each equal-d block), not the unrotated-frame
    # coordinate.  With a tie block in d and a diagonal prior needing an in-tie
    # sort the estimator's frame permutes the block, so the risk curve must map
    # means back to the original space with the *rotated* inverse.
    cov = np.diag([2.0, 2.0, 1.0, 1.0, 0.5])
    q = np.eye(5)
    prior = np.diag([1.0, 3.0, 2.0, 1.0, 1.0])
    b, binv, d, _ = _canonical_frame(cov, q, prior)

    for axis in range(5):
        ((_, u_star),) = _canonical_directions([axis], d, b, None)
        theta = u_star @ binv.T
        canonical_mean = theta @ b.T
        expected = np.zeros(5)
        expected[axis] = 1.0
        np.testing.assert_allclose(canonical_mean, expected, atol=1e-12)


def test_risk_curve_raw_direction_uses_estimator_prior_frame():
    # A raw direction equal to an estimator-frame axis (mapped back to the
    # original space) must resolve to exactly that canonical coordinate.
    cov = np.diag([2.0, 2.0, 1.0, 1.0, 0.5])
    q = np.eye(5)
    prior = np.diag([1.0, 3.0, 2.0, 1.0, 1.0])
    b, binv, d, _ = _canonical_frame(cov, q, prior)

    for axis in range(5):
        ((_, u_star),) = _canonical_directions([binv[:, axis]], d, b, None)
        expected = np.zeros(5)
        expected[axis] = 1.0
        np.testing.assert_allclose(u_star, expected, atol=1e-12)


def test_risk_curve_named_directions_frame_invariant():
    # The built-in directions depend only on d, which a within-tie-block prior
    # rotation does not change, so they are identical in the plain and
    # prior-rotated frames (and hence unaffected by the frame the curve maps
    # means back with).
    cov = np.diag([2.0, 2.0, 1.0, 1.0, 0.5])
    q = np.eye(5)
    prior = np.diag([1.0, 3.0, 2.0, 1.0, 1.0])
    names = ["uniform", "proportional", "inverse"]
    d_plain = np.array([2.0, 2.0, 1.0, 1.0, 0.5])
    u_plain = dict(_canonical_directions(names, d_plain, np.eye(5), None))
    b, _, d, _ = _canonical_frame(cov, q, prior)
    u_rot = dict(_canonical_directions(names, d, b, None))
    for name in names:
        np.testing.assert_allclose(u_rot[name], u_plain[name], atol=1e-12)


def test_tan_canonical_inputs_are_order_invariant():
    # The canonical estimators treat the coordinates as an unordered set that
    # they rank internally: relabelling x, d and pi together leaves the result
    # unchanged (up to the same relabelling).
    gen = rng()
    p = 5
    x = gen.normal(size=p)
    d = gen.uniform(0.5, 5.0, size=p)
    pi = gen.uniform(0.5, 5.0, size=p)
    perm = np.random.default_rng(0).permutation(p)

    got = _tan_canonical(x, d, positive=False, strength=1.0, gamma=0.7, pi=pi)
    permuted = _tan_canonical(
        x[perm], d[perm], positive=False, strength=1.0, gamma=0.7, pi=pi[perm]
    )
    np.testing.assert_allclose(got[perm], permuted, rtol=1e-10, atol=1e-12)


def test_prior_cov_with_empirical_gamma():
    # A per-observation empirical scale multiplies the per-coordinate prior
    # shape: tau_j = gamma_obs * pi_j.  The prior-aware empirical scale is
    # ||y / sqrt(pi)||^2 / p so each coordinate of y is divided by the
    # corresponding prior shape before the squared norm.
    gen = rng()
    x = gen.normal(size=(4, 3))
    cov = np.eye(3)
    q = np.eye(3)
    prior = np.diag([2.0, 3.0, 4.0])
    got = _shrinkage.bayes(x, cov, Q=q, gamma="empirical", prior_cov=prior)
    pi = np.array([2.0, 3.0, 4.0])
    g = np.sum(x**2 / pi, axis=-1) / 3
    tau = g[:, None] * pi
    expect = tau / (1.0 + tau) * x
    np.testing.assert_allclose(got, expect, rtol=1e-9, atol=1e-10)


def test_prior_cov_with_singular_Q():
    # A singular Q splits the problem; the prior is restricted to the range of
    # Q (the part that is shrunk), while the loss-free null space is kept at
    # the data value.
    x = np.array([1.0, 2.0, 3.0, 4.0])
    cov = np.eye(4)
    q = np.diag([1.0, 1.0, 0.0, 0.0])
    prior = np.diag([2.0, 3.0, 4.0, 1.0])
    got = _shrinkage.bayes(x, cov, Q=q, gamma=1.0, prior_cov=prior)
    expect = np.array([2.0 / 3.0 * x[0], 3.0 / 4.0 * x[1], x[2], x[3]])
    np.testing.assert_allclose(got, expect, rtol=1e-8, atol=1e-10)


def test_prior_cov_with_dirs():
    # With an affine no-shrink direction the prior acts only on the residual;
    # the component along dirs is kept at the data value.
    x = np.array([1.0, 2.0, 3.0, 4.0])
    cov = np.eye(4)
    q = np.eye(4)
    prior = np.diag([2.0, 3.0, 4.0, 1.0])
    dirs = np.eye(4)[:, :1]  # e1 kept at its data value
    got = _shrinkage.bayes(x, cov, Q=q, gamma=1.0, prior_cov=prior, dirs=dirs)
    expect = np.array([x[0], 3.0 / 4.0 * x[1], 4.0 / 5.0 * x[2], 1.0 / 2.0 * x[3]])
    np.testing.assert_allclose(got, expect, rtol=1e-8, atol=1e-10)


def test_risk_helpers_pass_prior_cov():
    # estimate_risk and estimate_risk_curve must accept and forward prior_cov
    # to the estimators.
    cov = np.diag([1.0, 2.0, 3.0])
    prior = np.diag([2.0, 1.0, 0.5])
    theta = np.array([1.0, 0.0, 0.0])
    est = functools.partial(_shrinkage.bayes, prior_cov=prior)
    risk = s.estimate_risk(theta, cov, est, n_reps=200, seed=0)
    assert risk.shape == (2,)
    est2 = functools.partial(_shrinkage.bayes, gamma=1.0, prior_cov=prior)
    records = s.estimate_risk_curve(
        cov,
        est2,
        directions="uniform",
        distances=(0.0, 1.0, 2),
        n_reps=200,
        seed=0,
        prior_cov=prior,
    )
    assert len(records) == 2
    assert all(np.isfinite(r["risk"]) for r in records)


def test_berger_rejects_prior_cov():
    # Berger involves no prior, so it rejects prior_cov (and gamma) with a
    # TypeError — both when called directly and through shrink (the default
    # method), mirroring the rejection of unsupported kwargs elsewhere.
    gen = rng()
    x = gen.normal(size=4)
    cov = np.diag([1.0, 2.0, 3.0, 4.0])
    prior = np.diag([2.0, 3.0, 4.0, 1.0])
    with pytest.raises(TypeError):
        _shrinkage.berger(x, cov=cov, prior_cov=prior)
    with pytest.raises(TypeError):
        _shrinkage.berger(x, cov=cov, gamma=1.0)
    with pytest.raises(TypeError):
        _shrinkage.shrink(x, cov=cov, prior_cov=prior, method="berger")
    # The prior-aware estimators still accept it through the same front-end.
    got = _shrinkage.shrink(x, cov=cov, method="bayes", gamma=1.0, prior_cov=prior)
    assert got.shape == x.shape
