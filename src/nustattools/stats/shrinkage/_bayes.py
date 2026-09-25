"""Bayes-rule shrinkage estimators for a multivariate normal mean.

This private module implements the non-minimax Bayes-rule estimators
(:func:`bayes` and :func:`robust_bayes`) and their canonical-form
implementations (:func:`_bayes_canonical` and :func:`_robust_bayes_canonical`).
Each estimator only ever needs to shrink a vector towards zero with independent
coordinates of varying variance (the canonical form); the shared machinery in
:mod:`nustattools.stats.shrinkage._core` validates and canonicalizes the
general problem.

"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._core import (
    GammaCallable,
    _check_gamma_nonnegative,
    _estimate,
    _prior_diagonal,
)


def _robust_bayes_canonical(
    y: NDArray[Any],
    d: NDArray[Any],
    *,
    strength: float,
    gamma: float | NDArray[Any],
    pi: NDArray[Any] | None = None,
) -> NDArray[Any]:
    """The robust generalised Bayes estimator :math:`\\delta^{\\mathrm{RB}}` in
    canonical form."""

    if len(d) < 3:
        return y
    p_eff = len(d)
    c_k = strength * (p_eff - 2)
    if c_k <= 0:
        return y

    pi = _prior_diagonal(d, pi)
    g = np.asarray(gamma, dtype=float)[..., None]
    d_plus_g = d + g * pi
    weight = d / d_plus_g
    s_val = np.sum(y**2 / d_plus_g, axis=-1)
    # When s_val = 0 (e.g. x = 0) the ratio is infinite so m = 1; suppress the
    # divide warning since the min() below maps it correctly.
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = c_k / s_val
    m_k = np.minimum(1.0, ratio)
    factor = 1.0 - m_k[..., None] * weight
    return cast(NDArray[Any], factor * y)


def robust_bayes(
    x: ArrayLike,
    cov: ArrayLike | None = None,
    *,
    Q: ArrayLike | None = None,
    strength: float = 1.0,
    gamma: float | str | GammaCallable | NDArray[Any] = 1.0,
    offset: ArrayLike | None = None,
    dirs: ArrayLike | None = None,
    prior_cov: ArrayLike | None = None,
) -> NDArray[Any]:
    """The robust generalised Bayes estimator :math:`\\delta^{\\mathrm{RB}}`.

    This is Berger's (1982) robust generalised Bayes estimator reviewed in
    [Tan2015]_, Section 2, Equation (7). It shrinks the coordinates in the
    direction of the Bayes rule with an empirical strength, but caps the
    shrinkage magnitude so the shrinkage ratio never exceeds one. In canonical
    coordinates:

    .. math::

        \\delta^\\mathrm{RB} = \\left(1 - \\min\\left(1,
            \\frac{\\text{strength}\\, (p_{eff}-2)}{s}\\right) W\\right) \\vec y

    where :math:`W = D(D + \\gamma\\Gamma)^{-1}` is the Bayes-rule weights and
    :math:`s = \\sum_j x_j^2/(d_j + \\gamma \\pi_j)`. The estimator is
    *non-minimax* (unlike :func:`minimax_bayes`): it is expected to provide
    significant risk reduction over the identity when the prior is
    well-specified but is robust to misspecification. If the number of
    dimensions :math:`p_{eff}` is less than three, it returns the MLE
    unmodified.

    Parameters
    ----------
    x : array_like
        Observed data.  A single vector of shape ``(p,)`` or a stack of
        observations of shape ``(..., p)``.  The estimator is applied to each
        observation over the last axis.
    cov : array_like, default=None
        The known covariance matrix of ``x``, of shape ``(p, p)``.  Must be
        symmetric and positive definite.  Defaults to the identity matrix.
    Q : array_like, default=None
        The known loss matrix, of shape ``(p, p)``.  May be positive
        semi-definite; see the :mod:`nustattools.stats.shrinkage` module
        docstring for how the loss-free null space is handled.  Defaults to the
        identity, i.e. squared-error loss.
    strength : float, default=1.0
        Shrinkage strength as a fraction of the critical value :math:`(k-2)_+`.
        ``strength = 0`` gives the identity estimator, ``strength = 1`` is
        Tan's version (with the constant :math:`(k-2)`) and ``strength = 2``
        Berger's original version (with :math:`2(k-2)`).  Values outside
        :math:`[0, 2]` are accepted but push the estimator further from its
        recommended operating range.
    gamma : float, str, callable, or numpy.ndarray, default=1.0
        Non-negative prior scale; see the :mod:`nustattools.stats.shrinkage`
        module docstring for the accepted forms. :math:`\\gamma = 0`
        corresponds to the spherically symmetric limiting form :math:`\\{1 -
        \\mathrm{strength}\\, (k-2)/(\\vec y^T D^{-1} \\vec y)\\}_+ \\vec y`
        while larger ``gamma`` shrinks coordinates more strongly in the
        direction of the Bayes rule.
    offset : array_like, default=None
        A point of shape ``(p,)`` towards which to shrink.  Defaults to zero,
        i.e. shrinking towards the origin.
    dirs : array_like, default=None
        A matrix of shape ``(p, k)`` whose columns span the affine direction
        of shrinkage. See the :mod:`nustattools.stats.shrinkage` module
        docstring for details.
    prior_cov : array_like, default=None
        The prior covariance matrix, of shape ``(p, p)``, in the same
        coordinates as ``x``i.  the fixed covariance :math:``\\Gamma`` of a Gaussian
        prior :math:`\\theta \\sim N(\\vec o, \\gamma \\Gamma)`.  Defaults to
        :math:`Q^{-1}` (a homoscedastic prior in canonical coordinates).
        See the :mod:`nustattools.stats.shrinkage` module docstring for details.

    Returns
    -------
    delta : numpy.ndarray
        The shrinkage estimate of the mean, with the same shape as ``x``.

    Notes
    -----
    The estimator is generally *not* minimax: its risk can exceed the minimax
    risk :math:`\\operatorname{Tr}[Q \\Sigma]`` when the true mean is far
    from the prior mean. However, it is robust to misspecification of the prior
    and can have substantially lower risk than any minimax estimator when the
    prior is well-specified.

    Examples
    --------

    >>> import numpy as np
    >>> import nustattools.stats.shrinkage as sh
    >>> rng = np.random.default_rng(0)
    >>> x = rng.normal(size=5)
    >>> sh.robust_bayes(x).shape
    (5,)

    """

    _check_gamma_nonnegative(gamma)

    return _estimate(
        x=x,
        cov=cov,
        q=Q,
        canonical_estimator=_robust_bayes_canonical,
        strength=strength,
        gamma=gamma,
        offset=offset,
        dirs=dirs,
        prior_cov=prior_cov,
    )


def _bayes_canonical(
    y: NDArray[Any],
    d: NDArray[Any],
    *,
    gamma: float | NDArray[Any],
    pi: NDArray[Any] | None = None,
) -> NDArray[Any]:
    """Bayes rule in canonical form under the prior
    ``Gamma = diag(gamma * pi)``.

    The canonical problem has diagonal covariance ``d`` and identity loss.
    Under the diagonal prior ``gamma * pi``, the posterior mean (Bayes rule)
    is:

        delta_j = y_j * (gamma * pi_j) / (d_j + gamma pi_j).

    Special cases:

    - ``gamma = 0``: degenerate prior (point mass at zero); the estimate
      is zero.
    - ``gamma = inf``: flat prior; the estimate is the MLE ``y``

    """

    pi = _prior_diagonal(d, pi)
    g = np.asarray(gamma, dtype=float)[..., None]
    with np.errstate(divide="ignore", invalid="ignore"):
        tau = np.where(pi == 0.0, 0.0, g * pi)
        factor = np.where(np.isfinite(tau), tau / (d + tau), 1.0)
    return cast(NDArray[Any], factor * y)


def bayes(
    x: ArrayLike,
    cov: ArrayLike | None = None,
    *,
    Q: ArrayLike | None = None,
    gamma: float | str | GammaCallable | NDArray[Any] = 1.0,
    offset: ArrayLike | None = None,
    dirs: ArrayLike | None = None,
    prior_cov: ArrayLike | None = None,
) -> NDArray[Any]:
    """Bayes rule shrinkage estimator for a multivariate normal mean.

    Applies the posterior-mean (Bayes rule) estimator under the prior
    :math:`\\theta \\sim N(\\vec o, \\gamma \\Gamma)` This is a *non-minimax*
    estimator: it does not dominate the identity estimator uniformly over the
    parameter space, but can have substantially lower risk when the true mean
    is close to the prior mean, i.e. the `offset`.

    Parameters
    ----------
    x : array_like
        Observed data.  A single vector of shape ``(p,)`` or a stack of
        observations of shape ``(..., p)``.  The estimator is applied to each
        observation over the last axis.
    cov : array_like, default=None
        The known covariance matrix of ``x``, of shape ``(p, p)``.  Must be
        symmetric and positive definite.  Defaults to the identity matrix.
    Q : array_like, default=None
        The known loss matrix, of shape ``(p, p)``.  May be positive
        semi-definite; see the :mod:`nustattools.stats.shrinkage` module
        docstring for how the loss-free null space is handled.  Defaults to the
        identity, i.e. squared-error loss.
    gamma : float, str, callable, or numpy.ndarray, default=1.0
        Non-negative prior scale; see the :mod:`nustattools.stats.shrinkage`
        module docstring for the accepted forms.
    offset : array_like, default=None
        A point of shape ``(p,)`` towards which to shrink.  Defaults to zero,
        i.e. shrinking towards the origin.
    dirs : array_like, default=None
        A matrix of shape ``(p, k)`` whose columns span the affine direction
        of shrinkage. See the :mod:`nustattools.stats.shrinkage` module
        docstring for details.
    prior_cov : array_like, default=None
        The prior covariance matrix, of shape ``(p, p)``, in the same
        coordinates as ``x``i.  the fixed covariance :math:``\\Gamma`` of a Gaussian
        prior :math:`\\theta \\sim N(\\vec o, \\gamma \\Gamma)`.  Defaults to
        :math:`Q^{-1}` (a homoscedastic prior in canonical coordinates).
        See the :mod:`nustattools.stats.shrinkage` module docstring for details.

    Returns
    -------
    delta : numpy.ndarray
        The shrinkage estimate of the mean, with the same shape as ``x``.

    Notes
    -----
    The estimator first transforms the problem to *canonical form* (diagonal
    covariance, identity loss), which is lossless, and applies the Bayes rule
    there.  In canonical coordinates the posterior mean under
    :math:`\\vec\\theta^* \\sim N(0, \\gamma \\operatorname{diag}(\\vec\\pi))` is

    .. math:: \\delta_j = \\frac{\\gamma \\pi_j}{d_j + \\gamma \\pi_j} \\, y_j,

    where :math:`d_j` is the variance of the :math:`j`-th canonical coordinate
    and :math:`\\pi_j` the :math:`j`-th diagonal of the explicit prior (all
    ones without ``prior_cov``). Coordinates with larger variance are shrunk
    less, which is the opposite of Berger's minimax estimator (which shrinks
    inversely proportional to variance).

    The Bayes rule is generally *not* minimax: its risk exceeds the minimax
    risk :math:`\\operatorname{Tr}[Q\\Sigma]` when the true mean is far from
    the prior mean. However, when the prior is well-specified (the true mean is
    near zero), the Bayes rule can have substantially lower risk than any
    minimax estimator.

    Examples
    --------

    >>> import numpy as np
    >>> import nustattools.stats.shrinkage as sh
    >>> rng = np.random.default_rng(0)
    >>> x = rng.normal(size=5)
    >>> sh.bayes(x).shape
    (5,)

    """

    _check_gamma_nonnegative(gamma)

    return _estimate(
        x=x,
        cov=cov,
        q=Q,
        canonical_estimator=_bayes_canonical,
        gamma=gamma,
        offset=offset,
        dirs=dirs,
        prior_cov=prior_cov,
    )
