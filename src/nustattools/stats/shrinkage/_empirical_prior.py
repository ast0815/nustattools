"""Per-observation prior-scale inference and the risk-cap gamma factories.

This private module derives the Bayesian prior scale ``gamma`` from the
canonical data ``y``, variances ``d`` and prior diagonal ``pi``: the empirical
preset :func:`_empirical_gamma`, the risk-cap factories
:func:`_max_rel_risk_gamma` / :func:`_max_abs_risk_gamma` (which refine the
closed-form scale toward the root of their risk-cap equation with
tolerance-driven, globally convergent Newton iterations), and the string
parser :func:`_parse_gamma_factory` used by the estimator front-ends.

Nothing in this module is public; estimators and the risk-estimation helpers in
the sibling modules build on it.

"""

from __future__ import annotations

import ast
import re
import warnings
from collections.abc import Callable
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

# np.finfo(float).eps trips a known pylint numpy false positive (E1101).
_EPSILON: float = np.finfo(float).eps  # pylint: disable=no-member


def _empirical_gamma(
    d: NDArray[Any], pi: NDArray[Any], y: NDArray[Any]
) -> NDArray[Any]:
    """Return the per-observation empirical prior scale
    :math:`\\| y / \\sqrt{\\pi} \\|^2 / p_{\\mathrm{eff}}`.

    ``y`` has shape ``(..., p_eff)`` (the centered canonical data) and ``pi``
    has shape ``(p_eff,)`` (the canonical diagonal of the prior covariance,
    all ones for the default homoscedastic prior).  The result has shape
    ``(...,)``: one prior scale for each observation, computed from the
    squared norm over the trailing (coordinate) axis only, after normalising
    each coordinate by the corresponding prior scale
    :math:`\\sqrt{\\pi_j}`.

    """

    p_eff = len(d)
    safe_pi = np.where(pi > 0, pi, 1.0)
    return cast(NDArray[Any], np.sum(y**2 / safe_pi, axis=-1) / p_eff)


_EMPIRICAL_GAMMA_PRESETS: dict[
    str, Callable[[NDArray[Any], NDArray[Any], NDArray[Any]], NDArray[Any]]
] = {
    "empirical": _empirical_gamma,
}


_MAX_BAYES_NORM_ITERS = 100


def _bayes_norm_root_solve(
    d: NDArray[Any],
    pi: NDArray[Any],
    y: NDArray[Any],
    target: NDArray[Any] | float,
    rtol: float,
) -> NDArray[Any]:
    """Refine the Bayes-rule prior scale toward the root of the norm equation.

    Returns, per observation, the prior scale :math:`\\gamma` that satisfies

    .. math::

        \\sum_j \\left(\\frac{d_j y_j}{d_j + \\gamma \\pi_j}\\right)^2 = B

    where ``target`` (the ``B`` of the equation) is the per-coordinate model
    constant of the caller
    (:math:`\\alpha` for the absolute-risk family,
    :math:`\\alpha \\sum_j d_j` for the
    relative-risk family).  The closed form

    .. math::

        \\gamma_0 = \\left(\\frac{1}{B}
            \\sum_j \\left(\\frac{d_j y_j}{\\pi_j}\\right)^2\\right)^{1/2}

    starts the iteration.  ``rtol = inf`` keeps :math:`\\gamma_0` exactly (the
    factories map a ``None`` tolerance to ``inf``), matching the historical
    closed form bit-for-bit; a finite ``rtol >= 0`` refines with vectorized
    Newton steps (quadratic convergence) with the scale clamped to
    :math:`\\ge 0`:

    .. math::

        \\gamma_{k+1} = \\max\\!\\left(
            \\gamma_k - \\frac{f(\\gamma_k)}{f'(\\gamma_k)}, 0\\right)

    Newton's method is used instead of a higher-order update because
    :math:`f` is strictly convex and strictly decreasing in
    :math:`\\gamma`, which
    makes it *globally* convergent (from the right it crosses the root once
    and then approaches monotonically from below; from the left it never
    overshoots).  A plain Halley or higher-order update is only cubically
    convergent near the root and can stall or enter a limit cycle when the
    closed-form seed :math:`\\gamma_0` is far above the root -- which happens
    precisely at small and medium data, where the asymptotic form is a poor
    approximation.  The Newton step also escapes the clamped
    :math:`\\gamma = 0`
    boundary toward a positive root (:math:`f(0) > 0` gives a negative step),
    while higher-order iterates may stay pinned at :math:`0`.

    The batch advances together until, for every observation, the relative
    residual :math:`|f(\\gamma)| / B` is at most ``rtol``, the scale is pinned
    at
    the boundary :math:`\\gamma = 0` with :math:`f(0) \\le 0` (the analytic
    optimum when
    there is no positive root), or no further progress is made at
    floating-point roundoff (so ``rtol = 0`` converges to machine precision
    instead of exhausting the iteration cap).  The iteration is capped at
    ``_MAX_BAYES_NORM_ITERS`` refinements; reaching the cap returns the best
    iterate and emits a :class:`RuntimeWarning` -- Newton is globally
    convergent for this strictly convex ``f``, so the cap is only a safety
    net.

    with ``d`` of shape ``(p,)``, ``pi`` of shape ``(p,)`` and ``y`` of shape
    ``(..., p)``.  The result has shape ``(...,)``.

    """

    g = np.maximum(np.sqrt(np.sum((d * y / pi) ** 2, axis=-1) / target), 0.0)
    if rtol == np.inf:
        return g
    rel_prev = np.full_like(g, np.nan)
    exhausted = True
    for _ in range(_MAX_BAYES_NORM_ITERS):
        dpg = d + g[..., np.newaxis] * pi
        num = (d * y) ** 2
        f = np.sum(num / dpg**2, axis=-1) - target
        rel = np.abs(f) / target
        boundary = (g == 0.0) & (f <= 0.0)
        stagnant = np.abs(rel - rel_prev) <= _EPSILON * (rel + rel_prev)
        if np.all((rel <= rtol) | boundary | stagnant):
            exhausted = False
            break
        f1 = -2.0 * np.sum(num * pi / dpg**3, axis=-1)
        zero = f1 == 0.0
        step = np.where(zero, 0.0, f / np.where(zero, 1.0, f1))
        g = np.maximum(g - step, 0.0)
        rel_prev = rel
    if exhausted:
        warnings.warn(
            "Prior-scale refinement could not meet rtol within "
            f"{_MAX_BAYES_NORM_ITERS} iterations; returning the best iterate.",
            RuntimeWarning,
            stacklevel=2,
        )
    return g


def _max_rel_risk_gamma(
    alpha: float,
    rtol: float | None = None,
) -> Callable[[NDArray[Any], NDArray[Any], NDArray[Any]], NDArray[Any]]:
    """Return a per-observation prior scale capping the relative risk increase.

    The returned callable ``g(d, pi, y)`` infers, for each observation, the
    prior scale that caps the per-observation *increase* of the relative risk
    at ``alpha`` (i.e. the relative risk itself is at most ``1 + alpha``) by
    solving

    .. math::

        \\sum_j \\left(\\frac{d_j y_j}{d_j + \\gamma \\pi_j}\\right)^2
            = \\alpha \\sum_j d_j

    with ``d`` of shape ``(p,)``, ``pi`` of shape ``(p,)`` (the canonical
    diagonal of the prior covariance) and ``y`` of shape ``(..., p)`` (the
    centered canonical data).  The result has shape ``(...,)``: one prior scale
    per observation.

    ``rtol`` is the relative-residual tolerance of the vectorized Newton
    refinement applied on top of the closed-form start

    .. math::

        \\gamma_0 = \\frac{1}{\\sqrt{\\alpha \\sum_j d_j}}
            \\left(\\sum_j \\left(\\frac{d_j y_j}{\\pi_j}\\right)^2\\right)^{1/2}

    Setting ``rtol = None`` (the default, equivalent to ``np.inf``) returns
    :math:`\\gamma_0` exactly, matching the historical closed form bit-for-bit;
    a
    finite ``rtol >= 0`` further refines :math:`\\gamma` until the relative
    residual :math:`|F(\\gamma)/B - 1|` (with :math:`B = \\alpha \\sum_j d_j`,
    see
    :func:`_bayes_norm_root_solve`) does not exceed ``rtol`` for every
    observation.  Raises :class:`ValueError` unless :math:`\\alpha > 0` and
    ``rtol`` is ``None``, ``np.inf``, or a non-negative number.

    """

    if alpha <= 0:
        msg = "alpha must be > 0."
        raise ValueError(msg)
    if rtol is not None and (np.isnan(rtol) or rtol < 0):
        msg = "rtol must be None, np.inf, or a non-negative number."
        raise ValueError(msg)

    def gamma(d: NDArray[Any], pi: NDArray[Any], y: NDArray[Any]) -> NDArray[Any]:
        target = cast(NDArray[Any], alpha * np.sum(d))
        r = np.inf if rtol is None else float(rtol)
        return _bayes_norm_root_solve(d, pi, y, target, r)

    return gamma


def _max_abs_risk_gamma(
    alpha: float,
    rtol: float | None = None,
) -> Callable[[NDArray[Any], NDArray[Any], NDArray[Any]], NDArray[Any]]:
    """Return a per-observation prior scale capping the absolute risk increase.

    The returned callable ``g(d, pi, y)`` infers, for each observation, the
    prior scale that caps the per-observation *absolute* increase of the risk
    at ``alpha`` by solving

    .. math::

        \\sum_j \\left(\\frac{d_j y_j}{d_j + \\gamma \\pi_j}\\right)^2
            = \\alpha

    with ``d`` of shape ``(p,)``, ``pi`` of shape ``(p,)`` (the canonical
    diagonal of the prior covariance) and ``y`` of shape ``(..., p)`` (the
    centered canonical data).  The result has shape ``(...,)``: one prior scale
    per observation.

    ``rtol`` is the relative-residual tolerance of the vectorized Newton
    refinement applied on top of the closed-form start

    .. math::

        \\gamma_0 = \\frac{1}{\\sqrt{\\alpha}}
            \\left(\\sum_j \\left(\\frac{d_j y_j}{\\pi_j}\\right)^2\\right)^{1/2}

    Setting ``rtol = None`` (the default, equivalent to ``np.inf``) returns
    :math:`\\gamma_0` exactly, matching the historical closed form bit-for-bit;
    a
    finite ``rtol >= 0`` further refines :math:`\\gamma` until the relative
    residual :math:`|F(\\gamma)/B - 1|` (with :math:`B = \\alpha`, see
    :func:`_bayes_norm_root_solve`) does not exceed ``rtol`` for every
    observation.  Raises :class:`ValueError` unless :math:`\\alpha > 0` and
    ``rtol`` is ``None``, ``np.inf``, or a non-negative number.

    """

    if alpha <= 0:
        msg = "alpha must be > 0."
        raise ValueError(msg)
    if rtol is not None and (np.isnan(rtol) or rtol < 0):
        msg = "rtol must be None, np.inf, or a non-negative number."
        raise ValueError(msg)

    def gamma(d: NDArray[Any], pi: NDArray[Any], y: NDArray[Any]) -> NDArray[Any]:
        r = np.inf if rtol is None else float(rtol)
        return _bayes_norm_root_solve(d, pi, y, alpha, r)

    return gamma


_GAMMA_FACTORIES: dict[
    str,
    Callable[..., Callable[[NDArray[Any], NDArray[Any], NDArray[Any]], NDArray[Any]]],
] = {
    "max_rel_risk": _max_rel_risk_gamma,
    "max_abs_risk": _max_abs_risk_gamma,
}

_FACTORY_PATTERN = re.compile(r"^(\w+)\(([^()]*)\)$")


def _parse_gamma_factory(
    spec: str,
) -> Callable[[NDArray[Any], NDArray[Any], NDArray[Any]], NDArray[Any]]:
    """Build a per-observation gamma callable from a factory string.

    ``spec`` has the form ``name(arg, ...)`` where ``name`` identifies a
    registered factory in :data:`_GAMMA_FACTORIES` and the arguments are
    numeric literals (e.g. ``"max_rel_risk(0.1)"``).  Raises
    :class:`ValueError` for a malformed specification or an unregistered name,
    and :class:`TypeError` for a literal argument that is not numeric or whose
    arity does not match the factory.

    """

    match = _FACTORY_PATTERN.match(spec)
    if match is None:
        msg = f"Unknown gamma specification '{spec}'."
        raise ValueError(msg)
    name, args_text = match.groups()
    if name not in _GAMMA_FACTORIES:
        names = ", ".join(_GAMMA_FACTORIES)
        msg = f"Unknown gamma factory '{name}'; available factories: {names}."
        raise ValueError(msg)
    args: list[float] = []
    for token in args_text.split(","):
        tok = token.strip()
        if not tok:
            msg = f"Invalid gamma factory argument in '{spec}'."
            raise ValueError(msg)
        try:
            value = ast.literal_eval(tok)
        except (ValueError, SyntaxError) as e:
            msg = (
                f"Gamma factory argument '{tok}' in '{spec}' must be a numeric literal."
            )
            raise ValueError(msg) from e
        if not isinstance(value, (int, float)):
            msg = (
                f"Gamma factory argument '{tok}' in '{spec}' must be a numeric literal."
            )
            raise TypeError(msg)
        args.append(float(value))
    return _GAMMA_FACTORIES[name](*args)
