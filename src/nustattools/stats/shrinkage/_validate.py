"""Input validation shared by the shrinkage estimators.

This private module holds every check that guards the public entry points of
:mod:`nustattools.stats.shrinkage` before any computation starts: shapes,
finiteness, definiteness, the ``strength``/``gamma`` arguments, and the
scale-relative zero-eigenvalue threshold the validators and the solvers use to
decide what counts as numerically zero.  It depends on nothing but NumPy (and
the machine epsilon of :mod:`nustattools.stats.shrinkage._empirical_prior`), so
it can be imported from anywhere without cycling back into the solvers.

Nothing here is public; the solvers in
:mod:`nustattools.stats.shrinkage._core` and the risk helpers import from it.

"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._empirical_prior import _EPSILON

# Symmetry tolerance for the symmetry gates.  ``np.allclose``'s default
# absolute floor (``atol=1e-8``) would make "is this matrix symmetric?" depend
# on the unit system: a matrix that is asymmetric by a factor of two in
# relative terms passes once every entry falls below ~1e-8.  The floor is
# therefore kept at the same magnitude but measured *relative to the matrix
# norm*, so the verdict is scale-invariant (like the zero-eigenvalue threshold
# below).  The element-wise relative part catches asymmetries between
# comparable entries even when the overall matrix is large.
_SYMMETRY_RTOL = 1e-5
_SYMMETRY_ATOL_SCALE = 1e-8


def _is_symmetric(a: NDArray[Any]) -> bool:
    """True when ``a`` is symmetric to within a scale-relative tolerance."""

    return bool(
        np.allclose(
            a,
            a.T,
            rtol=_SYMMETRY_RTOL,
            atol=_SYMMETRY_ATOL_SCALE * float(np.max(np.abs(a))),
        )
    )


def _validate_sympd(a: ArrayLike, shape: tuple[int, int], name: str) -> NDArray[Any]:
    """Validate that ``a`` is symmetric positive definite with the given shape."""

    aa = np.asarray(a)
    if aa.shape != shape:
        msg = f"{name} must have shape {shape}, got {aa.shape}."
        raise ValueError(msg)
    if not np.all(np.isfinite(aa)):
        msg = f"{name} must contain only finite values."
        raise ValueError(msg)
    if not _is_symmetric(aa):
        msg = f"{name} must be symmetric."
        raise ValueError(msg)
    try:
        np.linalg.cholesky(aa)
    except np.linalg.LinAlgError as e:
        msg = f"{name} must be positive definite."
        raise ValueError(msg) from e
    return aa


def _zero_eigenvalue_tolerance(w: NDArray[Any], p: int) -> float:
    """Numerically-zero eigenvalue threshold for a size-``p`` symmetric matrix.

    An eigenvalue of ``w`` is treated as zero when it lies below this
    tolerance, which scales the machine epsilon by ``p`` *and by the largest
    magnitude eigenvalue of ``w``*.  The threshold is therefore purely
    relative: rescaling the matrix by any positive factor never changes which
    of its eigenvalues count as zero, so every positive semi-definiteness and
    degeneracy decision made from it is scale-invariant.  An all-zero spectrum
    has a tolerance of zero, so only exactly-zero eigenvalues are then treated
    as zero.

    """

    maxw = float(np.max(np.abs(w))) if w.size else 0.0
    if maxw == 0.0:
        return 0.0
    return p * _EPSILON * maxw


def _is_positive_definite(q: NDArray[Any]) -> bool:
    """True when the symmetric matrix ``q`` is numerically positive definite.

    Every eigenvalue must exceed the scale-relative zero threshold of
    :func:`_zero_eigenvalue_tolerance`, the same predicate that decides
    between the direct and the split solve in :func:`_estimate`, so a caller
    deciding whether a canonical frame exists agrees with the solver.

    """

    w = np.linalg.eigvalsh(q)
    return bool(np.all(w > _zero_eigenvalue_tolerance(w, q.shape[0])))


def _validate_sympsd(a: ArrayLike, shape: tuple[int, int], name: str) -> NDArray[Any]:
    """Validate that ``a`` is symmetric positive semi-definite with the given shape."""

    aa = np.asarray(a)
    if aa.shape != shape:
        msg = f"{name} must have shape {shape}, got {aa.shape}."
        raise ValueError(msg)
    if not np.all(np.isfinite(aa)):
        # Before the symmetry test: a NaN entry fails `allclose(a, a.T)` too,
        # but reporting it as "not symmetric" hides what is actually wrong.
        msg = f"{name} must contain only finite values."
        raise ValueError(msg)
    if not _is_symmetric(aa):
        msg = f"{name} must be symmetric."
        raise ValueError(msg)
    if shape[0] == 0:
        return aa
    w = np.linalg.eigvalsh(aa)
    if np.min(w) < -_zero_eigenvalue_tolerance(w, shape[0]):
        msg = f"{name} must be positive semi-definite (all eigenvalues >= 0)."
        raise ValueError(msg)
    return aa


def _validate(
    x: ArrayLike, cov: ArrayLike | None, q: ArrayLike | None
) -> tuple[NDArray[Any], NDArray[Any], NDArray[Any], int]:
    """Validate and resolve the common estimator inputs.

    Returns ``(x, cov, q, p)`` where ``x`` has shape ``(..., p)``, ``cov`` is a
    symmetric positive definite ``(p, p)`` matrix and ``q`` is a symmetric
    positive semi-definite ``(p, p)`` matrix; ``cov`` is the identity if it was
    not given, as is ``q``.

    """

    xa = np.asarray(x)
    if xa.ndim < 1:
        msg = "x must have at least one dimension."
        raise ValueError(msg)
    if not np.all(np.isfinite(xa)):
        # Before anything computes with x: a NaN/inf input would otherwise
        # only surface as an all-NaN estimate.
        msg = "x must contain only finite values."
        raise ValueError(msg)
    p = xa.shape[-1]
    if cov is None:
        cova = np.eye(p)
    else:
        cova = _validate_sympd(cov, (p, p), "covariance matrix")
    if q is None:
        qa = np.eye(p)
    else:
        qa = _validate_sympsd(q, (p, p), "loss matrix Q")
    return xa, cova, qa, p


def _validate_offset(offset: ArrayLike | None, p: int) -> NDArray[Any] | None:
    """Validate the shrinkage target ``offset`` against the dimension ``p``.

    Returns the offset as a float array of shape ``(p,)``, or ``None`` when no
    offset was given.  Both the shape and the finiteness are checked here so
    every entry point reports the same message.

    """

    if offset is None:
        return None
    o = np.asarray(offset, dtype=float)
    if o.shape != (p,):
        msg = f"offset must have shape {(p,)}, got {o.shape}."
        raise ValueError(msg)
    if not np.all(np.isfinite(o)):
        msg = "offset must contain only finite values."
        raise ValueError(msg)
    return o


def _validate_dirs(dirs: ArrayLike, p: int) -> NDArray[Any]:
    """Validate spanning vectors and return them as an array of shape ``(p, k)``.

    ``dirs`` must have shape ``(p, k)`` with ``k >= 1`` and full column rank
    (its columns span the affine direction of shrinkage, i.e. they form a
    basis of the subspace).  Returns the matrix of shape ``(p, k)``.

    """

    dv = np.asarray(dirs, dtype=float)
    if dv.ndim != 2 or dv.shape[0] != p or dv.shape[1] < 1:
        msg = f"dirs must have shape ({p}, k) with k >= 1, got shape {dv.shape}."
        raise ValueError(msg)
    if not np.all(np.isfinite(dv)):
        msg = "dirs must contain only finite values."
        raise ValueError(msg)
    rank = np.linalg.matrix_rank(dv)
    if rank < dv.shape[1]:
        msg = (
            "dirs must have linearly independent columns (full column rank), "
            f"got rank {rank} for {dv.shape[1]} columns."
        )
        raise ValueError(msg)
    return dv


def _check_strength(strength: float, *, bounded: bool = True) -> None:
    """Raise ``ValueError`` if ``strength`` is not a usable shrinkage strength.

    NaN is always rejected: as a bare range comparison it satisfies neither
    side and would silently pass, giving each estimator a different all-NaN or
    limit result instead of the promised error.  With ``bounded`` (the
    default) the value must additionally lie in the minimax range
    ``[0, 2]``, which ``+/-inf`` violates as well.  The estimators that
    document out-of-range strengths pass ``bounded=False`` and keep every
    finite (and infinite) value they accept today.

    """

    if np.isnan(strength):
        msg = "strength must not be NaN."
        raise ValueError(msg)
    if bounded and not 0.0 <= strength <= 2.0:
        msg = "strength must be in [0, 2]."
        raise ValueError(msg)


def _check_gamma_nonnegative(
    gamma: float | str | Callable[..., Any] | NDArray[Any],
) -> None:
    """Raise ``ValueError`` if ``gamma`` contains negative or NaN values.

    String presets and callables are left to be resolved downstream; only
    numeric (scalar or array) values are validated here.  ``+inf`` stays
    allowed: it is the documented flat prior (and ``-inf`` is caught by the
    sign check).  NaN would otherwise slip through the sign comparison and be
    interpreted differently by every estimator (identity, MLE, or all-NaN
    output).

    """

    if isinstance(gamma, str) or callable(gamma):
        return
    values = np.asarray(gamma)
    if np.any(values < 0) or np.any(np.isnan(values)):
        msg = "gamma must be non-negative and not NaN."
        raise ValueError(msg)


def _check_gamma_shape(
    gamma: float | str | Callable[..., Any] | NDArray[Any] | None,
    batch_shape: tuple[int, ...],
) -> None:
    """Raise ``ValueError`` unless a numeric per-observation ``gamma`` fits ``x``.

    A scalar ``gamma`` applies to every observation; an array carries one scale
    per observation and must have exactly the leading batch shape of ``x``.
    NumPy would otherwise broadcast a mismatched one silently -- handing one
    row the scale meant for another -- or fail later with a raw broadcast
    error.  String presets and callables are resolved against the data
    downstream and are left alone here.

    """

    if gamma is None or isinstance(gamma, str) or callable(gamma):
        return
    arr = np.asarray(gamma)
    if arr.ndim == 0:
        return
    if arr.shape != batch_shape:
        msg = (
            f"gamma has shape {arr.shape}; it must be a scalar or have the "
            f"shape of the leading batch dimensions of x {batch_shape}."
        )
        raise ValueError(msg)
