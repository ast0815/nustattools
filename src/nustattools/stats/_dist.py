"""
Copyright (c) 2024 Lukas Koch. All rights reserved.

Statistical distributions that are useful, but not available in
``scipy.stats``.

"""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import chi, chi2, rv_continuous

from ._fmax import TestStatistic

# We need to define methods with the new distributions' parameters
# pylint: disable=arguments-differ

# We need to inherit from non-typed scipy
# mypy: disable-error-code="misc, no-any-return"


class Bee(rv_continuous):
    """A random variable representing the maximum of `df` chi distributions.

    Each :any:`chi <scipy.stats.chi>` disitribution has ``df = 1``.

    .. note::
        You probably do not need to use this class directly. Instead work with
        the instance :data:`bee`.

    Parameters
    ----------
    df : int
        The number of chi-distirbuted variables to take the maximum of.

    """

    def _cdf(self, x: ArrayLike, df: int) -> ArrayLike:
        return chi.cdf(x, df=1) ** df


#: Use this instance of :class:`Bee`
bee = Bee(name="bee", a=0)


class Bee2(rv_continuous):
    """A random variable representing the maximum of `df` chi2 distributions.

    Each :any:`chii2 <scipy.stats.chi2>` disitribution has ``df = 1``.

    .. note::
        You probably do not need to use this class directly. Instead work with
        the instance :data:`bee2`.

    Parameters
    ----------
    df : int
        The number of chi-distirbuted variables to take the maximum of.

    Notes
    -----

    This distribution is discussed in [Koch2021]_ in the context of robust test
    statistics.


    References
    ----------

    .. [Koch2021] L. Koch, "Robust test statistics for data sets with missing
        correlation information," Phys. Rev. D 103, 113008 (2021) , Vol. 103, No.
        11 p. 113008, https://arxiv.org/abs/2102.06172

    """

    def _cdf(self, x: ArrayLike, df: int) -> ArrayLike:
        return chi2.cdf(x, df=1) ** df


#: Use this instance of :class:`Bee2`
bee2 = Bee2(name="bee2", a=0)


class DF:
    """Helper class to get around limitations of `rv_continuous`.

    This class represents the shape parameter for the :class:`Cee` and
    :class:`Cee2` distributions.

    Parameters
    ----------

    *args : tuple of int
        The `df` for each chi(2) distribution

    """

    def __init__(self, *args: tuple[int]) -> None:
        self.df = args

    def __gt__(self, other: Any) -> bool:
        return True


class Cee(rv_continuous):
    """A random variable representing the maximum of multiple chi distributions.

    Each :any:`chi <scipy.stats.chi>` disitribution can have a different ``df``. If
    all ``df`` are equal to 1, this is identical to the :class:`Bee`
    distribution with ``df = len(k)``.

    .. note::
        You probably do not need to use this class directly. Instead work with
        the instance :data:`cee`.

    Parameters
    ----------
    k : DF or Iterable of DF
        Special class to pass variable length list of degrees of freedom of the
        chi-distirbuted variables to take the maximum of.

    Examples
    --------

    >>> cee(k=DF(1,2,3)).pdf(1)
    0.2501359390297275

    """

    def _cdf(self, x: NDArray[Any], k: Iterable[DF]) -> NDArray[Any]:
        # Translate each DF object into df tuple
        dof = [_k.df for _k in k]
        # Calculate and return cdf for each
        return np.array([np.prod([chi.cdf(x, df=n) for n in _k], axis=0) for _k in dof])


#: Use this instance of :class:`Cee`
cee = Cee(name="cee", a=0)


class Cee2(rv_continuous):
    """A random variable representing the maximum of multiple chi2 distributions.

    Each :any:`chi2 <scipy.stats.chi2>` disitribution can have a different ``df``. If
    all ``df`` are equal to 1, this is identical to a :class:`Bee2`
    disritbution with ``df = len(k)``.

    .. note::
        You probably do not need to use this class directly. Instead work with
        the instance :data:`cee2`.

    Parameters
    ----------
    k : DF or Iterable of DF
        Special class to pass variable length list of degrees of freedom of the
        chi-distirbuted variables to take the maximum of.

    Examples
    --------

    >>> cee2(k=DF(1,2,3)).pdf(1)
    0.12506796951321578

    Notes
    -----

    TODO: Add reference to paper.


    """

    def _cdf(self, x: NDArray[Any], k: Iterable[DF]) -> NDArray[Any]:
        # Translate each DF object into df tuple
        dof = [_k.df for _k in k]
        # Calculate and return cdf for each
        return np.array(
            [np.prod([chi2.cdf(x, df=n) for n in _k], axis=0) for _k in dof]
        )


#: Use this instance of :class:`Cee2`
cee2 = Cee2(
    name="cee2",
    a=0,
)


class RVTestStatistic(rv_continuous):
    """A random variable distributed as the expectation of a :class:`TestStatistic`.

    .. note::
        You probably do not need to use this class directly. Instead work with
        the instance :data:`rvteststatistic`.

    Parameters
    ----------
    statistic : TestStatistic or Iterable of TestStatistic

    Examples
    --------

    >>> ts = OptimalFMaxStatistic(k=[1,2,3])
    >>> rvteststatistic(statistic=ts).pdf(1)
    0.24601379637056994

    """

    def _cdf(self, x: NDArray[Any], statistic: Iterable[TestStatistic]) -> NDArray[Any]:
        # Calculate and return cdf for each statistic
        return np.array([s.cdf(x) for s in statistic])


#: Use this instance of :class:`RVTestStatistic`
rvteststatistic = RVTestStatistic(
    name="rvteststatistic",
)

__all__ = ["DF"]
# Export all distributions and their instances
_g = dict(globals().items())
for _s, _x in _g.items():
    # Only look at distribution classes
    if isinstance(_x, type) and issubclass(_x, rv_continuous) and _s != "rv_continuous":
        # Include the class
        __all__.append(_s)
        # And the instance
        __all__.append(_s.lower())
