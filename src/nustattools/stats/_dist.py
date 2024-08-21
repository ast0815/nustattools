"""
Copyright (c) 2024 Lukas Koch. All rights reserved.

Statistical distributions that are useful, but not available in
``scipy.stats``.

"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import chi, chi2, rv_continuous

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

    This distribution is discussed in [1]_ in the context of robust test
    statistics.


    References
    ----------

    .. [1] L. Koch, "Robust test statistics for data sets with missing
        correlation information," Phys. Rev. D 103, 113008 (2021) , Vol. 103, No.
        11 p. 113008, https://arxiv.org/abs/2102.06172

    """

    def _cdf(self, x: ArrayLike, df: int) -> ArrayLike:
        return chi2.cdf(x, df=1) ** df


#: Use this instance of :class:`Bee2`
bee2 = Bee2(name="bee2", a=0)


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
    k : list of int or int
        List of degrees of freedom of the chi-distirbuted variables to take the
        maximum of.

    """

    def _cdf(self, x: ArrayLike, k: ArrayLike) -> ArrayLike:
        k = np.atleast_1d(k)
        p = [chi.cdf(x, df=n) for n in k]
        return np.prod(p, axis=0)


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
    k : list of int or int
        List of degrees of freedom of the chi2-distirbuted variables to take the
        maximum of.

    Notes
    -----

    TODO: Add reference to paper.


    """

    def _cdf(self, x: ArrayLike, k: ArrayLike) -> ArrayLike:
        k = np.atleast_1d(k)
        p = [chi2.cdf(x, df=n) for n in k]
        return np.prod(p, axis=0)


#: Use this instance of :class:`Cee2`
cee2 = Cee2(name="cee2", a=0)

# Export all distributions and their instances
__all__ = []
_g = dict(globals().items())
for _s, _x in _g.items():
    # Only look at distribution classes
    if isinstance(_x, type) and issubclass(_x, rv_continuous) and _s != "rv_continuous":
        # Include the class
        __all__.append(_s)
        # And the instance
        __all__.append(_s.lower())
