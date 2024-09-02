"""Tools for fmax statistics"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Iterable, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import root
from scipy.stats import chi2


class TestStatistic(ABC):
    """General class for test statistcs.

    A TestStatistic must implement a way to calculate the test statistic from
    some data, as well as a way to calculate the CDF of the expected
    distribution of the test statistic if an assumed model is correct.

    """

    def calculate(self, data: ArrayLike) -> NDArray[Any]:
        """Calculate the test statistic from some given data.

        You can also call the object directly, e.g. ``statistic(data)`` instead
        of ``statistic.calculate(data)``.

        Parameters
        ----------
        data : numpy.ndarray
            The data to calculate the test statistic for. The test statstic is
            calculated over the last dimension of the array, so the return
            shape will be ``input_shape[:-1]``.

        Returns
        -------
        statistic_value : numpy.ndarray

        """
        return self._calculate(data)

    def cdf(self, statistic: ArrayLike) -> NDArray[Any]:
        """Calculate the CDF of the expected distribution of the test statistic.

        Assumes that certain model assumptions are true.

        Parameters
        ----------
        statistic : numpy.ndarray
            The test statistic to calculate the CDF for.

        Returns
        -------
        cdf : numpy.ndarray

        """

        return self._cdf(statistic)

    def __call__(self, data: ArrayLike) -> NDArray[Any]:
        return self._calculate(data)

    @abstractmethod
    def _calculate(self, data: ArrayLike) -> NDArray[Any]: ...

    @abstractmethod
    def _cdf(self, statistic: ArrayLike) -> NDArray[Any]: ...

    def __gt__(self, other: Any) -> Any:
        # Needed for use in RVTestStatistic
        return True


class FMaxStatistic(TestStatistic):
    """Test statistic that takes the maximum of functions of chi2 distributed data.

    Parameters
    ----------
    k : Iterable of int
        Number of degrees of freedom for the assumed chi2 distributions of the
        data points.
    funcs : Iterable of Callable or None, optional
        Functions to apply to each data point. Maximum will be taken from the
        function outputs. If not specified, the identity function``f(x) = x``
        will be used for all data points.
    invfuncs : Iterable of Callable or None, optional
        Inverse of the functions that are applied to each data point. These are
        used to calculate the CDF. If not specified, the inverse will be
        calculated numerically using the  `funcs`.

    Notes
    -----

    TODO: Cite paper.

    """

    def __init__(
        self,
        *,
        k: Iterable[int],
        funcs: Iterable[Callable[..., NDArray[Any]] | None] | None = None,
        inv_funcs: Iterable[Callable[..., NDArray[Any]] | None] | None = None,
    ) -> None:
        self.k = np.array(k)
        if funcs is None:
            funcs = [None for N in k]
        funcs_list = []

        def identity(x: ArrayLike) -> NDArray[Any]:
            return np.asarray(x)

        for f in funcs:
            if f is None:
                funcs_list.append(identity)
            else:
                funcs_list.append(f)
        self.funcs = funcs_list
        if inv_funcs is None:
            inv_funcs = [None for f in funcs]
        self.inv_funcs = inv_funcs

    def _calculate(self, data: ArrayLike) -> NDArray[Any]:
        x = np.asarray(data)
        y = np.ndarray(x.shape)  # type: NDArray[Any]
        for i, f in enumerate(self.funcs):
            y[..., i] = f(x[..., i])
        return np.asarray(np.max(y, axis=-1))

    def _cdf(self, statistic: ArrayLike) -> NDArray[Any]:
        z = np.asarray(statistic)
        M2 = np.ndarray((*z.shape, len(self.funcs)))  # type: NDArray[Any]
        for i, (f, invf) in enumerate(zip(self.funcs, self.inv_funcs)):
            if invf is None:
                m2 = np.ndarray(z.shape)  # type: NDArray[Any]
                for j, zz in enumerate(z.flat):

                    def rf(
                        x: NDArray[Any],
                        fun: Callable[[NDArray[Any]], NDArray[Any]] = f,
                        zz: NDArray[Any] = zz,
                    ) -> NDArray[Any]:
                        return cast(NDArray[Any], fun(x) - zz)

                    ret = root(rf, 0.5)
                    m2.flat[j] = ret.x[0]
                M2[..., i] = m2
            else:
                M2[..., i] = invf(z)
        cdf = chi2(df=self.k).cdf(M2)
        return np.asarray(np.prod(cdf, axis=-1))


class QMaxStatistic(FMaxStatistic):
    """FMaxStatistic where the functions are the assumed CDFs of the data.

    This is equivalent to taking the minimum of the individual p-values as the
    test statistic.

    Parameters
    ----------
    k : Iterable of int
        Number of degrees of freedom for the assumed chi2 distributions of the
        data points.

    Notes
    -----

    TODO: Cite paper.

    """

    def __init__(
        self,
        *,
        k: Iterable[int],
    ) -> None:
        funcs = []
        inv_funcs = []
        for n in k:

            def fun(x: ArrayLike, df: int = n) -> NDArray[Any]:
                return np.asarray(chi2(df=df).cdf(x))

            def ifun(x: ArrayLike, df: int = n) -> NDArray[Any]:
                return np.asarray(chi2(df=df).ppf(x))

            funcs.append(fun)
            inv_funcs.append(ifun)
        super().__init__(k=k, funcs=funcs, inv_funcs=inv_funcs)


class OptimalFMaxStatistic(FMaxStatistic):
    """FMaxStatistic that minimizes the maximum M-distance for any given p-value.

    Parameters
    ----------
    k : Iterable of int
        Number of degrees of freedom for the assumed chi2 distributions of the
        data points.

    Notes
    -----

    TODO: Cite paper.

    """

    def __init__(
        self,
        *,
        k: Iterable[int],
    ) -> None:
        funcs = []
        for n in k:

            def fun(x: ArrayLike, df: int = n) -> NDArray[Any]:
                return np.asarray(chi2(df=df).logcdf(x) - chi2(df=df).logpdf(x))

            funcs.append(fun)
        super().__init__(k=k, funcs=funcs)


__all__ = ["TestStatistic", "FMaxStatistic", "QMaxStatistic", "OptimalFMaxStatistic"]
