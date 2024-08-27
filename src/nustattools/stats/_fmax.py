"""Tools for fmax statistics"""

from __future__ import annotations

from typing import Any, Callable, Iterable

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import root
from scipy.stats import chi2


class FMaxStatistic:
    """bla"""

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

    def __call__(self, x: Iterable[ArrayLike]) -> NDArray[Any]:
        y = [f(z) for f, z in zip(self.funcs, x)]
        return np.asarray(np.max(y))

    def cdf(self, z: float) -> ArrayLike:
        M2 = []
        for f, invf in zip(self.funcs, self.inv_funcs):
            if invf is None:

                def rf(
                    x: NDArray[Any], fun: Callable[[NDArray[Any]], NDArray[Any]] = f
                ) -> NDArray[Any]:
                    return fun(x) - z

                ret = root(rf, 0.5)
                M2.append(ret.x[0])
            else:
                M2.append(invf(z))
        cdf = chi2(df=self.k).cdf(M2)
        return np.asarray(np.prod(cdf))


class OptimalFMaxStatistic(FMaxStatistic):
    """blub"""

    def __init__(
        self,
        *,
        k: Iterable[int],
        # funcs: Iterable[Callable[..., NDArray[Any]] | None] | None = None,
        # inv_funcs: Iterable[Callable[..., NDArray[Any]] | None] | None = None,
    ) -> None:
        funcs = []
        for n in k:

            def fun(x: ArrayLike, df: int = n) -> NDArray[Any]:
                return np.asarray(chi2(df=df).logcdf(x) - chi2(df=df).logpdf(x))

            funcs.append(fun)
        super().__init__(k=k, funcs=funcs)


__all__ = ["FMaxStatistic", "OptimalFMaxStatistic"]
