"""
Copyright (c) 2024 Lukas Koch. All rights reserved.

Statistical distributions that are useful, but not available in
``scipy.stats``.

"""

from __future__ import annotations

import numpy as np

from scipy.stats import chi, chi2, rv_continuous

__all__ = ["Bee", "Bee2", "Cee", "Cee2"]


class Bee(rv_continuous):
    def __init__(self, df=1):
        super().__init__(a=0)
        self.df = df
        self.chi = chi(df=1)

    def _cdf(self, x):
        p = self.chi.cdf(x) ** self.df
        return p


class Bee2(rv_continuous):
    def __init__(self, df=1):
        super().__init__(a=0)
        self.df = df
        self.chi2 = chi2(df=1)

    def _cdf(self, x):
        p = self.chi2.cdf(x) ** self.df
        return p


class Cee(rv_continuous):
    def __init__(self, k=[1]):
        super().__init__(a=0)
        self.k = np.atleast_1d(k)
        self.chi = [chi(df=_df) for _df in self.k]

    def _cdf(self, x):
        p = [c.cdf(x) for c in self.chi]
        return np.prod(p, axis=0)


class Cee2(rv_continuous):
    def __init__(self, k=[1]):
        super().__init__(a=0)
        self.k = np.atleast_1d(k)
        self.chi2 = [chi2(df=_df) for _df in self.k]

    def _cdf(self, x):
        p = [c.cdf(x) for c in self.chi2]
        return np.prod(p, axis=0)
