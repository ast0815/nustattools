from __future__ import annotations

from scipy.stats import chi, chi2

import nustattools.stats as s


def test_degenerate_bee():
    bee = s.bee(df=1)
    assert abs(bee.pdf(1) - chi(df=1).pdf(1)) < 1e-9
    assert abs(bee.cdf(1) - chi(df=1).cdf(1)) < 1e-9


def test_degenerate_bee2():
    bee2 = s.bee2(df=1)
    assert abs(bee2.pdf(1) - chi2(df=1).pdf(1)) < 1e-9
    assert abs(bee2.cdf(1) - chi2(df=1).cdf(1)) < 1e-9


def test_degenerate_cee():
    cee = s.cee(k=[1])
    assert abs(cee.pdf(1) - chi(df=1).pdf(1)) < 1e-9
    assert abs(cee.cdf(1) - chi(df=1).cdf(1)) < 1e-9


def test_degenerate_cee2():
    cee2 = s.cee2(k=[1])
    assert abs(cee2.pdf(1) - chi2(df=1).pdf(1)) < 1e-9
    assert abs(cee2.cdf(1) - chi2(df=1).cdf(1)) < 1e-9
