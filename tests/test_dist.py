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
    cee = s.cee(k=s.DF(1))
    assert abs(cee.pdf(1) - chi(df=1).pdf(1)) < 1e-9
    assert abs(cee.cdf(1) - chi(df=1).cdf(1)) < 1e-9


def test_degenerate_cee2():
    cee2 = s.cee2(k=s.DF(1))
    assert abs(cee2.pdf(1) - chi2(df=1).pdf(1)) < 1e-9
    assert abs(cee2.cdf(1) - chi2(df=1).cdf(1)) < 1e-9


def test_cee_shapes():
    cee = s.cee(k=[[s.DF(1), s.DF(2, 3), s.DF(4, 5, 6)]] * 4)
    ret = cee.pdf([[1, 2, 3]] * 4)
    assert ret.shape == (4, 3)


def test_cee2_shapes():
    cee2 = s.cee2(k=[[s.DF(1), s.DF(2, 3), s.DF(4, 5, 6)]] * 4)
    ret = cee2.pdf([[1, 2, 3]] * 4)
    assert ret.shape == (4, 3)


def test_rvteststatistic():
    stat = s.FMaxStatistic(k=[1])
    rv = s.rvteststatistic(statistic=stat)
    assert abs(rv.pdf(1) - chi2(df=1).pdf(1)) < 1e-9
    assert abs(rv.cdf(1) - chi2(df=1).cdf(1)) < 1e-9
