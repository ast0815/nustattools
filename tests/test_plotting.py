from __future__ import annotations

import numpy as np
import pytest
from matplotlib import pyplot as plt

import nustattools.plotting as p


def test_hinton():
    M = np.ones((3, 9))
    p.hinton(M)
    p.hinton(M, vmax=2)
    p.hinton(M, origin="lower")
    p.hinton(M, cmap="gray")
    p.hinton(M, legend=True)
    for s in ("circle", "square"):
        p.hinton(M, shape=s)
    with pytest.raises(ValueError, match="Unknown shape"):
        p.hinton(M, shape="")
    for o in ("upper", "lower"):
        p.hinton(M, origin=o)
    with pytest.raises(ValueError, match="Unknown origin"):
        p.hinton(M, origin="")
    fig, ax = plt.subplots()
    p.hinton(M, ax=ax)


def test_corplots():
    x = np.linspace(0, 10, 5)
    y = x
    u = x[:, np.newaxis]
    cov = np.eye(5) + u @ u.T
    p.corlines(x, y, cov)
    p.wedgeplot(x, y, np.ones_like(x))
    p.pcplot(x, y, cov)
    p.pcplot(x, y, cov, return_dict={})
    p.pcplot(x, y, cov, componentwidth=0.2)
    p.pcplot(x, y, cov, componentwidth=[0.2] * 5)
    p.pcplot(x, y, cov, componentwidth=[[0.2, 0.5]] * 5)
    p.pcplot(x, y, cov, drawcorlines=False)
    p.pcplot(x, y, cov, normalize=False)
    p.pcplot(x, y, cov, drawconditional=False)
    with pytest.warns(
        RuntimeWarning,
        match="Requested 5 principal components, but only 3 hatch styles are defined. Showing only 3 components.",
    ):
        p.pcplot(x, y, cov, components=0.999)
    with pytest.warns(
        RuntimeWarning,
        match="Requested 5 principal components, but only 3 hatch styles are defined. Showing only 3 components.",
    ):
        p.pcplot(x, y, cov, components=len(cov))
    p.pcplot(x, y, cov, components=len(cov), hatch=[(".", "/")] * len(cov))
    p.pcplot(x, y, cov, label_components=True)
