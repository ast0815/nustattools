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
