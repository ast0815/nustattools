from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt

import nustattools.plotting as p


def test_hinton():
    M = np.ones((3, 9))
    p.hinton(M)
    p.hinton(M, vmax=2)
    p.hinton(M, origin="lower")
    p.hinton(M, cmap="gray")
    p.hinton(M, legend=True)
    fig, ax = plt.subplots()
    p.hinton(M, ax=ax)
