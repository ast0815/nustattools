"""Plot matrices in a colourblind-friendly way."""

from __future__ import annotations

from typing import Any

import numpy as np
from matplotlib import collections, colormaps, patches, ticker
from matplotlib import pyplot as plt
from numpy.typing import NDArray


def hinton(
    matrix: NDArray[Any],
    *,
    vmax: float | None = None,
    shape: str = "circle",
    origin: str = "upper",
    cmap: str = "cividis",
    legend: bool = False,
    ax: None | Any = None,
) -> tuple[Any, Any]:
    """Draw Hinton diagram for visualizing a matrix with positive and negative values.

    Parameters
    ----------

    matrix : numpy.ndarray
        The matrix to be visualized.
    vmax : float, optional
        The upper limit of the value scale. `-vmax` will be used as the lower
        limit. Defaults to being inferred from the data.
    shape : str, default="circle"
        Either "circle" or "square".
        The shape of the symbols representing the matrix elements.
    origin : str, default="upper"
        Either "upper" or "lower".
        Where to put the 1st element of the 1st axis.
    cmap : str, default="cividis"
        The Matplotlib colormap to take the colors from.
        Should be perceptually uniform sequantial.
    legend : bool, default=False
        Draw a "legend" to the side of the plot, showing the range of values.
    ax : matplotlib.axes.Axes, optional
        Axes object to plot onto

    Returns
    -------
    col0, col1 : matplotlib.collections.PatchCollection
        The collections of patches for the negative and positive colors
        respectively

    Examples
    --------

    .. plot::
        :include-source: True

        Basic usage:

        >>> import numpy as np
        >>> from matplotlib import pyplot as plt
        >>> from nustattools import plotting as nuplt
        >>> rng = np.random.default_rng()
        >>> M = rng.uniform(size=(10,10)) - 0.5
        >>> nuplt.hinton(M)

    .. plot::
        :include-source: True

        Plot with a legend:

        >>> import numpy as np
        >>> from matplotlib import pyplot as plt
        >>> from nustattools import plotting as nuplt
        >>> rng = np.random.default_rng()
        >>> M = rng.uniform(size=(10,10)) - 0.5
        >>> nuplt.hinton(M, legend=True)
        >>> plt.tight_layout(pad=2)

    .. plot::
        :include-source: True

        Variants:

        >>> import numpy as np
        >>> from matplotlib import pyplot as plt
        >>> from nustattools import plotting as nuplt
        >>> rng = np.random.default_rng()
        >>> M = rng.uniform(size=(10,10)) - 0.5
        >>> nuplt.hinton(M, legend=True, shape="square", cmap="gray", origin="lower")
        >>> plt.tight_layout(pad=2)

    Notes
    -----

    Based on https://matplotlib.org/stable/gallery/specialty_plots/hinton_demo.html


    """
    ax = ax if ax is not None else plt.gca()

    if vmax is None:
        vmax = np.abs(matrix).max()

    colors = colormaps[cmap]((0, 0.5, 1))

    # Create the shapes
    patches0 = []
    patches1 = []

    def add_patch(x: float, y: float, w: float) -> None:
        size = np.sqrt(abs(w) / vmax)
        if shape == "circle":
            patch: patches.Patch = patches.Circle(
                (x, y),
                size / 2,
            )
        elif shape == "square":
            patch = patches.Rectangle(
                (x - size / 2, y - size / 2),
                size,
                size,
            )
        else:
            e = f"Unknown shape: {shape}"
            raise ValueError(e)

        if w > 0:
            patches1.append(patch)
        else:
            patches0.append(patch)

    for (y, x), w in np.ndenumerate(matrix):
        add_patch(x, y, w)

    # Add "legend"
    if legend:
        lw = int(np.ceil(np.max(matrix.shape) / 20))
        lx = matrix.shape[1] + lw
        lh = int(np.ceil(matrix.shape[0] * 0.45) * 2 - 1)
        ly = ((matrix.shape[0] - 1) / 2) - ((lh - 1) / 2)
        for i, w in enumerate(np.linspace(-vmax, vmax, lh)):
            y = ly + (lh - 1) * (w / vmax + 1) / 2
            ww = w if origin == "lower" else -w
            for x in range(lx, lx + lw):
                add_patch(x, y, ww)
            if i in (0, (lh - 1) / 2, lh - 1):
                ax.plot(
                    [x + 0.6, x + 0.6 + 0.2 * lw],
                    [y, y],
                    clip_on=False,
                    color="k",
                    linewidth=1,
                )
                if abs(ww) < 1e-12:
                    ww = 0
                ax.text(x + 0.6 + 0.2 * lw, y, f" {ww:.3g}", verticalalignment="center")
        ax.add_patch(
            patches.Rectangle(
                (lx - 0.6, ly - 0.6),
                lw + 0.2,
                lh + 0.2,
                facecolor=colors[1],
                edgecolor="k",
                clip_on=False,
            )
        )

    # Create collections and set colors, plots much faster than adding every single patch to the axes
    col0 = collections.PatchCollection(patches0)
    col0.set_facecolor(colors[0])
    col0.set_clip_on(False)
    col1 = collections.PatchCollection(patches1)
    col1.set_facecolor(colors[2])
    col1.set_clip_on(False)

    # Add collections to axes
    ax.add_collection(col0)
    ax.add_collection(col1)

    ax.patch.set_facecolor(colors[1])
    ax.set_aspect("equal", "box")
    edge = 0.1
    ax.set_xlim(-0.5 - edge, matrix.shape[1] - 0.5 + edge)
    ax.set_ylim(-0.5 - edge, matrix.shape[0] - 0.5 + edge)
    m: int = np.max(matrix.shape)
    ax.xaxis.set_major_locator(
        ticker.MaxNLocator(
            integer=True, min_n_ticks=1, nbins=1 + int(8 * matrix.shape[1] / m)
        )
    )
    ax.yaxis.set_major_locator(
        ticker.MaxNLocator(
            integer=True, min_n_ticks=1, nbins=1 + int(8 * matrix.shape[0] / m)
        )
    )
    if origin == "upper":
        ax.invert_yaxis()
        ax.xaxis.tick_top()
    elif origin == "lower":
        pass
    else:
        e = f"Unknown origin: {origin}"
        raise ValueError(e)

    return (col0, col1)


__all__ = ["hinton"]
