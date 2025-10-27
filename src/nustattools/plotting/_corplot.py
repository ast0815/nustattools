"""Plot correlated data"""

from __future__ import annotations

from typing import Any

import numpy as np
from matplotlib import pyplot as plt
from numpy.typing import NDArray


def corlines(
    x: NDArray[Any],
    y: NDArray[Any],
    ycov: NDArray[Any],
    *,
    corlinestyle: str = ":",
    cormarker: str = "_",
    ax: None | Any = None,
    **kwargs: Any,
) -> Any:
    """Plot data points with error bars and correlation lines.

    The correlation lines indicate the correlatio between neighbouring data
    points. They are attached to the vertical error bars at a relative height
    corresponding to the correlation coefficient between the data points. For
    positive correlations, they are attached on the same sides, for negative
    correlation at opposing sides.

    Parameters
    ----------

    x, y : numpy.ndarray
        The data x and y coordinates to be plotted.
    ycov : numpy.ndarray
        The covariance matrix describing the uncertainties of the y-values. The
        error bars will correspond the the square root of the diagonal entries.
    corlinestyle : str, default=":"
        The Matplotlib linestyle for the correlation lines.
    cormarker : str, default="_"
        The Matplotlib marker used where the correlation lines attach to the
        vertical error bars.
    ax : matplotlib.axes.Axes, optional
        Axes object to plot onto
    **kwargs : dict, optional
        All other keyword arguments are passed to :py:meth:`matplotlib.axes.Axes.errorbar`

    Returns
    -------
    matplotlib.container.ErrorbarContainer
        The return value of the :py:meth:`matplotlib.axes.Axes.errorbar` method.

    Notes
    -----

    Where the correlation lines attach to the vertical error bars, gives an
    indication of how much of the variance in the given data point is "caused"
    by the neighbouring data points. Also, if the value of the neighbouring
    data point is fixed to plus or minus 1 sigma away from its mean position,
    the mean of the given data point is shifted to the position where the
    correlation line attaches. Of course, this is a symmetric relationship and
    the "fixing" and "causing" can equally be read in the opposite direction.

    Examples
    --------

    .. plot::
        :include-source: True

        Basic usage:

        >>> import numpy as np
        >>> from matplotlib import pyplot as plt
        >>> from nustattools import plotting as nuplt
        >>> rng = np.random.default_rng()
        >>> x = np.linspace(0, 10, 5)
        >>> u = x[:,np.newaxis] / 4
        >>> u[-2] *= -1
        >>> cov = np.eye(5) + u@u.T
        >>> y = rng.multivariate_normal(np.zeros(5), cov)
        >>> nuplt.corlines(x, y, cov, marker="x")

    """

    if ax is None:
        ax = plt.gca()

    # Plot error bars
    yerr = np.sqrt(np.diag(ycov))
    fmt = kwargs.pop("fmt", " ")
    bars = ax.errorbar(x, y, yerr=yerr, fmt=fmt, **kwargs)
    color = bars.lines[0].get_color()

    # Get correlations between neighbours
    ycor = ycov / yerr[:, np.newaxis] / yerr[np.newaxis, :]
    ncor = np.diag(ycor, k=1)

    # Plot lines
    for i, c in enumerate(ncor):
        ax.plot(
            [x[i], x[i + 1]],
            [y[i] + yerr[i] * np.abs(c), y[i + 1] + yerr[i + 1] * c],
            color=color,
            linestyle=corlinestyle,
            marker=cormarker,
        )
        ax.plot(
            [x[i], x[i + 1]],
            [y[i] - yerr[i] * np.abs(c), y[i + 1] - yerr[i + 1] * c],
            color=color,
            linestyle=corlinestyle,
            marker=cormarker,
        )
    return bars


__all__ = ["corlines"]
