"""Plot correlated data"""

from __future__ import annotations

import itertools
from typing import Any

import numpy as np
from matplotlib import pyplot as plt
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import minimize


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
    yerr_safe = np.where(yerr > 0, yerr, 1e-12)
    ycor = ycov / yerr_safe[:, np.newaxis] / yerr_safe[np.newaxis, :]
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


def pcplot(
    x: NDArray[Any],
    y: NDArray[Any],
    ycov: NDArray[Any],
    *,
    componentwidth: Any = None,
    scaling: float | str = "mincor",
    poshatch: str = "/" * 5,
    neghatch: str = "\\" * 2,
    drawcorlines: bool = True,
    ax: Any = None,
    **kwargs: Any,
) -> Any:
    """Plot data points with 1st PCA component and correlation lines.

    The contribution of the first principle component is subtracted from the
    covariance and the remainder plotted with :py:func:`corlines`. Then the
    difference to the full covariance matrix is plotted with the type of infill
    indicating the direction of the first principle component.

    Parameters
    ----------

    x, y : numpy.ndarray
        The data x and y coordinates to be plotted.
    ycov : numpy.ndarray
        The covariance matrix describing the uncertainties of the y-values. The
        error bars will correspond the the square root of the diagonal entries.
    componentwidth : optional
        The width of the hatched areas indicating the 1st principal component
        in axes coordinates. Can be a single number, so it is equal for all
        data points; an iterable of numbers so it is different for each, or an
        iterable of pairs of numbers, so there is an asymmetric width for each.
    scaling: default="mincor"
        Determines how the length of the first principle component is scaled
        before removing its contribution from the covariance. If a
        :py:class:`float`, the contribution is scaled with that value. At 0.0,
        nothing is removed, at 1.0 the component is removed completely and the
        remaining covariance's rank will reduce by 1. If ``"mincor"``, the
        component will be scaled such that the overall correlation in the
        remaining covariance is minimized. If ``"second"``, the component will
        be scaled such that the remaining contribution of the first principle
        component is equal to the second principal component. If ``"last"``,
        the component will be scaled such that its contribution is equal to the
        last principle component.
    poshatch, neghatch: str, optional
        The Matplotlib hatch styles for the positive and negative directions of
        the first principal component.
    drawcorlines: default=True
        Whether to draw correlation lines of the remaining covariance.
    ax : matplotlib.axes.Axes, optional
        Axes object to plot onto
    **kwargs : dict, optional
        All other keyword arguments are passed to :py:func:`corlines`

    Returns
    -------
    matplotlib.container.ErrorbarContainer
        The return value of the :py:func:`corlines` function.

    Notes
    -----

    This plotting style is most useful for data where the first principle
    component dominates the covariance of the data.

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
        >>> nuplt.pcplot(x, y, cov, marker="x")


    """

    if not drawcorlines:
        kwargs.update({"corlinestyle": "", "cormarker": ""})

    yerr = np.sqrt(np.diag(ycov))
    yerr_safe = np.where(yerr > 0, yerr, 1e-12)
    ycor = ycov / yerr_safe[:, np.newaxis] / yerr_safe[np.newaxis, :]

    # Get principle components
    u, d, _ = np.linalg.svd(ycor)
    # Don't remove all of 1st principle component.
    # Otherwise the remaining K will be degenerate.
    # This also ensures that we do nothing if ycov in uncorrelated.
    u = u[:, 0]
    if isinstance(scaling, float):
        # Scale from 0 to maximum allowed
        u *= yerr * scaling * np.sqrt(d[0])
    elif scaling == "second":
        # Scale so remaining contribution is same as second PCA component
        u *= yerr * (np.sqrt(d[0] - d[1]))
    elif scaling == "last":
        # Scale so remaining contribution is same as last PCA component
        u *= yerr * (np.sqrt(d[0] - d[-1]))
    elif scaling == "mincor":
        # Scale to minimize total correlation in remaining covariance
        def fun(s: ArrayLike) -> Any:
            v = u * yerr * s * np.sqrt(d[0])
            V = v[:, np.newaxis] @ v[np.newaxis, :]
            # Ignore degenerate components
            L = (ycov - V)[d > 0, :][:, d > 0]

            with np.errstate(divide="ignore", invalid="ignore"):
                # Ignore divisions by zero when we scale by 1.0
                det = np.linalg.det(L)
                return np.prod(np.diag(L)) / det

        # Start close to scaling to lowest, non-zero PCA component
        # Ensures that we do nothing if everything is already uncorrelated
        dl = d[d > 0]
        ret = minimize(fun, x0=(dl[0] - dl[-1]), bounds=[(0.0, 1.0)])
        u *= yerr * ret.x * np.sqrt(d[0])
    else:
        e = f"Unknown scaling: {scaling}"
        raise ValueError(e)
    U = u[:, np.newaxis] @ u[np.newaxis, :]
    K = ycov - U

    if ax is None:
        ax = plt.gca()

    if componentwidth is None:
        # Try to guess a reasonable width from the data
        cw = min(np.min(np.diff(x)) * 0.9, (np.max(x) - np.min(x)) / 15)
        componentwidth = itertools.cycle([cw])

    try:
        cw_cycle = itertools.cycle(componentwidth)
    except TypeError:
        cw_cycle = itertools.cycle([componentwidth])

    # Plot error bars with correlation lines
    bars = corlines(x, y, K, ax=ax, **kwargs)

    # Plot first principle component
    Kerr = np.sqrt(np.diag(K))
    color = bars.lines[0].get_color()
    xx: list[float] = []
    yy: list[float] = []
    e_min: list[float] = []
    e_max: list[float] = []
    fill: list[bool] = []
    for i, (xs, ys, cw) in enumerate(zip(x, y, cw_cycle)):
        try:
            dxm = cw[0]
            dxp = cw[1]
        except (IndexError, TypeError):
            dxm = cw / 2
            dxp = cw / 2
        su = np.sign(u[i])
        su = 1 if su == 0 else su
        emin = Kerr[i] * su
        emax = yerr[i] * su
        # Turn every data point into three so we can use fill_between
        # and switch off filling in between points
        xx.extend((xs - dxm, xs + dxp, xs + dxp))
        yy.extend((ys,) * 3)
        e_min.extend((emin,) * 3)
        e_max.extend((emax,) * 3)
        fill.extend((True, True, False))

    xx_arr = np.array(xx)
    yy_arr = np.array(yy)
    e_min_arr = np.array(e_min)
    e_max_arr = np.array(e_max)
    fill_arr = np.array(fill)

    ax.fill_between(
        xx_arr,
        yy_arr + e_min_arr,
        yy_arr + e_max_arr,
        where=fill_arr,
        hatch=poshatch,
        facecolor="none",
        edgecolor=color,
    )
    ax.fill_between(
        xx_arr,
        yy_arr - e_min_arr,
        yy_arr - e_max_arr,
        where=fill_arr,
        hatch=neghatch,
        facecolor="none",
        edgecolor=color,
    )

    return bars


__all__ = ["corlines", "pcplot"]
