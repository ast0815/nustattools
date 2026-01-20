"""Plot correlated data"""

from __future__ import annotations

import itertools
from typing import Any
from warnings import warn

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.collections import PolyCollection
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
    zorder = bars.lines[0].zorder

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
            zorder=zorder,
        )
        ax.plot(
            [x[i], x[i + 1]],
            [y[i] - yerr[i] * np.abs(c), y[i + 1] - yerr[i + 1] * c],
            color=color,
            linestyle=corlinestyle,
            marker=cormarker,
            zorder=zorder,
        )
    return bars


def wedgeplot(
    x: NDArray[Any],
    y: NDArray[Any],
    dy: NDArray[Any],
    *,
    wedgewidth: Any = None,
    ax: Any = None,
    **kwargs: Any,
) -> Any:
    """Plot vertical wedges at the given data points with the given lengths.

    Parameters
    ----------

    x, y, dy : numpy.ndarray
        The data x and y coordinates and length of the wedges to be plotted.
    wedgewidth : optional
        The width of the wedges in axes coordinates. Can be a single number, so
        it is equal for all data points; an iterable of numbers so it is
        different for each, or an iterable of pairs of numbers, so there is an
        asymmetric width for each.
    ax : matplotlib.axes.Axes, optional
        Axes object to plot onto
    **kwargs : dict, optional
        All other keyword arguments are passed to :py:class:`matplotlib.collections.PolyCollection`


    Returns
    -------
    matplotlib.collections.PolyCollection


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
        >>> err = np.sqrt(np.diag(cov))
        >>> y = rng.multivariate_normal(np.zeros(5), cov)
        >>> up = nuplt.wedgeplot(x, y, err, color="C2")
        >>> down = nuplt.wedgeplot(x, y, -err, color="C3")
        >>> down.set_facecolor("C1")

    """

    if ax is None:
        ax = plt.gca()

    if wedgewidth is None:
        # Try to guess a reasonable width from the data
        ww = min(np.min(np.diff(x)) * 0.9, (np.max(x) - np.min(x)) / 15)  # type: ignore[operator]
        wedgewidth = itertools.cycle([ww])

    try:
        ww_cycle = itertools.cycle(wedgewidth)
    except TypeError:
        ww_cycle = itertools.cycle([wedgewidth])

    # Plot create wedges

    paths = []
    for xx, yy, dd, w in zip(x, y, dy, ww_cycle):
        try:
            dxm = w[0]
            dxp = w[1]
        except (IndexError, TypeError):
            dxm = w / 2
            dxp = w / 2
        points = [
            (xx - dxm, yy),
            (xx, yy + dd),
            (xx + dxp, yy),
        ]
        paths.append(points)
        # Make sure the axis is scaled to include everything
        ax.update_datalim(points)

    col = PolyCollection(paths, **kwargs)
    ax.add_collection(col)
    ax.autoscale()
    return col


def pcplot(
    x: NDArray[Any],
    y: NDArray[Any],
    ycov: NDArray[Any],
    *,
    componentwidth: Any = None,
    components: float | int = 0.5,
    target_quantile: float = 0.5,
    hatch: list[tuple[str, str]] | None = None,
    drawcorlines: bool = True,
    drawconditional: bool = True,
    normalize: bool = True,
    ax: Any = None,
    label_components: bool = False,
    return_dict: None | dict[Any, Any] = None,
    **kwargs: Any,
) -> Any:
    """Plot data points with 1st PCA component and correlation lines.

    The contribution of the first principal component is subtracted from the
    covariance and the remainder plotted with :py:func:`corlines`. Then the
    difference to the full covariance matrix is plotted with the type of infill
    indicating the direction of the first principal component.


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
    components : int or float, default=0.5
        How many components to show. If set to a ``float``, the number of
        components is set to cover the requested fraction of the total
        covariance. Cannot exceed the number of defined hatch styles.
    target_quantile : float, default=0.5
        Determines the scaling of the components to be removed and plotted
        separately. The target is chose as the given quantile of the original
        principal components. If the target is larger than the first untouched
        component (e.g. the 3rd componend when 2 components are plotted), the
        value of that untouched component is chosen as target.
    hatch : list of tuple of str, optional
        The Matplotlib hatch styles for the positive and negative  directions
        of the principal components.
    drawcorlines : default=True
        Whether to draw correlation lines of the remaining covariance.
    drawconditional : default=True
        Whether to draw the conditional uncertainty of each data point, i.e.
        the allowed variance if all other points are fixed.
    normalize : default=True
        If ``True``, the covariance is scaled such that all diagonals are 1,
        and the PCA is run on the correlation matrix. If ``False``, the PCA is
        run on the covariance matrix directly. In the latter case, different
        error scales for different data points will have a strong influence on
        the selection of the components.
    ax : matplotlib.axes.Axes, optional
        Axes object to plot onto
    label_components : default=False
        Whether to add labels to the principal components.
    return_dict : dict, optional
        Dictionary to store some of the intermediary steps of the covariance
        decompositions.
    **kwargs : dict, optional
        All other keyword arguments are passed to :py:func:`corlines`


    Returns
    -------

    matplotlib.container.ErrorbarContainer
        The return value of the :py:func:`corlines` function.


    Notes
    -----

    This plotting style is most useful for data where the first one or two
    principal components dominate the covariance of the data.

    The algorithm for plotting is as follows:

    1.  Calculate the principal components.

        - This is done by doing a Single Value Decomposition of the covariance
          matrix. If `normalize` is ``True``, the corresponding correlation
          matrix is used instead.

    2.  Determine the number principal components to be shown, ``N_pc``.

        - If an integer number is provided, this number is used
        - Otherwise the number is chosen so that those components together cover
          the provided fraction of the total covariance, i.e. the sum of
          singular values. Higher numbers mean more components.

    3.  Determine the amount of each component that will be removed.

        - Removing 100% of the principal components would make the remaining
          covariance matrix degenerate and thus strongly correlated. The aim is
          to make the plot of the remainder _less_ correlated, so the removed
          components need to be scaled down.
        - The scaling of the components is done so that the singular values
          of the first ``N_pc`` components after the subtraction are equal to
          the target value.
        - The target value is the specified quantile of the original singular
          values of the covariance matrix.
        - If the target value is larger (i.e. less subtraction) than the
          ``N_px+1``-th singular value, it is set to that value. So after the
          subtraction, all singular values will be no bigger than of the largest
          untouched principal component.

    4.  Subtract the scaled contributions of the first ``N_pc`` principal
        components. The remaining covariance is called ``K``.

    5.  Plot the data with covariance ``K`` using :py:func:`corlines`.

    6.  From smallest to largest, add the contribution of the subtracted
        principal components again. Plot the difference to the error bars of
        the previous covariance as hatched boxes.

    7.  If requested, determine the conditional uncertainty of each data point
        and plot those as wedges from the error bars of ``K`` pointing to the
        conditional uncertainties.


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
        >>> cov = np.eye(5)
        >>> u = x[:,np.newaxis] / 4
        >>> for i in range(3):
        >>>     u[i] *= -1
        >>>     cov = cov + u@u.T
        >>> y = rng.multivariate_normal(np.zeros(5), cov)
        >>> nuplt.pcplot(x, y, cov, marker="x")

    .. plot::
        :include-source: True

        Compare number of components:

        >>> import numpy as np
        >>> from matplotlib import pyplot as plt
        >>> from nustattools import plotting as nuplt
        >>> rng = np.random.default_rng()
        >>> x = np.linspace(0, 10, 5)
        >>> cov = np.eye(5)
        >>> u = x[:,np.newaxis] / 4
        >>> for i in range(4):
        >>>     u *= 2
        >>>     u[0] *= -1
        >>>     cov = cov + u@u.T
        >>>     u = np.roll(u, i+1)
        >>> y = rng.multivariate_normal(np.zeros(5), cov)
        >>> nuplt.pcplot(x, y, cov, componentwidth=1.4, components=1, label="1 components")
        >>> nuplt.pcplot(x, y, cov, componentwidth=[(0.4,0)], components=2, label="2 components")
        >>> nuplt.pcplot(x, y, cov, componentwidth=[(0,0.4)], components=3, label="3 components")
        >>> plt.legend()

    .. plot::
        :include-source: True

        Rank deficient covariance:

        >>> import numpy as np
        >>> from matplotlib import pyplot as plt
        >>> from nustattools import plotting as nuplt
        >>> rng = np.random.default_rng()
        >>> x = np.linspace(0, 10, 5)
        >>> cov = np.eye(5)
        >>> u = x[:,np.newaxis] / 4
        >>> for i in range(3):
        >>>     u[i] *= -1
        >>>     cov = cov + u@u.T
        >>> # Matrix to project to constant sum of data points
        >>> A = np.eye(5) - np.ones((5,5)) * 1/5
        >>> cov = A @ cov @ A.T
        >>> y = rng.multivariate_normal(np.zeros(5), cov)
        >>> nuplt.pcplot(x, y, cov)


    See also
    --------

    corlines : Plotting function for the remaining covariance

    """

    if not drawcorlines:
        kwargs.update({"corlinestyle": "", "cormarker": ""})

    yerr = np.sqrt(np.diag(ycov))
    yerr_safe = np.where(yerr > 0, yerr, 1e-12)
    if normalize:
        ycor = ycov / yerr_safe[:, np.newaxis] / yerr_safe[np.newaxis, :]
        yerrscale = yerr
    else:
        ycor = ycov
        yerrscale = 1.0

    # Conditional errors, i.e. if all other components are fixed
    # Make sure ycov is invertible by inflating the diagonal elements a tiny bit
    ycov_diag = np.diag(ycov)
    ycov_diag = np.where(ycov_diag == 0, np.max(ycov_diag), ycov_diag)
    ycov_safe = ycov + np.diag(ycov_diag) * 1e-12
    ycovinv = np.linalg.inv(ycov_safe)
    yconderr = 1 / np.sqrt(np.diag(ycovinv))

    # Get principal components
    q, d, _ = np.linalg.svd(ycor)

    # Initialize with default hatch styles if not provided
    if hatch is None:
        hatch = [("/" * 5, "\\" * 3), ("O" * 3, "." * 3), ("X" * 3, "+" * 3)]

    if isinstance(components, float):
        # Find index to cover specified fraction of total covariance
        D = np.cumsum(d)
        D = D / D[-1]
        n_comp = np.searchsorted(D, components) + 1
    else:
        n_comp = int(components)
    n_comp = max(n_comp, 1)

    if n_comp > len(hatch):
        m = (
            f"Requested {n_comp} principal components, but only {len(hatch)} "
            f"hatch styles are defined. Showing only {len(hatch)} components."
        )

        warn(m, RuntimeWarning, stacklevel=2)
        n_comp = len(hatch)

    # Scale the removed components
    if n_comp == len(d):
        target_covariance = 0
    else:
        target_covariance = np.quantile(d, target_quantile)
        # Make sure we are at least targeting the size of the first
        # untouched component
        target_covariance = min(d[n_comp], target_covariance)

    scaling_factors = np.sqrt(1 - target_covariance / d[:n_comp])

    K = ycov
    Us = []
    for i, s in enumerate(scaling_factors):
        u = q[:, i] * yerrscale * s * np.sqrt(d[i])

        # Remember covariance contributions of removed components
        Us.append(u[:, np.newaxis] @ u[np.newaxis, :])

        # Remove scaled components
        K = K - Us[-1]

    # Avoid numerical problems when all components are requested
    if n_comp == len(K):
        K = np.zeros_like(K)

    if ax is None:
        ax = plt.gca()

    if componentwidth is None:
        # Try to guess a reasonable width from the data
        cw = min(np.min(np.diff(x)) * 0.9, (np.max(x) - np.min(x)) / 15)  # type: ignore[operator]
        componentwidth = itertools.cycle([cw])

    # Plot error bars with correlation lines
    bars = corlines(x, y, K, ax=ax, **kwargs)
    color = bars.lines[0].get_color()
    zorder = bars.lines[0].zorder

    # Plot principal components
    Kerr = np.sqrt(np.diag(K))
    inner_err = Kerr
    for j in reversed(range(n_comp)):
        xx: list[float] = []
        yy: list[float] = []
        e_min: list[float] = []
        e_max: list[float] = []
        fill: list[bool] = []

        # Outer error band = Inner error band + component contribution
        outer_err = np.sqrt(inner_err**2 + np.diag(Us[j]))

        try:
            cw_cycle = itertools.cycle(componentwidth)
        except TypeError:
            cw_cycle = itertools.cycle([componentwidth])

        for i, (xs, ys, cw) in enumerate(zip(x, y, cw_cycle)):
            try:
                dxm = cw[0]
                dxp = cw[1]
            except (IndexError, TypeError):
                dxm = cw / 2
                dxp = cw / 2
            su = np.sign(q[:, j][i])
            su = 1 if su == 0 else su
            emin = inner_err[i] * su
            emax = outer_err[i] * su
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

        hatch_pair = hatch[j]

        # Draw component
        if label_components:
            comp_label = f"Principal component #{j+1}"
        else:
            comp_label = None

        # Make sure we have hatch styles before accessing them
        if hatch is not None:
            hatch_pair = hatch[j]

            ax.fill_between(
                xx_arr,
                yy_arr + e_min_arr,
                yy_arr + e_max_arr,
                where=fill_arr,
                hatch=hatch_pair[0],
                facecolor="none",
                edgecolor=color,
                zorder=zorder,
                label=comp_label,
            )
            ax.fill_between(
                xx_arr,
                yy_arr - e_min_arr,
                yy_arr - e_max_arr,
                where=fill_arr,
                hatch=hatch_pair[1],
                facecolor="none",
                edgecolor=color,
                zorder=zorder,
            )

        inner_err = outer_err

    if drawconditional:
        # Draw conditional probabilities
        yb = y + Kerr
        yd = -(Kerr - yconderr)
        tri_col_pos = wedgeplot(
            x, yb, yd, wedgewidth=componentwidth, closed=True, zorder=zorder
        )
        yb = y - Kerr
        yd = Kerr - yconderr
        tri_col_neg = wedgeplot(
            x, yb, yd, wedgewidth=componentwidth, closed=True, zorder=zorder
        )

        tri_col_pos.set_linewidth(1)
        tri_col_pos.set_color(color)
        tri_col_pos.set_facecolor("none")
        tri_col_neg.set_linewidth(1)
        tri_col_neg.set_color(color)
        tri_col_neg.set_facecolor("none")

    if return_dict is not None:
        return_dict.update(
            {
                "K": K,
                "q": q,
                "yconderr": yconderr,
                "d": d,
                "n_comp": n_comp,
            }
        )

    return bars


__all__ = ["corlines", "pcplot", "wedgeplot"]
