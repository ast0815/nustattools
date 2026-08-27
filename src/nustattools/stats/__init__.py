"""
Copyright (c) 2024 Lukas Koch. All rights reserved.

Potentially useful statistical tools that are not available in ``scipy.stats``.

References
----------

.. [Koch2021] L. Koch, "Robust test statistics for data sets with missing
    correlation information," Phys. Rev. D 103, 113008 (2021),
    https://arxiv.org/abs/2102.06172

.. [Koch2024] L. Koch "Hypothesis tests and model parameter estimation on
    data sets with missing correlation information",
    https://arxiv.org/abs/2410.22333

.. [Kessy2015] Kessy, Agnan / Lewin, Alex / Strimmer, Korbinian
    "Optimal whitening and decorrelation",
    The American Statistician 2018, Vol. 72, No. 4, pp. 309-314,
    Informa UK Limited, p. 309-314, https://arxiv.org/abs/1512.00809

"""

from __future__ import annotations

# ``shrinkage`` is imported so that the ``nustattools.stats.shrinkage``
# submodule is available, while only ``shrink`` is re-exported at package level.
from . import _derate, _dist, _fmax, shrinkage  # noqa: F401
from ._derate import *
from ._dist import *
from ._fmax import *
from .shrinkage import shrink

# Export all exports from the sub-modules
__all__ = _dist.__all__ + _derate.__all__ + _fmax.__all__ + ["shrink"]

# Some extra effort, so Sphinx picks up the data docstrings
# mypy: disable-error-code=name-defined
# pylint: disable=self-assigning-variable

#: Use this instance of :class:`Bee`.
bee = bee  # noqa: PLW0127
#: Use this instance of :class:`Bee2`.
bee2 = bee2  # noqa: PLW0127
#: Use this instance of :class:`Cee`.
cee = cee  # noqa: PLW0127
#: Use this instance of :class:`Cee2`.
cee2 = cee2  # noqa: PLW0127
#: Use this instance of :class:`RVTestStatistic`.
rvteststatistic = rvteststatistic  # noqa: PLW0127
