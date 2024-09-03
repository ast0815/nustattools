"""
Copyright (c) 2024 Lukas Koch. All rights reserved.

Potentially useful statistical tools that are not available in ``scipy.stats``.

"""

from __future__ import annotations

from . import _derate, _dist, _fmax
from ._derate import *  # noqa: F403
from ._dist import *  # noqa: F403
from ._fmax import *  # noqa: F403

# Export all exports from the sub-modules
__all__ = _dist.__all__ + _derate.__all__ + _fmax.__all__

# Some extra effort, so Sphinx picks up the data docstrings
# mypy: disable-error-code=name-defined
# pylint: disable=self-assigning-variable

#: Use this instance of :class:`Bee`.
bee = bee  # noqa: PLW0127, F405
#: Use this instance of :class:`Bee2`.
bee2 = bee2  # noqa: PLW0127, F405
#: Use this instance of :class:`Cee`.
cee = cee  # noqa: PLW0127, F405
#: Use this instance of :class:`Cee2`.
cee2 = cee2  # noqa: PLW0127, F405
#: Use this instance of :class:`RVTestStatistic`.
rvteststatistic = rvteststatistic  # noqa: PLW0127, F405
