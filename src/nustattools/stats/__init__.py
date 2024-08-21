"""
Copyright (c) 2024 Lukas Koch. All rights reserved.

Statistical distributions that are useful, but not available in
``scipy.stats``.

"""

from __future__ import annotations

from . import _dist
from ._dist import *  # noqa: F403

# Export all exports from the sub-modules
__all__ = _dist.__all__

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
