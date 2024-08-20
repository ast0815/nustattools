"""
Copyright (c) 2024 Lukas Koch. All rights reserved.

Statistical distributions that are useful, but not available in
``scipy.stats``.

"""

from __future__ import annotations

from . import _dist
from ._dist import *  # noqa: F403

__all__ = _dist.__all__
