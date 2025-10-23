"""
Copyright (c) 2024 Lukas Koch. All rights reserved.

Potentially useful statistical tools that are not available in ``scipy.stats``.

"""

from __future__ import annotations

from . import _hinton
from ._hinton import *  # noqa: F403

# Export all exports from the sub-modules
__all__ = _hinton.__all__

# Some extra effort, so Sphinx picks up the data docstrings
# mypy: disable-error-code=name-defined
# pylint: disable=self-assigning-variable
