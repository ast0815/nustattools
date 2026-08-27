"""
Copyright (c) 2024 Lukas Koch. All rights reserved.

Plotting tools for data with correlated uncertainties that are not available in
``matplotlib``.

References
----------

.. [Koch2026] L. Koch, "Plotting correlated data,"
    https://arxiv.org/abs/2601.20805

.. [Hinton1991] G. E. Hinton and T. Shallice, "Lesioning an Attractor Network:
    Investigations of Acquired Dyslexia," Psychological Review 98, no. 1, 74 (1991),
    https://doi.org/10.1037/0033-295x.98.1.74

"""

from __future__ import annotations

from . import _corplot, _hinton
from ._corplot import *  # noqa: F403
from ._hinton import *  # noqa: F403

# Export all exports from the sub-modules
__all__ = _hinton.__all__ + _corplot.__all__

# Some extra effort, so Sphinx picks up the data docstrings
# mypy: disable-error-code=name-defined
# pylint: disable=self-assigning-variable
