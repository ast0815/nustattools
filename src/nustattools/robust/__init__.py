"""
Copyright (c) 2024 Lukas Koch. All rights reserved.

Tools for robust statistical treatments, e.g. when dealing with unknown
covariances.

"""

from __future__ import annotations

from ._derate import derate_covariance

__all__ = ["derate_covariance"]
