from __future__ import annotations

import importlib.metadata

import nustattools as m


def test_version():
    assert importlib.metadata.version("nustattools") == m.__version__
