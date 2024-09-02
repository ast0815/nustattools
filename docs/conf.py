from __future__ import annotations

import importlib.metadata

from intersphinx_registry import get_intersphinx_mapping

project = "NuStatTools"
copyright = "2024, Lukas Koch"
author = "Lukas Koch"
version = release = importlib.metadata.version("nustattools")

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
    "sphinx.ext.viewcode",
]

source_suffix = [".rst", ".md"]
exclude_patterns = [
    "_build",
    "**.ipynb_checkpoints",
    "Thumbs.db",
    ".DS_Store",
    ".env",
    ".venv",
]

html_theme = "furo"

myst_enable_extensions = [
    "colon_fence",
]

intersphinx_mapping = get_intersphinx_mapping(packages={"python", "numpy", "scipy"})
# intersphinx_mapping.update({
#    'my-package' : ('<url>', None),
# })


nitpick_ignore = [
    ("py:class", "_io.StringIO"),
    ("py:class", "_io.BytesIO"),
    ("py:class", "optional"),
]

nitpick_ignore_regex = [
    ("py:class", "default=.*"),
]

always_document_param_types = True
