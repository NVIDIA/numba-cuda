# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Numba CUDA"
copyright = "2012-2024 Anaconda Inc. 2024, NVIDIA Corporation."
author = "NVIDIA Corporation"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ["numpydoc", "sphinx.ext.intersphinx", "sphinx.ext.autodoc"]

templates_path = ["_templates"]
exclude_patterns = []

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "llvmlite": ("https://llvmlite.readthedocs.io/en/latest/", None),
    "numba": ("https://numba.readthedocs.io/en/latest/", None),
}

# To prevent autosummary warnings
numpydoc_show_class_members = False

autodoc_typehints = "none"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

try:
    import nvidia_sphinx_theme  # noqa: F401

    html_theme = "nvidia_sphinx_theme"
except ImportError:
    html_theme = "sphinx_rtd_theme"

html_static_path = ["_static"]
html_favicon = "_static/numba-green-icon-rgb.svg"
html_show_sphinx = False
