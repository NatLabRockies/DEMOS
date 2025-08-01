# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os, sys
sys.path.insert(0, os.path.abspath("../../"))
sys.path.insert(0, os.path.abspath("../../demos"))

# Mock imports for dependencies that might not be available
autodoc_mock_imports = [
    'pydantic',
    'templates'
]

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'demos'
copyright = '2025, National Renewable Energy Laboratory'
author = 'National Renewable Energy Laboratory'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx_autodoc_typehints',
    'myst_parser',
    'sphinx.ext.viewcode',
    'sphinx_multiversion'
]

templates_path = ['_templates']
exclude_patterns = []

autodoc_typehints = 'description'
napoleon_google_docstring = True
napoleon_numpy_docstring  = True

autodoc_inherit_docstrings = True
autodoc_preserve_defaults = True

# Autodoc settings
autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'show-inheritance': True,
    'special-members': '__init__',
}


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
# html_theme = 'alabaster'
html_static_path = ['_static']

# -- MyST settings -----------------------------------------------------------
myst_enable_extensions = [
    "deflist",
    "html_admonition",
    "html_image",
]

# Sphinx multiversion config
smv_branch_whitelist = r'^(main|dev|yep/.*)$'
# Build only tags matching x.x.x (digits or 1–3 letters per segment):
smv_tag_whitelist = r'^(?:\d+|[a-z]{1,3})\.(?:\d+|[a-z]{1,3})\.(?:\d+|[a-z]{1,3})$'
